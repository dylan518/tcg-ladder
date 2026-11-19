#!/usr/bin/env python3
"""
Strategy embedding of Sender turns from LLM similarity judgments (align_triplets.py judge, same prompt/cache).

  collect   odd-one-out triplets over all Sender turns, balanced coverage (each turn appears about equally often),
            a third of them goal-matched (all three turns share the exact goal). Cached; re-runs never re-bill.
  fit       embedding X (one vector per turn) by maximum likelihood with squared distances:
              odd-one-out (i,j,k), odd=k:  P = exp(-d_ij) / (exp(-d_ij) + exp(-d_ik) + exp(-d_jk))
              2-choice probe (anchor a, c1, c2), chose c1:  P = exp(-d_a1) / (exp(-d_a1) + exp(-d_a2))
            dimension chosen by held-out judgment accuracy (judge self-consistency is the ceiling).
  place     a NEW turn: keep X fixed and fit only its vector from k comparisons against embedded turns.
            Validated by removing a turn's judgments, placing it from k of them, scoring its remaining ones.

Usage: python3 analysis/models/strategy_embed.py collect [--n 2500]
       python3 analysis/models/strategy_embed.py fit [--dims 2,3,4,6,8]
Output: out/strategy_embedding.npz (ids, X, dim), out/strategy_embedding.json (scores)
"""
import argparse
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align_triplets as A  # noqa: E402
import fullgame_sonnet as F  # noqa: E402

OUT = os.path.join(HERE, 'out')
MODEL = 'gpt-6.1-terra'
BASE = A.JUDGE.split('\n\nYou will compare')[0]
# one triplet question per embedding; every item is ONE round (goal + Sender moves)
QUESTIONS = {
    'strategy': (A.JUDGE, "Which turn's signalling strategy is the odd one out (the other two are most alike)?"),
    'strategy_noex': (A.JUDGE.replace("\n(e.g. go to the goal and wiggle toward the goal direction, wiggle away from it, mark the direction from the centre,\njust go to the goal, pause patterns, how many repeats)", ""),
                      "Which turn's signalling strategy is the odd one out (the other two are most alike)?"),
    'person': (BASE + "\n\nEach turn below is one round played by one person. People have personal signalling habits.",
               "Two of these three turns are most likely played by the SAME PERSON. Which turn is the odd one out (most likely someone else)?"),
    'partner': (BASE + "\n\nEach turn below is one round played by one person. Players were paired up and alternated Sender and "
                "Receiver, and partners pick up each other's way of signalling.",
                "Two of these three turns are most likely played by PARTNERS (two people who played together). Which turn is the odd one out?"),
    'success': (BASE + "\n\nEach turn below is one round played by one person. Imagine pairing up the people behind these turns: "
                "a pair succeeds when the Receiver reads the Sender's signal the way the Sender meant it, which needs the two to share "
                "the same way of showing the goal direction.",
                "Which two of these people would most likely SUCCEED together as partners? Answer with the odd one out (the third person)."),
}


def sample_triplets(T, n, seed=0):
    rng = random.Random(seed)
    ids = sorted(T)
    bygoal = defaultdict(list)
    for k, e in T.items():
        bygoal[(e['goal_tile'], e['goal_dir'])].append(k)
    out, pool = [], []
    while len(out) < n:
        if len(out) % 3 == 2:  # goal-matched
            out.append(rng.sample(bygoal[rng.choice(list(bygoal))], 3))
            continue
        if len(pool) < 3:
            pool = ids[:]; rng.shuffle(pool)
        trip = [pool.pop(), pool.pop(), pool.pop()]
        out.append(trip)
    return out


def cache_file(q):
    return os.path.join(OUT, f'align_cache_{MODEL}.jsonl' if q == 'strategy' else f'align_cache_{MODEL}_{q}.jsonl')


def collect(a):
    import openai
    system, question = QUESTIONS[a.question]
    T, _, _ = A.build()
    trips = sample_triplets(T, a.n)
    client = openai.OpenAI(max_retries=6, timeout=60)
    usage, lock = {}, threading.Lock()
    cache_path = cache_file(a.question)
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']

    def one(its):
        key = f"odd:{':'.join(its)}"
        if key not in cache:
            user = '\n\n'.join(f"Turn {i + 1}:\n{A.item(T[x])}" for i, x in enumerate(its)) + '\n\n' + question
            out = F.ask_openai(client, MODEL, system, user, A.ODD_SCHEMA, usage, lock)
            with lock:
                cache[key] = out
                with open(cache_path, 'a') as f:
                    f.write(json.dumps({'key': key, 'out': out}) + '\n')
        return cache[key]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, trips))
    print(f'{sum(r is not None for r in res)}/{len(res)} triplets judged')
    if usage:
        print(f"~${(usage.get('input_tokens', 0) * 1 + usage.get('cache_read_input_tokens', 0) * 0.05 + usage.get('output_tokens', 0) * 5) / 1e6:.2f}")


def load_judgments(q='strategy'):
    """-> ids, odd list [(i, j, k_odd)], two-choice list [(anchor, chosen, other)] as index triples"""
    T, _, _ = A.build()
    ids = sorted(T)
    ix = {k: n for n, k in enumerate(ids)}
    odd, two = [], []
    for l in open(cache_file(q)):
        r = json.loads(l)
        if not r['out']:
            continue
        parts = r['key'].split(':')
        if parts[0] == 'odd':
            its = [':'.join(parts[1 + 3 * j: 4 + 3 * j]) for j in range(3)]
            o = int(r['out']['odd_one_out']) - 1
            rest = [its[j] for j in range(3) if j != o]
            odd.append((ix[rest[0]], ix[rest[1]], ix[its[o]]))
        elif parts[0] == 'probe':
            an, c1, c2 = (':'.join(parts[1 + 3 * j: 4 + 3 * j]) for j in range(3))
            ch, ot = (c1, c2) if r['out']['more_similar'] == 'A' else (c2, c1)
            two.append((ix[an], ix[ch], ix[ot]))
    return ids, np.array(odd, dtype=int).reshape(-1, 3), np.array(two, dtype=int).reshape(-1, 3)


def nll_grad(X, odd, two, lam):
    """negative log-likelihood and gradient. odd rows (i, j, k): i,j were judged most alike. two rows (a, c, o): c chosen."""
    G = np.zeros_like(X)
    tot = lam * (X ** 2).sum()
    G += 2 * lam * X
    if len(odd):
        i, j, k = odd.T
        dij = ((X[i] - X[j]) ** 2).sum(1); dik = ((X[i] - X[k]) ** 2).sum(1); djk = ((X[j] - X[k]) ** 2).sum(1)
        L = np.stack([-dij, -dik, -djk], 1)
        m = L.max(1, keepdims=True)
        P = np.exp(L - m); P /= P.sum(1, keepdims=True)
        tot += -(np.log(P[:, 0] + 1e-12)).sum()
        # d nll / d d_xy = (1 - P0) for dij, -P1 for dik, -P2 for djk  (since nll = d_ij + logsumexp(-d))
        wij, wik, wjk = 1 - P[:, 0], -P[:, 1], -P[:, 2]
        for (a_, b_, w) in ((i, j, wij), (i, k, wik), (j, k, wjk)):
            diff = 2 * (X[a_] - X[b_]) * w[:, None]
            np.add.at(G, a_, diff); np.add.at(G, b_, -diff)
    if len(two):
        a_, c, o = two.T
        dac = ((X[a_] - X[c]) ** 2).sum(1); dao = ((X[a_] - X[o]) ** 2).sum(1)
        z = dac - dao  # nll = log(1 + exp(z))
        tot += np.logaddexp(0, z).sum()
        s = 1 / (1 + np.exp(-z))
        for (p, q, w) in ((a_, c, s), (a_, o, -s)):
            diff = 2 * (X[p] - X[q]) * w[:, None]
            np.add.at(G, p, diff); np.add.at(G, q, -diff)
    return tot, G


def fit_X(n, dim, odd, two, lam=0.01, steps=3000, lr=0.05, seed=0, X0=None, free=None):
    """Adam on X; if free is given (index array), only those rows are updated (placing new turns)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.3, (n, dim)) if X0 is None else X0.copy()
    m = np.zeros_like(X); v = np.zeros_like(X)
    for t in range(1, steps + 1):
        _, G = nll_grad(X, odd, two, lam)
        if free is not None:
            mask = np.zeros(len(X), bool); mask[free] = True; G[~mask] = 0
        m = 0.9 * m + 0.1 * G; v = 0.999 * v + 0.001 * G ** 2
        X -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return X


def acc(X, odd, two):
    out = {}
    if len(odd):
        i, j, k = odd.T
        dij = ((X[i] - X[j]) ** 2).sum(1); dik = ((X[i] - X[k]) ** 2).sum(1); djk = ((X[j] - X[k]) ** 2).sum(1)
        out['odd'] = float(np.mean((dij < dik) & (dij < djk)))
    if len(two):
        a_, c, o = two.T
        out['two'] = float(np.mean(((X[a_] - X[c]) ** 2).sum(1) < ((X[a_] - X[o]) ** 2).sum(1)))
    return out


def fit(a):
    ids, odd, two = load_judgments(a.question)
    n = len(ids)
    print(f'{n} turns, {len(odd)} odd-one-out, {len(two)} two-choice judgments')
    rng = np.random.default_rng(0)
    po, pt = rng.permutation(len(odd)), rng.permutation(len(two))
    cut_o, cut_t = len(odd) // 5, len(two) // 5
    res = {'n_odd': len(odd), 'n_two': len(two), 'cv': {}}
    for dim in [int(d) for d in a.dims.split(',')]:
        X = fit_X(n, dim, odd[po[cut_o:]], two[pt[cut_t:]], lam=a.lam)
        res['cv'][dim] = {'train': acc(X, odd[po[cut_o:]], two[pt[cut_t:]]), 'held_out': acc(X, odd[po[:cut_o]], two[pt[:cut_t]])}
        print(f"dim {dim}: held-out odd-one-out {res['cv'][dim]['held_out']['odd']:.0%} (chance 33%), "
              f"2-choice {res['cv'][dim]['held_out'].get('two', float('nan')):.0%} (chance 50%) | train {res['cv'][dim]['train']['odd']:.0%}", flush=True)
    best = max(res['cv'], key=lambda d: res['cv'][d]['held_out']['odd'] + res['cv'][d]['held_out'].get('two', 0))
    X = fit_X(n, best, odd, two, lam=a.lam)
    res['dim'] = best

    # placement from a subsample: hide every judgment involving item q, refit others? (costly) -> keep X fixed, re-place q
    place = defaultdict(list)
    qs = rng.choice(n, size=min(60, n), replace=False)
    for q in qs:
        oq = odd[(odd == q).any(1)]; tq = two[(two == q).any(1)]
        allq = [('o', r) for r in oq] + [('t', r) for r in tq]
        if len(allq) < 25:
            continue
        perm = rng.permutation(len(allq))
        test = [allq[p] for p in perm[:10]]
        for k in (5, 10, 20):
            train = [allq[p] for p in perm[10:10 + k]]
            X0 = X.copy(); X0[q] = rng.normal(0, 0.3, best)
            to = np.array([r for t, r in train if t == 'o'], int).reshape(-1, 3); tt = np.array([r for t, r in train if t == 't'], int).reshape(-1, 3)
            Xq = fit_X(n, best, to, tt, lam=a.lam, steps=400, X0=X0, free=np.array([q]))
            eo = np.array([r for t, r in test if t == 'o'], int).reshape(-1, 3); et = np.array([r for t, r in test if t == 't'], int).reshape(-1, 3)
            sc = acc(Xq, eo, et)
            place[k].append((sc.get('odd', np.nan), len(eo), sc.get('two', np.nan), len(et)))
    res['placement'] = {}
    for k, v in place.items():
        o = sum(a_ * n_ for a_, n_, _, _ in v if n_) / max(1, sum(n_ for _, n_, _, _ in v))
        t_ = sum(b_ * m_ for _, _, b_, m_ in v if m_) / max(1, sum(m_ for _, _, _, m_ in v))
        res['placement'][k] = {'odd': o, 'two': t_, 'n_items': len(v)}
        print(f'place a turn from {k} comparisons: its other judgments predicted odd-one-out {o:.0%}, 2-choice {t_:.0%} (items={len(v)})')
    full = acc(X, odd, two)
    print(f'dim {best} on all judgments: odd {full["odd"]:.0%}, 2-choice {full.get("two", float("nan")):.0%}; judge self-consistency (odd, reordered) 84%')
    tag = '' if a.question == 'strategy' else f'_{a.question}'
    np.savez(os.path.join(OUT, f'strategy_embedding{tag}.npz'), ids=np.array(ids), X=X, dim=best)
    json.dump(res, open(os.path.join(OUT, f'strategy_embedding{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['collect', 'fit'])
    ap.add_argument('--n', type=int, default=2500)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--dims', default='2,3,4,6,8')
    ap.add_argument('--lam', type=float, default=0.01)
    ap.add_argument('--question', default='strategy', choices=list(QUESTIONS))
    a = ap.parse_args()
    collect(a) if a.cmd == 'collect' else fit(a)
