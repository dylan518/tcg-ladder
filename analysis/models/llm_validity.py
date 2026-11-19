#!/usr/bin/env python3
"""
Validity of LLM-predicted Sender turns, measured in the LLM-judgment strategy embedding (strategy_embed.py).

Each predicted turn (fullgame_sonnet per-turn run: the model saw the player's real game so far and wrote the moves)
is PLACED in the fixed embedding from k odd-one-out judgments (same judge prompt) against random embedded turns.
Everything is compared in goal-removed space (vector minus the mean of its goal's 8 real turns).

Predictors of a player's actual turn, scored by distance to it (lower = better) and by 2AFC vs a same-goal decoy:
  llm      the model's predicted turn (placed)
  repeat   the player's own previous Sender turn
  drift    a + b_own * own previous + b_partner * partner's latest (fitted leave-one-run-out)
  mean     the run-independent population mean (0 in goal-removed space)
Placement noise check: real turns placed the same way vs their fitted vectors.

Usage: python3 analysis/models/llm_validity.py [--preds out/fullgame_sonnet_per_turn_gpt-6.1-terra_5.jsonl] [--k 12]
"""
import argparse
import hashlib
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
import readable as RD  # noqa: E402
import strategy_embed as E  # noqa: E402

OUT = os.path.join(HERE, 'out')


def item_text(goal_tile, goal_dir, tokens):
    return f"goal: {goal_tile}, pointing {goal_dir}\nmoves: {F.moves(tokens)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default=os.path.join(OUT, 'fullgame_sonnet_per_turn_gpt-6.1-terra_5.jsonl'))
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--n_check', type=int, default=40)
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    import openai
    client = openai.OpenAI(max_retries=6, timeout=60)
    usage, lock = {}, threading.Lock()
    T, _, _ = A.build()
    d = np.load(os.path.join(OUT, 'strategy_embedding.npz'))
    ids = list(d['ids']); X = d['X']; dim = int(d['dim'])
    ix = {k: i for i, k in enumerate(ids)}
    games, names = F.load_players()
    inv = {(run, al): p for run, m in names.items() for p, al in m.items()}

    # ---- new items: LLM-predicted sender turns
    new = []  # (name, text, true turn id)
    for r in map(json.loads, open(a.preds)):
        if r['role'] != 'sender' or not r['out']:
            continue
        tid = f"{r['run']}:{r['round']}:{inv[(r['run'], r['player'])]}"
        e = T[tid]
        new.append((f'llm:{tid}', item_text(e['goal_tile'], e['goal_dir'], RD.realize(r['out'])), tid))
    rng = random.Random(0)
    check = rng.sample(ids, a.n_check)
    for tid in check:
        new.append((f'real:{tid}', A.item(T[tid]), tid))
    tag = os.path.basename(a.preds).replace('.jsonl', '')
    cache_path = os.path.join(OUT, f'llm_validity_cache.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']
    system, question = E.QUESTIONS['strategy']
    jobs = []
    for nm, text, tid in new:
        rr = random.Random(nm)
        for j in range(a.k):
            others = rr.sample([x for x in ids if x != tid], 2)
            pos = rr.randrange(3)
            jobs.append((nm, text, others, pos))

    def one(job):
        nm, text, others, pos = job
        texts = [A.item(T[o]) for o in others]
        texts.insert(pos, text)
        key = f"place:{nm}:{others[0]}:{others[1]}:{pos}:{hashlib.md5(text.encode()).hexdigest()[:10]}"
        if key not in cache:
            user = '\n\n'.join(f"Turn {i + 1}:\n{t}" for i, t in enumerate(texts)) + '\n\n' + question
            out = F.ask_openai(client, E.MODEL, system, user, A.ODD_SCHEMA, usage, lock)
            with lock:
                cache[key] = out
                with open(cache_path, 'a') as f:
                    f.write(json.dumps({'key': key, 'out': out}) + '\n')
        return cache[key]
    with ThreadPoolExecutor(a.workers) as ex:
        outs = list(ex.map(one, jobs))

    # ---- place each new item: X fixed, fit its vector from its odd-one-out constraints
    n0 = len(ids)
    names_new = [nm for nm, _, _ in new]
    nix = {nm: n0 + i for i, nm in enumerate(names_new)}
    odd = []
    for (nm, text, others, pos), out in zip(jobs, outs):
        if not out:
            continue
        slots = [ix[o] for o in others]
        slots.insert(pos, nix[nm])
        o = int(out['odd_one_out']) - 1
        rest = [slots[j] for j in range(3) if j != o]
        odd.append((rest[0], rest[1], slots[o]))
    odd = np.array(odd, int)
    Xa = np.vstack([X, np.random.default_rng(0).normal(0, 0.3, (len(new), dim))])
    Xa = E.fit_X(len(Xa), dim, odd, np.zeros((0, 3), int), lam=0.01, steps=600, X0=Xa, free=np.arange(n0, len(Xa)))
    V = {nm: Xa[nix[nm]] for nm in names_new}

    # ---- goal-removed space
    goal = defaultdict(list)
    for k, e in T.items():
        goal[(e['goal_tile'], e['goal_dir'])].append(k)
    gm = {g: np.mean([X[ix[k]] for k in ks], 0) for g, ks in goal.items()}
    res_of = lambda k: X[ix[k]] - gm[(T[k]['goal_tile'], T[k]['goal_dir'])]
    gr = lambda v, k: v - gm[(T[k]['goal_tile'], T[k]['goal_dir'])]

    # placement noise
    pe = [np.linalg.norm(V[f'real:{t}'] - X[ix[t]]) for t in check]
    typical = np.mean([np.linalg.norm(res_of(x) - res_of(y)) for x in ids[:60] for y in ids[60:120]])
    print(f'placement check: real turns placed from {a.k} judgments land {np.mean(pe):.2f} from their fitted vectors '
          f'(typical distance between two random turns {typical:.2f})')

    # ---- drift model, leave-one-run-out
    def history(k):
        e = T[k]
        mine = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['s'] and T[x]['round'] < e['round']], key=lambda x: T[x]['round'])
        part = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['r'] and T[x]['round'] < e['round']], key=lambda x: T[x]['round'])
        return (mine[-1] if mine else None), (part[-1] if part else None)
    rows = []
    for nm, _, tid in new:
        if not nm.startswith('llm:'):
            continue
        own, part = history(tid)
        if own is None or part is None:
            continue
        rows.append(dict(tid=tid, run=T[tid]['run'], phase=T[tid]['phase'], y=res_of(tid), llm=gr(V[nm], tid), own=res_of(own), part=res_of(part)))

    def fit_drift(train):
        Y = np.concatenate([r['y'] for r in train])
        M = np.column_stack([np.concatenate([r['own'] for r in train]), np.concatenate([r['part'] for r in train]), np.ones(len(Y))])
        return np.linalg.lstsq(M, Y, rcond=None)[0]
    for run in sorted({r['run'] for r in rows}):
        b = fit_drift([r for r in rows if r['run'] != run])
        for r in rows:
            if r['run'] == run:
                r['drift'] = b[0] * r['own'] + b[1] * r['part'] + b[2]
    for r in rows:
        r['repeat'] = r['own']; r['mean'] = np.zeros(dim)

    # decoys: a same-goal real turn by someone else (goal-removed)
    for r in rows:
        e = T[r['tid']]
        dec = [x for x in goal[(e['goal_tile'], e['goal_dir'])] if not (T[x]['run'] == e['run'] and T[x]['s'] == e['s'])]
        r['decoy'] = res_of(random.Random(r['tid']).choice(dec))

    print(f"\nPredicting {len(rows)} real Sender turns ({tag}); goal-removed strategy embedding")
    print(f"{'predictor':10s} mean distance to actual   closer to actual than to a same-goal decoy   by phase (distance)")
    out = {}
    for p in ('llm', 'repeat', 'drift', 'mean'):
        dist = np.array([np.linalg.norm(r[p] - r['y']) for r in rows])
        afc = np.mean([np.linalg.norm(r[p] - r['y']) < np.linalg.norm(r[p] - r['decoy']) for r in rows])
        byph = {ph: float(np.mean([np.linalg.norm(r[p] - r['y']) for r in rows if r['phase'] == ph])) for ph in (1, 2, 3)}
        out[p] = {'dist': float(dist.mean()), 'afc': float(afc), 'by_phase': byph}
        bs = [dist[np.random.default_rng(i).integers(0, len(dist), len(dist))].mean() for i in range(1000)]
        print(f"{p:10s} {dist.mean():.2f} [{np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}]          {afc:.0%}"
              f"                                          " + ' '.join(f'{ph}:{v:.2f}' for ph, v in byph.items()))
    # paired comparison llm vs others
    for p in ('repeat', 'drift'):
        diff = np.array([np.linalg.norm(r[p] - r['y']) - np.linalg.norm(r['llm'] - r['y']) for r in rows])
        print(f"  llm closer than {p}: {np.mean(diff > 0):.0%} of turns, mean advantage {diff.mean():+.2f}")
    json.dump(out, open(os.path.join(OUT, f'llm_validity_{tag}.json'), 'w'), indent=1)
    if usage:
        print(f"~${(usage.get('input_tokens', 0) * 1 + usage.get('cache_read_input_tokens', 0) * 0.05 + usage.get('output_tokens', 0) * 5) / 1e6:.2f}")


if __name__ == '__main__':
    main()
