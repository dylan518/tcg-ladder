#!/usr/bin/env python3
"""
Single-turn embeddings of Sender turns + a fixed VALIDITY benchmark. An embedding sees ONE turn only (its goal and
moves); validity = how well distances between turn vectors predict known relations.

Embeddings
  triplet   3-D ML embedding from LLM odd-one-out / 2-choice judgments (strategy_embed.py)
  rel_text  text-embedding-3-large of a GOAL-RELATIVE rewrite: every tile is named by where it lies from the goal tile,
            relative to the goal direction (toward / toward-right / right / away-right / away / ... , near/far), so turns
            with different goals are comparable. Deterministic coordinate change, no strategy coding.
  abs_text  text-embedding-3-large of the raw readable moves + goal
  rel_bow   bag of goal-relative step tokens (TF-IDF), no API

Validity (higher = better)
  next        same person, their NEXT turn vs a same-goal decoy: P(d(anchor, next) < d(anchor, decoy))  [LLM judge 73%]
  same        same person, a random other turn vs decoy                                          [judge 57%]
  partner_out current partner's nearest turn vs other-group stranger with the same goal           [judge 55%]
  partner_in  current partner vs same-group non-partner with the same goal                        [judge 52-55%]
  succ_pair   same-goal SUCCESS/FAIL pairs: pick the turn more similar to its Receiver's own most recent Sender turn
                                                                                                  [judge, no context 58%]
  pair_r      pairing level: Spearman r(partner similarity minus same-goal-stranger similarity, pairing success rate)
Usage: python3 analysis/models/turn_embed.py
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align_triplets as A  # noqa: E402
import common as C  # noqa: E402
import judge_tasks as J  # noqa: E402
import readable as RD  # noqa: E402

OUT = os.path.join(HERE, 'out')
REL8 = ['toward', 'toward-right', 'right', 'away-right', 'away', 'away-left', 'left', 'toward-left']


def rel_name(tile, goal, gdir):
    if tile == goal:
        return 'GOAL'
    (x, y), (gx, gy) = RD.XY[tile], RD.XY[goal]
    dx, dy = x - gx, y - gy
    ang = math.degrees(math.atan2(dx, dy)) % 360  # clockwise from up
    rel = (ang - 45 * C.DIRS.index(gdir)) % 360
    return f"{REL8[int(round(rel / 45)) % 8]}-{'near' if max(abs(dx), abs(dy)) == 1 else 'far'}"


def rel_text(e):
    g, d = e['goal_tile'], e['goal_dir']
    out = []
    for s in RD.steps_from_tokens(e['s_tokens'])['steps']:
        t = [rel_name(x, g, d) for x in s['tiles']]
        if s['type'] == 'go_to':
            out.append(f'went to {t[0]}')
        elif s['type'] == 'wiggle':
            out.append(f"wiggled {t[0]}<->{t[1]} x{s['repeats']}")
        elif s['type'] == 'loop':
            out.append(f"looped {'->'.join(t)} x{s['repeats']}")
        elif s['type'] == 'pause':
            out.append(('long pause' if s['repeats'] >= 2 else 'paused') + f' on {t[0]}')
    out.append('time ran out' if e['s_tokens'][-1] == 'TIMEOUT' else 'pressed done')
    return '; '.join(out)


def embed_openai(texts, cache_name):
    path = os.path.join(OUT, cache_name)
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if list(d['texts']) == texts:
            return d['X']
    import openai
    client = openai.OpenAI(max_retries=6, timeout=60)
    X = []
    for i in range(0, len(texts), 100):
        r = client.embeddings.create(model='text-embedding-3-large', input=texts[i:i + 100])
        X += [x.embedding for x in r.data]
    X = np.array(X)
    np.savez(path, texts=np.array(texts, dtype=object), X=X)
    return X


def bow(texts):
    toks = [[w for w in t.replace(';', ' ').split() if not w.startswith('x')] for t in texts]
    vocab = sorted({w for t in toks for w in t})
    ix = {w: i for i, w in enumerate(vocab)}
    M = np.zeros((len(texts), len(vocab)))
    for r, t in enumerate(toks):
        for w in t:
            M[r, ix[w]] += 1
    df = (M > 0).sum(0)
    return M * np.log(len(texts) / df)


def validity(ids, X, T, tasks, rec_prev, pairings):
    ix = {k: i for i, k in enumerate(ids)}
    Z = X - X.mean(0)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sim = lambda a, b: float(Z[ix[a]] @ Z[ix[b]])
    res = {}
    for t in ('next', 'same', 'partner_out', 'partner_in'):
        qs = tasks[t]
        res[t] = float(np.mean([sim(q['anchor'], q['right']) > sim(q['anchor'], q['wrong']) for q in qs]))
    ok = []
    for q in tasks['success']:
        a, b = q['right'], q['wrong']
        if rec_prev.get(a) and rec_prev.get(b):
            sa, sb = sim(a, rec_prev[a]), sim(b, rec_prev[b])
            ok.append(1.0 if sa > sb else 0.5 if sa == sb else 0.0)
    res['succ_pair'] = float(np.mean(ok))
    xs, ys = [], []
    for (run, pr), d in pairings.items():
        cross = [sim(a, b) for a in d['A'] for b in d['B']]
        strangers = [sim(a, s) for a in d['A'] + d['B'] for s in d['strangers'].get(a, [])]
        xs.append(np.mean(cross) - np.mean(strangers)); ys.append(d['succ'])
    rx, ry = np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys))
    res['pair_r'] = float(np.corrcoef(rx, ry)[0, 1])
    res['pair_n'] = len(xs)
    return res


def main():
    T, _, _ = A.build()
    ids = sorted(T)
    same, succ, group, extra = J.build(T)
    tasks = {'same': same, 'success': succ, **extra}
    # Receiver's own most recent Sender turn before this round (any partner)
    rec_prev = {}
    for k, e in T.items():
        own = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['r'] and T[x]['round'] < e['round']]
        if own:
            rec_prev[k] = max(own, key=lambda x: T[x]['round'])
    # pairings: both directions of each (run, unordered pair) within a phase
    bygoal = defaultdict(list)
    for k, e in T.items():
        bygoal[(e['goal_tile'], e['goal_dir'])].append(k)
    pairings = {}
    for k, e in T.items():
        key = (e['run'], tuple(sorted([e['s'], e['r']])) + (e['phase'],))
        d = pairings.setdefault(key, {'A': [], 'B': [], 'succ_l': [], 'strangers': {}})
        (d['A'] if e['s'] == key[1][0] else d['B']).append(k)
        d['succ_l'].append(e['succ'])
        d['strangers'][k] = [x for x in bygoal[(e['goal_tile'], e['goal_dir'])] if T[x]['run'] != e['run']]
    pairings = {k: {**d, 'succ': float(np.mean(d['succ_l']))} for k, d in pairings.items() if d['A'] and d['B']}

    rel = [rel_text(T[k]) for k in ids]
    absx = [f"goal: {T[k]['goal_tile']}, pointing {T[k]['goal_dir']}; moves: " + A.F.moves(T[k]['s_tokens']) for k in ids]
    print('example goal-relative rewrite:\n  ', absx[0], '\n  ->', rel[0], '\n')
    E = {}
    d = np.load(os.path.join(OUT, 'strategy_embedding.npz'))
    tri = {k: x for k, x in zip(d['ids'], d['X'])}
    E['triplet (LLM judgments, 3-D)'] = np.array([tri[k] for k in ids])
    E['rel_text (text-embedding-3-large, goal-relative)'] = embed_openai(rel, 'turn_embed_rel.npz')
    E['abs_text (text-embedding-3-large, raw moves)'] = embed_openai(absx, 'turn_embed_abs.npz')
    E['rel_bow (goal-relative step counts, no API)'] = bow(rel)
    rows = {}
    print(f"{'embedding':52s} next  same  partner_out partner_in succ_pair pair_r(n={len(pairings)})")
    print(f"{'LLM judge (pairwise prompts, for reference)':52s} 73%   57%   55%         52-55%     58% (no ctx) -")
    for name, X in E.items():
        r = validity(ids, X, T, tasks, rec_prev, pairings)
        rows[name] = r
        print(f"{name:52s} {r['next']:.0%}   {r['same']:.0%}   {r['partner_out']:.0%}         {r['partner_in']:.0%}        {r['succ_pair']:.0%}       {r['pair_r']:+.2f}")
    json.dump(rows, open(os.path.join(OUT, 'turn_embed_validity.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
