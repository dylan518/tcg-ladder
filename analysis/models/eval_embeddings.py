#!/usr/bin/env python3
"""
Validity of the single-turn LLM-triplet embeddings, one per triplet question (strategy / person / partner / success;
strategy_embed.py). Every embedding places ONE round (goal + Sender moves). Same benchmark for all:

  next, same, partner_out, partner_in   identity / partner tests (judge_tasks question sets; chance 50%)
  succ_rec      AUC: is the turn closer to signals this Receiver decoded correctly before than to ones they failed?
                (any partner, earlier rounds; receiver-cluster bootstrap CI)
  succ_rec_pp   same, current partner only
  pair_r        Spearman r(pairing alignment = partner similarity minus same-goal-stranger similarity, pairing success)
Usage: python3 analysis/models/eval_embeddings.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align_triplets as A  # noqa: E402
import judge_tasks as J  # noqa: E402
import turn_embed as TE  # noqa: E402

OUT = os.path.join(HERE, 'out')


def succ_auc(ids, X, T, partner_only, rng, B=2000):
    ix = {k: i for i, k in enumerate(ids)}
    Z = X - X.mean(0); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    s = []
    for k in ids:
        e = T[k]
        prev = [x for x in T if T[x]['run'] == e['run'] and T[x]['r'] == e['r'] and T[x]['round'] < e['round']
                and (not partner_only or T[x]['s'] == e['s'])]
        w = [x for x in prev if T[x]['succ']]; l = [x for x in prev if not T[x]['succ']]
        if w and l:
            s.append((np.mean([Z[ix[k]] @ Z[ix[x]] for x in w]) - np.mean([Z[ix[k]] @ Z[ix[x]] for x in l]), e['succ'], (e['run'], e['r'])))
    sc = np.array([a for a, _, _ in s]); y = np.array([b for _, b, _ in s]); cl = [c for _, _, c in s]
    auc = lambda sc, y: float(np.mean(sc[y == 1][:, None] > sc[y == 0][None, :]))
    groups = sorted(set(cl)); members = {g: [i for i, c in enumerate(cl) if c == g] for g in groups}
    bs = []
    for _ in range(B):
        idx = [i for g in rng.choice(len(groups), len(groups)) for i in members[groups[g]]]
        if len(set(y[idx])) == 2:
            bs.append(auc(sc[idx], y[idx]))
    return auc(sc, y), np.percentile(bs, [2.5, 97.5]).tolist(), len(s)


def main():
    T, _, _ = A.build()
    ids = sorted(T)
    same, succ, group, extra = J.build(T)
    tasks = {'same': same, 'success': succ, **extra}
    rec_prev = {}
    for k, e in T.items():
        own = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['r'] and T[x]['round'] < e['round']]
        if own:
            rec_prev[k] = max(own, key=lambda x: T[x]['round'])
    from collections import defaultdict
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
    rng = np.random.default_rng(0)
    res = {}
    print(f"{'triplet question':10s} dim  next  same  partner_out partner_in | success AUC any partner [CI]   | partner only [CI]          | pair_r")
    for q in ('strategy', 'strategy_noex', 'person', 'partner', 'success'):
        path = os.path.join(OUT, 'strategy_embedding.npz' if q == 'strategy' else f'strategy_embedding_{q}.npz')
        if not os.path.exists(path):
            continue
        d = np.load(path)
        emb = {k: x for k, x in zip(d['ids'], d['X'])}
        X = np.array([emb[k] for k in ids])
        v = TE.validity(ids, X, T, tasks, rec_prev, pairings)
        a1 = succ_auc(ids, X, T, False, rng); a2 = succ_auc(ids, X, T, True, rng)
        res[q] = {**v, 'succ_rec': a1, 'succ_rec_pp': a2, 'dim': int(d['dim'])}
        print(f"{q:10s} {int(d['dim']):3d}  {v['next']:.0%}   {v['same']:.0%}   {v['partner_out']:.0%}         {v['partner_in']:.0%}       | "
              f"{a1[0]:.2f} [{a1[1][0]:.2f}-{a1[1][1]:.2f}] n={a1[2]}       | {a2[0]:.2f} [{a2[1][0]:.2f}-{a2[1][1]:.2f}] n={a2[2]} | {v['pair_r']:+.2f}")
    json.dump(res, open(os.path.join(OUT, 'eval_embeddings.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
