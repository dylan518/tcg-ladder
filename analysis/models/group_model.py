#!/usr/bin/env python3
"""
A generative model of how a group builds shared understanding in the Tacit Communication Game, fitted to the real
tournaments and simulated on their real partner schedules.

State: each player's signalling style = a point in the goal-removed LLM-judgment strategy embedding (3-D).
Components (all fitted on real data):
  (1) STYLE UPDATE   x_next = mu + a_own (x_own_prev - mu) + a_part (x_partner_latest - mu) + a_win (c_win - mu) + eps
                     c_win = centroid of the player's own earlier SUCCESSFUL sends (sender learning from feedback);
                     eps ~ N(0, residual covariance).
  (2) LEGIBILITY     L(x, goal room) = P(a history-free decoder reads the signal correctly)   [logistic, fitted on
                     the 240 neutral-decoder results from cf_encode_decode.py]
  (3) SUCCESS        P(success) = logistic(b0 + b_L * logit L + b_room * room - b_mis * ||x - e_rs||)
                     e_rs = the Receiver's expectation of this Sender's style: the last style of this Sender the
                     Receiver has seen, else the Receiver's own latest sending style (egocentric projection) - so
                     understanding is partner-specific and must be rebuilt when partners change.
Simulation: each tournament from the players' real first Sender turns, real schedule (who sends to whom, which goal),
N runs. Compared with the real data: success by phase (incl. new vs reunited partners in phase 3), legibility by phase,
style distance between partners vs non-partners. Ablations remove one component at a time.
Figures -> paper/figures/fig7_model.pdf/png ; numbers -> analysis/models/out/group_model.json
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import convergence as CV  # noqa: E402
import readable as RD  # noqa: E402

OUT = os.path.join(HERE, 'out')
DV = {'up': (0, 1), 'up-right': (1, 1), 'right': (1, 0), 'down-right': (1, -1), 'down': (0, -1), 'down-left': (-1, -1), 'left': (-1, 0), 'up-left': (-1, 1)}


def room(e):
    x, y = RD.XY[e['goal_tile']]; dx, dy = DV[e['goal_dir']]
    return float(abs(x + dx) <= 1 and abs(y + dy) <= 1)


def logit_fit(X, y, l2=1e-2, iters=3000, lr=0.1):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return w


def sig(z):
    return 1 / (1 + np.exp(-z))


def load_data():
    T, R = CV.load()
    enc = {}
    for l in open(os.path.join(OUT, 'cf_cache_gpt-6.1-terra.jsonl')):
        r = json.loads(l)
        if r['key'].startswith('enc:') and r['out']:
            _, run, rnd, s = r['key'].split(':', 3)
            tid = f'{run}:{rnd}:{s}'
            if tid in T:
                e = T[tid]
                enc[tid] = int(r['out']['tile'] == e['goal_tile'] and r['out']['direction'] == e['goal_dir'])
    return T, R, enc


def fit(T, R, enc):
    mu = np.mean(list(R.values()), 0)
    order = sorted(T, key=lambda k: (T[k]['run'], T[k]['round']))
    # ---- (2) legibility
    ks = [k for k in order if k in enc]
    wL = logit_fit(np.column_stack([np.array([R[k] for k in ks]), [room(T[k]) for k in ks]]), np.array([enc[k] for k in ks]))
    L = lambda x, rm: sig(wL[0] + x @ wL[1:4] + wL[4] * rm)
    # ---- (1) style update + (3) success, walking through the real history
    Ys, Xs, srows = [], [], []
    for k in order:
        e = T[k]
        prev = [x for x in order if T[x]['run'] == e['run'] and T[x]['round'] < e['round']]
        mine = [x for x in prev if T[x]['s'] == e['s']]
        part = [x for x in prev if T[x]['s'] == e['r']]
        if mine and part:
            wins = [x for x in mine if T[x]['succ']]
            cw = np.mean([R[x] for x in wins], 0) if wins else mu
            Ys.append(R[k] - mu); Xs.append(np.stack([R[mine[-1]] - mu, R[part[-1]] - mu, cw - mu]))
        # receiver's expectation of this sender
        seen = [x for x in prev if T[x]['s'] == e['s'] and T[x]['r'] == e['r']]
        own_r = [x for x in prev if T[x]['s'] == e['r']]
        if seen:
            ex = R[seen[-1]]
        elif own_r:
            ex = R[own_r[-1]]
        else:
            continue
        lg = L(R[k], room(e))
        srows.append((np.log(lg / (1 - lg)), room(e), np.linalg.norm(R[k] - ex), e['succ']))
    Y = np.array(Ys); X = np.array(Xs)  # (n, 3 terms, dim)
    M = X.transpose(0, 2, 1).reshape(-1, 3)  # pooled over dimensions
    a = np.linalg.lstsq(M, Y.reshape(-1), rcond=None)[0]
    resid = Y - np.einsum('t,ntd->nd', a, X)
    cov = np.cov(resid.T)
    S = np.array(srows)
    wS = logit_fit(S[:, :3], S[:, 3])
    return dict(mu=mu, wL=wL, a=a, cov=cov, wS=wS, L=L, n_style=len(Y), n_succ=len(S))


def schedule(T, run):
    return sorted([k for k in T if T[k]['run'] == run], key=lambda k: (T[k]['round'], T[k]['s']))


def simulate(T, R, P, run, rng, use=('own', 'part', 'win', 'mis')):
    mu, a, cov, wS, L = P['mu'], P['a'], P['cov'], P['wS'], P['L']
    a = np.array([a[0] if 'own' in use else 0, a[1] if 'part' in use else 0, a[2] if 'win' in use else 0])
    sch = schedule(T, run)
    style, wins, seen = {}, defaultdict(list), {}
    first = {}
    for k in sch:  # real first sender turn of each player = initial style
        first.setdefault(T[k]['s'], R[k])
    out = []
    for k in sch:
        e = T[k]; s, r = e['s'], e['r']
        if s not in style:
            x = first[s].copy()
        else:
            part = style.get(r, first.get(r, mu))
            cw = np.mean(wins[s], 0) if wins[s] else mu
            x = mu + a[0] * (style[s] - mu) + a[1] * (part - mu) + a[2] * (cw - mu) + rng.multivariate_normal(np.zeros(len(mu)), cov)
        ex = seen.get((r, s), style.get(r, first.get(r, mu)))
        lg = L(x, room(e))
        z = wS[0] + wS[1] * np.log(lg / (1 - lg)) + wS[2] * room(e) + (wS[3] * np.linalg.norm(x - ex) if 'mis' in use else wS[3] * P['mean_mis'])
        succ = int(rng.random() < sig(z))
        style[s] = x; seen[(r, s)] = x
        if succ:
            wins[s].append(x)
        out.append(dict(run=run, round=e['round'], phase=e['phase'], s=s, r=r, x=x, succ=succ, leg=lg, reunion=e.get('reunion', 0)))
    return out


def summarize(rows, T=None):
    res = {'succ': {}, 'leg': {}, 'dist': {}}
    for ph in (1, 2, 3):
        v = [r for r in rows if r['phase'] == ph]
        res['succ'][ph] = float(np.mean([r['succ'] for r in v]))
        res['leg'][ph] = float(np.mean([r['leg'] for r in v]))
        part, non = [], []
        pairs = {(r['run'], r['s']): r['r'] for r in v}
        by = defaultdict(list)
        for r in v:
            by[(r['run'], r['s'])].append(r['x'])
        keys = list(by)
        for i, p in enumerate(keys):
            for q in keys[i + 1:]:
                if p[0] != q[0]:
                    continue
                d = np.mean([np.linalg.norm(x - y) for x in by[p] for y in by[q]])
                (part if pairs.get(p) == q[1] else non).append(d)
        res['dist'][ph] = (float(np.mean(part)) if part else np.nan, float(np.mean(non)) if non else np.nan)
    for lab in (0, 1):
        v = [r for r in rows if r['phase'] == 3 and r['reunion'] == lab]
        res.setdefault('ph3_by_reunion', {})[lab] = float(np.mean([r['succ'] for r in v])) if v else np.nan
    return res


def main():
    T, R, enc = load_data()
    pairs = defaultdict(set)
    for e in T.values():
        pairs[(e['run'], e['phase'])].add(tuple(sorted([e['s'], e['r']])))
    for e in T.values():
        pr = tuple(sorted([e['s'], e['r']]))
        e['reunion'] = int(e['phase'] == 3 and pr in pairs[(e['run'], 1)] | pairs[(e['run'], 2)])
    P = fit(T, R, enc)
    # mean mismatch (used when the mismatch component is ablated)
    P['mean_mis'] = 1.9
    print('fitted: style update own %.2f  partner %.2f  own-successes %.2f  (n=%d)' % (*P['a'], P['n_style']))
    print('        success: b0 %.2f  b_legibility %.2f  b_room %.2f  b_mismatch %.2f  (n=%d)' % (*P['wS'], P['n_succ']))
    real_rows = [dict(run=e['run'], phase=e['phase'], s=e['s'], r=e['r'], x=R[k], succ=e['succ'], leg=P['L'](R[k], room(e)), reunion=e['reunion']) for k, e in T.items()]
    real = summarize(real_rows)
    rng = np.random.default_rng(0)
    N = 300
    variants = {'full model': ('own', 'part', 'win', 'mis'), 'no partner pull': ('own', 'win', 'mis'),
                'no learning from success': ('own', 'part', 'mis'), 'no partner-specific expectations': ('own', 'part', 'win')}
    sims = {}
    for name, use in variants.items():
        runs = [summarize(sum([simulate(T, R, P, run, rng, use) for run in (1, 2, 3)], [])) for _ in range(N)]
        sims[name] = runs
    def band(runs, f):
        v = np.array([f(r) for r in runs]); return float(v.mean()), float(np.percentile(v, 5)), float(np.percentile(v, 95))
    print('\n                                   success ph1/ph2/ph3        phase-3 new vs reunited   legibility ph1/ph2/ph3     partner-vs-nonpartner distance gap ph1/ph2/ph3')
    fmt = lambda r: (f"{r['succ'][1]:.0%}/{r['succ'][2]:.0%}/{r['succ'][3]:.0%}", f"{r['ph3_by_reunion'][0]:.0%} vs {r['ph3_by_reunion'][1]:.0%}",
                     f"{r['leg'][1]:.0%}/{r['leg'][2]:.0%}/{r['leg'][3]:.0%}", '/'.join(f"{r['dist'][p][1] - r['dist'][p][0]:+.2f}" for p in (1, 2, 3)))
    print(f"{'REAL':34s} " + '   '.join(fmt(real)))
    out = {'params': {'a': P['a'].tolist(), 'wS': P['wS'].tolist(), 'wL': P['wL'].tolist()}, 'real': real, 'sims': {}}
    for name, runs in sims.items():
        mean = {'succ': {p: band(runs, lambda r: r['succ'][p]) for p in (1, 2, 3)},
                'leg': {p: band(runs, lambda r: r['leg'][p]) for p in (1, 2, 3)},
                'gap': {p: band(runs, lambda r: r['dist'][p][1] - r['dist'][p][0]) for p in (1, 2, 3)},
                'reun': {lab: band(runs, lambda r: r['ph3_by_reunion'][lab]) for lab in (0, 1)}}
        out['sims'][name] = mean
        print(f"{name:34s} {mean['succ'][1][0]:.0%}/{mean['succ'][2][0]:.0%}/{mean['succ'][3][0]:.0%}"
              f"                   {mean['reun'][0][0]:.0%} vs {mean['reun'][1][0]:.0%}"
              f"               {mean['leg'][1][0]:.0%}/{mean['leg'][2][0]:.0%}/{mean['leg'][3][0]:.0%}"
              f"                 " + '/'.join(f"{mean['gap'][p][0]:+.2f}" for p in (1, 2, 3)))
    json.dump(out, open(os.path.join(OUT, 'group_model.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
