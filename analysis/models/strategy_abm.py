#!/usr/bin/env python3
"""
Strategy-level simulation of the tournaments (coarse agent-based model).

Instead of key presses, each Sender turn is one strategy choice and each round one success/fail draw:
  strategies  GO goal-only | WF wiggle toward the goal direction | WB wiggle away | WS wiggle sideways |
              CD point from the centre | X anything else (codes from analyze_tournament.code_turn)
  menu        cardinal goals: GO WF WB CD X;  diagonal goals: GO WS WB CD X  (a corner wiggle can't be diagonal)

Choice model (conditional logit over the menu, per goal type g = cardinal/diagonal):
  U(k) = base[g,k] + b_win  * own wins with k on g-goals as Sender
                   + b_use  * own uses of k on g-goals as Sender (habit)
                   + b_soc  * wins with k on g-goals seen as Receiver (copying what worked on you)
Outcome model (logistic):
  logit P(success) = a[g,k] + c_rcv * times the Receiver has seen k on g-goals (any partner, with feedback)
                            + c_pair * times THIS pair has done k on g-goals together
                            + c_self * times the Receiver itself used k on g-goals as Sender
                            + c_new  * [fewer than 2 games so far with this partner]
  (+ a_room * [the goal direction points into the board from the goal tile] in every variant: pointing off the
   board, e.g. left tile pointing left, roughly halves success)
Variants switch groups of learning terms off (fixed at 0):
  static    no learning at all (only base rates)            -> what goal mix + strategy mix alone predict
  sender    sender reinforcement only (b_win, b_use)
  receiver  receiver learning only (c_rcv, c_pair)
  private   everything except carrying things across partners (b_soc=0, c_rcv=0; keeps c_pair)
  full      all terms above
  convention  full + c_self (Receiver decodes better a strategy it itself uses as Sender: reads through its own
            convention) + c_new (first two games with a new partner)
Each variant is fit by maximum likelihood (small L2) on the human rounds, then the real schedules (who plays whom,
which goal, in which round) are replayed N times with simulated choices and outcomes. Compared: success by phase, by
goal type, and strategy mix by phase. --loo refits leaving each run out and simulates that run.

Usage: python3 analysis/models/strategy_abm.py [--sims 2000] [--loo]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

CODE = {'goal-only': 'GO', 'goal-wiggle-F': 'WF', 'goal-wiggle-B': 'WB', 'goal-wiggle-S': 'WS', 'centre-dir': 'CD'}
MENU = {0: ['GO', 'WF', 'WB', 'CD', 'X'], 1: ['GO', 'WS', 'WB', 'CD', 'X']}  # 0 cardinal, 1 diagonal
XY = {v: k for k, v in C.XY_TILE.items()}
VEC = {'up': (0, 1), 'up-right': (1, 1), 'right': (1, 0), 'down-right': (1, -1), 'down': (0, -1), 'down-left': (-1, -1), 'left': (-1, 0), 'up-left': (-1, 1)}
CELLS = [(g, k) for g in (0, 1) for k in MENU[g]]
CI = {c: i for i, c in enumerate(CELLS)}
NC = len(CELLS)
LEARN = ['b_win', 'b_use', 'b_soc', 'c_rcv', 'c_pair', 'c_self', 'c_new', 'a_room']
VARIANTS = {'static': ['a_room'], 'sender': ['b_win', 'b_use', 'a_room'], 'receiver': ['c_rcv', 'c_pair', 'a_room'],
            'private': ['b_win', 'b_use', 'c_pair', 'a_room'], 'full': LEARN[:5] + ['a_room'],
            'convention': LEARN}


def load_rounds():
    out = []
    for t in C.load_turns():
        if t['role'] != 'sender':
            continue
        g = int('-' in t['goal_direction'])
        k = CODE.get(C.sender_strategy(t['tokens'], t['goal_tile'], t['goal_direction']), 'X')
        if k not in MENU[g]:
            k = 'X'
        gx, gy = XY[t['goal_tile']]; dx, dy = VEC[t['goal_direction']]
        out.append({'room': int(abs(gx + dx) <= 1 and abs(gy + dy) <= 1), 'run': t['run'], 'round': t['round'], 'phase': t['phase'], 's': t['player'], 'r': t['partner'],
                    'g': g, 'k': k, 'succ': t['success']})
    return sorted(out, key=lambda x: (x['run'], x['round'], x['s']))


class History:
    def __init__(self):
        self.win = defaultdict(float); self.use = defaultdict(float); self.soc = defaultdict(float)
        self.rcv = defaultdict(float); self.pair = defaultdict(float)

    def feats(self, s, r, g):
        """per menu item: (win, use, soc) for the sender's choice and (rcv, pair) for the outcome"""
        pk = tuple(sorted((s, r)))
        npair = sum(v for (p, gg, kk), v in self.pair.items() if p == pk)
        return np.array([[self.win[s, g, k], self.use[s, g, k], self.soc[s, g, k], self.rcv[r, g, k], self.pair[pk, g, k],
                          self.use[r, g, k], float(npair < 2)] for k in MENU[g]])

    def update(self, s, r, g, k, succ):
        self.use[s, g, k] += 1
        self.win[s, g, k] += succ
        self.soc[r, g, k] += succ
        self.rcv[r, g, k] += 1
        self.pair[tuple(sorted((s, r))), g, k] += 1


def unpack(theta, free):
    base, a = theta[:NC], theta[NC:2 * NC]
    lw = dict.fromkeys(LEARN, 0.0)
    for i, n in enumerate(free):
        lw[n] = theta[2 * NC + i]
    return base, a, lw


def nll(theta, rounds, free, l2=0.5):
    base, a, lw = unpack(theta, free)
    h = History(); tot = 0.0
    for x in rounds:
        if x.get('_new_run'):
            h = History()
        g, k = x['g'], x['k']
        F = h.feats(x['s'], x['r'], g)
        idx = [CI[g, m] for m in MENU[g]]
        u = base[idx] + lw['b_win'] * F[:, 0] + lw['b_use'] * F[:, 1] + lw['b_soc'] * F[:, 2]
        u = u - u.max()
        j = MENU[g].index(k)
        tot -= u[j] - np.log(np.exp(u).sum())
        z = a[CI[g, k]] + lw['c_rcv'] * F[j, 3] + lw['c_pair'] * F[j, 4] + lw['c_self'] * F[j, 5] + lw['c_new'] * F[j, 6] + lw['a_room'] * x['room']
        p = 1 / (1 + np.exp(-z))
        tot -= np.log(p if x['succ'] else 1 - p)
        h.update(x['s'], x['r'], g, k, x['succ'])
    return tot + l2 * np.sum(theta ** 2) * 0.1


def mark_runs(rounds):
    prev = None
    for x in rounds:
        x['_new_run'] = x['run'] != prev; prev = x['run']
    return rounds


def fit(rounds, free):
    rounds = mark_runs(rounds)
    th0 = np.zeros(2 * NC + len(free))
    res = minimize(nll, th0, args=(rounds, free), method='L-BFGS-B')
    return res.x, res.fun


def simulate(theta, free, schedule, n, rng):
    """schedule: one run's rounds (s, r, g, phase) in order -> per-sim lists of (phase, g, k, succ)"""
    base, a, lw = unpack(theta, free)
    sims = []
    for _ in range(n):
        h = History(); out = []
        for x in schedule:
            g = x['g']
            F = h.feats(x['s'], x['r'], g)
            idx = [CI[g, m] for m in MENU[g]]
            u = base[idx] + lw['b_win'] * F[:, 0] + lw['b_use'] * F[:, 1] + lw['b_soc'] * F[:, 2]
            p = np.exp(u - u.max()); p /= p.sum()
            j = rng.choice(len(p), p=p); k = MENU[g][j]
            z = a[CI[g, k]] + lw['c_rcv'] * F[j, 3] + lw['c_pair'] * F[j, 4] + lw['c_self'] * F[j, 5] + lw['c_new'] * F[j, 6] + lw['a_room'] * x['room']
            succ = int(rng.random() < 1 / (1 + np.exp(-z)))
            h.update(x['s'], x['r'], g, k, succ)
            out.append((x['phase'], g, k, succ))
        sims.append(out)
    return sims


def summarize(items):
    """items: list of (phase, g, k, succ) -> success by phase, by goal type x phase, strategy share by phase"""
    by = defaultdict(list)
    for ph, g, k, s in items:
        by['ph', ph].append(s); by['g', g, ph].append(s); by['k', ph].append(k)
    ph = {p: np.mean(by['ph', p]) for p in (1, 2, 3)}
    gp = {(g, p): np.mean(by['g', g, p]) for g in (0, 1) for p in (1, 2, 3) if by['g', g, p]}
    mix = {p: {k: by['k', p].count(k) / len(by['k', p]) for k in ['GO', 'WF', 'WB', 'WS', 'CD', 'X']} for p in (1, 2, 3)}
    return ph, gp, mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--loo', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'strategy_abm.json'))
    a = ap.parse_args()
    rounds = load_rounds()
    runs = sorted({x['run'] for x in rounds})
    real_ph, real_gp, real_mix = summarize([(x['phase'], x['g'], x['k'], x['succ']) for x in rounds])
    print(f"REAL      success by phase {' / '.join(f'{real_ph[p]:.0%}' for p in (1, 2, 3))}   "
          f"cardinal {' / '.join(f'{real_gp[0, p]:.0%}' for p in (1, 2, 3))}   diagonal {' / '.join(f'{real_gp[1, p]:.0%}' for p in (1, 2, 3))}")
    rng = np.random.default_rng(0)
    results = {'real': {'phase': real_ph, 'mix': real_mix}}
    n_obs = len(rounds) * 2
    for name, free in VARIANTS.items():
        sims_all = []
        if a.loo:
            for held in runs:
                th, _ = fit([x for x in rounds if x['run'] != held], free)
                sch = [x for x in rounds if x['run'] == held]
                for sm in simulate(th, free, sch, a.sims, rng):
                    sims_all.append(sm)
            fitted = None
        else:
            th, f = fit(list(rounds), free)
            fitted = {n: round(float(th[2 * NC + i]), 3) for i, n in enumerate(free)}
            for held in runs:
                sch = [x for x in rounds if x['run'] == held]
                sims_all += simulate(th, free, sch, a.sims, rng)
        # mean over simulated sessions (pool runs per simulation index)
        ph, gp, mix = summarize([it for sm in sims_all for it in sm])
        per_sim_ph3 = [np.mean([s for p, _, _, s in sm if p == 3]) - np.mean([s for p, _, _, s in sm if p == 1]) for sm in sims_all]
        lo, hi = np.percentile(per_sim_ph3, [5, 95])
        print(f"{name:9s} success by phase {' / '.join(f'{ph[p]:.0%}' for p in (1, 2, 3))}   "
              f"cardinal {' / '.join(f'{gp[0, p]:.0%}' for p in (1, 2, 3))}   diagonal {' / '.join(f'{gp[1, p]:.0%}' for p in (1, 2, 3))}"
              f"   phase3-phase1 per session 90% [{lo:+.0%}, {hi:+.0%}]" + (f"   fitted {fitted}" if fitted else ''))
        results[name] = {'phase': ph, 'mix': mix, 'fitted': fitted, 'ph3_minus_ph1_90': [lo, hi]}
    print('\nstrategy mix by phase (real vs full):')
    for p in (1, 2, 3):
        print(f"  phase {p}: " + ', '.join(f"{k} {real_mix[p][k]:.0%}/{results['full']['mix'][p][k]:.0%}" for k in ['GO', 'WF', 'WB', 'WS', 'CD', 'X']))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out.replace('.json', '_loo.json') if a.loo else a.out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
