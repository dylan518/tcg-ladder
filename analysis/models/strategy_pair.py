#!/usr/bin/env python3
"""
Strategy-pair model: predict each player's STRATEGY at each turn, then whether the pair succeeds follows from whether
the Sender's strategy and the Receiver's strategy fit together.

Sender strategy (what the signal says about the goal; analyze_tournament.code_turn):
  WF wiggle on the goal tile toward the goal direction | WB wiggle away from it | WS wiggle sideways |
  CD emphasise the tile in the goal direction from the centre | GO just go to the goal | X anything else
  menu: cardinal goal GO WF WB CD X; diagonal goal GO WS WB CD X
Receiver strategy (how they read what they saw; analyze_tournament.reading):
  signal seen (WF/WB/WS/CD): F point toward the wiggle / emphasised direction | B point away | S point sideways
  no signal (GO/X):          up leave the triangle pointing up | guess turn it somewhere
  plus tile: put it where the Sender ended (end) or elsewhere (other)
Outcome: P(success | Sender strategy, Receiver strategy) from the human table (smoothed); e.g. WF+F 97%, WB+B 64%,
CD+F 79%, any mismatch 0%. Receivers who don't use the Sender's end tile never succeeded.

Predictors (conditional logit, fit on two runs, scored on the third = held out):
  Sender  base[goal type, k] + own_last (k = my last Sender strategy for this goal type)
          + own_win (k = my last Sender strategy that worked) + saw_win (k = last strategy that worked on me as Receiver)
          + partner_last (k = the strategy this partner last sent me)
  Receiver base[signal class, k] + own_last (my last reading in this class) + own_win (my last reading that worked)
          + self (k = F if my own last wiggle as Sender was WF/CD, B if WB: reading through my own convention)
          + pair_win (the reading that last worked with this partner)
Variants: static (base only), own (own_* terms), social (+ saw_win/partner_last/pair_win), full (+ self).
Then the real schedules are replayed N times with predicted strategies; success by phase with 90% bands per run, and
how often a no-learning (static) session shows a phase pattern as extreme as the real one (is it variance?).

Usage: python3 analysis/models/strategy_pair.py [--sims 2000]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import analyze_tournament as AT  # noqa: E402

SCODE = {'goal-only': 'GO', 'goal-wiggle-F': 'WF', 'goal-wiggle-B': 'WB', 'goal-wiggle-S': 'WS', 'centre-dir': 'CD'}
SMENU = {0: ['GO', 'WF', 'WB', 'CD', 'X'], 1: ['GO', 'WS', 'WB', 'CD', 'X']}
SIGNAL = {'WF', 'WB', 'WS', 'CD'}
RMENU = {1: ['F', 'B', 'S'], 0: ['up', 'guess']}  # 1 = signal seen
S_FEATS = ['own_last', 'own_win', 'saw_win', 'partner_last']
R_FEATS = ['own_last', 'own_win', 'self', 'pair_win']
VARIANTS = {'static': ([], []), 'own': (['own_last', 'own_win'], ['own_last', 'own_win']),
            'social': (['own_last', 'own_win', 'saw_win', 'partner_last'], ['own_last', 'own_win', 'pair_win']),
            'full': (S_FEATS, R_FEATS)}


def load():
    T = C.load_turns()
    rec = {(t['run'], t['round'], t['player']): t for t in T if t['role'] == 'receiver'}
    out = []
    for t in T:
        if t['role'] != 'sender':
            continue
        r = rec[(t['run'], t['round'], t['partner'])]
        code, ang, _ = AT.code_turn({'sender_moves': C.tokens_to_moves(t['tokens']), 'goal_tile': t['goal_tile'], 'goal_direction': t['goal_direction']})
        g = int('-' in t['goal_direction'])
        s = SCODE.get(code, 'X')
        if s not in SMENU[g]:
            s = 'X'
        end = C.replay([x for x in t['tokens'] if x in 'UDLR'])[0]
        rd = AT.reading(ang, t['receiver_direction'])
        sig = int(s in SIGNAL)
        if sig:
            rk = {'reads-F': 'F', 'reads-B': 'B', 'reads-S': 'S'}.get(rd, 'S')
        else:
            rk = 'up' if t['receiver_direction'] == 'up' else 'guess'
        out.append({'run': t['run'], 'round': t['round'], 'phase': t['phase'], 's': t['player'], 'r': t['partner'], 'g': g,
                    'S': s, 'R': rk, 'tile_end': int(r['receiver_tile'] == end), 'succ': t['success']})
    return sorted(out, key=lambda x: (x['run'], x['round'], x['s']))


class Hist:
    def __init__(self):
        self.s_last, self.s_win, self.saw_win, self.sent_to = {}, {}, {}, {}
        self.r_last, self.r_win, self.pair_win, self.wiggle = {}, {}, {}, {}

    def s_feats(self, x):
        p, g = x['s'], x['g']
        f = {'own_last': self.s_last.get((p, g)), 'own_win': self.s_win.get((p, g)), 'saw_win': self.saw_win.get((p, g)),
             'partner_last': self.sent_to.get((x['r'], p, g))}
        return np.array([[float(f[n] == k) for n in S_FEATS] for k in SMENU[g]])

    def r_feats(self, x, S):
        p, sig = x['r'], int(S in SIGNAL)
        own = self.wiggle.get(p)
        selfk = {'WF': 'F', 'CD': 'F', 'WB': 'B'}.get(own) if sig else None
        f = {'own_last': self.r_last.get((p, sig)), 'own_win': self.r_win.get((p, sig)), 'self': selfk,
             'pair_win': self.pair_win.get((tuple(sorted((p, x['s']))), sig))}
        return np.array([[float(f[n] == k) for n in R_FEATS] for k in RMENU[sig]])

    def update(self, x, S, R, succ):
        p, q, g, sig = x['s'], x['r'], x['g'], int(S in SIGNAL)
        self.s_last[p, g] = S; self.sent_to[p, q, g] = S
        self.r_last[q, sig] = R
        if S in ('WF', 'WB', 'CD'):
            self.wiggle[p] = S
        if succ:
            self.s_win[p, g] = S; self.saw_win[q, g] = S; self.r_win[q, sig] = R
            self.pair_win[tuple(sorted((p, q))), sig] = R


SB = [(g, k) for g in (0, 1) for k in SMENU[g]]
RB = [(c, k) for c in (1, 0) for k in RMENU[c]]
SBI, RBI = {c: i for i, c in enumerate(SB)}, {c: i for i, c in enumerate(RB)}


def unpack(th, sf, rf):
    sb, rb = th[:len(SB)], th[len(SB):len(SB) + len(RB)]
    i = len(SB) + len(RB)
    sw = np.zeros(len(S_FEATS)); rw = np.zeros(len(R_FEATS))
    for n in sf:
        sw[S_FEATS.index(n)] = th[i]; i += 1
    for n in rf:
        rw[R_FEATS.index(n)] = th[i]; i += 1
    return sb, rb, sw, rw


def probs(sb, rb, sw, rw, h, x, S=None):
    g = x['g']
    us = sb[[SBI[g, k] for k in SMENU[g]]] + h.s_feats(x) @ sw
    ps = np.exp(us - us.max()); ps /= ps.sum()
    if S is None:
        return ps, None
    sig = int(S in SIGNAL)
    ur = rb[[RBI[sig, k] for k in RMENU[sig]]] + h.r_feats(x, S) @ rw
    pr = np.exp(ur - ur.max()); pr /= pr.sum()
    return ps, pr


def nll(th, rounds, sf, rf):
    sb, rb, sw, rw = unpack(th, sf, rf)
    tot, h, prev = 0.0, None, None
    for x in rounds:
        if x['run'] != prev:
            h, prev = Hist(), x['run']
        ps, pr = probs(sb, rb, sw, rw, h, x, x['S'])
        tot -= np.log(ps[SMENU[x['g']].index(x['S'])] + 1e-12) + np.log(pr[RMENU[int(x['S'] in SIGNAL)].index(x['R'])] + 1e-12)
        h.update(x, x['S'], x['R'], x['succ'])
    return tot + 0.05 * np.sum(th ** 2)


def fit(rounds, sf, rf):
    from scipy.optimize import minimize
    th = np.zeros(len(SB) + len(RB) + len(sf) + len(rf))
    return minimize(nll, th, args=(rounds, sf, rf), method='L-BFGS-B').x


def score(th, rounds, sf, rf):
    """held-out per-turn prediction (true history): log-loss and accuracy for Sender and Receiver strategy"""
    sb, rb, sw, rw = unpack(th, sf, rf)
    h, prev = None, None
    ls = lr = acs = acr = 0.0
    for x in rounds:
        if x['run'] != prev:
            h, prev = Hist(), x['run']
        ps, pr = probs(sb, rb, sw, rw, h, x, x['S'])
        js, jr = SMENU[x['g']].index(x['S']), RMENU[int(x['S'] in SIGNAL)].index(x['R'])
        ls -= np.log2(ps[js]); lr -= np.log2(pr[jr]); acs += ps.argmax() == js; acr += pr.argmax() == jr
        h.update(x, x['S'], x['R'], x['succ'])
    n = len(rounds)
    return {'S_bits': ls / n, 'S_acc': acs / n, 'R_bits': lr / n, 'R_acc': acr / n}


def outcome_table(rounds):
    """P(success | S, R) among receivers who used the end tile, Laplace-smoothed toward 0; P(end tile)"""
    cnt = defaultdict(lambda: [0, 0])
    for x in rounds:
        if x['tile_end']:
            cnt[x['S'], x['R']][0] += x['succ']; cnt[x['S'], x['R']][1] += 1
    tab = {k: (v[0] + 0.0) / (v[1] + 1) for k, v in cnt.items()}
    return tab, np.mean([x['tile_end'] for x in rounds])


def simulate(th, sf, rf, sched, tab, p_end, n, rng):
    sb, rb, sw, rw = unpack(th, sf, rf)
    sims = []
    for _ in range(n):
        h, out = Hist(), []
        for x in sched:
            ps, _ = probs(sb, rb, sw, rw, h, x)
            S = SMENU[x['g']][rng.choice(len(ps), p=ps)]
            _, pr = probs(sb, rb, sw, rw, h, x, S)
            R = RMENU[int(S in SIGNAL)][rng.choice(len(pr), p=pr)]
            succ = int(rng.random() < p_end and rng.random() < tab.get((S, R), 0.0))
            h.update(x, S, R, succ)
            out.append((x['phase'], S, R, succ))
        sims.append(out)
    return sims


def by_phase(items):
    return [np.mean([s for p, _, _, s in items if p == ph]) for ph in (1, 2, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sims', type=int, default=1000)
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'strategy_pair.json'))
    a = ap.parse_args()
    rounds = load()
    runs = sorted({x['run'] for x in rounds})
    rng = np.random.default_rng(0)
    real = by_phase([(x['phase'], x['S'], x['R'], x['succ']) for x in rounds])
    real_run = {r: by_phase([(x['phase'], x['S'], x['R'], x['succ']) for x in rounds if x['run'] == r]) for r in runs}
    tab, p_end = outcome_table(rounds)
    print('outcome table P(success | S, R, end tile):', {f'{k[0]}+{k[1]}': round(v, 2) for k, v in sorted(tab.items())}, f' P(end tile) {p_end:.2f}')
    print(f"\nREAL success by phase {' / '.join(f'{v:.0%}' for v in real)}  (per run: " +
          '; '.join(f"{r}: " + '/'.join(f'{v:.0%}' for v in real_run[r]) for r in runs) + ')')
    res = {'real': real, 'outcome_table': {f'{k[0]}+{k[1]}': v for k, v in tab.items()}}
    print(f"\n{'variant':8s} | held-out prediction: Sender strategy acc / bits | Receiver strategy acc / bits | simulated success by phase (90% band per session) | P(static-like dip&rise)")
    real_dip, real_rise = real[1] - real[0], real[2] - real[1]
    for name, (sf, rf) in VARIANTS.items():
        sc = defaultdict(float); sims = []
        for held in runs:
            tr = [x for x in rounds if x['run'] != held]; te = [x for x in rounds if x['run'] == held]
            th = fit(tr, sf, rf)
            s = score(th, te, sf, rf)
            for k, v in s.items():
                sc[k] += v * len(te) / len(rounds)
            tab_h, pe_h = outcome_table(tr)
            sims += [(held, sm) for sm in simulate(th, sf, rf, te, tab_h, pe_h, a.sims, rng)]
        ph = by_phase([it for _, sm in sims for it in sm])
        per = np.array([by_phase(sm) for _, sm in sims])
        band = np.percentile(per, [5, 95], axis=0)
        extreme = np.mean((per[:, 1] - per[:, 0] <= real_dip) & (per[:, 2] - per[:, 1] >= real_rise))
        print(f"{name:8s} | {sc['S_acc']:.0%} / {sc['S_bits']:.2f}                       | {sc['R_acc']:.0%} / {sc['R_bits']:.2f}                    | "
              + ' / '.join(f'{ph[i]:.0%} [{band[0, i]:.0%}-{band[1, i]:.0%}]' for i in range(3)) + f" | {extreme:.1%}")
        res[name] = {'pred': dict(sc), 'sim_phase': ph, 'band': band.tolist(), 'p_dip_and_rise_as_extreme': extreme,
                     'fitted_all': dict(zip([f'{a_}:{b_}' for a_, b_ in SB] + [f'R{a_}:{b_}' for a_, b_ in RB] + [f'S:{n}' for n in sf] + [f'R:{n}' for n in rf], np.round(fit(rounds, sf, rf), 2).tolist()))}
    w = res['full']['fitted_all']
    print('\nfull model learning weights (all runs):', {k: v for k, v in w.items() if k.startswith(('S:', 'R:'))})
    mix = defaultdict(lambda: defaultdict(int))
    for x in rounds:
        mix[x['phase']][x['R']] += 1
    print('real Receiver readings by phase:', {p: {k: round(v / sum(m.values()), 2) for k, v in sorted(m.items())} for p, m in sorted(mix.items())})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
