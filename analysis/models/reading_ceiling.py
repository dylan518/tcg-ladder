#!/usr/bin/env python3
"""Exploratory: do signals set a ceiling on understanding, and does reading improve within it?
Uses the neutral-reader results cached by cf_encode_decode.py and the coded rounds of each tournament.
  1. real success split by whether the neutral reader read the signal (by phase)
  2. Receivers meeting a NEW partner in phase 2 vs phase 3 (real vs neutral reader)
  3. first vs last five rounds of phase 2 with a new partner
  4. spread of individual success rates as Sender vs as Receiver
  5. bootstrap of the Sender-minus-Receiver effect in the 2 x 2 counterfactual
Usage: python3 analysis/models/reading_ceiling.py [--model gpt-6.1-terra]"""
import argparse, collections, csv, glob, json, os
import numpy as np
from scipy.stats import fisher_exact

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(model):
    cache = {}
    for l in open(os.path.join(HERE, 'out', f'cf_cache_{model}.jsonl')):
        d = json.loads(l)
        if d['key'].startswith('enc:'):
            cache[d['key']] = d['out']
    rows = []
    for run, f in enumerate(sorted(glob.glob(os.path.join(ROOT, 'analysis', '2026-11-11-*_tournament', 'coded_rounds.csv'))), 1):
        partners = collections.defaultdict(dict)  # receiver -> {phase: sender}
        R = list(csv.DictReader(open(f)))
        for r in R:
            partners[r['receiver']][int(r['phase'])] = r['sender']
        for r in R:
            ph, s, rc = int(r['phase']), r['sender'], r['receiver']
            earlier = [partners[rc].get(p) for p in range(1, ph)]
            status = 'first' if ph == 1 else 'continuing' if earlier[-1] == s else 'reunited' if s in earlier else 'new'
            o = cache[f"enc:{run}:{r['round']}:{s}"]
            rows.append(dict(run=run, ph=ph, rip=int(r['round_in_phase']), status=status, real=int(r['success']),
                             neu=int(o['tile'] == r['goal_tile'] and o['direction'] == r['goal_direction']),
                             tile=int(r['tile_ok']), dir=int(r['dir_ok']), s=(run, s), rc=(run, rc)))
    return rows


def frac(v, k):
    n = sum(r[k] for r in v)
    return f"{n}/{len(v)} = {n / len(v):.0%}"


def fisher(a, b, k='real'):
    return fisher_exact([[sum(r[k] for r in a), len(a) - sum(r[k] for r in a)], [sum(r[k] for r in b), len(b) - sum(r[k] for r in b)]])[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='gpt-6.1-terra'); a = ap.parse_args()
    rows = load(a.model)
    print('1. real success by whether the neutral reader read the signal')
    for ph in (1, 2, 3, None):
        v = [r for r in rows if ph is None or r['ph'] == ph]
        print(f"   phase {ph or 'all'}: readable {frac([r for r in v if r['neu']], 'real')}   unreadable {frac([r for r in v if not r['neu']], 'real')}")
    rd, un = [r for r in rows if r['neu']], [r for r in rows if not r['neu']]
    print(f"   Fisher p = {fisher(rd, un):.5f}; tile right: readable {frac(rd, 'tile')}, unreadable {frac(un, 'tile')}; direction right: {frac(rd, 'dir')}, {frac(un, 'dir')}")
    print('2. new partner, phase 2 vs phase 3')
    p2, p3 = [r for r in rows if r['ph'] == 2 and r['status'] == 'new'], [r for r in rows if r['ph'] == 3 and r['status'] == 'new']
    for lbl, v in (('phase 2', p2), ('phase 3', p3)):
        print(f"   {lbl}: real {frac(v, 'real')}   neutral {frac(v, 'neu')}   real on readable signals {frac([r for r in v if r['neu']], 'real')}")
    print(f"   real success phase 3 vs 2: Fisher p = {fisher(p3, p2):.3f}")
    print('3. phase 2, new partner: first vs last five rounds')
    for lbl, v in (('rounds 1-5', [r for r in p2 if r['rip'] <= 5]), ('rounds 6-10', [r for r in p2 if r['rip'] > 5])):
        print(f"   {lbl}: real {frac(v, 'real')}   neutral {frac(v, 'neu')}")
    print('4. individual success rates')
    for role in ('s', 'rc'):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[role]].append(r['real'])
        m = [np.mean(v) for v in g.values()]
        print(f"   as {'Sender' if role == 's' else 'Receiver'}: range {min(m):.0%}-{max(m):.0%}, SD {np.std(m, ddof=1):.2f}")
    print('5. counterfactual: Sender effect minus Receiver effect (Receivers resampled)')
    dec = [json.loads(l) for l in open(os.path.join(HERE, 'out', f'cf_rows_{a.model}.jsonl'))]
    byR = collections.defaultdict(list)
    for r in dec:
        if r['kind'] == 'decode':
            byR[(r['run'], r['R'])].append(r)

    def eff(v):
        c = {(s, q): np.mean([x['correct'] for x in v if x['sig'] == s and x['rec'] == q]) for s in ('early_sig', 'late_sig') for q in ('early_R', 'late_R')}
        se = (c['late_sig', 'early_R'] + c['late_sig', 'late_R'] - c['early_sig', 'early_R'] - c['early_sig', 'late_R']) / 2
        re = (c['early_sig', 'late_R'] + c['late_sig', 'late_R'] - c['early_sig', 'early_R'] - c['late_sig', 'early_R']) / 2
        return se - re
    keys, rng = list(byR), np.random.default_rng(0)
    B = [eff([x for i in rng.integers(0, len(keys), len(keys)) for x in byR[keys[i]]]) for _ in range(3000)]
    print(f"   difference {eff([x for v in byR.values() for x in v]):+.3f}, 95% CI [{np.percentile(B, 2.5):+.3f}, {np.percentile(B, 97.5):+.3f}]")


if __name__ == '__main__':
    main()
