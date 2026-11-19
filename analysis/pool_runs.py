#!/usr/bin/env python3
"""Pool coded tournament runs and compare across them.
Usage: python3 analysis/pool_runs.py analysis/<run1> analysis/<run2> ...   (folders written by analyze_tournament.py)"""
import sys, os
import pandas as pd
from scipy.stats import fisher_exact

runs = []
for i, d in enumerate(sys.argv[1:], 1):
    df = pd.read_csv(os.path.join(d, 'coded_rounds.csv'))
    df['run'] = i
    runs.append(df)
cr = pd.concat(runs)
cr['diag'] = cr.goal_direction.str.contains('-')
cr['reunion'] = cr.separation.astype(str).eq('1')
cr['pair_id'] = cr.run.astype(str) + ':' + cr.phase.astype(str) + ':' + cr.pair

def rate(g): return f"{g.success.mean():.0%} ({int(g.success.sum())}/{len(g)})"
def fisher(a, b):
    t = [[a.success.sum(), len(a) - a.success.sum()], [b.success.sum(), len(b) - b.success.sum()]]
    return fisher_exact(t)[1]

print(f'{len(cr)} games from {cr.run.nunique()} runs, {cr.pair_id.nunique()} partnerships')
print(f'Overall success {rate(cr)}; tile right {cr.tile_ok.mean():.0%}; direction right {cr.dir_ok.mean():.0%}')
for r, g in cr.groupby('run'): print(f'  run {r}: {rate(g)}')
print('\nCardinal vs diagonal goal direction:')
a, b = cr[~cr.diag], cr[cr.diag]
print(f'  cardinal {rate(a)}   diagonal {rate(b)}   Fisher p = {fisher(a, b):.3f}')

print('\nReunited vs new partner (phases 2-3):')
later = cr[cr.phase > 1]
for ph, g in later.groupby('phase'):
    re_, new = g[g.reunion], g[~g.reunion]
    print(f'  phase {ph}: reunited {rate(re_) if len(re_) else "-"}   new {rate(new) if len(new) else "-"}')
re_, new = later[later.reunion], later[~later.reunion]
print(f'  all: reunited {rate(re_)}   new {rate(new)}   Fisher p = {fisher(re_, new):.3f}')
p1 = cr[cr.phase == 1]
print(f'  phase 1 (everyone new) {rate(p1)};  new partners in phase 3 only: {rate(later[(later.phase == 3) & ~later.reunion])}')

print('\nSender strategies (pooled):')
for s, g in cr.groupby('strategy'):
    print(f'  {s:14s} {len(g):3d} games  success {rate(g)}   direction right {g.dir_ok.mean():.0%}')
gf, gs = cr[cr.strategy == 'goal-wiggle-F'], cr[cr.strategy == 'goal-wiggle-S']
print(f'  goal-wiggle-F vs -S: Fisher p = {fisher(gf, gs):.4f}')

print('\nStrategy share by phase (pooled, % of Sender turns):')
tab = pd.crosstab(cr.phase, cr.strategy, normalize='index').mul(100).round(0)
print('  ' + tab.to_string().replace('\n', '\n  '))

print('\nHow Receivers read goal wiggles (pooled):')
gw = cr[cr.strategy.isin(['goal-wiggle-F', 'goal-wiggle-B'])]
toward = (gw.receiver_reading == 'reads-F').sum(); away = (gw.receiver_reading == 'reads-B').sum(); side = (gw.receiver_reading == 'reads-S').sum()
print(f'  triangle pointed TOWARD the wiggle arm {toward}x, AWAY from it {away}x, sideways {side}x')
for (r, ph), g in gw.groupby(['run', 'phase']):
    print(f'    run {r} phase {ph}: toward {(g.receiver_reading == "reads-F").sum()}, away {(g.receiver_reading == "reads-B").sum()}')

print('\nCentre-dir users (one per run so far) and their partners:')
cd = cr[cr.strategy == 'centre-dir']
for name, g in cd.groupby(['run', 'sender']):
    print(f'  run {name[0]} {name[1]}: {len(g)} turns')
    for ph, h in g.groupby('phase'):
        print(f'    phase {ph} -> receiver {h.receiver.iloc[0]} ({"reunited" if h.reunion.iloc[0] else "new"}): {rate(h)}')
