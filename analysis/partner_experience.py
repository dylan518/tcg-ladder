#!/usr/bin/env python3
"""Does more time with the same partner raise the chance of success?  Pools coded tournament runs.

Usage: python3 analysis/partner_experience.py analysis/<run1> analysis/<run2> ... [--out analysis/pooled]

For every game: partner_games = games this pair had already played together before this one (a pair reunited in a
later phase carries over its earlier games, so reunions give 10-19), and round = overall practice (1-30).
Within a single partnership these two move together; reunions are what pull them apart.

Models (logistic regression, standard errors clustered by partnership because games within a pair are not independent):
  1. success ~ partner_games
  2. success ~ partner_games + diagonal goal
  3. success ~ partner_games + diagonal goal + round (overall practice)
  4. same-goal comparison: linear probability model with a fixed effect per goal (tile + direction) and per run, so pairs
     are compared on the identical goal. Runs 1-2 used one fixed goal order (goal = round there); later runs shuffle
     the order per session, which also lets general practice (round) be separated from goal difficulty.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ap = argparse.ArgumentParser()
ap.add_argument('runs', nargs='+')
ap.add_argument('--out', default='analysis/pooled')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cr = pd.concat([pd.read_csv(os.path.join(d, 'coded_rounds.csv')).assign(run=i) for i, d in enumerate(a.runs, 1)])
cr['pair_key'] = cr.run.astype(str) + ':' + cr.apply(lambda r: '&'.join(sorted([r.sender, r.receiver])), axis=1)
cr = cr.sort_values(['run', 'round'])
cr['partner_games'] = cr.groupby('pair_key').cumcount()          # games already played together
cr['diagonal'] = cr.goal_direction.str.contains('-').astype(int)
cr['partnership'] = cr.run.astype(str) + ':' + cr.phase.astype(str) + ':' + cr.pair  # cluster unit (one partnership spell)
cr['success'] = cr.success.astype(int)
cr['goal'] = cr.goal_tile + '/' + cr.goal_direction  # goal identity (orders are shuffled per run from 2026-11-11 on)

L = []
p = L.append
p(f'{len(cr)} games, {cr.run.nunique()} runs, {cr.pair_key.nunique()} distinct pairs, {cr.partnership.nunique()} partnership spells')
p(f'partner_games ranges 0-{cr.partner_games.max()}; games with 10+ prior games together (reunions): {(cr.partner_games >= 10).sum()}')
p('')
p('Success by games already played with this partner:')
bins = [(-1, 1, '0-1'), (1, 3, '2-3'), (3, 5, '4-5'), (5, 7, '6-7'), (7, 9, '8-9'), (9, 14, '10-14 (reunions)'), (14, 19, '15-19 (reunions)')]
rows = []
for lo, hi, lab in bins:
    g = cr[(cr.partner_games > lo) & (cr.partner_games <= hi)]
    if len(g):
        rows.append((lab, g.success.mean(), len(g), (lo + hi + 1) / 2))
        p(f'  {lab:18s} {g.success.mean():4.0%}  ({g.success.sum()}/{len(g)})')
p('')

def fit(formula):
    m = smf.logit(formula, data=cr).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(cr.partnership)[0]})
    return m

for i, f in enumerate(['success ~ partner_games', 'success ~ partner_games + diagonal', 'success ~ partner_games + diagonal + round'], 1):
    m = fit(f)
    p(f'Model {i}: {f}')
    for v in m.params.index:
        if v == 'Intercept':
            continue
        orr = np.exp(m.params[v]); lo, hi = np.exp(m.conf_int().loc[v])
        p(f'  {v:14s} odds ratio {orr:.2f} per unit  (95% CI {lo:.2f}-{hi:.2f})  p = {m.pvalues[v]:.3f}')
    p('')

groups = pd.factorize(cr.partnership)[0]
m4 = smf.ols('success ~ partner_games + C(goal) + C(run)', data=cr).fit(cov_type='cluster', cov_kwds={'groups': groups})
lo, hi = m4.conf_int().loc['partner_games'] * 100
p(f'Model 4 (same goal: goal + run fixed effects): {m4.params["partner_games"] * 100:+.1f} percentage points per game together '
  f'(95% CI {lo:.1f} to {hi:.1f}), p = {m4.pvalues["partner_games"]:.3f}  -> 10 more games together = {m4.params["partner_games"] * 1000:+.0f} points')
later = cr[cr.phase > 1].copy(); later['reunited'] = (later.partner_games >= 10).astype(int)
m5 = smf.ols('success ~ reunited + C(goal) + C(run)', data=later).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(later.partnership)[0]})
lo, hi = m5.conf_int().loc['reunited'] * 100
p(f'Reunited vs new pair on the same goal (phases 2-3): {m5.params["reunited"] * 100:+.0f} points (95% CI {lo:.0f} to {hi:.0f}), p = {m5.pvalues["reunited"]:.3f}')
p(f'Raw (no goal control): reunited {later[later.reunited == 1].success.mean():.0%} vs new {later[later.reunited == 0].success.mean():.0%}')
p('')

m = fit('success ~ partner_games + diagonal')
pg = np.arange(0, 20)
pred = m.predict(pd.DataFrame({'partner_games': pg, 'diagonal': 0}))
pred_d = m.predict(pd.DataFrame({'partner_games': pg, 'diagonal': 1}))
p('Model 2 predicted success, cardinal goal: ' + ', '.join(f'{k} games {pred[k]:.0%}' for k in (0, 5, 9, 15, 19)))
p('Model 2 predicted success, diagonal goal: ' + ', '.join(f'{k} games {pred_d[k]:.0%}' for k in (0, 5, 9, 15, 19)))

txt = '\n'.join(L)
open(os.path.join(a.out, 'partner_experience.txt'), 'w').write(txt + '\n')
print(txt)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(pg, pred * 100, color='#2b4c7e', lw=2, label='model: up/down/left/right goal')
ax.plot(pg, pred_d * 100, color='#e39a4a', lw=2, ls='--', label='model: diagonal goal')
for lab, rate, n, x in rows:
    ax.scatter(x, rate * 100, s=n * 3, color='#555', alpha=.55, zorder=3)
ax.axvline(9.5, color='#bbb', ls=':', lw=1)
ax.text(9.7, 92, 'beyond here: reunited pairs', fontsize=8, color='#777')
ax.set_xlabel('games already played with this partner'); ax.set_ylabel('% success')
ax.set_ylim(0, 100); ax.set_xlim(-0.5, 19.5)
ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Success vs time with partner (dots: observed, size = games)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(a.out, 'partner_experience.png'), dpi=150)
print(f'\nwrote {a.out}/partner_experience.txt and partner_experience.png')
