#!/usr/bin/env python3
"""
How signalling styles move over time, in the single-turn LLM strategy embedding (strategy_embed.py).

Goal removed first: each turn's vector minus the mean vector of all turns with the same goal (every goal occurs 8x),
so distances reflect style, not where the goal was.

  spread      mean distance between turns of different players in the same run, by phase (collapse = falls)
  partner     mean distance between partners' turns vs same-run non-partners vs other-run players, by phase and by
              games played together
  model       next Sender turn of player i (3-D) regressed on: i's own previous turn, the CURRENT PARTNER's most recent
              turn, and the mean of everyone else in the run so far; placebo = a random same-run non-partner instead of
              the partner. Pull coefficients pooled over dimensions, SEs clustered by player.
Figures -> analysis/figures/convergence_*.png
"""
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import align_triplets as A  # noqa: E402

FIG = os.path.join(HERE, '..', 'figures')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'


def load(path='strategy_embedding.npz'):
    T, _, _ = A.build()
    d = np.load(os.path.join(HERE, 'out', path))
    emb = {k: x for k, x in zip(d['ids'], d['X'])}
    goal = defaultdict(list)
    for k, e in T.items():
        goal[(e['goal_tile'], e['goal_dir'])].append(k)
    gm = {g: np.mean([emb[k] for k in ks], 0) for g, ks in goal.items()}
    R = {k: emb[k] - gm[(e['goal_tile'], e['goal_dir'])] for k, e in T.items()}
    return T, R


def boot_mean(v, rng, B=2000):
    v = np.asarray(v)
    if len(v) < 2:
        return np.nan, np.nan
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    os.makedirs(FIG, exist_ok=True)
    T, R = load(os.environ.get('EMB', 'strategy_embedding.npz'))
    rng = np.random.default_rng(0)
    d = lambda a, b: float(np.linalg.norm(R[a] - R[b]))
    runs = sorted({e['run'] for e in T.values()})
    partner_of = {}
    for k, e in T.items():
        partner_of[(e['run'], e['phase'], e['s'])] = e['r']
        partner_of[(e['run'], e['phase'], e['r'])] = e['s']

    # ---------- distances by phase
    cat = defaultdict(list)
    for a in T:
        for b in T:
            if a >= b:
                continue
            ea, eb = T[a], T[b]
            if ea['phase'] != eb['phase'] or ea['s'] == eb['s'] and ea['run'] == eb['run']:
                continue
            ph = ea['phase']
            if ea['run'] != eb['run']:
                cat['other run', ph].append(d(a, b))
            elif partner_of.get((ea['run'], ph, ea['s'])) == eb['s']:
                cat['partners', ph].append(d(a, b))
            else:
                cat['same run, not partners', ph].append(d(a, b))
    print('mean distance between two players\' turns in the same phase (goal removed):')
    for c in ('partners', 'same run, not partners', 'other run'):
        print(f'  {c:24s} ' + '  '.join(f'ph{p}: {np.mean(cat[c, p]):.3f}' for p in (1, 2, 3)))

    # ---------- partner distance by games together (within a pairing: B's turns vs A's turns at similar time)
    gt = defaultdict(list)
    for a, ea in T.items():
        for b, eb in T.items():
            if eb['run'] == ea['run'] and eb['s'] == ea['r'] and eb['r'] == ea['s'] and abs(eb['round'] - ea['round']) <= 1:
                gt[min(ea['pg'] // 2, 6)].append(d(a, b))

    # ---------- per-player centroids by phase (for trajectories)
    X_all = np.array(list(R.values()))
    mu = X_all.mean(0)
    U, S, Vt = np.linalg.svd(X_all - mu, full_matrices=False)
    P = lambda v: (v - mu) @ Vt[:2].T
    print('PCA variance share (goal-removed):', np.round(S ** 2 / (S ** 2).sum(), 2))

    # ---------- convergence model
    rows = []
    for k, e in T.items():
        mine = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['s'] and T[x]['round'] < e['round']], key=lambda x: T[x]['round'])
        part = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['r'] and T[x]['round'] < e['round']], key=lambda x: T[x]['round'])
        others = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] not in (e['s'], e['r']) and T[x]['round'] < e['round']]
        if not mine or not part or not others:
            continue
        # placebo: a random same-run player who is neither i nor the partner, their most recent turn
        cands = sorted({T[x]['s'] for x in others})
        pl = rng.choice(cands)
        plt_ = sorted([x for x in others if T[x]['s'] == pl], key=lambda x: T[x]['round'])
        rows.append(dict(y=R[k], own=R[mine[-1]], partner=R[part[-1]], placebo=R[plt_[-1]],
                         group=np.mean([R[x] for x in others], 0), player=(e['run'], e['s']), phase=e['phase'], pg=e['pg']))
    import statsmodels.api as sm

    def fit(which, subset=lambda r: True):
        rr = [r for r in rows if subset(r)]
        Y = np.concatenate([r['y'] for r in rr])
        Xm = np.column_stack([np.concatenate([r['own'] for r in rr]), np.concatenate([r[which] for r in rr])])
        dim = np.concatenate([np.arange(len(r['y'])) for r in rr])
        Xm = np.column_stack([Xm, np.eye(len(rr[0]['y']))[dim]])  # per-dimension intercepts
        g = np.concatenate([[hash(r['player']) % 10 ** 8] * len(r['y']) for r in rr])
        m = sm.OLS(Y, Xm).fit(cov_type='cluster', cov_kwds={'groups': g})
        return m.params[:2], m.bse[:2], m.pvalues[:2], len(rr)
    print('\nnext turn ~ own previous + other-player latest (pooled dims, SE clustered by player):')
    res = {}
    for which, lab in (('partner', 'current partner'), ('placebo', 'random non-partner (placebo)')):
        b, se, p, n = fit(which)
        res[which] = (b, se)
        print(f'  with {lab:30s} own {b[0]:+.2f} (p={p[0]:.3f})  other {b[1]:+.2f} (p={p[1]:.3f})  n={n}')
    by_phase = {}
    for ph in (1, 2, 3):
        b, se, p, n = fit('partner', lambda r: r['phase'] == ph)
        by_phase[ph] = (b, se)
        print(f'  phase {ph}: own {b[0]:+.2f} (p={p[0]:.3f})  partner {b[1]:+.2f} (p={p[1]:.3f})  n={n}')

    # ================= FIGURES
    # 1) trajectories per run: player centroid per phase, arrows 1->2->3, faint turns
    fig, axs = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.4), sharex=True, sharey=True)
    for ax, run in zip(axs, runs):
        players = sorted({e['s'] for e in T.values() if e['run'] == run})
        for i, pl in enumerate(players):
            col = SERIES[i % len(SERIES)]
            ks = [k for k, e in T.items() if e['run'] == run and e['s'] == pl]
            pts = np.array([P(R[k]) for k in ks])
            ax.scatter(pts[:, 0], pts[:, 1], s=10, color=col, alpha=0.18, lw=0)
            cen = [P(np.mean([R[k] for k in ks if T[k]['phase'] == ph], 0)) for ph in (1, 2, 3) if any(T[k]['phase'] == ph for k in ks)]
            cen = np.array(cen)
            ax.plot(cen[:, 0], cen[:, 1], color=col, lw=2)
            ax.scatter(cen[:-1, 0], cen[:-1, 1], s=28, color=col, edgecolor='white', lw=1.5, zorder=3)
            ax.annotate('', xy=cen[-1], xytext=cen[-2], arrowprops=dict(arrowstyle='-|>', color=col, lw=2, mutation_scale=14))
        ax.set_title(f'Run {run}', color=INK, fontsize=11, loc='left')
        style(ax); ax.grid(color=GRID, lw=0.6)
    allp = np.array([P(v) for v in R.values()])
    lo, hi = np.percentile(allp, 1, axis=0), np.percentile(allp, 99, axis=0)
    axs[0].set_xlim(lo[0] - 0.3, hi[0] + 0.3); axs[0].set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    axs[0].set_ylabel('style axis 2', color=MUTED); [a.set_xlabel('style axis 1', color=MUTED) for a in axs]
    fig.suptitle('Each player\'s average signalling style, phase 1 → 2 → 3 (goal removed; dots = single turns)', color=INK, fontsize=12, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'convergence_trajectories.png'), dpi=160); plt.close(fig)

    # 2) distance by phase for the three relations
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, c in enumerate(('partners', 'same run, not partners', 'other run')):
        m = [np.mean(cat[c, p]) for p in (1, 2, 3)]
        ci = np.array([boot_mean(cat[c, p], rng) for p in (1, 2, 3)])
        ax.fill_between([1, 2, 3], ci[:, 0], ci[:, 1], color=SERIES[i], alpha=0.12, lw=0)
        ax.plot([1, 2, 3], m, color=SERIES[i], lw=2, marker='o', ms=6, mec='white', mew=1.5)
        ax.text(3.06, m[-1], c, color=INK, fontsize=9, va='center')
    ax.set_xticks([1, 2, 3], ['phase 1', 'phase 2', 'phase 3']); ax.set_xlim(0.85, 3.9)
    ax.set_ylabel('distance between two players\' turns\n(lower = more alike)', color=MUTED)
    ax.set_title('Do styles collapse? Distance between players, by relation and phase', color=INK, fontsize=11, loc='left')
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'convergence_distance_by_phase.png'), dpi=160); plt.close(fig)

    # 3) partner distance by games played together
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs = sorted(gt)
    m = [np.mean(gt[x]) for x in xs]; ci = np.array([boot_mean(gt[x], rng) for x in xs])
    base = np.mean(cat['other run', 1] + cat['other run', 2] + cat['other run', 3])
    ax.axhline(base, color=MUTED, lw=1, ls='--'); ax.text(xs[0], base + 0.03, 'strangers (other run)', color=MUTED, fontsize=9, va='bottom')
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color=SERIES[0], alpha=0.12, lw=0)
    ax.plot(xs, m, color=SERIES[0], lw=2, marker='o', ms=6, mec='white', mew=1.5)
    ax.set_xticks(xs, [f'{2 * x}-{2 * x + 1}' if x < 6 else '12+' for x in xs])
    ax.set_xlabel('games played together', color=MUTED); ax.set_ylabel('distance between partners\' turns', color=MUTED)
    ax.set_title('Partners over a pairing: do they get closer?', color=INK, fontsize=11, loc='left')
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'convergence_partners_by_games.png'), dpi=160); plt.close(fig)

    # 4) model coefficients
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labs = ['own previous turn', 'other player\'s latest turn']
    for j, (which, lab, col) in enumerate((('partner', 'other = current partner', SERIES[0]), ('placebo', 'other = random non-partner', SERIES[1]))):
        b, se = res[which]
        y = np.arange(2) + (0.15 if j == 0 else -0.15)
        ax.errorbar(b, y, xerr=1.96 * se, fmt='o', color=col, ms=7, mec='white', mew=1.5, lw=2, capsize=0, label=lab)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(2), labs); ax.invert_yaxis()
    ax.set_xlabel('pull on the next turn (regression weight, 95% CI)', color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title('What predicts a player\'s next signalling style?', color=INK, fontsize=11, loc='left')
    style(ax); ax.grid(axis='x', color=GRID); ax.grid(axis='y', visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'convergence_model.png'), dpi=160); plt.close(fig)
    print('\nfigures in', os.path.abspath(FIG))


if __name__ == '__main__':
    main()
