#!/usr/bin/env python3
"""Publication figures for paper/writeup.tex -> paper/figures/*.pdf (+ .png previews). All numbers recomputed from
analysis outputs in analysis/models/out/."""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, 'analysis', 'models')
OUT = os.path.join(MOD, 'out')
FIG = os.path.join(ROOT, 'paper', 'figures')
sys.path.insert(0, MOD)
import align_triplets as A  # noqa: E402
import convergence as CV  # noqa: E402
import fullgame_sonnet as F  # noqa: E402
import readable as RD  # noqa: E402

S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'axes.edgecolor': GRID, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': MUTED,
                     'text.color': INK, 'pdf.fonttype': 42})
rng = np.random.default_rng(0)


def style(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=170, bbox_inches='tight')
    plt.close(fig)


def boot(v, B=3000):
    v = np.asarray(v, float)
    return np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)], [2.5, 97.5])


# ---------------------------------------------------------------- 1. the game (example turn)
def fig_board():
    T, _, _ = A.build()
    # a clean "pointer" turn: wiggle from goal toward the goal direction
    ex = next(k for k, e in T.items() if e['goal_tile'] == 'top' and e['goal_dir'] == 'down' and 'wiggled top<->center x5' in F.moves(e['s_tokens']))
    e = T[ex]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    for (name, (x, y)) in RD.XY.items():
        ax.add_patch(Rectangle((x - 0.47, y - 0.47), 0.94, 0.94, fc='#f6f5f2' if name != e['goal_tile'] else '#dbe9fb', ec=GRID, lw=1))
    gx, gy = RD.XY[e['goal_tile']]
    ax.annotate('', xy=(gx, gy - 0.38), xytext=(gx, gy + 0.05), arrowprops=dict(arrowstyle='-|>', color=S[0], lw=2.2, mutation_scale=14))
    ax.text(gx + 0.62, gy + 0.62, 'goal tile +\ndirection', ha='left', va='top', fontsize=7.5, color=S[0])
    # sender path: centre -> top, then wiggle top<->centre
    path = [(0, 0), (0, 1)] + [(0, 0), (0, 1)] * 2
    for (x0, y0), (x1, y1), off in zip(path[:-1], path[1:], np.linspace(-0.12, 0.12, len(path) - 1)):
        ax.add_patch(FancyArrowPatch((x0 + off, y0 + (0.1 if y1 > y0 else -0.1)), (x1 + off, y1 + (-0.1 if y1 > y0 else 0.1)),
                                     arrowstyle='-|>', mutation_scale=9, color=S[1], lw=1.4, alpha=0.9))
    ax.text(0, -1.75, 'Sender: go to the goal, then step back and\nforth toward the goal direction', ha='center', fontsize=8, color=MUTED)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-2.1, 1.6); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('One round (Sender view)', loc='left')
    ax = axs[1]
    ax.axis('off'); ax.set_title('Tournament design', loc='left')
    rows = [('Phase 1', 'partner A, 10 rounds', 'roles alternate'),
            ('Phase 2', 'new partner B', '(1 pair in run 2 kept A)'),
            ('Phase 3', 'new partner C', 'or back to A (run 1; 1 pair run 2)')]
    for i, (a, b, c) in enumerate(rows):
        y = 0.82 - i * 0.27
        ax.add_patch(Rectangle((0.0, y - 0.09), 0.98, 0.2, transform=ax.transAxes, fc='#f6f5f2', ec=GRID))
        ax.text(0.03, y + 0.035, a, transform=ax.transAxes, fontsize=9, weight="bold")
        ax.text(0.03, y - 0.055, f"{b}; {c}", transform=ax.transAxes, fontsize=7.5, color=MUTED)
    ax.text(0.0, 0.0, '3 tournaments, 16 players, 240 rounds.\nReceivers only learn SUCCESS / FAIL.', transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, 'fig1_game')


# ---------------------------------------------------------------- 2. success: real vs neutral decoder (encoding)
def fig_success():
    rows = [json.loads(l) for l in open(os.path.join(OUT, 'cf_rows_gpt-6.1-terra.jsonl'))]
    enc = [r for r in rows if r['kind'] == 'encode']
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for key, lab, col in (('real', 'real Receivers', S[1]), ('correct', 'same neutral decoder\n(no history)', S[0])):
        m = [np.mean([r[key] for r in enc if r['phase'] == p]) for p in (1, 2, 3)]
        ci = np.array([boot([r[key] for r in enc if r['phase'] == p]) for p in (1, 2, 3)])
        ax.fill_between([1, 2, 3], ci[:, 0] * 100, ci[:, 1] * 100, color=col, alpha=0.12, lw=0)
        ax.plot([1, 2, 3], np.array(m) * 100, color=col, lw=2, marker='o', ms=5.5, mec='white', mew=1.3)
        ax.text(3.08, m[-1] * 100, lab, fontsize=8, va='center', color=INK)
    ax.set_xticks([1, 2, 3], ['phase 1', 'phase 2', 'phase 3']); ax.set_xlim(0.8, 4.3)
    ax.set_ylabel('rounds decoded correctly (%)'); ax.set_ylim(0, 80)
    ax.set_title('Signals get clearer every phase', loc='left')
    style(ax); save(fig, 'fig2_success')


# ---------------------------------------------------------------- 3. counterfactual decomposition
def fig_cf():
    R = [json.loads(l) for l in open(os.path.join(OUT, 'cf_rows_gpt-6.1-terra.jsonl'))]
    R = [r for r in R if r['kind'] == 'decode']
    keys = sorted({(r['run'], r['R']) for r in R})

    def cells(rows):
        return {(s, c): np.mean([x['correct'] for x in rows if x['sig'] == s and x['rec'] == c] or [np.nan]) for s in ('early_sig', 'late_sig') for c in ('early_R', 'late_R')}

    def eff(c):
        se = (c['late_sig', 'early_R'] + c['late_sig', 'late_R'] - c['early_sig', 'early_R'] - c['early_sig', 'late_R']) / 2
        re = (c['early_sig', 'late_R'] + c['late_sig', 'late_R'] - c['early_sig', 'early_R'] - c['late_sig', 'early_R']) / 2
        ix = (c['late_sig', 'late_R'] - c['late_sig', 'early_R']) - (c['early_sig', 'late_R'] - c['early_sig', 'early_R'])
        return np.array([se, re, ix])
    c0 = cells(R); e0 = eff(c0)
    by = defaultdict(list)
    for r in R:
        by[(r['run'], r['R'])].append(r)
    bs = np.array([eff(cells([x for i in rng.integers(0, len(keys), len(keys)) for x in by[keys[i]]])) for _ in range(3000)])
    lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axs[0]
    for j, (s, lab, col) in enumerate((('early_sig', 'phase-1 signals', S[1]), ('late_sig', 'phase-3 signals', S[0]))):
        v = [c0[s, 'early_R'] * 100, c0[s, 'late_R'] * 100]
        ax.plot([0, 1], v, color=col, lw=2, marker='o', ms=6, mec='white', mew=1.3)
        ax.text(1.06, v[1], lab, fontsize=8, va='center')
    ax.set_xticks([0, 1], ['early\nReceiver', 'late\nReceiver'])
    ax.set_xlim(-0.3, 1.8); ax.set_ylim(30, 62); ax.set_ylabel('decoded correctly (%)')
    ax.set_title('Same signals, different Receivers', loc='left'); style(ax)
    ax = axs[1]
    labs = ['Sender effect\n(encoding)', 'Receiver effect\n(decoding)', 'Interaction\n(shared convention)']
    y = np.arange(3)[::-1]
    ax.errorbar(e0 * 100, y, xerr=[(e0 - lo) * 100, (hi - e0) * 100], fmt='o', color=S[0], ms=6, mec='white', mew=1.3, lw=2, capsize=0)
    ax.axvline(0, color=MUTED, lw=1)
    for yy, v, h in zip(y, e0, hi):
        ax.text(h * 100 + 1, yy, f'{v * 100:+.0f} pts', va='center', fontsize=8)
    ax.set_xlim(-15, 32)
    ax.set_yticks(y, labs); ax.set_xlabel('change in success (percentage points, 95% CI)')
    ax.set_title('Where the phase-3 gain comes from', loc='left'); style(ax, grid='x')
    fig.tight_layout(); save(fig, 'fig3_counterfactual')


# ---------------------------------------------------------------- 4. embedding map with verbal groups
def fig_embedding():
    from sklearn.cluster import KMeans
    T, _, _ = A.build()
    d = np.load(os.path.join(OUT, 'strategy_embedding.npz'))
    ids = list(d['ids']); X = d['X']
    lab = KMeans(4, n_init=20, random_state=0).fit(X).labels_
    names = {3: 'Pointers', 1: 'Right axis,\nunclear end', 2: 'Still /\nminimal', 0: 'Busy /\ncircling'}
    order = [3, 1, 2, 0]
    Xc = X - X.mean(0); U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False); Z = Xc @ Vt[:2].T
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axs[0]
    for j, c in enumerate(order):
        m = lab == c
        ax.scatter(Z[m, 0], Z[m, 1], s=14, color=S[j], alpha=0.75, lw=0)
        cx, cy = Z[m].mean(0)
        ax.text(cx, cy, names[c], fontsize=8, ha='center', va='center', weight='bold', color=INK,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=S[j], lw=1, alpha=0.9))
    lo_, hi_ = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
    ax.set_xlim(lo_[0] - 0.4, hi_[0] + 0.4); ax.set_ylim(lo_[1] - 0.4, hi_[1] + 0.4)
    ax.set_xlabel('embedding axis 1'); ax.set_ylabel('embedding axis 2')
    ax.set_title('240 Sender turns placed by LLM similarity judgments', loc='left'); style(ax, grid=None)
    ax = axs[1]
    succ = [100 * np.mean([T[k]['succ'] for k, l in zip(ids, lab) if l == c]) for c in order]
    n = [int((lab == c).sum()) for c in order]
    y = np.arange(4)[::-1]
    ax.barh(y, succ, color=[S[j] for j in range(4)], height=0.55)
    for yy, v, nn in zip(y, succ, n):
        ax.text(v + 1, yy, f'{v:.0f}%  (n={nn})', va='center', fontsize=8)
    ax.set_yticks(y, [names[c].replace('\n', ' ') for c in order]); ax.set_xlim(0, 60)
    ax.set_xlabel('rounds that succeeded (%)'); ax.set_title('Success by group', loc='left'); style(ax, grid='x')
    fig.tight_layout(); save(fig, 'fig4_embedding')


# ---------------------------------------------------------------- 5. convergence
def fig_convergence():
    T, R = CV.load()
    partner_of = {}
    for k, e in T.items():
        partner_of[(e['run'], e['phase'], e['s'])] = e['r']; partner_of[(e['run'], e['phase'], e['r'])] = e['s']
    cat = defaultdict(list)
    ks = list(T)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            ea, eb = T[a], T[b]
            if ea['phase'] != eb['phase'] or (ea['s'] == eb['s'] and ea['run'] == eb['run']):
                continue
            dd = float(np.linalg.norm(R[a] - R[b])); ph = ea['phase']
            if ea['run'] != eb['run']:
                cat['other tournament', ph].append(dd)
            elif partner_of.get((ea['run'], ph, ea['s'])) == eb['s']:
                cat['partners', ph].append(dd)
            else:
                cat['same tournament, not partners', ph].append(dd)
    # model
    import statsmodels.api as sm
    rows = []
    rr = np.random.default_rng(0)
    for k, e in T.items():
        mine = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['s'] and T[x]['round'] < e['round']], key=lambda x: T[x]['round'])
        part = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['r'] and T[x]['round'] < e['round']], key=lambda x: T[x]['round'])
        others = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] not in (e['s'], e['r']) and T[x]['round'] < e['round']]
        if not mine or not part or not others:
            continue
        pl = rr.choice(sorted({T[x]['s'] for x in others}))
        plt_ = sorted([x for x in others if T[x]['s'] == pl], key=lambda x: T[x]['round'])
        rows.append(dict(y=R[k], own=R[mine[-1]], partner=R[part[-1]], placebo=R[plt_[-1]], player=(e['run'], e['s'])))

    def fit(which):
        Y = np.concatenate([r['y'] for r in rows]); D = len(rows[0]['y'])
        M = np.column_stack([np.concatenate([r['own'] for r in rows]), np.concatenate([r[which] for r in rows]),
                             np.eye(D)[np.concatenate([np.arange(D)] * len(rows))]])
        g = np.concatenate([[hash(r['player']) % 10 ** 8] * D for r in rows])
        m = sm.OLS(Y, M).fit(cov_type='cluster', cov_kwds={'groups': g})
        return m.params[:2], m.bse[:2]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={'width_ratios': [1.2, 1]})
    ax = axs[0]
    for i, c in enumerate(('partners', 'same tournament, not partners', 'other tournament')):
        m = [np.mean(cat[c, p]) for p in (1, 2, 3)]; ci = np.array([boot(cat[c, p], 1500) for p in (1, 2, 3)])
        ax.fill_between([1, 2, 3], ci[:, 0], ci[:, 1], color=S[i], alpha=0.12, lw=0)
        ax.plot([1, 2, 3], m, color=S[i], lw=2, marker='o', ms=5.5, mec='white', mew=1.3)
        ax.text(3.08, m[-1] + (0.04 if i == 1 else (-0.04 if i == 2 else 0)), c, fontsize=7.5, va='center')
    ax.set_xticks([1, 2, 3], ['phase 1', 'phase 2', 'phase 3']); ax.set_xlim(0.8, 4.6)
    ax.set_ylabel('style distance between two players'); ax.set_title('Partners stay closest', loc='left'); style(ax)
    ax = axs[1]
    for j, (w, lab, col) in enumerate((('partner', 'other = current partner', S[0]), ('placebo', 'other = random non-partner', S[1]))):
        b, se = fit(w)
        y = np.array([1, 0]) + (0.13 if j == 0 else -0.13)
        ax.errorbar(b, y, xerr=1.96 * se, fmt='o', color=col, ms=6, mec='white', mew=1.3, lw=2, capsize=0, label=lab)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks([1, 0], ['own previous\nturn', "other player's\nlatest turn"])
    ax.set_xlabel('weight on next turn (95% CI)'); ax.legend(frameon=False, fontsize=7.5, loc='center left', bbox_to_anchor=(0.0, 0.5))
    ax.set_title('What predicts the next turn', loc='left'); style(ax, grid='x')
    fig.tight_layout(); save(fig, 'fig5_convergence')


# ---------------------------------------------------------------- 6. LLM models of players
def fig_llm():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axs[0]
    models = ['History-free\nbaseline', 'Claude\nHaiku 5.5', 'Claude\nSonnet 5.5', 'GPT-6.1\nTerra']
    snd = [43, 41, 50, 57]; rec = [49, 43, 57, 63]
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2 - 0.01, snd, w, color=S[0], label='Sender strategy')
    ax.bar(x + w / 2 + 0.01, rec, w, color=S[2], label='Receiver reading')
    for xx, a, b in zip(x, snd, rec):
        ax.text(xx - w / 2, a + 1, f'{a}', ha='center', fontsize=7.5); ax.text(xx + w / 2, b + 1, f'{b}', ha='center', fontsize=7.5)
    ax.set_xticks(x, models, fontsize=8); ax.set_ylim(0, 75); ax.set_ylabel('next-turn prediction accuracy (%)')
    ax.legend(frameon=False, fontsize=8, loc='upper left'); ax.set_title('Predicting each real turn from real history', loc='left'); style(ax)
    ax = axs[1]
    labs = ['LLM prediction vs\nplayer\'s previous turn', 'Same-goal stranger vs\nplayer\'s previous turn\n(control)', 'LLM next turn vs\nplayer\'s REAL next turn']
    vals = [70, 53, 58]
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=[S[0], MUTED, S[1]], height=0.55)
    ax.axvline(50, color=INK, lw=1, ls='--')
    for yy, v in zip(y, vals):
        ax.text(v + 1, yy, f'{v}%', va='center', fontsize=8)
    ax.set_yticks(y, labs, fontsize=7.5); ax.set_xlim(0, 85)
    ax.set_xlabel('first option judged closer (%)'); ax.set_title('Judge head-to-heads (n=224)', loc='left'); style(ax, grid='x')
    fig.tight_layout(); save(fig, 'fig6_llm')


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_board, fig_success, fig_cf, fig_embedding, fig_convergence, fig_llm):
        f(); print('done', f.__name__, flush=True)
