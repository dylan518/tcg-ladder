#!/usr/bin/env python3
"""Figures added for the APA report / final talk (shared-understanding framing).
   figH_hypotheses   predictions of the three accounts (code / pact / asymmetric)
   figG_gap          real success vs neutral-reader legibility, phase-2 gap annotated (with n)
   figR_reunion      phase-3 new vs reunited partners: success and neutral legibility
   figM_model        diagram of the style-update model with fitted weights
   figS_sim          fitted group model simulated on the real schedules vs real data
Run after make_figures.py (reuses its style helpers)."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import make_figures as MF

S, INK, MUTED, GRID, OUT, style, save, boot = MF.S, MF.INK, MF.MUTED, MF.GRID, MF.OUT, MF.style, MF.save, MF.boot
PH = [1, 2, 3]
PHL = ['Phase 1\nfirst partner', 'Phase 2\nnew partner', 'Phase 3\nnew or reunited']


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ------------------------------------------------------------------ hypotheses
def fig_hyp():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    acc = [
        ('A. Code', 'signalling and reading\nboth carry over',
         [30, 42, 52], [36, 46, 56]),
        ('B. Pact', 'neither carries over:\nresets with each new partner',
         [30, 24, 30], [36, 36, 36]),
        ('C. Asymmetric', 'signalling carries over,\nreading must be relearned',
         [30, 24, 48], [36, 46, 60]),
    ]
    for ax, (t, sub, succ, leg) in zip(axs, acc):
        ax.plot(PH, leg, color=S[0], lw=2, marker='o', ms=5, mec='white', mew=1.2)
        ax.plot(PH, succ, color=S[1], lw=2, marker='o', ms=5, mec='white', mew=1.2, ls='--' if t != 'C. Asymmetric' else '-')
        if t.startswith('C'):
            ax.annotate('', xy=(2, succ[1] + 2), xytext=(2, leg[1] - 2), arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1))
            ax.text(2.08, (succ[1] + leg[1]) / 2, 'gap', fontsize=7.5, color=MUTED, va='center')
        ax.set_title(t, loc='left', fontsize=9.5, fontweight='bold')
        ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=7, color=MUTED, va='bottom') if False else None
        ax.text(1, 75, sub, fontsize=7, color=MUTED, va='top')
        ax.set_xticks(PH, ['P1', 'P2', 'P3']); ax.set_ylim(10, 78); ax.set_xlim(0.7, 3.3)
        style(ax)
    axs[0].set_ylabel('predicted % correct')
    axs[2].text(3.1, 60, 'neutral\nreader', color=S[0], fontsize=7.5, va='center')
    axs[2].text(3.1, 46, 'real\npartner', color=S[1], fontsize=7.5, va='center')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'figH_hypotheses')


# ------------------------------------------------------------------ phase-2 gap
def fig_gap():
    rows = [json.loads(l) for l in open(os.path.join(OUT, 'cf_rows_gpt-6.1-terra.jsonl'))]
    enc = [r for r in rows if r['kind'] == 'encode']
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    vals = {}
    for key, lab, col in (('correct', 'neutral reader (no history)', S[0]), ('real', 'real partner (Receiver)', S[1])):
        m, lo, hi, ns = [], [], [], []
        for p in PH:
            v = [r[key] for r in enc if r['phase'] == p]
            k, n = sum(v), len(v)
            a, b = wilson(k, n)
            m.append(k / n * 100); lo.append(a * 100); hi.append(b * 100); ns.append(n)
        vals[key] = m
        ax.fill_between(PH, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(PH, m, color=col, lw=2.2, marker='o', ms=6, mec='white', mew=1.3)
        for x, y in zip(PH, m):
            ax.text(x, y + (4 if key == 'correct' else -7), f'{y:.0f}%', ha='center', fontsize=8, color=col, fontweight='bold')
        ax.text(3.12, m[-1], lab, fontsize=7.8, va='center')
    ax.annotate('', xy=(2, vals['real'][1] + 1.5), xytext=(2, vals['correct'][1] - 1.5),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=1.1))
    ax.text(1.62, (vals['real'][1] + vals['correct'][1]) / 2, 'readable but\nmisread', fontsize=7.5, ha='center', va='center', color=INK)
    ax.set_xticks(PH, PHL); ax.set_xlim(0.75, 4.35); ax.set_ylim(0, 82)
    ax.set_ylabel('signals decoded correctly (%)')
    ax.set_title('Signals get clearer; new partners lag behind', loc='left')
    ax.text(0.75, -26, 'n = 80 Sender turns per phase; bands = 95% Wilson CIs.', fontsize=7, color=MUTED)
    style(ax); save(fig, 'figG_gap')


# ------------------------------------------------------------------ reunion
def fig_reunion():
    # phase-3 counts (PROJECT_NOTES 2026-11-12): success new 16/40, reunited 23/40;
    # neutral (no-history) decoder on the same signals: new 19/40 (48%), reunited 30/40 (75%)
    data = {'real partner success': (16, 23), 'neutral reader': (19, 30)}
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    x = np.arange(2)
    w = 0.36
    for j, (lab, (kn, kr)) in enumerate(data.items()):
        col = S[1] if j == 0 else S[0]
        for i, k in enumerate((kn, kr)):
            a, b = wilson(k, 40)
            xx = x[i] + (j - 0.5) * w
            ax.bar(xx, k / 40 * 100, w * 0.92, color=col, alpha=0.85 if i else 0.55)
            ax.plot([xx, xx], [a * 100, b * 100], color=INK, lw=1)
            ax.text(xx, b * 100 + 2, f'{k / 40 * 100:.0f}%', ha='center', fontsize=7.5)
    ax.set_xticks(x, ['new partner\n(n = 40 rounds)', 'reunited with\nphase-1 partner (n = 40)'])
    ax.set_ylim(0, 100); ax.set_ylabel('% correct in phase 3')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S[1], label='real partner'), Patch(color=S[0], label='neutral reader')],
              frameon=False, fontsize=7.5, loc='upper left')
    ax.set_title('Reunions help, but their signals were\nclearer to anyone (confound)', loc='left', fontsize=9)
    style(ax); save(fig, 'figR_reunion')


# ------------------------------------------------------------------ model diagram
def fig_model():
    g = json.load(open(os.path.join(OUT, 'group_model.json')))
    a = g['params']['a']
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')

    def box(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.15', fc=fc, ec=GRID, lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8.5)
    box(0.0, 2.85, 3.0, 0.85, 'my last turn', '#eef4fc')
    box(0.0, 1.55, 3.0, 0.85, "my partner's last turn", '#fdeee6')
    box(0.0, 0.25, 3.0, 0.85, 'my past turns\nthat worked', '#e8f6f0')
    box(6.6, 1.55, 3.2, 0.85, 'my next turn\n(position in style map)', '#f6f5f2')
    for y, w_, col, lab in ((3.27, a[0], S[0], 'own'), (1.97, a[1], S[1], 'partner'), (0.67, a[2], S[2], 'success')):
        ax.add_patch(FancyArrowPatch((3.05, y), (6.55, 1.97), arrowstyle='-|>', mutation_scale=12, lw=1 + 10 * w_, color=col, alpha=0.8,
                                     connectionstyle='arc3,rad=0'))
        ax.text(4.55, (y + 1.97) / 2 + 0.18, f'{w_:.2f}', fontsize=8.5, color=col, fontweight='bold', ha='center')
    ax.text(0.1, -0.25, 'Weights fitted on 224 Sender turns (goal removed). Replacing the partner with a random non-partner gives -0.08.',
            fontsize=7, color=MUTED)
    save(fig, 'figM_model')


# ------------------------------------------------------------------ model simulation vs real
def fig_sim():
    g = json.load(open(os.path.join(OUT, 'group_model.json')))
    real, sims = g['real'], g['sims']
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7))
    ax = axs[0]
    ax.plot(PH, [real['succ'][str(p)] * 100 for p in PH], color=INK, lw=2.2, marker='o', ms=6, mec='white', label='real players')
    for name, col in (('full model', S[0]), ('no partner pull', S[3])):
        m = [sims[name]['succ'][str(p)][0] * 100 for p in PH]
        lo = [sims[name]['succ'][str(p)][1] * 100 for p in PH]
        hi = [sims[name]['succ'][str(p)][2] * 100 for p in PH]
        ax.fill_between(PH, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(PH, m, color=col, lw=1.8, marker='o', ms=4.5, mec='white', label=name)
    ax.set_xticks(PH, ['P1', 'P2', 'P3']); ax.set_ylim(0, 65); ax.set_ylabel('success (%)')
    ax.set_title('Success curve: not reproduced', loc='left'); ax.legend(frameon=False, fontsize=7.5, loc='upper left'); style(ax)
    ax = axs[1]
    rg = [real['dist'][str(p)][1] - real['dist'][str(p)][0] for p in PH]
    ax.plot(PH, rg, color=INK, lw=2.2, marker='o', ms=6, mec='white', label='real players')
    for name, col in (('full model', S[0]), ('no partner pull', S[3])):
        m = [sims[name]['gap'][str(p)][0] for p in PH]
        lo = [sims[name]['gap'][str(p)][1] for p in PH]
        hi = [sims[name]['gap'][str(p)][2] for p in PH]
        ax.fill_between(PH, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(PH, m, color=col, lw=1.8, marker='o', ms=4.5, mec='white', label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(PH, ['P1', 'P2', 'P3']); ax.set_ylabel('partners closer than non-partners\n(style distance)')
    ax.set_title('Partner closeness: reproduced', loc='left'); style(ax)
    fig.text(0.01, -0.04, 'Model simulated 300 times on each tournament\'s real partner schedule; bands = 95% of simulations.', fontsize=7, color=MUTED)
    fig.tight_layout(); save(fig, 'figS_sim')


if __name__ == '__main__':
    fig_hyp(); fig_gap(); fig_reunion(); fig_model(); fig_sim()
    print('ok')


# ------------------------------------------------------------------ group lexicon (hand-coded forms by tournament and phase)
def fig_lexicon():
    import csv, glob
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(ROOT, 'analysis', '2026-11-11-*_tournament', 'coded_rounds.csv')))
    groups = [('pointer wiggle\n(toward goal direction)', {'goal-wiggle-F'}, S[0]),
              ('side wiggle', {'goal-wiggle-S'}, '#8fbcf2'),
              ('backward wiggle', {'goal-wiggle-B'}, S[3]),
              ('centre mark', {'centre-dir'}, S[2]),
              ('go to goal only', {'goal-only'}, '#b9b6ad'),
              ('other', {'wiggle-other', 'other', 'none'}, '#e4e3df')]
    data = {}
    for i, f in enumerate(files, 1):
        for row in csv.DictReader(open(f)):
            data.setdefault((i, int(row['phase'])), []).append(row['strategy'])
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.8), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1.15]})
    for j, ax in enumerate(axs):
        for p in PH:
            v = sum((data[(r, p)] for r in (1, 2, 3)), []) if j == 3 else data[(j + 1, p)]
            bottom = 0
            for lab, keys, col in groups:
                h = 100 * sum(x in keys for x in v) / len(v)
                ax.bar(p, h, 0.72, bottom=bottom, color=col, edgecolor='white', lw=0.6)
                if lab.startswith('pointer') and h >= 12:
                    ax.text(p, bottom + h / 2, f'{h:.0f}%', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
                bottom += h
        ax.set_xticks(PH, ['P1', 'P2', 'P3'])
        ax.set_title('All tournaments' if j == 3 else f'Tournament {j + 1}', loc='left', fontsize=9, fontweight='bold' if j == 3 else 'normal')
        style(ax)
    axs[0].set_ylabel('% of Sender turns'); axs[0].set_ylim(0, 100)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l.replace('\n', ' ')) for l, _, c in groups], loc='lower center', ncol=3,
               frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, 'figL_lexicon')


if __name__ == '__main__':
    fig_lexicon()


# ------------------------------------------------------------------ Figure 2 redo: task + design
def fig_task():
    from matplotlib.patches import Rectangle, Polygon, Circle
    fig = plt.figure(figsize=(7.2, 3.3))
    # --- boards
    def board(ax, title, col):
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((i, j), 0.96, 0.96, fc='#f6f5f2', ec='#d9d6ce', lw=1))
        ax.set_xlim(-0.1, 3.05); ax.set_ylim(-0.75, 3.35); ax.set_aspect('equal'); ax.axis('off')
        ax.text(0, 3.12, title, fontsize=9, fontweight='bold', color=col)
    axS = fig.add_axes([0.0, 0.08, 0.25, 0.86]); board(axS, 'SENDER sees the goal', S[0])
    axS.add_patch(Rectangle((0, 1), 0.96, 0.96, fc='#dbe9fb', ec='#d9d6ce', lw=1))      # goal tile: middle-left, facing right
    axS.add_patch(Polygon([[0.74, 1.48], [0.3, 1.27], [0.3, 1.69]], closed=True, fc='none', ec=S[0], lw=1.6, ls='--'))
    axS.add_patch(Circle((1.48, 1.48), 0.15, fc=INK))
    axS.add_patch(FancyArrowPatch((1.3, 1.78), (0.55, 1.78), arrowstyle='-|>', mutation_scale=10, lw=1.6, color=S[1]))
    axS.add_patch(FancyArrowPatch((0.55, 1.14), (1.45, 1.14), arrowstyle='<|-|>', mutation_scale=10, lw=1.6, color=S[1]))
    axS.text(1.5, 2.55, '1. go to the goal tile', fontsize=6.8, color=S[1], ha='center')
    axS.text(1.5, 0.42, '2. step back and forth toward\nthe way the goal points', fontsize=6.8, color=S[1], ha='center', va='center')
    axS.text(0, -0.3, 'Moves a circle with the\narrow keys (~9 s). No words.', fontsize=7, color=MUTED, va='top')
    axR = fig.add_axes([0.25, 0.08, 0.25, 0.86]); board(axR, 'RECEIVER watches, then acts', S[1])
    axR.add_patch(Polygon([[0.74, 1.48], [0.3, 1.27], [0.3, 1.69]], closed=True, fc=S[1], ec='none'))
    axR.text(0, -0.3, 'Places and turns a triangle.\nBoth see only SUCCESS / FAIL.', fontsize=7, color=MUTED, va='top')
    # --- design timeline (one 6-player tournament)
    ax = fig.add_axes([0.53, 0.1, 0.46, 0.84]); ax.axis('off'); ax.set_xlim(0, 3.2); ax.set_ylim(-0.6, 7.2)
    ax.text(0, 6.9, 'TOURNAMENT DESIGN (example, 6 players)', fontsize=8, fontweight='bold', color=INK)
    players = 'ABCDEF'
    pairs = [['AB', 'CD', 'EF'], ['AC', 'BE', 'DF'], ['AB', 'CD', 'EF']]
    reun = [False, False, True]
    titles = ['Phase 1\nrounds 1-10', 'Phase 2\nrounds 11-20', 'Phase 3\nrounds 21-30']
    pc = ['#2a78d6', '#eb6834', '#1baf7a']
    for p in range(3):
        x = p * 1.05 + 0.05
        ax.text(x + 0.47, 6.1, titles[p], ha='center', va='center', fontsize=7.2, color=INK)
        for q, pr in enumerate(pairs[p]):
            y = 4.7 - q * 1.55
            ax.add_patch(FancyBboxPatch((x, y), 0.94, 1.15, boxstyle='round,pad=0.02,rounding_size=0.08',
                                        fc=pc[q] if not (p == 1) else '#f6f5f2', ec=pc[q] if p != 1 else '#b9b6ad', lw=1.2, alpha=0.18 if p != 1 else 1,
                                        hatch='////' if reun[p] else None))
            ax.text(x + 0.47, y + 0.58, f'{pr[0]} & {pr[1]}', ha='center', va='center', fontsize=8.5, fontweight='bold', color=INK)
    ax.text(1.62, 1.35, 'all new partners', ha='center', va='top', fontsize=6.8, color=MUTED)
    ax.text(2.67, 1.35, 'new, or reunited with\nphase-1 partner (hatched)', ha='center', va='top', fontsize=6.8, color=MUTED)
    ax.text(0, -0.45, 'Roles alternate every round. 3 tournaments\n(6, 6, 4 players), 240 rounds in total.', fontsize=6.8, color=MUTED)
    save(fig, 'fig1_game')


# ------------------------------------------------------------------ methods explainer: the three measurement tools
def fig_tools():
    fig = plt.figure(figsize=(7.2, 2.9))
    def panel(x, title):
        ax = fig.add_axes([x, 0.02, 0.31, 0.9]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
        ax.text(0, 9.6, title, fontsize=9, fontweight='bold', color=INK, va='top')
        return ax
    def bx(ax, x, y, w, h, t, fc, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.3', fc=fc, ec=GRID, lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK)
    def ar(ax, a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, lw=1.2, color=MUTED))
    # A neutral reader
    ax = panel(0.0, 'A. Neutral reader (legibility)')
    bx(ax, 0.2, 6.6, 4.2, 1.7, 'a real Sender\nturn (moves)', '#eef4fc')
    bx(ax, 5.4, 6.6, 4.4, 1.7, 'rules + 5 games\nfrom OTHER groups', '#f6f5f2')
    ar(ax, (2.3, 6.5), (4.3, 5.1)); ar(ax, (7.6, 6.5), (5.7, 5.1))
    bx(ax, 2.2, 3.4, 5.6, 1.6, 'LLM Receiver with no\nhistory with this Sender', '#fdeee6')
    ar(ax, (5.0, 3.3), (5.0, 2.3))
    bx(ax, 1.6, 0.6, 6.8, 1.6, 'correct tile + direction?\n= legible to an outsider', '#e8f6f0')
    # B counterfactual 2x2
    ax = panel(0.345, 'B. Counterfactual 2 x 2')
    ax.text(5.9, 8.3, 'Receiver history cut at', ha='center', fontsize=6.8, color=MUTED)
    ax.text(4.1, 7.5, 'end of\nphase 1', ha='center', va='center', fontsize=6.8); ax.text(7.7, 7.5, 'start of\nphase 3', ha='center', va='center', fontsize=6.8)
    ax.text(1.1, 5.3, 'phase-1\nsignals', ha='center', va='center', fontsize=6.8); ax.text(1.1, 2.9, 'phase-3\nsignals', ha='center', va='center', fontsize=6.8)
    cells = [[38, 43], [52, 56]]
    for i in range(2):
        for j in range(2):
            v = cells[i][j]
            ax.add_patch(FancyBboxPatch((2.4 + j * 3.6, 4.3 - i * 2.4), 3.3, 2.0, boxstyle='round,pad=0.03,rounding_size=0.2',
                                        fc=plt.cm.Blues(0.15 + (v - 35) / 40), ec='white'))
            ax.text(4.05 + j * 3.6, 5.3 - i * 2.4, f'{v}%', ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
    ax.text(0.2, 0.9, 'Sender effect = down the columns (+14)\nReceiver effect = across the rows (+5)', fontsize=6.8, color=INK, va='center')
    # C style map
    ax = panel(0.69, 'C. Style map (similarity judgments)')
    for k, (x, odd) in enumerate([(0.3, False), (3.4, True), (6.5, False)]):
        bx(ax, x, 7.0, 2.8, 1.3, f'turn {k + 1}' + ('\n(odd one out)' if odd else ''), '#fdeee6' if odd else '#eef4fc', fs=6.8)
    ar(ax, (5.0, 6.9), (5.0, 5.9))
    ax.text(5.0, 5.4, '4,209 judgments by an LLM judge', ha='center', fontsize=6.8, color=MUTED)
    ar(ax, (5.0, 5.0), (5.0, 4.1))
    rng_ = np.random.default_rng(3)
    pts = np.clip(rng_.normal(0, 0.8, (40, 2)), -1.6, 1.6)
    ax.scatter(5 + pts[:, 0] * 1.3, 2.4 + pts[:, 1] * 0.7, s=6, color=S[0], alpha=0.7)
    ax.text(5.0, 0.4, 'each turn = a point; alike turns sit close', ha='center', fontsize=6.8, color=INK)
    save(fig, 'figT_tools')


if __name__ == '__main__':
    fig_task(); fig_tools()
