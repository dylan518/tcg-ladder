#!/usr/bin/env python3
"""
Label communicative strategies in circle -> triangle TCG trials and trace how they
spread through the partner ladder.

The Sender controls a circle (cannot rotate) and must show the Receiver both WHERE their
triangle goes (target cell T) and WHICH WAY it points (direction d). This script reads
trials.jsonl written by server.js and produces:

  labeled_trials.csv     one row per trial: sender strategy, receiver reading, features
  pairing_strategies.csv one row per participant x pairing: dominant strategy as sender
  adoptions.csv          each first use of a strategy, and whether the adopter had seen it
                         from an earlier partner (contact-explained transmission)
  exposure_summary.csv   P(use strategy | previously exposed) vs P(use | not exposed)
  figures/*.png          strategy timeline, success by strategy, ladder network

Usage:  python3 analysis/label_strategies.py data/<session>   [--out <dir>]

------------------------------------------------------------------------------------
Sender strategy labels (relative to the triangle's true direction d)
  wiggle-F   steps off T toward the direction the triangle points, then back (out-and-back)
  wiggle-B   out-and-back opposite to the pointing direction
  wiggle-L/R out-and-back perpendicular (left/right of the pointing direction)
  wiggle-mix several out-and-backs with no dominant direction
  exit-F     no out-and-back; leaves T for the last time moving in the pointing direction,
             on a detour (not simply heading to the sender's own goal)
  enter-F    no out-and-back; arrives on T moving in the pointing direction via a detour
             (not on a shortest path from the centre)
  pause      sits on T (>= 1 s) without a directional cue
  pass       moves over T without pausing or a directional cue
  no-target  never visits T
  none       key never placed / no moves (timeout)

Receiver reading labels: the receiver's final triangle direction relative to the sender's
main signal direction (direction of the out-and-backs, or of a deliberate exit/entry move):
  reads-F (points along the signal), reads-B (opposite), reads-L/R, reads-none (no signal
  or receiver did not rotate/place), so a pair is coordinated on a convention when e.g.
  sender wiggle-F meets receiver reads-F.

Angles: 0 = up, 90 = right, 180 = down, 270 = left (clockwise, as in the game).
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

MOVE_ANGLE = {(0, 1): 0, (1, 0): 90, (0, -1): 180, (-1, 0): 270}
REL = {0: 'F', 90: 'R', 180: 'B', 270: 'L'}
PAUSE_MS = 1000
NON_STRATEGIES = {'none', 'no-target'}


# ---------------------------------------------------------------- per-trial features

def sender_track(path):
    """Positions the sender's circle occupied, with arrival times (ms from turn onset)."""
    steps = [p for p in path if p['role'] == 'sender']
    track = []
    for p in steps:
        if p['a'] in ('start', 'up', 'down', 'left', 'right'):
            track.append(((p['x'], p['y']), p['t']))
    end_t = steps[-1]['t'] if steps else 0
    if track and track[0][0] != (0, 0) and steps[0]['a'] != 'start':
        track.insert(0, ((0, 0), track[0][1]))  # key appeared at the centre
    return track, end_t


def move_angle(a, b):
    return MOVE_ANGLE.get((b[0] - a[0], b[1] - a[1]))


def rel(angle, d):
    return None if angle is None else REL[(angle - d) % 360]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def label_trial(r):
    G = tuple(r['senderGoal'][:2])
    T = tuple(r['receiverGoal'][:2])
    d = r['receiverGoal'][2]
    track, end_t = sender_track(r['path'])
    f = {'target': T, 'direction': d}

    if len(track) == 0:
        f.update(sender_label='none', n_visits=0, n_excursions=0, dwell_ms=0, signal_dir=None)
        return f

    cells = [c for c, _ in track]
    times = [t for _, t in track] + [end_t]
    visits = [i for i, c in enumerate(cells) if c == T and (i == 0 or cells[i - 1] != T)]
    dwell = sum(times[i + 1] - times[i] for i, c in enumerate(cells) if c == T)
    # out-and-back: T -> neighbour -> T
    exc = [move_angle(cells[i], cells[i + 1]) for i in range(len(cells) - 2)
           if cells[i] == T and cells[i + 1] != T and cells[i + 2] == T]
    f.update(n_visits=len(visits), n_excursions=len(exc), dwell_ms=int(dwell))

    i_entry = next((i for i in range(1, len(cells)) if cells[i] == T), None)
    entry = move_angle(cells[i_entry - 1], cells[i_entry]) if i_entry else None
    i_exits = [i for i in range(len(cells) - 1) if cells[i] == T and cells[i + 1] != T]
    last_exit = move_angle(cells[i_exits[-1]], cells[i_exits[-1] + 1]) if i_exits else None
    # an entry/exit move is only a deliberate cue if it departs from the efficient route:
    # entering T on a shortest path from the centre, or leaving T toward the sender's own
    # goal, is what anyone would do and says nothing about direction
    exit_deliberate = bool(i_exits) and manhattan(cells[i_exits[-1] + 1], G) >= manhattan(T, G)
    entry_deliberate = i_entry is not None and manhattan((0, 0), T) < i_entry
    f.update(exit_deliberate=exit_deliberate, entry_deliberate=entry_deliberate)
    f.update(entry_rel=rel(entry, d), exit_rel=rel(last_exit, d),
             excursion_rels=' '.join(rel(a, d) for a in exc))

    if not visits:
        label, signal = 'no-target', None
    elif exc:
        cnt = Counter(exc)
        top, n = cnt.most_common(1)[0]
        if n / len(exc) > 0.5:
            label, signal = 'wiggle-' + rel(top, d), top
        else:
            label, signal = 'wiggle-mix', None
    elif last_exit is not None and exit_deliberate and rel(last_exit, d) == 'F':
        label, signal = 'exit-F', last_exit
    elif entry is not None and entry_deliberate and rel(entry, d) == 'F':
        label, signal = 'enter-F', entry
    elif dwell >= PAUSE_MS:
        label, signal = 'pause', None
    else:
        label, signal = 'pass', None
    f.update(sender_label=label, signal_dir=signal)
    return f


def receiver_reading(r, signal_dir):
    placed = any(p['role'] == 'receiver' and p['a'] != 'timeout' for p in r['path'])
    if not placed or signal_dir is None:
        return 'reads-none'
    return 'reads-' + REL[(r['receiverFinal'][2] - signal_dir) % 360]


# ---------------------------------------------------------------- ladder-level analysis

def dominant(labels):
    """Main strategy in a set of trials. A directional cue wins if it is used in at least a
    third of the trials, because it is often impossible (e.g. wiggling off the board edge)
    and the sender then falls back to pausing."""
    useful = [l for l in labels if l not in NON_STRATEGIES]
    directional = [l for l in useful if l not in ('pause', 'pass')]
    if directional and len(directional) >= len(useful) / 3:
        return Counter(directional).most_common(1)[0][0]
    pool = useful or labels
    return Counter(pool).most_common(1)[0][0] if pool else None


def analyse(df):
    # pairing order per participant (rounds are ints; free play sorts after by time)
    df['round_key'] = pd.to_numeric(df['round'], errors='coerce')
    df = df.sort_values(['round_key', 'timestamp']).reset_index(drop=True)

    # participant x pairing table, as sender
    rows = []
    for (room, sender), g in df.groupby(['room', 'sender'], sort=False):
        partner = g['receiver'].iloc[0]
        rows.append({
            'participant': sender, 'partner': partner, 'room': room, 'round': g['round'].iloc[0],
            'round_key': g['round_key'].iloc[0], 'first_ts': g['timestamp'].min(),
            'n_sender_trials': len(g), 'dominant': dominant(list(g['sender_label'])),
            'labels': ' '.join(g['sender_label']), 'success_as_sender': g['success'].mean(),
        })
    ps = pd.DataFrame(rows).sort_values(['round_key', 'first_ts'])

    # within-pair convergence: do the two partners end up using the same strategy?
    conv = {}
    for room, g in df.groupby('room'):
        half = g[g['trial'] > g['trial'].max() / 2]
        doms = [dominant(list(h['sender_label'])) for _, h in half.groupby('sender')]
        conv[room] = int(len(doms) == 2 and doms[0] == doms[1])
    ps['pair_converged_2nd_half'] = ps['room'].map(conv)

    # exposure: strategies a participant has watched as RECEIVER, trial by trial
    seen_before = defaultdict(set)   # participant -> labels seen so far (from partners)
    used_before = defaultdict(set)   # participant -> labels used so far (as sender)
    first_user = {}                  # label -> (participant, round, trial)
    sources = {}                     # (participant, label) -> partners they saw it from
    adoptions, exposure_rows = [], []
    labels_all = sorted(set(df['sender_label']) - NON_STRATEGIES)
    for _, r in df.iterrows():
        s, rc, lab = r['sender'], r['receiver'], r['sender_label']
        if lab not in NON_STRATEGIES:
            for L in labels_all:
                exposure_rows.append({'participant': s, 'round': r['round'], 'label': L,
                                      'exposed': L in seen_before[s], 'used': int(lab == L)})
            if lab not in used_before[s]:
                origin = first_user.get(lab)
                adoptions.append({
                    'participant': s, 'round': r['round'], 'trial': r['trial'], 'room': r['room'], 'label': lab,
                    'innovation': origin is None,
                    'seen_from_partner_before': lab in seen_before[s],
                    'seen_from': ' '.join(sorted(sources[(s, lab)])) if (s, lab) in sources else '',
                    'first_used_by': origin[0] if origin else s,
                })
                first_user.setdefault(lab, (s, r['round'], r['trial']))
            used_before[s].add(lab)
            seen_before[rc].add(lab)
            sources.setdefault((rc, lab), set()).add(s)
    adoptions = pd.DataFrame(adoptions)
    ex = pd.DataFrame(exposure_rows)
    if len(ex):
        exposure = (ex.groupby(['label', 'exposed'])['used'].agg(['mean', 'count']).unstack('exposed'))
        exposure.columns = [f'{a}_{"exposed" if b else "unexposed"}' for a, b in exposure.columns]
        exposure = exposure.reset_index()
    else:
        exposure = pd.DataFrame()
    return df, ps, adoptions, exposure


# ---------------------------------------------------------------- figures

def figures(df, ps, out, ladder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    labels = sorted(set(ps['dominant'].dropna()))
    cmap = plt.get_cmap('tab10')
    color = {l: cmap(i % 10) for i, l in enumerate(labels)}

    # 1) strategy timeline: participants x pairings
    people = list(dict.fromkeys(ps.sort_values('first_ts')['participant']))
    rounds = sorted(ps['round'].unique(), key=lambda x: (pd.to_numeric(x, errors='coerce'), str(x)))
    fig, ax = plt.subplots(figsize=(1.4 * len(rounds) + 3, 0.5 * len(people) + 1.5))
    for i, p in enumerate(people):
        for j, rd in enumerate(rounds):
            row = ps[(ps['participant'] == p) & (ps['round'] == rd)]
            if row.empty:
                continue
            lab = row['dominant'].iloc[0]
            ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, color=color.get(lab, '#ddd')))
            ax.text(j + 0.47, i + 0.45, f"{lab}\nw/ {row['partner'].iloc[0]}", ha='center', va='center', fontsize=7)
    ax.set_xlim(0, len(rounds)); ax.set_ylim(len(people), 0)
    ax.set_xticks(np.arange(len(rounds)) + 0.47); ax.set_xticklabels([f'round {r}' for r in rounds])
    ax.set_yticks(np.arange(len(people)) + 0.45); ax.set_yticklabels(people)
    ax.set_title('Dominant sender strategy per pairing')
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'strategy_timeline.png'), dpi=150); plt.close(fig)

    # 2) success by strategy
    g = df.groupby('sender_label')['success'].agg(['mean', 'count']).sort_values('mean')
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(g) + 1.2))
    ax.barh(g.index, g['mean'], color=[color.get(l, '#999') for l in g.index])
    for i, (m, n) in enumerate(zip(g['mean'], g['count'])):
        ax.text(m + 0.01, i, f'{m:.0%} (n={n})', va='center', fontsize=8)
    ax.set_xlim(0, 1.15); ax.set_xlabel('Vault cracked'); ax.set_title('Success by sender strategy')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'success_by_strategy.png'), dpi=150); plt.close(fig)

    # 3) ladder network coloured by each participant's final dominant strategy
    try:
        import networkx as nx
    except ImportError:
        return
    G = nx.Graph()
    for _, r in ps.iterrows():
        G.add_edge(r['participant'], r['partner'], round=r['round'])
    final = ps.sort_values('first_ts').groupby('participant')['dominant'].last()
    pos = nx.spring_layout(G, seed=1) if ladder != 'chain' else {p: (i, 0) for i, p in enumerate(people)}
    fig, ax = plt.subplots(figsize=(max(5, len(people) * 0.9), 4 if ladder != 'chain' else 2.2))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#bbb')
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels={(a, b): f"r{d['round']}" for a, b, d in G.edges(data=True)}, font_size=7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[color.get(final.get(n), '#ddd') for n in G.nodes], node_size=700)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=l) for l, c in color.items()]
    ax.legend(handles=handles, fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title('Ladder (edges = pairings, colour = final strategy)'); ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'ladder_network.png'), dpi=150); plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('session_dir', help='data/<session> folder containing trials.jsonl')
    ap.add_argument('--out', help='output folder (default: <session_dir>/analysis)')
    ap.add_argument('--all-levels', action='store_true', help='also label non circle->triangle trials')
    a = ap.parse_args()
    out = a.out or os.path.join(a.session_dir, 'analysis')
    os.makedirs(out, exist_ok=True)

    recs = [json.loads(l) for l in open(os.path.join(a.session_dir, 'trials.jsonl'))]
    if not a.all_levels:
        recs = [r for r in recs if r['senderShape'] == 2 and r['receiverShape'] == 3]
    if not recs:
        raise SystemExit('No circle -> triangle trials found.')
    rows = []
    for r in recs:
        f = label_trial(r)
        f['receiver_label'] = receiver_reading(r, f.get('signal_dir'))
        rows.append({k: r[k] for k in ('session', 'design', 'room', 'round', 'trial', 'sender', 'receiver',
                                       'success', 'receiverLoc', 'receiverOri', 'timestamp')} | f)
    df = pd.DataFrame(rows)
    df, ps, adoptions, exposure = analyse(df)

    df.drop(columns=['round_key']).to_csv(os.path.join(out, 'labeled_trials.csv'), index=False)
    ps.drop(columns=['round_key', 'first_ts']).to_csv(os.path.join(out, 'pairing_strategies.csv'), index=False)
    adoptions.to_csv(os.path.join(out, 'adoptions.csv'), index=False)
    exposure.to_csv(os.path.join(out, 'exposure_summary.csv'), index=False)
    figures(df, ps, os.path.join(out, 'figures'), df['design'].iloc[0])

    # console summary
    print(f'{len(df)} circle->triangle trials, {len(set(df["sender"]) | set(df["receiver"]))} participants, '
          f'{df["room"].nunique()} pairings, success {df["success"].mean():.0%}\n')
    print('Sender strategies:'); print(df['sender_label'].value_counts().to_string(), '\n')
    print('Coordination (sender strategy x receiver reading):')
    print(pd.crosstab(df['sender_label'], df['receiver_label']).to_string(), '\n')
    if len(adoptions):
        n_inn = int(adoptions['innovation'].sum())
        copied = adoptions[~adoptions['innovation']]
        print(f'Strategy first-uses: {len(adoptions)} ({n_inn} innovations, {len(copied)} adoptions of an existing strategy)')
        if len(copied):
            print(f'  adoptions preceded by seeing it from a partner: {copied["seen_from_partner_before"].mean():.0%}')
    print(f'\nWrote results to {out}')


if __name__ == '__main__':
    main()
