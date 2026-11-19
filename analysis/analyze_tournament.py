#!/usr/bin/env python3
"""
Code sender strategies and analyse a random-partner tournament session (8-direction triangle task).

Usage:  python3 analysis/analyze_tournament.py data/<session> [--out analysis/<session>]

Reads rounds.csv (one row per game) and assignments.csv (slot -> name). Writes to --out:
  coded_rounds.csv      one row per game: features, sender strategy code, receiver reading, accuracy
  strategy_by_player.csv  each player's most-used strategy per phase (as Sender)
  transmission.csv      for phases 2-3: did a player switch to their previous partner's strategy?
  summary.txt           the numbers quoted in the write-up
  figures/*.png

------------------------------------------------------------------------------------------------
How a Sender turn is coded (board coordinates: x right, y up; centre = (0,0); directions as vectors)

The Sender's circle starts in the centre. From the key presses we get the sequence of tiles visited and
how long the circle stayed on each. Two things are extracted:

  * WIGGLES: out-and-back steps (A->B->A) counted per board edge. A goal wiggle has the goal tile at one
    end of the edge; its ARM is the other tile, measured from the goal outward (so goal<->neighbour
    back-and-forth gives the same arm whichever tile the run starts on).
  * a DWELL: the non-final tile the circle rested on longest (>= 1.5 s counts as deliberate).

Strategy codes (first rule that applies):
  none          the circle never moved (or the turn timed out without moves)
  centre-dir    the direction is shown by a tile seen FROM THE CENTRE: the Sender emphasises the tile that
                lies in the triangle's direction from the centre (>= 2 wiggles on edges touching it, or a long
                dwell there, at least as much as any goal wiggling), that tile is not the goal, and the Sender
                also visits the goal tile.
  goal-wiggle-F oscillation on the goal tile; the arm points the way the triangle should point (forward)
  goal-wiggle-B oscillation on the goal tile; the arm points the opposite way (the Receiver should point
                from the arm into the goal)
  goal-wiggle-S oscillation on the goal tile, arm sideways (45-135 degrees off the target direction)
  goal-only     reaches the goal tile and stays/ends there with no oscillation or deliberate dwell elsewhere
                (tile shown, direction not signalled)
  wiggle-other  >= 2 out-and-backs on one edge that is neither on the goal tile nor centre-referenced
  other         anything else (laps around the board, wandering, never reaching the goal)

Diagonal arms: moves are orthogonal only, so a diagonal can only be shown with two arms (e.g. stepping
left and down from the hub); the arm vector is the mean of the arms, so two orthogonal arms make a diagonal.

Receiver reading (games whose code carries a direction signal): the angle between the Receiver's final
triangle and the Sender's signal vector: reads-F (same way, within 22.5 deg), reads-B (opposite),
reads-S (in between), or no-signal.
"""

import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict


TILE_XY = {'top-left': (-1, 1), 'top': (0, 1), 'top-right': (1, 1), 'left': (-1, 0), 'center': (0, 0),
           'right': (1, 0), 'bottom-left': (-1, -1), 'bottom': (0, -1), 'bottom-right': (1, -1)}
DIR_VEC = {'up': (0, 1), 'up-right': (1, 1), 'right': (1, 0), 'down-right': (1, -1), 'down': (0, -1),
           'down-left': (-1, -1), 'left': (-1, 0), 'up-left': (-1, 1)}
DWELL_MS = 1500
MOVE_RE = re.compile(r'(\w+)\((-?\d),(-?\d)\)@(\d+)')


def angle(v):
    """compass angle of a vector, 0 = up, clockwise"""
    return math.degrees(math.atan2(v[0], v[1])) % 360


def adiff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def visits(moves):
    """[(tile, arrive_ms, dwell_ms)] for the Sender's circle, consecutive duplicates merged"""
    ev = [(m.group(1), (int(m.group(2)), int(m.group(3))), int(m.group(4))) for m in MOVE_RE.finditer(moves)]
    if not ev:
        return []
    end = ev[-1][2]
    out = []
    for i, (a, xy, t) in enumerate(ev):
        if a in ('stop', 'timeout'):
            break
        nxt = ev[i + 1][2] if i + 1 < len(ev) else end
        if out and out[-1][0] == xy:
            out[-1][2] += nxt - t
        else:
            out.append([xy, t, nxt - t])
    return out


def edge_wiggles(seq):
    """{frozenset({A, B}): number of out-and-backs} — consecutive A->B->A steps on each board edge"""
    tiles = [s[0] for s in seq]
    w = Counter()
    for a, b, c in zip(tiles, tiles[1:], tiles[2:]):
        if a == c and a != b:
            w[frozenset((a, b))] += 1
    return w


def code_turn(row):
    seq = visits(row['sender_moves'])
    G, d = TILE_XY[row['goal_tile']], DIR_VEC[row['goal_direction']]
    dang = angle(d)
    C = (0, 0)
    Cd = (C[0] + d[0], C[1] + d[1])  # the tile in the triangle's direction from the centre
    feat = {'n_moves': max(len(seq) - 1, 0), 'visited_goal': any(s[0] == G for s in seq),
            'ends_on_goal': bool(seq) and seq[-1][0] == G}
    if len(seq) <= 1:
        return 'none', None, feat
    w = edge_wiggles(seq)
    mids = seq[1:-1]
    dwell = max(mids, key=lambda s: s[2]) if mids else None
    dwell_tile = dwell[0] if dwell and dwell[2] >= DWELL_MS else None

    # wiggles on the goal tile: arms = the neighbours it goes back and forth to
    goal_arms = {next(iter(e - {G})): n for e, n in w.items() if G in e}
    goal_w = sum(goal_arms.values())
    # centre-referenced emphasis on Cd: wiggles on edges touching Cd (when Cd is not the goal), or a long dwell there
    cd_w = 0 if Cd == G else sum(n for e, n in w.items() if Cd in e) + (2 if dwell_tile == Cd else 0)

    if cd_w >= 2 and cd_w >= goal_w and feat['visited_goal']:
        return 'centre-dir', dang, feat
    if goal_w >= 2:
        vx = sum((a[0] - G[0]) * n for a, n in goal_arms.items()) / goal_w
        vy = sum((a[1] - G[1]) * n for a, n in goal_arms.items()) / goal_w
        if (vx, vy) == (0, 0):
            return 'wiggle-other', None, feat
        va = angle((vx, vy))
        diff = adiff(va, dang)
        if diff <= 22.5:
            return 'goal-wiggle-F', va, feat
        if diff >= 157.5:
            return 'goal-wiggle-B', va, feat
        return 'goal-wiggle-S', va, feat
    if w and max(w.values()) >= 2:
        return 'wiggle-other', None, feat
    if feat['ends_on_goal']:
        return 'goal-only', None, feat
    return 'other', None, feat


def reading(sig_angle, rec_dir):
    if sig_angle is None or rec_dir not in DIR_VEC:
        return 'no-signal'
    diff = adiff(angle(DIR_VEC[rec_dir]), sig_angle)
    return 'reads-F' if diff <= 22.5 else 'reads-B' if diff >= 157.5 else 'reads-S'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: F401  (plotting/table deps only needed here)
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--out')
    a = ap.parse_args()
    sid = os.path.basename(a.session.rstrip('/'))
    out = a.out or os.path.join('analysis', sid)
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)

    df = pd.read_csv(os.path.join(a.session, 'rounds.csv'), dtype=str)
    names = {}
    ap_ = os.path.join(a.session, 'assignments.csv')
    if os.path.exists(ap_):
        for r in csv.DictReader(open(ap_)):
            if r.get('name'):
                names[r['player_id']] = r['name']
    nm = lambda p: names.get(p, p)
    for c in ['round', 'phase', 'success', 'sender_phase_ms', 'receiver_phase_ms', 'attempt']:
        df[c] = pd.to_numeric(df[c])
    # replayed games: keep the last attempt only
    df = df.sort_values(['round', 'group', 'attempt']).drop_duplicates(['round', 'group'], keep='last')

    rows = []
    for _, r in df.iterrows():
        code, sig, feat = code_turn(r)
        rows.append({
            'round': r['round'], 'phase': r['phase'], 'round_in_phase': (r['round'] - 1) % 10 + 1,
            'pair': r['group'], 'sender': nm(r['sender_id']), 'receiver': nm(r['receiver_id']),
            'goal_tile': r['goal_tile'], 'goal_direction': r['goal_direction'],
            'receiver_tile': r['receiver_tile'], 'receiver_direction': r['receiver_direction'],
            'tile_ok': int(r['receiver_tile'] == r['goal_tile']),
            'dir_ok': int(r['receiver_direction'] == r['goal_direction']),
            'success': r['success'], 'strategy': code, 'signal_angle': sig,
            'receiver_reading': reading(sig, r['receiver_direction']),
            'sender_s': r['sender_phase_ms'] / 1000, 'receiver_s': r['receiver_phase_ms'] / 1000,
            'separation': r.get('degrees_of_separation', ''), 'sender_moves': r['sender_moves'],
            'n_moves': feat['n_moves'], 'ends_on_goal': int(feat['ends_on_goal']),
        })
    cr = pd.DataFrame(rows).sort_values(['sender', 'round'])
    cr.to_csv(os.path.join(out, 'coded_rounds.csv'), index=False)

    L = []
    p = L.append
    n = len(cr)
    p(f'Session {sid}: {n} games, {cr.sender.nunique()} players, {cr["round"].max()} rounds')
    p(f'Players: {", ".join(sorted(set(cr.sender) | set(cr.receiver)))}')
    p(f'Success {cr.success.sum()}/{n} = {cr.success.mean():.0%}; tile right {cr.tile_ok.mean():.0%}; direction right {cr.dir_ok.mean():.0%}; '
      f'direction right when tile right {cr[cr.tile_ok == 1].dir_ok.mean():.0%}')
    p('Chance: tile 1/8 = 13%, direction 1/8 = 13%, both 1.6%')
    p('')
    p('By phase (new random partner each phase):')
    for ph, g in cr.groupby('phase'):
        p(f'  phase {ph}: success {g.success.mean():.0%} ({g.success.sum()}/{len(g)}), tile {g.tile_ok.mean():.0%}, direction {g.dir_ok.mean():.0%}, '
          f'sender {g.sender_s.median():.1f}s, receiver {g.receiver_s.median():.1f}s (median)')
    diag = cr.goal_direction.str.contains('-')
    p(f'Goal direction cardinal (up/down/left/right): success {cr[~diag].success.mean():.0%} ({cr[~diag].success.sum()}/{(~diag).sum()}), '
      f'direction right {cr[~diag].dir_ok.mean():.0%};  diagonal: success {cr[diag].success.mean():.0%} ({cr[diag].success.sum()}/{diag.sum()}), direction right {cr[diag].dir_ok.mean():.0%}')
    p('Within a partnership (rounds 1-5 vs 6-10 of each phase):')
    for half, g in cr.groupby(cr.round_in_phase > 5):
        p(f'  {"rounds 6-10" if half else "rounds 1-5"}: success {g.success.mean():.0%} ({g.success.sum()}/{len(g)})')
    p('')
    p('Pairs (both directions of play) by phase:')
    for (ph, pair), g in cr.groupby(['phase', 'pair']):
        who = ' & '.join(sorted(set(g.sender)))
        sep = g.separation.iloc[0]
        p(f'  phase {ph} {who:22s} {g.success.sum()}/{len(g)}' + (f'   separation {sep}' if ph > 1 else ''))
    p('')
    p('Sender strategies (all games):')
    for s, g in cr.groupby('strategy'):
        p(f'  {s:14s} {len(g):3d} games ({len(g) / n:.0%})  success {g.success.mean():.0%}  tile {g.tile_ok.mean():.0%}  direction {g.dir_ok.mean():.0%}')
    p('')
    p('How Receivers read a goal wiggle (arm direction vs triangle):')
    gw = cr[cr.strategy.str.startswith('goal-wiggle')]
    for s, g in gw.groupby('strategy'):
        c = Counter(g.receiver_reading)
        p(f'  {s}: ' + ', '.join(f'{k} {v}' for k, v in sorted(c.items())))
    p('  (goal-wiggle-F + reads-F = forward convention; goal-wiggle-B + reads-F would mean the Receiver pointed toward the arm)')
    p('')

    # each player's dominant strategy per phase
    dom = (cr.groupby(['sender', 'phase']).strategy
           .agg(lambda s: Counter(s).most_common(1)[0][0]).unstack())
    dom.to_csv(os.path.join(out, 'strategy_by_player.csv'))
    p('Most-used Sender strategy per player and phase:')
    p('  ' + dom.to_string().replace('\n', '\n  '))
    p('')

    # transmission: in phase k+1, does a player use the strategy their phase-k partner used most?
    partner = {}
    for (ph, pair), g in cr.groupby(['phase', 'pair']):
        a_, b_ = sorted(set(g.sender))[:2] if len(set(g.sender)) == 2 else (None, None)
        if a_:
            partner[(a_, ph)] = b_
            partner[(b_, ph)] = a_
    tr = []
    for (pl, ph), prev in partner.items():
        if (pl, ph + 1) not in partner:
            continue
        mine_before = dom.loc[pl, ph] if ph in dom.columns else None
        theirs = dom.loc[prev, ph] if prev in dom.index else None
        mine_after = dom.loc[pl, ph + 1] if (ph + 1) in dom.columns else None
        tr.append({'player': pl, 'from_phase': ph, 'own_before': mine_before, 'partner': prev, 'partner_strategy': theirs,
                   'own_after': mine_after, 'changed': mine_after != mine_before,
                   'adopted_partner': mine_after == theirs and mine_before != theirs})
    tr = pd.DataFrame(tr).sort_values(['from_phase', 'player'])
    tr.to_csv(os.path.join(out, 'transmission.csv'), index=False)
    p('Transmission (next phase vs previous partner):')
    for _, t in tr.iterrows():
        tag = 'ADOPTED partner\'s' if t.adopted_partner else ('changed' if t.changed else 'kept')
        p(f'  {t.player:10s} phase {t.from_phase}->{t.from_phase + 1}: {t.own_before} -> {t.own_after}  (partner {t.partner}: {t.partner_strategy}) {tag}')
    p(f'  adopted partner\'s strategy: {int(tr.adopted_partner.sum())} of {int((tr.own_before != tr.partner_strategy).sum())} cases where it differed')
    p('')
    sep = cr[cr.phase > 1].copy()
    if len(sep):
        p('Success by degrees of separation (phases 2-3):')
        for s, g in sep.groupby('separation'):
            p(f'  separation {s}: {g.success.mean():.0%} ({g.success.sum()}/{len(g)})')
    txt = '\n'.join(L)
    open(os.path.join(out, 'summary.txt'), 'w').write(txt + '\n')
    print(txt)

    # ---- figures
    fig, ax = plt.subplots(figsize=(8, 3.2))
    by_r = cr.groupby('round').agg(success=('success', 'mean'), tile=('tile_ok', 'mean'), direction=('dir_ok', 'mean'))
    roll = by_r.rolling(3, center=True, min_periods=1).mean()
    for c, col, ls, lw in [('tile', '#7aa6c2', '-', 2), ('success', '#2b4c7e', '-', 2.5), ('direction', '#e39a4a', '--', 2)]:
        ax.plot(roll.index, roll[c] * 100, label=c, color=col, lw=lw, ls=ls)
    for x in (10.5, 20.5):
        ax.axvline(x, color='#bbb', ls='--', lw=1)
    ax.set_xlabel('round (dashed: new partner)'); ax.set_ylabel('% of games (3-round average)')
    ax.set_ylim(0, 105); ax.legend(frameon=False, ncol=3, loc='lower right'); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figures', 'accuracy_by_round.png'), dpi=150); plt.close(fig)

    order = cr.strategy.value_counts().index.tolist()
    players = sorted(cr.sender.unique())
    cmap = dict(zip(order, ['#2b4c7e', '#e39a4a', '#5b9a68', '#b5574b', '#8a6fb0', '#7aa6c2', '#c9b458', '#999999'][:len(order)]))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(players) + 1.2))
    for i, pl in enumerate(players):
        for _, r in cr[cr.sender == pl].iterrows():
            ax.scatter(r['round'], i, s=90, color=cmap[r.strategy], marker='o' if r.success else 'X', edgecolor='white', lw=0.5)
    for x in (10.5, 20.5):
        ax.axvline(x, color='#bbb', ls='--', lw=1)
    ax.set_yticks(range(len(players))); ax.set_yticklabels(players); ax.set_xlabel('round (circle = success, X = fail)')
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=cmap[s], label=s) for s in order], frameon=False,
              bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'figures', 'strategy_timeline.png'), dpi=150); plt.close(fig)
    print(f'\nwrote {out}/coded_rounds.csv, strategy_by_player.csv, transmission.csv, summary.txt, figures/')


if __name__ == '__main__':
    main()
