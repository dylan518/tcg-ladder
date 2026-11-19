#!/usr/bin/env python3
"""
Per-action player transcripts for a user simulator: each player's whole game (all rounds, all partners), one line per
key press with the real time since their previous action and the resulting state.

Usage: python3 analysis/models/actions.py data/<s1> data/<s2> ... [--out analysis/models/data/transcripts.jsonl]

Round block (what this player experienced in that round):
  ROUND 12 | phase 2 | game 2 with this partner (new partner) | you are SENDER | goal: top-right, pointing up-right | limit 9 s
  you: 2.6 U top | 0.3 R top-right | ... | 0.5 END
  partner: 1.8 U top | 0.4 R top-right | 0.6 RR pointing up-right | 0.3 END
  result: partner placed top-right pointing up-right: SUCCESS
Action line "<dt> <A> <state>": dt = seconds since the player's previous action (first action: since the turn began),
A in U D L R (move; state = tile after it, unchanged at an edge), RL RR (rotate 45 degrees; state = "pointing D"),
END (pressed Space), TIMEOUT (time ran out; dt = time from the last action to the limit).
Receivers never see the goal (only SUCCESS/FAIL); Senders see the goal and watch the Receiver's moves.

Output JSONL: one object per player per session: {session, run, player, rounds: [{round, phase, role, header, you,
partner, result, ...raw fields}]}. The model is trained on rendered text with loss only on the "you:" action lines.
"""
import argparse
import csv
import json
import os
from collections import Counter

import pandas as pd

TILE = {(-1, 1): 'top-left', (0, 1): 'top', (1, 1): 'top-right', (-1, 0): 'left', (0, 0): 'center',
        (1, 0): 'right', (-1, -1): 'bottom-left', (0, -1): 'bottom', (1, -1): 'bottom-right'}
DIRS = ['up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left']
CODE = {'up': 'U', 'down': 'D', 'left': 'L', 'right': 'R', 'rotateleft': 'RL', 'rotateright': 'RR'}


def action_list(kp_rows, limit_ms):
    """keypress rows (one role, one game) -> [(dt_s, A, state)]"""
    out, last = [], 0
    for r in kp_rows:
        a, t = r['action'], int(r['t'])
        if a == 'start':
            last = t
            continue
        if a in CODE:
            A = CODE[a]
            state = TILE[(int(r['x']), int(r['y']))] if A in 'UDLR' else 'pointing ' + DIRS[(int(r['angle']) // 45) % 8]
        elif a == 'stop':
            A, state = 'END', ''
        elif a == 'timeout':
            A, state = 'TIMEOUT', ''
        else:
            continue
        out.append((round((t - last) / 1000, 1), A, state))
        last = t
    return out


def fmt_actions(acts):
    return ' | '.join(f'{dt:.1f} {A}' + (f' {s}' if s else '') for dt, A, s in acts) or '(nothing)'


def build(sessions):
    players = []
    for run, sdir in enumerate(sessions, 1):
        sid = os.path.basename(sdir.rstrip('/'))
        names = {r['player_id']: r['name'] for r in csv.DictReader(open(os.path.join(sdir, 'assignments.csv'))) if r.get('name')}
        sess = json.load(open(os.path.join(sdir, 'session.json')))
        turn_ms, first_ms = sess.get('turnMs', 9000), sess.get('firstTurnMs', 20000) or sess.get('turnMs', 9000)
        rounds = pd.read_csv(os.path.join(sdir, 'rounds.csv'), dtype=str)
        rounds['round'] = rounds['round'].astype(int); rounds['attempt'] = rounds['attempt'].astype(int)
        rounds = rounds.sort_values(['round', 'group', 'attempt']).drop_duplicates(['round', 'group'], keep='last').sort_values('round')
        kp = pd.read_csv(os.path.join(sdir, 'keypresses.csv'), dtype=str)
        kp = kp[kp.room.isin(set(rounds.room))]
        kp['t'] = kp.t.astype(int)
        ids = sorted(set(rounds.sender_id) | set(rounds.receiver_id))
        per = {p: [] for p in ids}
        pair_games, seen_partners, first_role = Counter(), {p: [] for p in ids}, {p: set() for p in ids}
        for _, r in rounds.iterrows():
            k = kp[kp.room == r.room].sort_values('t', kind='stable')
            sp, rp = r.sender_id, r.receiver_id
            key = tuple(sorted([sp, rp]))
            pg = pair_games[key]; pair_games[key] += 1
            for me, role in ((sp, 'sender'), (rp, 'receiver')):
                partner = rp if me == sp else sp
                lim = first_ms if role not in first_role[me] else turn_ms
                first_role[me].add(role)
                new = partner not in seen_partners[me]
                if new:
                    seen_partners[me].append(partner)
                you = action_list(k[k.role == role].to_dict('records'), lim)
                other = action_list(k[k.role != role].to_dict('records'), lim)
                header = (f"ROUND {r['round']} | phase {r.phase} | game {pg + 1} with this partner" + (' (new partner)' if pg == 0 else '')
                          + f" | you are {role.upper()}"
                          + (f" | goal: {r.goal_tile}, pointing {r.goal_direction}" if role == 'sender' else '')
                          + f" | limit {lim // 1000} s")
                placed = f"partner placed {r.receiver_tile} pointing {r.receiver_direction}" if role == 'sender' else f"you placed {r.receiver_tile} pointing {r.receiver_direction}"
                result = f"result: {placed}: {'SUCCESS' if r.success == '1' else 'FAIL'}"
                per[me].append({'round': int(r['round']), 'phase': int(r.phase), 'role': role, 'partner': names.get(partner, partner),
                                'partner_games': pg, 'limit_s': lim // 1000, 'goal_tile': r.goal_tile, 'goal_direction': r.goal_direction,
                                'receiver_tile': r.receiver_tile, 'receiver_direction': r.receiver_direction, 'success': int(r.success),
                                'header': header, 'you': you, 'partner_actions': other, 'result': result})
        for p in ids:
            players.append({'session': sid, 'run': run, 'player': names.get(p, p), 'rounds': per[p]})
    return players


def render_round(rd, you_text=None):
    """text of one round block; you_text overrides the player's own action line (for generation)"""
    you = you_text if you_text is not None else fmt_actions(rd['you'])
    if rd['role'] == 'sender':
        return f"{rd['header']}\nyou: {you}\npartner: {fmt_actions(rd['partner_actions'])}\n{rd['result']}\n"
    return f"{rd['header']}\npartner: {fmt_actions(rd['partner_actions'])}\nyou: {you}\n{rd['result']}\n"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'transcripts.jsonl'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    P = build(a.sessions)
    with open(a.out, 'w') as f:
        for p in P:
            f.write(json.dumps(p) + '\n')
    n_act = sum(len(rd['you']) for p in P for rd in p['rounds'])
    print(f"{len(P)} player transcripts, {sum(len(p['rounds']) for p in P)} turns, {n_act} own actions -> {a.out}")
    p = P[0]
    print('\nexample (first 3 rounds of', p['player'], '):\n' + ''.join(render_round(rd) for rd in p['rounds'][:3]))
    print('chars per transcript:', [len(''.join(render_round(rd) for rd in q['rounds'])) for q in P])
