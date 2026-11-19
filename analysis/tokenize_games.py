#!/usr/bin/env python3
"""
Turn every Sender and Receiver turn into a token sequence, for next-token / outcome prediction (n-gram, decision
model, LLM).

Usage: python3 analysis/tokenize_games.py data/<session> [data/<session> ...] [--out analysis/tokens]

Writes
  turns.jsonl   one JSON object per turn (context + tokens + outcome), in play order
  turns.txt     the same, one readable line per turn
  vocab.json    token list with descriptions

TOKENS (what a player did during their turn, in order)
  U D L R        move the key up / down / left / right (absolute board directions; a press against the edge still
                 appears as a token, though the key does not move)
  RL RR          rotate the triangle left / right (Receivers only; the Sender's circle cannot rotate)
  W1 W2 W3       a wait before the next action: W1 0.4-1 s, W2 1-2 s, W3 > 2 s (under 0.4 s = no token: fluent
                 pressing). Thresholds from the data: median gap between presses 0.46 s, 75% < 1 s, 90% < 2 s.
                 The wait before the very first action (time to start) is included the same way.
  END            the player pressed Space to finish the turn
  TIMEOUT        the turn's time ran out

GOAL-RELATIVE ("canonical") Sender tokens
  The whole board is rotated by a multiple of 90 degrees so that the goal direction becomes up (cardinal goals) or
  up-right (diagonal goals); moves are rotated with it, and the goal tile is reported in the rotated frame. Then
  "wiggle toward the direction" has the same tokens for every goal. Receivers are NOT canonicalised: they do not know
  the goal, and rotating by the true goal would leak it.

CONTEXT (inputs a predictor may use; not predicted)
  role, goal_tile, goal_direction (Sender only), round, phase, round_in_phase, partner_games (games this pair had
  already played together, reunions included), player/partner names, and for Receiver turns the Sender's token
  sequence (what the Receiver watched). Outcome fields: success, receiver final tile and direction.
"""
import argparse
import csv
import json
import os
from collections import Counter

import pandas as pd

MOVE = {'up': 'U', 'down': 'D', 'left': 'L', 'right': 'R', 'rotateleft': 'RL', 'rotateright': 'RR'}
WAITS = [(400, None), (1000, 'W1'), (2000, 'W2'), (float('inf'), 'W3')]
DIRS = ['up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left']
TILE_XY = {'top-left': (-1, 1), 'top': (0, 1), 'top-right': (1, 1), 'left': (-1, 0), 'center': (0, 0),
           'right': (1, 0), 'bottom-left': (-1, -1), 'bottom': (0, -1), 'bottom-right': (1, -1)}
XY_TILE = {v: k for k, v in TILE_XY.items()}
ROT_MOVE = {'U': 'R', 'R': 'D', 'D': 'L', 'L': 'U'}  # one quarter turn clockwise


def wait_token(ms):
    for hi, tok in WAITS:
        if ms < hi:
            return tok


def tokens_for(presses):
    """presses: [(action, t_ms)] in order, starting with ('start', 0)"""
    out, last = [], 0
    for a, t in presses:
        if a == 'start':
            last = t
            continue
        w = wait_token(t - last)
        if w:
            out.append(w)
        if a == 'stop':
            out.append('END')
        elif a == 'timeout':
            out.append('TIMEOUT')
        elif a in MOVE:
            out.append(MOVE[a])
        last = t
    return out


def quarter_turns_to_canonical(direction):
    """number of clockwise quarter turns that bring the goal direction to up (cardinal) or up-right (diagonal)"""
    i = DIRS.index(direction)
    return (-(i // 2)) % 4


def rotate_xy(xy, k):
    x, y = xy
    for _ in range(k):
        x, y = y, -x  # clockwise quarter turn
    return (x, y)


def canonical(tokens, goal_tile, goal_dir):
    k = quarter_turns_to_canonical(goal_dir)
    toks = []
    for t in tokens:
        for _ in range(k):
            t = ROT_MOVE.get(t, t)
        toks.append(t)
    cdir = DIRS[(DIRS.index(goal_dir) + 2 * k) % 8]
    ctile = XY_TILE[rotate_xy(TILE_XY[goal_tile], k)]
    return toks, ctile, cdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sessions', nargs='+')
    ap.add_argument('--out', default='analysis/tokens')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    turns = []
    for run, sdir in enumerate(a.sessions, 1):
        sid = os.path.basename(sdir.rstrip('/'))
        names = {}
        if os.path.exists(os.path.join(sdir, 'assignments.csv')):
            for r in csv.DictReader(open(os.path.join(sdir, 'assignments.csv'))):
                if r.get('name'):
                    names[r['player_id']] = r['name']
        rounds = pd.read_csv(os.path.join(sdir, 'rounds.csv'), dtype=str)
        rounds['round'] = rounds['round'].astype(int)
        rounds['attempt'] = rounds['attempt'].astype(int)
        rounds = rounds.sort_values(['round', 'group', 'attempt']).drop_duplicates(['round', 'group'], keep='last')
        kp = pd.read_csv(os.path.join(sdir, 'keypresses.csv'), dtype={'t': int})
        kp = kp[kp.room.isin(set(rounds.room))]
        pair_games = Counter()
        for _, r in rounds.sort_values('round').iterrows():
            key = tuple(sorted([r.sender_id, r.receiver_id]))
            pg = pair_games[key]
            pair_games[key] += 1
            k = kp[kp.room == r.room]
            seqs = {}
            for role in ('sender', 'receiver'):
                kk = k[k.role == role].sort_values('t', kind='stable')
                seqs[role] = tokens_for(list(zip(kk.action, kk.t)))
            ctoks, ctile, cdir = canonical(seqs['sender'], r.goal_tile, r.goal_direction)
            base = {'session': sid, 'run': run, 'round': int(r['round']), 'phase': int(r.phase),
                    'round_in_phase': (int(r['round']) - 1) % 10 + 1, 'pair': r.group, 'partner_games': pg,
                    'goal_tile': r.goal_tile, 'goal_direction': r.goal_direction,
                    'success': int(r.success), 'receiver_tile': r.receiver_tile, 'receiver_direction': r.receiver_direction}
            turns.append({**base, 'role': 'sender', 'player': names.get(r.sender_id, r.sender_id),
                          'partner': names.get(r.receiver_id, r.receiver_id), 'tokens': seqs['sender'],
                          'canon_tokens': ctoks, 'canon_goal_tile': ctile, 'canon_goal_direction': cdir})
            turns.append({**base, 'role': 'receiver', 'player': names.get(r.receiver_id, r.receiver_id),
                          'partner': names.get(r.sender_id, r.sender_id), 'tokens': seqs['receiver'],
                          'saw_sender_tokens': seqs['sender']})

    with open(os.path.join(a.out, 'turns.jsonl'), 'w') as f:
        for t in turns:
            f.write(json.dumps(t) + '\n')
    with open(os.path.join(a.out, 'turns.txt'), 'w') as f:
        for t in turns:
            ctx = f"run{t['run']} r{t['round']:02d} {t['role'][:4]:4s} {t['player'][:10]:10s} pg={t['partner_games']:2d}"
            if t['role'] == 'sender':
                ctx += f" goal={t['goal_tile']}/{t['goal_direction']}  canon={t['canon_goal_tile']}/{t['canon_goal_direction']}"
                f.write(f"{ctx} | {' '.join(t['tokens'])}  || canon: {' '.join(t['canon_tokens'])}\n")
            else:
                f.write(f"{ctx} -> {t['receiver_tile']}/{t['receiver_direction']} {'WIN' if t['success'] else 'miss'} | {' '.join(t['tokens'])}\n")
    vocab = {'U': 'move up', 'D': 'move down', 'L': 'move left', 'R': 'move right', 'RL': 'rotate left (Receiver)',
             'RR': 'rotate right (Receiver)', 'W1': 'wait 0.4-1 s', 'W2': 'wait 1-2 s', 'W3': 'wait > 2 s',
             'END': 'pressed Space to end the turn', 'TIMEOUT': 'turn timed out'}
    json.dump(vocab, open(os.path.join(a.out, 'vocab.json'), 'w'), indent=1)

    s = [t for t in turns if t['role'] == 'sender']; r = [t for t in turns if t['role'] == 'receiver']
    cnt = Counter(tok for t in turns for tok in t['tokens'])
    print(f'{len(turns)} turns ({len(s)} Sender, {len(r)} Receiver) from {len(a.sessions)} sessions')
    print('token counts:', dict(cnt.most_common()))
    print(f"mean length: Sender {sum(len(t['tokens']) for t in s) / len(s):.1f} tokens, Receiver {sum(len(t['tokens']) for t in r) / len(r):.1f}")
    print(f"wrote {a.out}/turns.jsonl, turns.txt, vocab.json")


if __name__ == '__main__':
    main()
