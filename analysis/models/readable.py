"""
Readable-move encoding of a turn: the raw key tokens grouped into what the player was doing.

  Sender steps   go_to X            moved to tile X (shortest path; intermediate tiles are implied)
                 wiggle [A, B] ×n   back and forth between A and B, n times out-and-back
                 loop [A, B, C..] ×n  went round the same small cycle of tiles n times
                 pause              stayed put for over 1 s (long pause: over 2 s)
                 end: done | timeout
  Receiver steps the same moves, plus  turn left/right ×n (now pointing D), and the final placement.

steps_from_tokens() turns raw tokens into steps; describe() renders steps as one readable line; realize() turns steps
back into raw tokens (shortest paths, vertical first), so generated steps can be scored with the same strategy coder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

XY = {v: k for k, v in C.XY_TILE.items()}
MOVE = {'U': (0, 1), 'D': (0, -1), 'L': (-1, 0), 'R': (1, 0)}


def walk(tokens):
    """-> list of (tile, pause) visits in order (pause: None / 'pause' / 'long pause' while on that tile), plus turn events.
    Returns events: ('tile', name, pause) and ('turn', dir_name_after) for Receivers."""
    x = y = 0
    a = 0
    events = [['tile', 'center', None]]
    pending = None
    for t in tokens:
        if t in ('W2', 'W3'):
            pending = 'long pause' if t == 'W3' else 'pause'
            continue
        if t == 'W1':
            continue
        if pending:  # the pause happened on the current tile, before this action
            last = next(e for e in reversed(events) if e[0] == 'tile')
            last[2] = 'long pause' if 'long pause' in (last[2], pending) else 'pause'
            pending = None
        if t in MOVE:
            nx, ny = max(-1, min(1, x + MOVE[t][0])), max(-1, min(1, y + MOVE[t][1]))
            if (nx, ny) != (x, y):
                x, y = nx, ny
                events.append(['tile', C.XY_TILE[(x, y)], None])
        elif t in ('RL', 'RR'):
            a = (a + (1 if t == 'RR' else -1)) % 8
            events.append(['turn', t, C.DIRS[a]])
    if pending:
        last = next(e for e in reversed(events) if e[0] == 'tile')
        last[2] = 'long pause' if 'long pause' in (last[2], pending) else 'pause'
    return events


def _shortest(a, b):
    return abs(XY[a][0] - XY[b][0]) + abs(XY[a][1] - XY[b][1])


def steps_from_tokens(tokens):
    ev = walk(tokens)
    end = 'timeout' if tokens and tokens[-1] == 'TIMEOUT' else 'done'
    steps = []
    i = 0
    tiles = [e for e in ev]
    # leading pause on the centre
    if tiles and tiles[0][0] == 'tile' and tiles[0][2]:
        steps.append({'type': 'pause', 'tiles': ['center'], 'repeats': 2 if tiles[0][2] == 'long pause' else 1})
    i = 1
    cur = 'center'
    while i < len(tiles):
        e = tiles[i]
        if e[0] == 'turn':
            # group consecutive turns in the same rotational direction
            j = i
            while j < len(tiles) and tiles[j][0] == 'turn' and tiles[j][1] == e[1]:
                j += 1
            steps.append({'type': 'turn_right' if e[1] == 'RR' else 'turn_left', 'tiles': [], 'repeats': j - i, 'pointing': tiles[j - 1][2]})
            i = j
            continue
        # tile visits: look for a repeating cycle starting at the current tile
        run = [cur]
        k = i
        while k < len(tiles) and tiles[k][0] == 'tile':
            run.append(tiles[k][1])
            if tiles[k][2]:
                break  # a pause ends the movement run
            k += 1
        best = None
        for p in (2, 3, 4):
            m = 0
            while m + p < len(run) and run[m + p] == run[m]:
                m += 1
            reps = (m + p - 1) // p  # complete returns to the start (m + p tiles follow the cycle)
            if reps >= 2 and (best is None or reps * p > best[1] * best[0]):
                best = (p, reps)
        if best:
            p, reps = best
            cyc = run[:p]
            steps.append({'type': 'wiggle' if p == 2 else 'loop', 'tiles': cyc, 'repeats': reps})
            used = p * reps
            i += used
            cur = run[used]
        else:
            # plain movement: go to the next tile at which direction of travel bends away from the shortest path
            j = 1
            while j + 1 < len(run) and _shortest(run[0], run[j + 1]) == j + 1:
                j += 1
            steps.append({'type': 'go_to', 'tiles': [run[j]], 'repeats': 1})
            i += j
            cur = run[j]
        # pause on the tile we stopped at
        last_tile_ev = tiles[i - 1] if i - 1 < len(tiles) else None
        if last_tile_ev and last_tile_ev[0] == 'tile' and last_tile_ev[2]:
            steps.append({'type': 'pause', 'tiles': [cur], 'repeats': 2 if last_tile_ev[2] == 'long pause' else 1})
    return {'steps': steps, 'end': end}


def rel(a, b):
    """compass direction from tile a to tile b (for neighbouring or diagonal tiles), in the triangle's direction words"""
    dx, dy = XY[b][0] - XY[a][0], XY[b][1] - XY[a][1]
    sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    return {(0, 1): 'up', (1, 1): 'up-right', (1, 0): 'right', (1, -1): 'down-right', (0, -1): 'down',
            (-1, -1): 'down-left', (-1, 0): 'left', (-1, 1): 'up-left'}.get((sx, sy), 'same place')


def describe(st, role='sender'):
    parts = []
    for s in st['steps']:
        t = s['type']
        if t == 'go_to':
            parts.append(f"went to {s['tiles'][0]}")
        elif t == 'wiggle':
            parts.append(f"wiggled {s['tiles'][0]}<->{s['tiles'][1]} x{s['repeats']} ({s['tiles'][1]} is {rel(s['tiles'][0], s['tiles'][1])} of {s['tiles'][0]})")
        elif t == 'loop':
            parts.append(f"looped {'->'.join(s['tiles'])} x{s['repeats']}")
        elif t == 'pause':
            parts.append(('long pause' if s['repeats'] >= 2 else 'paused') + f" on {s['tiles'][0]}")
        elif t in ('turn_left', 'turn_right'):
            parts.append(f"turned {'left' if t == 'turn_left' else 'right'} x{s['repeats']}" + (f" (now pointing {s['pointing']})" if s.get('pointing') else ''))
    parts.append('time ran out' if st['end'] == 'timeout' else 'pressed done')
    return '; '.join(parts)


def _path(a, b):
    (x, y), (gx, gy) = XY[a], XY[b]
    out = []
    while y < gy: out.append('U'); y += 1
    while y > gy: out.append('D'); y -= 1
    while x < gx: out.append('R'); x += 1
    while x > gx: out.append('L'); x -= 1
    return out


def realize(st):
    """steps -> raw tokens (shortest paths, vertical first). Unknown tiles are skipped. A turn step that states the
    direction it ends up pointing ("now pointing D") is realized as the rotations that reach D, in the stated turning
    direction; otherwise its count is used."""
    toks, cur, ang = [], 'center', 0
    for s in st.get('steps', []):
        t, tl = s.get('type'), [x for x in s.get('tiles', []) if x in XY]
        n = max(1, int(s.get('repeats', 1) or 1))
        if t == 'go_to' and tl:
            toks += _path(cur, tl[0]); cur = tl[0]
        elif t in ('wiggle', 'loop') and len(tl) >= 2:
            toks += _path(cur, tl[0]); cur = tl[0]
            for _ in range(n):
                for nxt in tl[1:] + [tl[0]]:
                    toks += _path(cur, nxt); cur = nxt
        elif t == 'pause':
            toks.append('W3' if n >= 2 else 'W2')
        elif t in ('turn_left', 'turn_right'):
            if s.get('pointing') in C.DIRS:
                tgt = C.DIRS.index(s['pointing'])
                n = (ang - tgt) % 8 if t == 'turn_left' else (tgt - ang) % 8
            toks += ['RL' if t == 'turn_left' else 'RR'] * n
            ang = (ang + (-n if t == 'turn_left' else n)) % 8
    toks.append('TIMEOUT' if st.get('end') == 'timeout' else 'END')
    return toks


def parse(text):
    """inverse of describe(): readable line -> steps (tolerant: unknown pieces are skipped)"""
    import re
    tiles = '|'.join(sorted(C.TILES, key=len, reverse=True))
    steps, end = [], 'done'
    for part in [p.strip() for p in text.replace('\n', ';').split(';') if p.strip()]:
        m = re.match(rf'went to ({tiles})$', part)
        if m:
            steps.append({'type': 'go_to', 'tiles': [m.group(1)], 'repeats': 1}); continue
        m = re.match(rf'wiggled ({tiles})<->({tiles}) x(\d+)(?: \(.*\))?$', part)
        if m:
            steps.append({'type': 'wiggle', 'tiles': [m.group(1), m.group(2)], 'repeats': int(m.group(3))}); continue
        m = re.match(rf'looped ((?:{tiles})(?:->(?:{tiles}))+) x(\d+)$', part)
        if m:
            steps.append({'type': 'loop', 'tiles': m.group(1).split('->'), 'repeats': int(m.group(2))}); continue
        m = re.match(rf'(long pause|paused) on ({tiles})$', part)
        if m:
            steps.append({'type': 'pause', 'tiles': [m.group(2)], 'repeats': 2 if m.group(1) == 'long pause' else 1}); continue
        m = re.match(r'turned (left|right) x(\d+)(?: \(now pointing ([a-z-]+)\))?', part)
        if m:
            st = {'type': 'turn_' + m.group(1), 'tiles': [], 'repeats': int(m.group(2))}
            if m.group(3) in C.DIRS:
                st['pointing'] = m.group(3)  # the stated result wins over the count when realizing
            steps.append(st); continue
        if part.startswith('time ran out'):
            end = 'timeout'
        elif part.startswith('pressed done'):
            end = 'done'
    return {'steps': steps, 'end': end}
