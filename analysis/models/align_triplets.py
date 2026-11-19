#!/usr/bin/env python3
"""
Subjective strategy similarity judged by an LLM, goal-controlled, to measure alignment between partners.

Items = sender turns (goal + moves in readable form). Two kinds of judgment:

  probe   anchor = player A's sender turn; candidates = (P) the partner B's nearest sender turn in the same pairing,
          (Q) a control turn with EXACTLY the same goal as P, by a player who never played A or B in this pairing
          (other run; same phase when possible). "Which candidate uses the more similar signalling strategy to the
          anchor?" P(partner chosen) > 0.5 = partner alignment beyond what the goal forces.
  odd     odd-one-out triplets (random + same-goal), for an embedding later; asked twice in different orders in the
          pilot to measure judge consistency.

Alignment is then related to games played together (speed) and to success (pair success rate, next-round success).

Usage: python3 analysis/models/align_triplets.py [--model gpt-6.1-terra] [--n_odd 100] [--workers 32]
"""
import argparse
import json
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fullgame_sonnet as F  # noqa: E402

OUT = os.path.join(HERE, 'out')
JUDGE = """You are judging how real people played a two-player communication game on a 3x3 board.
Tiles: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right.
The Sender sees a goal (a tile, and a direction the Receiver's triangle must point) and can only communicate by moving a
circle around the board; the Receiver watches and then places and turns a triangle. Moves are written as steps:
went to X; wiggled A<->B xN (back and forth, with which way B is from A); looped; paused (>1 s) / long pause (>2 s);
pressed done / time ran out.

You will compare Sender turns by their SIGNALLING STRATEGY: what the moves are designed to tell the Receiver and how
(e.g. go to the goal and wiggle toward the goal direction, wiggle away from it, mark the direction from the centre,
just go to the goal, pause patterns, how many repeats). Goals differ between turns: judge the strategy RELATIVE TO EACH
TURN'S OWN GOAL, not whether the tiles are the same."""

PROBE_SCHEMA = {'type': 'object', 'properties': {'more_similar': {'type': 'string', 'enum': ['A', 'B']}},
                'required': ['more_similar'], 'additionalProperties': False}
ODD_SCHEMA = {'type': 'object', 'properties': {'odd_one_out': {'type': 'string', 'enum': ['1', '2', '3']}},
              'required': ['odd_one_out'], 'additionalProperties': False}


def item(e):
    return f"goal: {e['goal_tile']}, pointing {e['goal_dir']}\nmoves: {F.moves(e['s_tokens'])}"


def tid(e):
    return f"{e['run']}:{e['round']}:{e['s']}"


def build(seed=0, n_odd=100):
    rng = random.Random(seed)
    games, _ = F.load_players()
    T = {tid(e): e for evl in games.values() for r, e in evl if r == 'sender'}
    bygoal = defaultdict(list)
    for e in T.values():
        bygoal[(e['goal_tile'], e['goal_dir'])].append(e)
    probes = []
    for a in T.values():
        b_turns = [e for e in T.values() if e['run'] == a['run'] and e['s'] == a['r'] and e['r'] == a['s']]  # partner sending to A
        if not b_turns:
            continue
        p = min(b_turns, key=lambda e: abs(e['round'] - a['round']))
        pool = [e for e in bygoal[(p['goal_tile'], p['goal_dir'])] if e['run'] != a['run']]
        same_ph = [e for e in pool if e['phase'] == p['phase']]
        q = rng.choice(same_ph or pool)
        swap = rng.random() < 0.5
        probes.append({'anchor': tid(a), 'partner': tid(p), 'control': tid(q), 'swap': swap})
    ids = sorted(T)
    odds = []
    for i in range(n_odd):
        if i % 2:
            g = rng.choice(list(bygoal))
            trip = [tid(e) for e in rng.sample(bygoal[g], 3)]
        else:
            trip = rng.sample(ids, 3)
        odds.append({'items': trip, 'order': [0, 1, 2], 'rep': 0})
        perm = [1, 2, 0] if rng.random() < 0.5 else [2, 0, 1]
        odds.append({'items': trip, 'order': perm, 'rep': 1})
    return T, probes, odds


def build_full(seed=1, n_other=3):
    """pilot probes + extra comparisons. Each probe: anchor, 'good' candidate, 'base' candidate (exactly the same goal);
    chose_good = 1 if the judge picks 'good'.
      po  partner vs other-run stranger   (pair + group alignment; n_other per anchor, the pilot's control first)
      ps  partner vs same-run stranger    (pair alignment beyond the group)
      so  same-run stranger vs other-run stranger (group convention beyond chance)"""
    T, pilot, odds = build()
    rng = random.Random(seed)
    bygoal = defaultdict(list)
    for e in T.values():
        bygoal[(e['goal_tile'], e['goal_dir'])].append(e)
    out = []
    for p in pilot:
        a, P = T[p['anchor']], T[p['partner']]
        g = bygoal[(P['goal_tile'], P['goal_dir'])]
        other = [e for e in g if e['run'] != a['run']]
        other.sort(key=lambda e: (e['phase'] != P['phase'], rng.random()))
        ctrls = [p['control']] + [tid(e) for e in other if tid(e) != p['control']][:n_other - 1]
        for j, c in enumerate(ctrls):
            out.append({'type': 'po', 'anchor': p['anchor'], 'good': p['partner'], 'base': c, 'swap': p['swap'] if j == 0 else rng.random() < 0.5})
        same = [e for e in g if e['run'] == a['run'] and e['s'] not in (a['s'], a['r'])]
        for e in same:
            out.append({'type': 'ps', 'anchor': p['anchor'], 'good': p['partner'], 'base': tid(e), 'swap': rng.random() < 0.5})
            o = rng.choice(other)
            out.append({'type': 'so', 'anchor': p['anchor'], 'good': tid(e), 'base': tid(o), 'swap': rng.random() < 0.5})
    return T, out, odds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-6.1-terra')
    ap.add_argument('--n_odd', type=int, default=100)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--full', action='store_true', help='all probe types, several comparisons per anchor')
    a = ap.parse_args()
    import openai
    client = openai.OpenAI(max_retries=6, timeout=60)
    usage, lock = {}, threading.Lock()
    if a.full:
        T, probes, odds = build_full()
    else:
        T, probes, odds = build(n_odd=a.n_odd)
        probes = [{'type': 'po', 'anchor': p['anchor'], 'good': p['partner'], 'base': p['control'], 'swap': p['swap']} for p in probes]
    cache_path = os.path.join(OUT, f'align_cache_{a.model}.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']

    def ask(key, user, schema):
        if key not in cache:
            out = F.ask_openai(client, a.model, JUDGE, user, schema, usage, lock)
            with lock:
                cache[key] = out
                with open(cache_path, 'a') as f:
                    f.write(json.dumps({'key': key, 'out': out}) + '\n')
        return cache[key]

    def do_probe(p):
        c1, c2 = (p['base'], p['good']) if p['swap'] else (p['good'], p['base'])
        user = (f"ANCHOR turn:\n{item(T[p['anchor']])}\n\nCandidate A:\n{item(T[c1])}\n\nCandidate B:\n{item(T[c2])}\n\n"
                "Which candidate uses the more similar signalling strategy to the anchor (relative to each turn's own goal)?")
        out = ask(f"probe:{p['anchor']}:{c1}:{c2}", user, PROBE_SCHEMA)
        if not out:
            return None
        chose = c1 if out['more_similar'] == 'A' else c2
        return int(chose == p['good'])

    def do_odd(o):
        its = [o['items'][j] for j in o['order']]
        user = '\n\n'.join(f"Turn {i + 1}:\n{item(T[x])}" for i, x in enumerate(its)) + \
               "\n\nWhich turn's signalling strategy is the odd one out (the other two are most alike)?"
        out = ask(f"odd:{':'.join(its)}", user, ODD_SCHEMA)
        return its[int(out['odd_one_out']) - 1] if out else None

    with ThreadPoolExecutor(a.workers) as ex:
        pr = list(ex.map(do_probe, probes))
        od = list(ex.map(do_odd, odds))

    rows = []
    for p, r in zip(probes, pr):
        if r is None:
            continue
        an = T[p['anchor']]
        partner = [e for e in T.values() if e['run'] == an['run'] and e['s'] == an['r'] and e['r'] == an['s']]
        rows.append({**p, 'chose_good': r, 'run': an['run'], 'phase': an['phase'], 'pg': an['pg'], 'round': an['round'],
                     'pair': '-'.join(sorted([an['s'], an['r']])), 'succ_anchor': an['succ'],
                     'pair_success': sum(e['succ'] for e in partner + [x for x in T.values() if x['run'] == an['run'] and x['s'] == an['s'] and x['r'] == an['r']]) /
                                     max(1, len(partner) + sum(1 for x in T.values() if x['run'] == an['run'] and x['s'] == an['s'] and x['r'] == an['r']))})
    tag = a.model + ('_full' if a.full else '')
    with open(os.path.join(OUT, f'align_probes_{tag}.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    agree = [od[i] == od[i + 1] for i in range(0, len(od), 2) if od[i] and od[i + 1]]
    res = {'model': a.model, 'n_probes': len(rows),
           'p_good_by_type': {t: sum(r['chose_good'] for r in rows if r['type'] == t) / max(1, sum(1 for r in rows if r['type'] == t)) for t in ('po', 'ps', 'so')},
           'odd_consistency': sum(agree) / len(agree) if agree else None, 'odd_n': len(agree)}
    if usage:
        pr_ = F.PRICES.get(a.model, (1, 1, 0.05, 5))
        res['cost'] = (usage.get('input_tokens', 0) * pr_[0] + usage.get('cache_read_input_tokens', 0) * pr_[2] + usage.get('output_tokens', 0) * pr_[3]) / 1e6
    json.dump(res, open(os.path.join(OUT, f'align_{tag}.json'), 'w'), indent=1)
    with open(os.path.join(OUT, f'align_odd_{tag}.jsonl'), 'w') as f:
        for o, r in zip(odds, od):
            f.write(json.dumps({**o, 'odd': r}) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
