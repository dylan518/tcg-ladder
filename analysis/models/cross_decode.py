#!/usr/bin/env python3
"""
Cross-decoding: can Sonnet (the best receiver model) read what the Qwen SFT readable-v2 Sender is trying to say?

For every held-out Sender turn, the Qwen SFT model (trained on the other two runs) generated a Sender turn for the real
goal. Here that generated turn replaces what the human Sender did, and Sonnet (same 40-shot readable receiver prompt as
api_fewshot.py, examples from the other runs) predicts where the Receiver puts the triangle. Score: would it succeed?
Reference: the same Sonnet receiver reading the REAL human Senders (cached in out/api_fewshot_receiver_*_readable.jsonl).

Usage: python3 analysis/models/cross_decode.py [--preds out/server/sft_readable_v2/predictions.jsonl] [--limit N]
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import api_fewshot as AF  # noqa: E402
import common as C  # noqa: E402
import readable as RD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default=os.path.join(HERE, 'out', 'server', 'sft_readable_v2', 'predictions.jsonl'))
    ap.add_argument('--model', default='claude-sonnet-5-5')
    ap.add_argument('--shots', type=int, default=40)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    turns = C.load_turns()
    if os.path.basename(a.preds).startswith('api_fewshot_sender'):  # a Claude Sender run (readable steps, id run:round:sender:player)
        gen = {}
        for r in map(json.loads, open(a.preds)):
            if r['out']:
                run, rnd, _, pl = r['id'].split(':', 3)
                gen[(int(run), int(rnd), pl)] = RD.realize(r['out'])
        tag = os.path.basename(a.preds)[len('api_fewshot_sender_'):-len('.jsonl')]
    else:
        gen = {(p['run'], p['round'], p['player']): p['gen'] for p in map(json.loads, open(a.preds)) if p['role'] == 'sender'}
        tag = os.path.basename(os.path.dirname(a.preds))
    path = os.path.join(AF.OUT, f'cross_decode_{tag}_{a.model}.jsonl')
    done = {json.loads(l)['id']: json.loads(l) for l in open(path)} if os.path.exists(path) else {}
    human_dec = {}
    hp = os.path.join(AF.OUT, f'api_fewshot_receiver_{a.model}_readable.jsonl')
    if os.path.exists(hp):
        human_dec = {r['id']: r['out'] for r in map(json.loads, open(hp)) if r['out']}
    client = anthropic.Anthropic(max_retries=6)
    usage, lock = {}, threading.Lock()
    rows = []
    for held, train, test in C.folds(turns):
        rng = random.Random(held)  # same examples as the human-sender receiver run
        system = (AF.RULES + AF.READABLE_LEGEND + '\n\nExamples of real Receiver turns (other groups):\n'
                  + AF.receiver_examples(train, a.shots, rng, AF.rd) + '\n\nAnswer with the tile and direction where this Receiver put the triangle.')
        # receiver turns of the held-out run, paired with the Qwen sender turn for the same round/sender
        todo = []
        for t in test:
            if t['role'] != 'receiver' or (t['run'], t['round'], t['partner']) not in gen:
                continue
            q = dict(t, saw_sender_tokens=gen[(t['run'], t['round'], t['partner'])])
            todo.append((t, q))
        if a.limit:
            todo = todo[:a.limit]

        def one(item):
            t, q = item
            tid = AF.turn_id(t)
            if tid in done:
                return done[tid]
            out = AF.ask(client, a.model, system, C.receiver_query(q, turns, AF.rd), AF.REC_SCHEMA, usage, lock)
            r = {'id': tid, 'run': t['run'], 'qwen_sender': RD.describe(RD.steps_from_tokens(q['saw_sender_tokens'])), 'out': out}
            with lock, open(path, 'a') as f:
                f.write(json.dumps(r) + '\n')
            return r

        if not todo:
            continue
        rows.append((todo[0], one(todo[0])))
        with ThreadPoolExecutor(a.workers) as ex:
            rows += list(zip(todo[1:], ex.map(one, todo[1:])))

    n = ok = tile = dirn = 0
    qend = 0
    h_ok = h_n = 0
    by_phase = {}
    for (t, q), r in rows:
        if not r['out']:
            continue
        n += 1
        s = r['out']['tile'] == t['goal_tile'] and r['out']['direction'] == t['goal_direction']
        ok += s; tile += r['out']['tile'] == t['goal_tile']; dirn += r['out']['direction'] == t['goal_direction']
        qend += C.replay([x for x in q['saw_sender_tokens'] if x in 'UDLR'])[0] == t['goal_tile']
        by_phase.setdefault(t['phase'], [0, 0, 0]); by_phase[t['phase']][0] += s; by_phase[t['phase']][1] += t['success']; by_phase[t['phase']][2] += 1
        hd = human_dec.get(AF.turn_id(t))
        if hd:
            h_n += 1; h_ok += hd['tile'] == t['goal_tile'] and hd['direction'] == t['goal_direction']
    print(f'{tag} Sender -> {a.model} Receiver, n={n} held-out rounds')
    print(f'  success {ok / n:.0%} (goal tile {tile / n:.0%}, goal direction {dirn / n:.0%}); sender ended on goal tile {qend / n:.0%}')
    if h_n:
        print(f'  reference: same Sonnet receiver reading the REAL human senders: success {h_ok / h_n:.0%} (n={h_n}); real human pairs {sum(t["success"] for (t, _), _ in rows) / len(rows):.0%}')
    print('  by phase (model pair vs real humans): ' + ', '.join(f'{k}: {v[0] / v[2]:.0%} vs {v[1] / v[2]:.0%}' for k, v in sorted(by_phase.items())))
    if usage:
        cost = (usage.get('input_tokens', 0) * 2 + usage.get('cache_creation_input_tokens', 0) * 2.5 +
                usage.get('cache_read_input_tokens', 0) * 0.1 + usage.get('output_tokens', 0) * 10) / 1e6
        print(f'  ~${cost:.2f}')


if __name__ == '__main__':
    main()
