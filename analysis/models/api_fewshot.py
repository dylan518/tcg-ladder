#!/usr/bin/env python3
"""
Approach A: a Claude model, prompted with many real human turns (few-shot / "multishot"), predicts what a human
Receiver does and generates human-like Sender turns. Leave-one-run-out: the examples always come from the OTHER runs.

Usage:
  python3 analysis/models/api_fewshot.py --task receiver|sender|both [--limit N] [--shots 40] [--model claude-sonnet-5-5]
Needs ANTHROPIC_API_KEY (or an `ant auth login` profile). Predictions are cached in analysis/models/out/ so re-running
never re-bills a finished turn.

Prompt layout: system = game rules + token legend + the few-shot block (cached with cache_control, identical for every
turn in a fold); user = this turn's context (pair history so far + what the Sender did / the goal).
Thinking stays on (it can't be turned off on this model); effort "low" because each answer is short.
"""
import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import readable as RD  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

RULES = """You are simulating a human participant in a two-player communication game on a 3x3 board.
Each round one player is the Sender and the other the Receiver. Only the Sender sees the goal: a tile on the board
and one of 8 directions (up, up-right, right, down-right, down, down-left, left, up-left) that the Receiver's
triangle must point. The Sender can only communicate by moving their own circle around the board with arrow keys;
then the Receiver moves and rotates a triangle. Both keys start in the centre; the triangle starts pointing up.
Players cannot talk. They play 10 rounds with one partner, alternating roles, then switch partners.

A turn is written as tokens:
  U D L R      move up / down / left / right (pressing against the edge does nothing)
  RL RR        rotate the triangle 45 degrees left / right (Receiver only)
  W1 W2 W3     a pause before the next key: W1 0.4-1 s, W2 1-2 s, W3 over 2 s (no token = a quick press)
  END          pressed Space to finish the turn;  TIMEOUT  the turn's ~9 second limit ran out

These are REAL human players: about a third of rounds succeed, people often misread each other, and
partners slowly develop private conventions. Predict what THIS human actually did, not the ideal answer."""


READABLE_LEGEND = """A turn is written as the steps the player took (readable form):
  went to X              moved the key to tile X
  wiggled A<->B xN       went back and forth between tiles A and B, N times
  looped A->B->C->B xN   went round the same small circuit of tiles N times
  paused / long pause    stayed still for over 1 s / over 2 s
  turned left/right xN   rotated the triangle 45 degrees N times (Receiver only), with the direction it then points
  pressed done / time ran out   how the turn ended (each turn has a ~9 second limit)
Tiles: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right."""


def rd(tokens):
    return RD.describe(RD.steps_from_tokens(tokens))


def receiver_examples(train, k, rng, R=C.raw):
    rec = [t for t in train if t['role'] == 'receiver']
    ex = rng.sample(rec, min(k, len(rec)))
    return '\n'.join(f"Sender did: {R(t['saw_sender_tokens'])}\n  -> Receiver placed the triangle on {t['receiver_tile']}, pointing "
                     f"{t['receiver_direction']} (goal was {t['goal_tile']}/{t['goal_direction']}; {'success' if t['success'] else 'fail'})"
                     for t in ex)


def sender_examples(train, k, rng, R=C.raw):
    snd = [t for t in train if t['role'] == 'sender']
    ex = rng.sample(snd, min(k, len(snd)))
    return '\n'.join(f"Goal {t['goal_tile']}/{t['goal_direction']} (round {t['round']}, {t['partner_games']} games with partner) -> "
                     f"{R(t['tokens'])}" for t in ex)


REC_SCHEMA = {'type': 'object', 'properties': {'tile': {'type': 'string', 'enum': C.TILES}, 'direction': {'type': 'string', 'enum': C.DIRS}},
              'required': ['tile', 'direction'], 'additionalProperties': False}
SND_SCHEMA = {'type': 'object', 'properties': {'tokens': {'type': 'string', 'description': 'space-separated tokens ending with END or TIMEOUT'}},
              'required': ['tokens'], 'additionalProperties': False}


SND_STEPS_SCHEMA = {'type': 'object', 'properties': {
    'steps': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'type': {'type': 'string', 'enum': ['go_to', 'wiggle', 'loop', 'pause']},
        'tiles': {'type': 'array', 'items': {'type': 'string', 'enum': C.TILES}},
        'repeats': {'type': 'integer'}}, 'required': ['type', 'tiles', 'repeats'], 'additionalProperties': False}},
    'end': {'type': 'string', 'enum': ['done', 'timeout']}}, 'required': ['steps', 'end'], 'additionalProperties': False}


def ask(client, model, system, user, schema, usage, lock):
    resp = client.messages.create(
        model=model, max_tokens=4000,
        system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
        messages=[{'role': 'user', 'content': user}],
        output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': schema}},
    )
    with lock:
        for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
            usage[k] = usage.get(k, 0) + (getattr(resp.usage, k, 0) or 0)
    if resp.stop_reason == 'refusal':
        return None
    text = next(b.text for b in resp.content if b.type == 'text')
    return json.loads(text)


def turn_id(t):
    return f"{t['run']}:{t['round']}:{t['role']}:{t['player']}"


def run(task, args):
    turns = C.load_turns()
    client = anthropic.Anthropic(max_retries=6)
    readable = args.format == 'readable'
    R = rd if readable else C.raw
    legend = READABLE_LEGEND if readable else ''
    path = os.path.join(OUT, f'api_fewshot_{task}_{args.model}{"_readable" if readable else ""}.jsonl')
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l)
            done[r['id']] = r
    usage, lock = {}, threading.Lock()
    rows = []
    for held, train, test in C.folds(turns):
        rng = random.Random(held)
        if task == 'receiver':
            system = RULES + legend + '\n\nExamples of real Receiver turns (other groups):\n' + receiver_examples(train, args.shots, rng, R) + \
                '\n\nAnswer with the tile and direction where this Receiver put the triangle.'
            todo = [t for t in test if t['role'] == 'receiver']
        else:
            system = RULES + legend + '\n\nExamples of real Sender turns (other groups):\n' + sender_examples(train, args.shots, rng, R) + \
                ('\n\nGive the steps this Sender took (go_to / wiggle / loop / pause, tiles, repeats) and how the turn ended.' if readable
                 else '\n\nWrite the token sequence this Sender produced.')
            todo = [t for t in test if t['role'] == 'sender']
        if args.limit:
            todo = todo[:args.limit]

        def one(t):
            tid = turn_id(t)
            if tid in done:
                return done[tid]
            user = C.receiver_query(t, turns, R) if task == 'receiver' else C.sender_query(t, turns, R)
            schema = REC_SCHEMA if task == 'receiver' else (SND_STEPS_SCHEMA if readable else SND_SCHEMA)
            out = ask(client, args.model, system, user, schema, usage, lock)
            r = {'id': tid, 'run': t['run'], 'out': out}
            with lock, open(path, 'a') as f:
                f.write(json.dumps(r) + '\n')
            return r

        if not todo:
            continue
        rows.append((todo[0], one(todo[0])))  # first request alone writes the cache; the rest read it
        with ThreadPoolExecutor(args.workers) as ex:
            for t, r in zip(todo[1:], ex.map(one, todo[1:])):
                rows.append((t, r))

    if task == 'receiver':
        sc = C.receiver_scores([{'turn': t, 'pred_tile': r['out']['tile'], 'pred_dir': r['out']['direction']} for t, r in rows if r['out']])
        print(f"[receiver] {args.model} ({args.format}), {args.shots} shots, n={sc['n']}: match human tile+dir {sc['match_both']:.0%}, tile {sc['match_tile']:.0%}, "
              f"direction {sc['match_dir']:.0%} | model success {sc['pred_success']:.0%} vs human {sc['human_success']:.0%}")
    else:
        gen = (lambda o: RD.realize(o)) if readable else (lambda o: C.parse_tokens(o['tokens']))
        model_p = C.sender_profile([(gen(r['out']), t['goal_tile'], t['goal_direction']) for t, r in rows if r['out']])
        human_p = C.sender_profile([(t['tokens'], t['goal_tile'], t['goal_direction']) for t, _ in rows])
        print(f"[sender] {args.model} ({args.format}), {args.shots} shots, n={model_p['n']}")
        for k in ('mean_len', 'timeout_rate', 'ends_on_goal'):
            print(f"   {k:13s} model {model_p[k]:.2f}  human {human_p[k]:.2f}")
        print(f"   strategy JS divergence {C.js_divergence(model_p['strategy'], human_p['strategy']):.3f} bits (0 = same mix)")
        for s in sorted(set(model_p['strategy']) | set(human_p['strategy'])):
            print(f"     {s:14s} model {model_p['strategy'].get(s, 0):4.0%}  human {human_p['strategy'].get(s, 0):4.0%}")
    if usage:
        cost = (usage.get('input_tokens', 0) * 2 + usage.get('cache_creation_input_tokens', 0) * 2.5 +
                usage.get('cache_read_input_tokens', 0) * 0.1 + usage.get('output_tokens', 0) * 10) / 1e6
        print(f'   tokens this run: {usage}  ~${cost:.2f} (list prices: Sonnet 5.5 $2/$10 per MTok)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='both', choices=['receiver', 'sender', 'both'])
    ap.add_argument('--limit', type=int, default=0, help='turns per fold (pilot)')
    ap.add_argument('--shots', type=int, default=40)
    ap.add_argument('--model', default='claude-sonnet-5-5')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--format', default='readable', choices=['readable', 'raw'])
    a = ap.parse_args()
    for task in (['receiver', 'sender'] if a.task == 'both' else [a.task]):
        run(task, a)
