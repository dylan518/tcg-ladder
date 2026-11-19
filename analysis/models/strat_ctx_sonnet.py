#!/usr/bin/env python3
"""
Sonnet with whole-game context at the strategy level (strat_ctx.py). Few-shot = complete game transcripts of players
from the OTHER runs (all rounds, all partners); the query = this player's own game so far + the current round. Sonnet
returns a probability for each option (Sender strategy or Receiver reading). Same scoring as strat_ctx_sft.py.

Usage: python3 analysis/models/strat_ctx_sonnet.py [--shots 4] [--mode per_turn|closed|both] [--sims 2] [--limit N]
Cached in analysis/models/out/strat_ctx_sonnet_*.jsonl so re-runs never re-bill.
"""
import argparse
import json
import math
import os
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import anthropic
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat_ctx as X  # noqa: E402
import strategy_pair as SP  # noqa: E402

OUT = os.path.join(HERE, 'out')
TASK = ("\n\nYou will see another player's game so far and the start of their current round. Predict what THIS real person "
        "does next, as a probability for each option (real people often misread each other; conventions spread and change "
        "between partners). Probabilities must sum to 1.")


def schema(opts):
    return {'type': 'object', 'properties': {k: {'type': 'number'} for k in opts}, 'required': list(opts), 'additionalProperties': False}


def ask(client, model, system, user, opts_text, usage, lock):
    resp = client.messages.create(
        model=model, max_tokens=3000,
        system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
        messages=[{'role': 'user', 'content': user}],
        output_config={'effort': 'low', 'format': {'type': 'json_schema', 'schema': schema(opts_text)}})
    with lock:
        for k in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
            usage[k] = usage.get(k, 0) + (getattr(resp.usage, k, 0) or 0)
    d = json.loads(next(b.text for b in resp.content if b.type == 'text'))
    p = np.array([max(float(d.get(o, 0)), 1e-4) for o in opts_text])
    return (p / p.sum()).tolist()


def query(ctx, pre, role, opts_text):
    return (f"GAME SO FAR:\n{ctx[len(X.INTRO):] or '(first round)'}\nCURRENT ROUND:\n{pre}...\n\n"
            f"Options for what comes after \"{'you sent:' if role == 'sender' else 'you read it:'}\": " + ' / '.join(opts_text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='claude-sonnet-5-5')
    ap.add_argument('--shots', type=int, default=4)
    ap.add_argument('--mode', default='per_turn', choices=['per_turn', 'closed', 'both'])
    ap.add_argument('--sims', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    events, names = X.load_players()
    runs = sorted({r for r, _ in events})
    allr = SP.load()
    client = anthropic.Anthropic(max_retries=6)
    usage, lock = {}, threading.Lock()
    cache_path = os.path.join(OUT, f'strat_ctx_sonnet_cache_{a.model}_{a.shots}.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['p']

    def cached_ask(key, system, user, opts_text):
        if key in cache:
            return cache[key]
        p = ask(client, a.model, system, user, opts_text, usage, lock)
        with lock:
            cache[key] = p
            with open(cache_path, 'a') as f:
                f.write(json.dumps({'key': key, 'p': p}) + '\n')
        return p

    pt = defaultdict(lambda: [0, 0.0, 0])
    closed = []
    rng = np.random.default_rng(0)
    for held in runs:
        train = [(p, ev) for (r, p), ev in events.items() if r != held]
        shots = random.Random(held).sample(train, min(a.shots, len(train)))
        system = X.INTRO + 'EXAMPLE GAMES (other players, other groups):\n\n' + '\n'.join(
            f"--- example player {i + 1} ---\n" + X.full_text(ev, names)[len(X.INTRO):] for i, (_, ev) in enumerate(shots)) + TASK
        te = {p: ev for (r, p), ev in events.items() if r == held}
        if a.mode in ('per_turn', 'both'):
            jobs = []
            for p, evl in te.items():
                ctx = X.INTRO
                for role, ev in evl[:a.limit or None]:
                    pre, tgt, suf = X.render(role, ev, names)
                    opts = X.options(role, ev)
                    jobs.append((role, ev, opts, ctx, pre))
                    ctx += pre + tgt + suf

            def one(j):
                role, ev, opts, ctx, pre = j
                ot = [X.option_text(role, k) for k in opts]
                return cached_ask(f"pt:{held}:{ev['round']}:{role}:{ev['s'] if role == 'sender' else ev['r']}", system, query(ctx, pre, role, ot), ot)
            first = one(jobs[0])
            with ThreadPoolExecutor(a.workers) as ex:
                probs = [first] + list(ex.map(one, jobs[1:]))
            for (role, ev, opts, _, _), pr in zip(jobs, probs):
                truth = ev['S'] if role == 'sender' else ev['R']
                jx = opts.index(truth)
                pt[role][0] += 1; pt[role][1] -= math.log2(pr[jx]); pt[role][2] += int(np.argmax(pr) == jx)
            print(f'fold {held} per-turn done', flush=True)
        if a.mode in ('closed', 'both'):
            tab, p_end = SP.outcome_table([x for x in allr if x['run'] != held])
            sched = sorted({(ev['round'], ev['s']): ev for evl in te.values() for role, ev in evl if role == 'sender'}.items())
            byround = defaultdict(list)
            for (rnd, _), ev in sched:
                byround[rnd].append(ev)
            for sim in range(a.sims):
                ctx = {p: X.INTRO for p in te}
                out = []
                for rnd in sorted(byround)[: a.limit or None]:
                    evs = byround[rnd]
                    def send(ev):
                        pre, _, _ = X.render('sender', ev, names)
                        so = X.options('sender', ev); ot = [X.option_text('sender', k) for k in so]
                        return so, cached_ask(f"cl:{held}:{sim}:{rnd}:{ev['s']}:S", system, query(ctx[ev['s']], pre, 'sender', ot), ot)
                    with ThreadPoolExecutor(a.workers) as ex:
                        sres = list(ex.map(send, evs))
                    picks = []
                    for ev, (so, ps) in zip(evs, sres):
                        S = so[rng.choice(len(so), p=ps)]
                        picks.append((ev, S, X.sim_signal(S, ev, rng)))
                    def recv(item):
                        ev, S, sev = item
                        pre, _, _ = X.render('receiver', sev, names, S=S, R='F')
                        ro = X.options('receiver', sev, S); ot = [X.option_text('receiver', k) for k in ro]
                        return ro, cached_ask(f"cl:{held}:{sim}:{rnd}:{ev['r']}:R:{S}", system, query(ctx[ev['r']], pre, 'receiver', ot), ot)
                    with ThreadPoolExecutor(a.workers) as ex:
                        rres = list(ex.map(recv, picks))
                    for (ev, S, sev), (ro, pr) in zip(picks, rres):
                        R = ro[rng.choice(len(ro), p=pr)]
                        succ = int(rng.random() < p_end and rng.random() < tab.get((S, R), 0.0))
                        ctx[ev['s']] += ''.join(X.render('sender', sev, names, S=S, R=R, succ=succ))
                        ctx[ev['r']] += ''.join(X.render('receiver', sev, names, S=S, R=R, succ=succ))
                        out.append({'run': held, 'sim': sim, 'round': rnd, 'phase': ev['phase'], 'S': S, 'R': R, 'succ': succ,
                                    'match': int(X.match(S, R)), 'signal': int(S in SP.SIGNAL), 'real_succ': ev['succ']})
                closed += out
                print(f"fold {held} closed sim {sim}: success {sum(o['succ'] for o in out)}/{len(out)} (real {sum(o['real_succ'] for o in out)})", flush=True)

    res = {'model': a.model, 'shots': a.shots}
    if pt:
        res['per_turn'] = {r: {'n': v[0], 'bits': v[1] / v[0], 'acc': v[2] / v[0]} for r, v in pt.items()}
        print('per-turn:', {r: f"acc {v['acc']:.0%} bits {v['bits']:.2f} (n={v['n']})" for r, v in res['per_turn'].items()},
              ' | history-free baseline: sender 43% / 2.07, receiver 49% / 1.40')
    if closed:
        by = defaultdict(list)
        for o in closed:
            by['s', o['phase']].append(o['succ']); by['r', o['phase']].append(o['real_succ'])
            if o['signal']:
                by['m', o['phase']].append(o['match'])
        res['closed'] = {ph: {'sim_success': float(np.mean(by['s', ph])), 'real_success': float(np.mean(by['r', ph])),
                              'sim_match': float(np.mean(by['m', ph])) if by['m', ph] else None} for ph in (1, 2, 3) if by['s', ph]}
        print('closed by phase:', {ph: f"success {v['sim_success']:.0%} (real {v['real_success']:.0%}), match {v['sim_match']:.0%}" for ph, v in res['closed'].items()},
              ' | real match 59/30/69%')
        with open(os.path.join(OUT, f'strat_ctx_sonnet_closed_{a.model}.jsonl'), 'w') as f:
            for o in closed:
                f.write(json.dumps(o) + '\n')
    json.dump(res, open(os.path.join(OUT, f'strat_ctx_sonnet_{a.model}_{a.mode}.json'), 'w'), indent=1)
    if usage:
        cost = (usage.get('input_tokens', 0) * 2 + usage.get('cache_creation_input_tokens', 0) * 2.5 +
                usage.get('cache_read_input_tokens', 0) * 0.1 + usage.get('output_tokens', 0) * 10) / 1e6
        print(f'~${cost:.2f}')


if __name__ == '__main__':
    main()
