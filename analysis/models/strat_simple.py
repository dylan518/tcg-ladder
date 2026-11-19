#!/usr/bin/env python3
"""
Major simplification: each round is one line with the strategy and SUCCESS/FAIL (strat_ctx.py transcripts).
Few-shot = whole games of 5 players from the OTHER runs. Query = this player's game so far + the current round up to
"you sent:" / "you read it:". The model answers the player's strategy (Sender) or reading (Receiver) and whether the
round succeeds. Leave-one-run-out.

Scored against history-free baselines: strategy 43% (Sender) / 49% (Receiver) from strategy_pair, and "always FAIL"
for the result (66%).

Usage: python3 analysis/models/strat_simple.py [--model claude-haiku-5-5] [--shots 5] [--runs 2] [--limit N]
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
import api_fewshot as AF  # noqa: E402
import strat_ctx as X  # noqa: E402

OUT = os.path.join(HERE, 'out')
PRICES = {'claude-sonnet-5-5': (2, 2.5, 0.1, 10), 'claude-haiku-5-5': (0.1, 0.125, 0.01, 0.5)}
TASK = ("\n=== YOUR TASK ===\nYou will get another real player's game so far and the start of their current round. Emulate this "
        "real player: predict the strategy they use this round and whether the round ends in SUCCESS or FAIL.")


def schema(opts):
    return {'type': 'object', 'properties': {'strategy': {'type': 'string', 'enum': opts}, 'result': {'type': 'string', 'enum': ['SUCCESS', 'FAIL']}},
            'required': ['strategy', 'result'], 'additionalProperties': False}


def system_prompt(events, names, held, shots):
    train = sorted(k for k in events if k[0] != held)
    pick = random.Random(held).sample(train, min(shots, len(train)))
    return X.INTRO + '\n'.join(f"=== EXAMPLE: one real player's whole game (another group) ===\n" + X.full_text(events[k], names)[len(X.INTRO):]
                               for k in pick) + TASK


def user_prompt(evl, i, names):
    role, ev = evl[i]
    FIRST = '(this is their first round)\n'
    past = ''.join(''.join(X.render(r, e, names)) for r, e in evl[:i])
    pre = X.render(role, ev, names)[0]
    what = 'you sent' if role == 'sender' else 'you read it'
    return f"THIS PLAYER'S GAME SO FAR:\n{past or FIRST}\nCURRENT ROUND:\n{pre}...\n\nGive \"{what}\" and the result."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='claude-haiku-5-5')
    ap.add_argument('--shots', type=int, default=5)
    ap.add_argument('--runs', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    import anthropic
    events, names = X.load_players()
    runs = [int(r) for r in a.runs.split(',')] if a.runs else sorted({k[0] for k in events})
    client = anthropic.Anthropic(max_retries=6)
    usage, lock = {}, threading.Lock()
    cache_path = os.path.join(OUT, f'strat_simple_cache_{a.model}_{a.shots}.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']

    rows = []
    for held in runs:
        system = system_prompt(events, names, held, a.shots)
        mine = {k: v for k, v in events.items() if k[0] == held}
        jobs = [(k, i) for k, evl in mine.items() for i in range(len(evl[:a.limit or None]))]

        def one(job):
            k, i = job
            role, ev = mine[k][i]
            opts = [X.option_text(role, o) for o in X.options(role, ev)]
            key = f"{held}:{ev['round']}:{role}:{k[1]}"
            if key not in cache:
                out = AF.ask(client, a.model, system, user_prompt(mine[k], i, names), schema(opts), usage, lock)
                with lock:
                    cache[key] = out
                    with open(cache_path, 'a') as f:
                        f.write(json.dumps({'key': key, 'out': out}) + '\n')
            return k, i, opts, cache[key]
        res = [one(jobs[0])]
        with ThreadPoolExecutor(a.workers) as ex:
            res += list(ex.map(one, jobs[1:]))
        for k, i, opts, out in res:
            role, ev = mine[k][i]
            truth = X.option_text(role, ev['S'] if role == 'sender' else ev['R'])
            out = out or {'strategy': None, 'result': 'FAIL'}
            rows.append({'run': held, 'round': ev['round'], 'phase': ev['phase'], 'role': role, 'player': names[held][k[1]],
                         'pred': out['strategy'], 'truth': truth, 'pred_succ': int(out['result'] == 'SUCCESS'), 'succ': ev['succ']})
        print(f'fold {held} done ({len(res)} turns)', flush=True)

    mean = lambda v: sum(v) / len(v) if v else float('nan')
    res = {'model': a.model, 'shots': a.shots}
    for role in ('sender', 'receiver'):
        r = [x for x in rows if x['role'] == role]
        res[role] = {'n': len(r), 'strategy_acc': mean([x['pred'] == x['truth'] for x in r]),
                     'result_acc': mean([x['pred_succ'] == x['succ'] for x in r]),
                     'both_acc': mean([x['pred'] == x['truth'] and x['pred_succ'] == x['succ'] for x in r])}
    res['success_by_phase'] = {ph: {'pred': mean([x['pred_succ'] for x in rows if x['phase'] == ph]),
                                    'real': mean([x['succ'] for x in rows if x['phase'] == ph])} for ph in (1, 2, 3)}
    res['result_baseline_always_fail'] = mean([1 - x['succ'] for x in rows])
    if usage:
        pr = PRICES.get(a.model, PRICES['claude-sonnet-5-5'])
        res['cost'] = (usage.get('input_tokens', 0) * pr[0] + usage.get('cache_creation_input_tokens', 0) * pr[1] +
                       usage.get('cache_read_input_tokens', 0) * pr[2] + usage.get('output_tokens', 0) * pr[3]) / 1e6
    tag = f'{a.model}_{a.shots}' + (f'_runs{a.runs}' if a.runs else '') + (f'_lim{a.limit}' if a.limit else '')
    with open(os.path.join(OUT, f'strat_simple_rows_{tag}.jsonl'), 'w') as f:
        for x in rows:
            f.write(json.dumps(x) + '\n')
    json.dump(res, open(os.path.join(OUT, f'strat_simple_{tag}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    print('baselines: strategy sender 43% / receiver 49% (history-free); result "always FAIL"', f"{res['result_baseline_always_fail']:.0%}")


if __name__ == '__main__':
    main()
