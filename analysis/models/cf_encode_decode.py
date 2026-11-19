#!/usr/bin/env python3
"""
Counterfactuals: does the phase-3 rise come from better ENCODING (senders' signals get clearer), better DECODING
(receivers get better at reading), or a shared convention (late senders + late receivers)?

A simulated Receiver (fullgame_sonnet prompt: rules + 5 complete real games from the other runs, as moves) decodes a
real Sender turn and answers tile + direction; scored against the real goal.

  encode   fixed decoder with NO history ("first round") reads every real Sender turn -> success by the turn's phase.
  decode   2x2 per held-out run: signals = real Sender turns from phase 1 (early) or phase 3 (late) of that run;
           receiver = each real player R conditioned on R's real game up to the end of phase 1 (early R) or up to the
           start of phase 3 (late R). Signals R received itself are excluded. Success in each cell:
             sender effect   = late signals - early signals (same receivers)
             receiver effect = late R - early R (same signals)
             interaction     = convention shared by late senders and late receivers
Usage: python3 analysis/models/cf_encode_decode.py [--model gpt-6.1-terra] [--per_cell 8]
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
import fullgame_sonnet as F  # noqa: E402

OUT = os.path.join(HERE, 'out')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-6.1-terra')
    ap.add_argument('--per_cell', type=int, default=8, help='signals per receiver per early/late set')
    ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    F.PROMPT_V = 'v1'
    import openai
    client = openai.OpenAI(max_retries=6, timeout=60)
    usage, lock = {}, threading.Lock()
    games, names = F.load_players()
    cache_path = os.path.join(OUT, f'cf_cache_{a.model}.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']
    systems = {run: F.system_prompt(games, names, run, 5) for run in sorted({k[0] for k in games})}

    def decode(key, run, history, ev):
        if key not in cache:
            evl = history + [('receiver', ev)]
            out = F.ask_openai(client, a.model, systems[run], F.user_prompt(evl, len(evl) - 1, names), AF.REC_SCHEMA, usage, lock)
            with lock:
                cache[key] = out
                with open(cache_path, 'a') as f:
                    f.write(json.dumps({'key': key, 'out': out}) + '\n')
        out = cache[key]
        return None if not out else int(out['tile'] == ev['goal_tile'] and out['direction'] == ev['goal_dir'])

    senders = {}
    for (run, p), evl in games.items():
        for role, ev in evl:
            if role == 'sender':
                senders[(run, ev['round'], ev['s'])] = ev
    jobs = []
    # ---- encode: no-history decoder on every real Sender turn
    for (run, rnd, s), ev in senders.items():
        jobs.append(('encode', f"enc:{run}:{rnd}:{s}", run, [], ev, {'phase': ev['phase'], 'real': ev['succ']}))
    # ---- decode 2x2
    rng = random.Random(0)
    for (run, R), evl in games.items():
        early_hist = [x for x in evl if x[1]['phase'] == 1]
        late_hist = [x for x in evl if x[1]['phase'] <= 2]
        received = {(x[1]['round'], x[1]['s']) for x in evl if x[0] == 'receiver'}
        for sp, label in ((1, 'early_sig'), (3, 'late_sig')):
            pool = [ev for (r_, rnd, s), ev in senders.items() if r_ == run and ev['phase'] == sp and s != R and (rnd, s) not in received]
            for ev in rng.sample(pool, min(a.per_cell, len(pool))):
                for hist, hl in ((early_hist, 'early_R'), (late_hist, 'late_R')):
                    # this receiver decodes another player's signal: relabel the round as R's, partner = the real sender
                    ev2 = dict(ev, r=R, pg=sum(1 for x in hist if x[1]['s'] == ev['s'] or x[1]['r'] == ev['s']) // 2)
                    jobs.append(('decode', f"dec:{run}:{R}:{hl}:{ev['round']}:{ev['s']}", run, hist, ev2,
                                 {'sig': label, 'rec': hl, 'R': R, 'run': run}))
    print(f"{sum(j[0] == 'encode' for j in jobs)} encode + {sum(j[0] == 'decode' for j in jobs)} decode queries", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        got = list(ex.map(lambda j: decode(j[1], j[2], j[3], j[4]), jobs))
    rows = [{'kind': j[0], **j[5], 'correct': g} for j, g in zip(jobs, got) if g is not None]
    with open(os.path.join(OUT, f'cf_rows_{a.model}.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    m = lambda v: sum(v) / len(v) if v else float('nan')
    enc = [r for r in rows if r['kind'] == 'encode']
    print('\nENCODE (same no-history decoder reads every real Sender turn): success by phase of the turn')
    for ph in (1, 2, 3):
        v = [r for r in enc if r['phase'] == ph]
        print(f"  phase {ph}: decoder {m([r['correct'] for r in v]):.0%}   real receivers {m([r['real'] for r in v]):.0%}   (n={len(v)})")
    dec = [r for r in rows if r['kind'] == 'decode']
    cell = {(s, r): m([x['correct'] for x in dec if x['sig'] == s and x['rec'] == r]) for s in ('early_sig', 'late_sig') for r in ('early_R', 'late_R')}
    n = {(s, r): sum(1 for x in dec if x['sig'] == s and x['rec'] == r) for s in ('early_sig', 'late_sig') for r in ('early_R', 'late_R')}
    print('\nDECODE 2x2 (success):           early receiver   late receiver')
    for s in ('early_sig', 'late_sig'):
        print(f"  {s:12s} (phase {'1' if s == 'early_sig' else '3'} signals)   {cell[s, 'early_R']:.0%} (n={n[s, 'early_R']})        {cell[s, 'late_R']:.0%} (n={n[s, 'late_R']})")
    se = (cell['late_sig', 'early_R'] + cell['late_sig', 'late_R'] - cell['early_sig', 'early_R'] - cell['early_sig', 'late_R']) / 2
    re = (cell['early_sig', 'late_R'] + cell['late_sig', 'late_R'] - cell['early_sig', 'early_R'] - cell['late_sig', 'early_R']) / 2
    ix = (cell['late_sig', 'late_R'] - cell['late_sig', 'early_R']) - (cell['early_sig', 'late_R'] - cell['early_sig', 'early_R'])
    print(f"\n  sender (encoding) effect   {se:+.0%}\n  receiver (decoding) effect {re:+.0%}\n  interaction (shared convention) {ix:+.0%}")
    if usage:
        print(f"~${(usage.get('input_tokens', 0) * 1 + usage.get('cache_read_input_tokens', 0) * 0.05 + usage.get('output_tokens', 0) * 5) / 1e6:.2f}")


if __name__ == '__main__':
    main()
