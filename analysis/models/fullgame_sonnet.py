#!/usr/bin/env python3
"""
Sonnet with whole-game context written as the actual moves (readable form, no strategy labels).

Few-shot = 5 complete real play-throughs (every round, every partner) of players from the OTHER runs. Query = this
player's own game so far, then the current round up to their turn. Each round block:

  ROUND 12 | phase 2 | partner P4, game 3 with them | you are SENDER | goal: top-right, pointing up-right
  you: went to top-right; wiggled top-right<->top x2 (top is left of top-right); pressed done
  partner: went to top-right; turned right x1 (now pointing up-right); pressed done
  result: partner placed top-right pointing up-right: SUCCESS

Receivers never see the goal (only SUCCESS/FAIL), as in the game.
Answer: Sender -> the steps of their turn (realised to keys and coded afterwards); Receiver -> tile + direction.

Usage: python3 analysis/models/fullgame_sonnet.py --show            (write an example prompt to the Desktop)
       python3 analysis/models/fullgame_sonnet.py --mode per_turn|closed [--shots 5] [--sims 1] [--limit N]
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
import common as C  # noqa: E402
import readable as RD  # noqa: E402
import analyze_tournament as AT  # noqa: E402
import api_fewshot as AF  # noqa: E402
import strategy_pair as SP  # noqa: E402

OUT = os.path.join(HERE, 'out')
PRICES = {'claude-sonnet-5-5': (2, 2.5, 0.1, 10), 'claude-haiku-5-5': (0.1, 0.125, 0.01, 0.5), 'gpt-6.1-terra': (1, 1, 0.05, 5)}  # prompts <=100K tokens
TASK = {'v1': 'Say what this person does on their turn.',
        'v3': "Emulate this real player's activity: the goal is to predict their exact moves on this turn as closely as possible."}
PROMPT_V = 'v3'  # v1 = original task line; v2 = long 'emulate this person' text (never run); v3 = short emulate/predict-exact-moves line; bump whenever the prompt changes (cache + output names)

RULES = """You are simulating one real person playing a two-player communication game on a 3x3 board.
Tiles: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right.
Each round one player is the Sender, the other the Receiver. Only the Sender sees the goal: a tile, and one of 8
directions (up, up-right, right, down-right, down, down-left, left, up-left) the Receiver's triangle must point.
The Sender can only communicate by moving their circle around the board (about 9 seconds); the Receiver watches,
then moves a triangle to a tile and turns it. Both start in the centre, the triangle pointing up. No talking.
Players play several rounds with one partner, alternating roles, then switch to a new partner (phases 1, 2, 3).
Receivers only learn SUCCESS or FAIL, never what the goal was.

Moves are written as steps:
  went to X              moved to tile X
  wiggled A<->B xN       went back and forth between A and B N times (with which way B is from A)
  looped A->B->C xN      went round the same little circuit N times
  paused / long pause    stood still over 1 s / over 2 s
  turned left/right xN   rotated the triangle 45 degrees N times (Receiver), and where it ended up pointing
  pressed done / time ran out
These are REAL people: about a third of rounds succeed, partners often misread each other, and conventions form,
spread and clash as people switch partners."""


def load_players():
    T = C.load_turns()
    names = {}
    for t in sorted(T, key=lambda t: (t['run'], t['player'])):
        names.setdefault(t['run'], {}).setdefault(t['player'], f"P{len(names[t['run']]) + 1}")
    snd = {(t['run'], t['round'], t['player']): t for t in T if t['role'] == 'sender'}
    rec = {(t['run'], t['round'], t['player']): t for t in T if t['role'] == 'receiver'}
    games = {}
    for (run, rnd, sp), s in sorted(snd.items()):
        r = rec[(run, rnd, s['partner'])]
        ev = {'run': run, 'round': rnd, 'phase': s['phase'], 'pg': s['partner_games'], 's': sp, 'r': s['partner'],
              'goal_tile': s['goal_tile'], 'goal_dir': s['goal_direction'], 's_tokens': s['tokens'], 'r_tokens': r['tokens'],
              'rec_tile': s['receiver_tile'], 'rec_dir': s['receiver_direction'], 'succ': s['success']}
        games.setdefault((run, sp), []).append(('sender', ev))
        games.setdefault((run, s['partner']), []).append(('receiver', ev))
    for k in games:
        games[k].sort(key=lambda x: x[1]['round'])
    return games, names


def moves(tokens):
    return RD.describe(RD.steps_from_tokens(tokens))


def render(role, ev, names, upto_turn=False):
    """one round block from this player's view; upto_turn=True stops where this player has to act"""
    nm = names[ev['run']]
    other = ev['r'] if role == 'sender' else ev['s']
    head = (f"ROUND {ev['round']} | phase {ev['phase']} | partner {nm[other]}, game {ev['pg'] + 1} with them"
            + (' (new partner)' if ev['pg'] == 0 else '') + f" | you are {role.upper()}")
    if role == 'sender':
        head += f" | goal: {ev['goal_tile']}, pointing {ev['goal_dir']}"
        if upto_turn:
            return head + '\nyou: '
        return (f"{head}\nyou: {moves(ev['s_tokens'])}\npartner: {moves(ev['r_tokens'])}\n"
                f"result: partner placed {ev['rec_tile']} pointing {ev['rec_dir']}: {'SUCCESS' if ev['succ'] else 'FAIL'}\n")
    head += f"\npartner: {moves(ev['s_tokens'])}"
    if upto_turn:
        return head + '\nyou: '
    return (f"{head}\nyou: {moves(ev['r_tokens'])}\n"
            f"result: you placed {ev['rec_tile']} pointing {ev['rec_dir']}: {'SUCCESS' if ev['succ'] else 'FAIL'}\n")


def game_text(evl, names):
    return '\n'.join(render(role, ev, names) for role, ev in evl)


def system_prompt(games, names, held, shots):
    train = sorted(k for k in games if k[0] != held)
    pick = random.Random(held).sample(train, min(shots, len(train)))
    ex = '\n\n'.join(f"=== EXAMPLE: one real player's whole game (another group) ===\n{game_text(games[k], names)}" for k in pick)
    return RULES + '\n\n' + ex + ("\n\n=== YOUR TASK ===\nYou will get another real player's game so far and the start of their "
                                 "current round. " + TASK[PROMPT_V])


def user_prompt(evl, i, names):
    role, ev = evl[i]
    past = '\n'.join(render(r, e, names) for r, e in evl[:i])
    ask = ('Give the steps this Sender takes (go_to / wiggle / loop / pause, with tiles and repeats) and how the turn ends.'
           if role == 'sender' else 'Give the tile where this Receiver puts the triangle and the direction it points.')
    return f"THIS PLAYER'S GAME SO FAR:\n{past or '(this is their first round)'}\n\nCURRENT ROUND:\n{render(role, ev, names, upto_turn=True)}...\n\n{ask}"


def show(path):
    games, names = load_players()
    held = 2
    sysp = system_prompt(games, names, held, 5)
    key = sorted(k for k in games if k[0] == held)[0]
    evl = games[key]
    i = next(j for j, (role, ev) in enumerate(evl) if role == 'receiver' and ev['phase'] == 2 and ev['pg'] >= 2)
    role, ev = evl[i]
    out = (f"EXAMPLE SONNET PROMPT — whole real play-throughs, written as moves (analysis/models/fullgame_sonnet.py)\n"
           f"Held-out run {held}; predicting player {names[held][key[1]]}'s turn in round {ev['round']} (phase 2, Receiver).\n"
           f"System = rules + 5 complete real games from runs 1 and 3 (cached). User = this player's game so far + current round.\n\n"
           f"================ SYSTEM PROMPT ================\n{sysp}\n\n"
           f"================ USER MESSAGE ================\n{user_prompt(evl, i, names)}\n\n"
           f"================ ANSWER FORMAT ================\nJSON {{\"tile\": one of the 9 tiles, \"direction\": one of the 8 directions}}\n\n"
           f"================ WHAT THE REAL PLAYER DID (hidden from the model) ================\n"
           f"you: {moves(ev['r_tokens'])}\nresult: you placed {ev['rec_tile']} pointing {ev['rec_dir']}: {'SUCCESS' if ev['succ'] else 'FAIL'}\n")
    open(path, 'w').write(out)
    return len(sysp), len(out)


def code_S(tokens, goal_tile, goal_dir):
    """Sender tokens -> (strategy code, signal angle, end tile), same coder as strategy_pair"""
    code, ang, _ = AT.code_turn({'sender_moves': C.tokens_to_moves(tokens), 'goal_tile': goal_tile, 'goal_direction': goal_dir})
    S = SP.SCODE.get(code, 'X')
    if S not in SP.SMENU[int('-' in goal_dir)]:
        S = 'X'
    return S, ang, C.replay([x for x in tokens if x in 'UDLR'])[0]


def code_R(S, ang, rec_dir):
    if S in SP.SIGNAL:
        return {'reads-F': 'F', 'reads-B': 'B', 'reads-S': 'S'}.get(AT.reading(ang, rec_dir), 'S')
    return 'up' if rec_dir == 'up' else 'guess'


def rec_tokens(tile, d):
    """Receiver answer (tile, direction) -> keys, so it can be shown in later rounds' history"""
    k = C.DIRS.index(d)
    st = [{'type': 'go_to', 'tiles': [tile], 'repeats': 1}]
    if k:
        st.append({'type': 'turn_right' if k <= 4 else 'turn_left', 'tiles': [], 'repeats': 1, 'pointing': d})
    return [t if t not in ('RR', 'RL') else t for t in RD.realize({'steps': st, 'end': 'done'})]


def ask_openai(client, model, system, user, schema, usage, lock):
    """OpenAI chat completions with a strict JSON schema; prompt caching is automatic for the shared system prefix"""
    resp = client.chat.completions.create(
        model=model, reasoning_effort='low',
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        response_format={'type': 'json_schema', 'json_schema': {'name': 'answer', 'schema': schema, 'strict': True}})
    u = resp.usage
    cached = getattr(getattr(u, 'prompt_tokens_details', None), 'cached_tokens', 0) or 0
    with lock:
        usage['input_tokens'] = usage.get('input_tokens', 0) + u.prompt_tokens - cached
        usage['cache_read_input_tokens'] = usage.get('cache_read_input_tokens', 0) + cached
        usage['output_tokens'] = usage.get('output_tokens', 0) + u.completion_tokens
    msg = resp.choices[0].message
    if getattr(msg, 'refusal', None) or not msg.content:
        return None
    return json.loads(msg.content)


def run(a):
    games, names = load_players()
    runs = [int(r) for r in a.runs.split(',')] if a.runs else sorted({k[0] for k in games})
    if a.model.startswith('gpt-'):
        import openai
        client = openai.OpenAI(max_retries=6, timeout=60)  # one request hung ~10 min on the default 600 s timeout
    else:
        import anthropic
        client = anthropic.Anthropic(max_retries=6)
    usage, lock = {}, threading.Lock()
    cache_path = os.path.join(OUT, f'fullgame_sonnet_cache_{a.model}_{a.shots}' + ('' if PROMPT_V == 'v1' else f'_{PROMPT_V}') + '.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']

    def ask(key, system, user, schema):
        if key in cache:
            return cache[key]
        out = (ask_openai if a.model.startswith('gpt-') else AF.ask)(client, a.model, system, user, schema, usage, lock)
        with lock:
            cache[key] = out
            with open(cache_path, 'a') as f:
                f.write(json.dumps({'key': key, 'out': out}) + '\n')
        return out

    def sender_tokens(out):
        return RD.realize(out) if out else ['TIMEOUT']

    def receiver_ans(out):
        return (out['tile'], out['direction']) if out else ('center', 'up')

    pt_rows, closed = [], []
    for held in runs:
        system = system_prompt(games, names, held, a.shots)
        mine = {k: v for k, v in games.items() if k[0] == held}
        if a.mode in ('per_turn', 'both'):
            jobs = [(k, i) for k, evl in mine.items() for i in range(len(evl[:a.limit or None]))]

            def one(job):
                k, i = job
                role, ev = mine[k][i]
                out = ask(f"pt:{held}:{ev['round']}:{role}:{k[1]}", system, user_prompt(mine[k], i, names),
                          AF.SND_STEPS_SCHEMA if role == 'sender' else AF.REC_SCHEMA)
                return k, i, out
            res = [one(jobs[0])]
            with ThreadPoolExecutor(a.workers) as ex:
                res += list(ex.map(one, jobs[1:]))
            for k, i, out in res:
                role, ev = mine[k][i]
                S_real, ang_real, end_real = code_S(ev['s_tokens'], ev['goal_tile'], ev['goal_dir'])
                row = {'run': held, 'round': ev['round'], 'phase': ev['phase'], 'role': role, 'player': names[held][k[1]], 'out': out}
                if role == 'sender':
                    toks = sender_tokens(out)
                    S_p, _, end_p = code_S(toks, ev['goal_tile'], ev['goal_dir'])
                    row.update(S_real=S_real, S_pred=S_p, end_goal_real=int(end_real == ev['goal_tile']),
                               end_goal_pred=int(end_p == ev['goal_tile']), timeout_pred=int(toks[-1] == 'TIMEOUT'),
                               timeout_real=int(ev['s_tokens'][-1] == 'TIMEOUT'), len_pred=len(toks), len_real=len(ev['s_tokens']),
                               pred_text=RD.describe(out) if out else None, real_text=moves(ev['s_tokens']))
                else:
                    tile, d = receiver_ans(out)
                    row.update(tile_pred=tile, dir_pred=d, tile_real=ev['rec_tile'], dir_real=ev['rec_dir'],
                               succ_pred=int(tile == ev['goal_tile'] and d == ev['goal_dir']), succ_real=ev['succ'],
                               R_real=code_R(S_real, ang_real, ev['rec_dir']), R_pred=code_R(S_real, ang_real, d),
                               signal=int(S_real in SP.SIGNAL))
                pt_rows.append(row)
            print(f'fold {held} per-turn done ({len(res)} turns)', flush=True)

        if a.mode in ('closed', 'both'):
            byround = defaultdict(dict)
            for k, evl in mine.items():
                for role, ev in evl:
                    if role == 'sender':
                        byround[ev['round']][ev['s']] = ev
            for sim in range(a.sims):
                hist = {k: [] for k in mine}
                out_rows = []
                for rnd in sorted(byround)[:a.limit or None]:
                    evs = list(byround[rnd].values())

                    def send(ev):
                        h = hist[(held, ev['s'])] + [('sender', ev)]
                        return sender_tokens(ask(f"cl:{held}:{sim}:{rnd}:{ev['s']}:S", system, user_prompt(h, len(h) - 1, names), AF.SND_STEPS_SCHEMA))
                    with ThreadPoolExecutor(a.workers) as ex:
                        stoks = list(ex.map(send, evs))
                    sevs = [dict(ev, s_tokens=t) for ev, t in zip(evs, stoks)]

                    def recv(sev):
                        h = hist[(held, sev['r'])] + [('receiver', sev)]
                        return receiver_ans(ask(f"cl:{held}:{sim}:{rnd}:{sev['r']}:R", system, user_prompt(h, len(h) - 1, names), AF.REC_SCHEMA))
                    with ThreadPoolExecutor(a.workers) as ex:
                        ans = list(ex.map(recv, sevs))
                    for ev, sev, (tile, d) in zip(evs, sevs, ans):
                        succ = int(tile == ev['goal_tile'] and d == ev['goal_dir'])
                        done = dict(sev, r_tokens=rec_tokens(tile, d), rec_tile=tile, rec_dir=d, succ=succ)
                        hist[(held, ev['s'])].append(('sender', done))
                        hist[(held, ev['r'])].append(('receiver', done))
                        S, ang, end = code_S(sev['s_tokens'], ev['goal_tile'], ev['goal_dir'])
                        R = code_R(S, ang, d)
                        out_rows.append({'run': held, 'sim': sim, 'round': rnd, 'phase': ev['phase'], 'S': S, 'R': R, 'succ': succ,
                                         'match': int((S, R) in {('WF', 'F'), ('WB', 'B'), ('CD', 'F'), ('WS', 'S')}),
                                         'signal': int(S in SP.SIGNAL), 'end_goal': int(end == ev['goal_tile']),
                                         'tile_ok': int(tile == ev['goal_tile']), 'real_succ': ev['succ'],
                                         'sender_text': moves(sev['s_tokens'])})
                closed += out_rows
                print(f"fold {held} closed sim {sim}: success {sum(o['succ'] for o in out_rows)}/{len(out_rows)} "
                      f"(real {sum(o['real_succ'] for o in out_rows)})", flush=True)

    report(a, pt_rows, closed, usage)


def report(a, pt_rows, closed, usage):
    import numpy as np
    mean = lambda v: float(np.mean(v)) if len(v) else float('nan')
    res = {'model': a.model, 'shots': a.shots}
    tag = f'{a.model}_{a.shots}' + ('' if PROMPT_V == 'v1' else f'_{PROMPT_V}') + (f'_runs{a.runs}' if a.runs else '') + (f'_lim{a.limit}' if a.limit else '')
    if pt_rows:
        snd = [r for r in pt_rows if r['role'] == 'sender']
        rec = [r for r in pt_rows if r['role'] == 'receiver']
        sig = [r for r in rec if r['signal']]
        res['per_turn'] = {
            'sender': {'n': len(snd), 'strategy_acc': mean([r['S_pred'] == r['S_real'] for r in snd]),
                       'end_goal_pred': mean([r['end_goal_pred'] for r in snd]), 'end_goal_real': mean([r['end_goal_real'] for r in snd]),
                       'timeout_pred': mean([r['timeout_pred'] for r in snd]), 'timeout_real': mean([r['timeout_real'] for r in snd]),
                       'mix_pred': {k: mean([r['S_pred'] == k for r in snd]) for k in ['GO', 'WF', 'WB', 'WS', 'CD', 'X']},
                       'mix_real': {k: mean([r['S_real'] == k for r in snd]) for k in ['GO', 'WF', 'WB', 'WS', 'CD', 'X']}},
            'receiver': {'n': len(rec), 'exact': mean([r['tile_pred'] == r['tile_real'] and r['dir_pred'] == r['dir_real'] for r in rec]),
                         'tile': mean([r['tile_pred'] == r['tile_real'] for r in rec]), 'dir': mean([r['dir_pred'] == r['dir_real'] for r in rec]),
                         'reading_acc_when_signal': mean([r['R_pred'] == r['R_real'] for r in sig]),
                         'succ_pred': mean([r['succ_pred'] for r in rec]), 'succ_real': mean([r['succ_real'] for r in rec]),
                         'succ_by_phase': {ph: [mean([r['succ_pred'] for r in rec if r['phase'] == ph]), mean([r['succ_real'] for r in rec if r['phase'] == ph])] for ph in (1, 2, 3)}}}
        with open(os.path.join(OUT, f'fullgame_sonnet_per_turn_{tag}.jsonl'), 'w') as f:
            for r in pt_rows:
                f.write(json.dumps(r) + '\n')
    if closed:
        by = defaultdict(list)
        for o in closed:
            by['s', o['phase']].append(o['succ']); by['r', o['phase']].append(o['real_succ'])
            by['e', o['phase']].append(o['end_goal']); by['t', o['phase']].append(o['tile_ok'])
            if o['signal']:
                by['m', o['phase']].append(o['match'])
        res['closed'] = {ph: {'sim_success': mean(by['s', ph]), 'real_success': mean(by['r', ph]), 'sim_match': mean(by['m', ph]),
                              'signal_rate': len(by['m', ph]) / len(by['s', ph]), 'sender_end_goal': mean(by['e', ph]),
                              'receiver_tile_ok': mean(by['t', ph]), 'n': len(by['s', ph])} for ph in (1, 2, 3) if by['s', ph]}
        mix = defaultdict(lambda: defaultdict(int))
        for o in closed:
            mix[o['phase']]['S:' + o['S']] += 1; mix[o['phase']]['R:' + o['R']] += 1
        res['closed_mix'] = {ph: dict(m) for ph, m in mix.items()}
        with open(os.path.join(OUT, f'fullgame_sonnet_closed_{tag}.jsonl'), 'w') as f:
            for o in closed:
                f.write(json.dumps(o) + '\n')
    if usage:
        res['usage'] = usage
        pr = PRICES.get(a.model, PRICES['claude-sonnet-5-5'])  # $/MTok: input, cache write (5 min), cache read, output
        res['cost'] = (usage.get('input_tokens', 0) * pr[0] + usage.get('cache_creation_input_tokens', 0) * pr[1] +
                       usage.get('cache_read_input_tokens', 0) * pr[2] + usage.get('output_tokens', 0) * pr[3]) / 1e6
    json.dump(res, open(os.path.join(OUT, f'fullgame_sonnet_{tag}_{a.mode}.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--path', default=os.path.expanduser('~/Desktop/sonnet_fullgame_prompt.txt'))
    ap.add_argument('--mode', default='per_turn', choices=['per_turn', 'closed', 'both'])
    ap.add_argument('--model', default='claude-sonnet-5-5')
    ap.add_argument('--shots', type=int, default=5)
    ap.add_argument('--sims', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='smoke: turns per player (per_turn) / rounds (closed)')
    ap.add_argument('--runs', default='', help='comma list of held-out runs (default all)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--prompt', default=PROMPT_V, choices=list(TASK))
    a = ap.parse_args()
    PROMPT_V = a.prompt
    if a.show:
        print(show(a.path))
    else:
        run(a)
