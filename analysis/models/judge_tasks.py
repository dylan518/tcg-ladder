#!/usr/bin/env python3
"""
Two LLM judgment tasks on Sender turns with a known right answer, scored directly (accuracy), leave-one-run-out.

  same     anchor = player X's Sender turn; candidates = another turn by X (different round) vs a decoy turn by a
           different player with EXACTLY the same goal as X's other turn (same phase when possible).
           "Which candidate was played by the same person as the anchor?"
  success  two Sender turns with exactly the same goal, one SUCCESS and one FAIL. "Which one succeeded?"

Shots: 0 (zero-shot) or k solved examples from the OTHER runs in the system prompt (cached).
Usage: python3 analysis/models/judge_tasks.py [--model gpt-6.1-terra] [--shots 0,10] [--n 240]
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
import align_triplets as A  # noqa: E402
import fullgame_sonnet as F  # noqa: E402

OUT = os.path.join(HERE, 'out')
BASE = A.JUDGE.split('\n\nYou will compare')[0]
TASK = {
    'same': ("\n\nTASK: each player has a personal style of signalling. You get an ANCHOR turn and two candidate turns. "
             "One candidate was played by the same person as the anchor (in a different round, usually with a different goal); "
             "the other by someone else. The two candidates always have the same goal, so the goal tells you nothing. "
             "Which candidate was played by the same person as the anchor?"),
    'group': ("\n\nTASK: players played in separate groups (tournaments); within a group, players switch partners and can pick up "
              "each other's signalling habits. You get an ANCHOR turn and two candidate turns, played by two OTHER people. One "
              "candidate's player was in the same group as the anchor's player; the other was in a different group. The two "
              "candidates always have the same goal. Which candidate's player was in the same group as the anchor?"),
    'next': ("\n\nTASK: each player has a personal style of signalling. You get an ANCHOR turn and two candidate turns. One "
             "candidate is the very next (or previous) Sender turn by the same person as the anchor; the other was played by someone "
             "else. The two candidates always have the same goal. Which candidate was played by the same person as the anchor?"),
    'partner_out': ("\n\nTASK: players are paired up and alternate Sender and Receiver, and partners can pick up each other's way of "
                    "signalling. You get an ANCHOR turn and two candidate turns. One candidate was played by the anchor player's "
                    "CURRENT PARTNER (close in time); the other by someone from another group who never met them. The two candidates "
                    "always have the same goal. Which candidate was played by the anchor's partner?"),
    'partner_in': ("\n\nTASK: players are paired up and alternate Sender and Receiver, and partners can pick up each other's way of "
                   "signalling. You get an ANCHOR turn and two candidate turns. One candidate was played by the anchor player's "
                   "CURRENT PARTNER (close in time); the other by another player in the same group who is not their partner right now. "
                   "The two candidates always have the same goal. Which candidate was played by the anchor's partner?"),
    'pairnext': ("\n\nTASK: you see one full round played by a pair (the Sender's moves, where the Receiver put the triangle, "
                 "and the result). In the next round they swap roles. Two candidate Sender turns follow: one is the real next turn "
                 "of this pair (the previous Receiver, now sending); the other comes from a different pair. The two candidates "
                 "always have the same goal. Which candidate is this pair's next turn?"),
    'succ_ctx': ("\n\nTASK: you get two Sender turns with exactly the same goal, each with its RECEIVER's last few rounds as "
                 "Receiver with the same partner (what signal they saw, the goal, where they put the triangle, the result), so you "
                 "can see how that Receiver tends to read signals. In one of the two rounds the Receiver then placed the triangle "
                 "correctly (SUCCESS); in the other the round failed. Which turn succeeded?"),
    'success': ("\n\nTASK: you get two Sender turns with exactly the same goal. In one the Receiver then placed the triangle "
                "correctly (SUCCESS); in the other the round failed. Which turn succeeded?")}
SCHEMA = {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['A', 'B']}}, 'required': ['answer'], 'additionalProperties': False}


def build(T, seed=0):
    rng = random.Random(seed)
    bygoal = defaultdict(list)
    for k, e in T.items():
        bygoal[(e['goal_tile'], e['goal_dir'])].append(k)
    same = []
    for k, e in T.items():
        mine = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['s'] and x != k]
        s = rng.choice(mine)
        g = T[s]
        dec = [x for x in bygoal[(g['goal_tile'], g['goal_dir'])] if not (T[x]['run'] == e['run'] and T[x]['s'] == e['s'])]
        dec.sort(key=lambda x: (T[x]['phase'] != g['phase'], rng.random()))
        same.append({'task': 'same', 'anchor': k, 'right': s, 'wrong': dec[0], 'swap': rng.random() < 0.5, 'run': e['run'], 'phase': e['phase']})
    succ = []
    for g, ks in bygoal.items():
        w = [x for x in ks if T[x]['succ']]; l = [x for x in ks if not T[x]['succ']]
        pairs = [(a, b) for a in w for b in l]
        rng.shuffle(pairs)
        for a, b in pairs[:max(1, len(pairs) // 2)]:
            succ.append({'task': 'success', 'anchor': None, 'right': a, 'wrong': b, 'swap': rng.random() < 0.5,
                         'run': T[a]['run'], 'phase': T[a]['phase'], 'other_run': T[b]['run']})
    group = []
    for k, e in T.items():
        others = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] != e['s']]
        s_ = rng.choice(others)
        g = T[s_]
        dec = [x for x in bygoal[(g['goal_tile'], g['goal_dir'])] if T[x]['run'] != e['run']]
        dec.sort(key=lambda x: (T[x]['phase'] != g['phase'], rng.random()))
        group.append({'task': 'group', 'anchor': k, 'right': s_, 'wrong': dec[0], 'swap': rng.random() < 0.5, 'run': e['run'], 'phase': e['phase']})
    def ctx_round(x):
        e = T[x]
        return (f"goal: {e['goal_tile']}, pointing {e['goal_dir']}\nSender moves: {F.moves(e['s_tokens'])}\n"
                f"Receiver put the triangle on {e['rec_tile']} pointing {e['rec_dir']}: {'SUCCESS' if e['succ'] else 'FAIL'}")
    extra = defaultdict(list)
    for k, e in T.items():
        mine = sorted([x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['s'] and x != k], key=lambda x: T[x]['round'])
        after = [x for x in mine if T[x]['round'] > e['round']]
        nxt = after[0] if after else mine[-1]
        g = T[nxt]
        dec = [x for x in bygoal[(g['goal_tile'], g['goal_dir'])] if not (T[x]['run'] == e['run'] and T[x]['s'] == e['s'])]
        dec.sort(key=lambda x: (T[x]['phase'] != g['phase'], rng.random()))
        extra['next'].append({'task': 'next', 'anchor': k, 'right': nxt, 'wrong': dec[0], 'swap': rng.random() < 0.5, 'run': e['run'], 'phase': e['phase']})
        bt = [x for x in T if T[x]['run'] == e['run'] and T[x]['s'] == e['r'] and T[x]['r'] == e['s']]  # partner sending to X
        if bt:
            p_ = min(bt, key=lambda x: abs(T[x]['round'] - e['round']))
            g = T[p_]
            same_goal = bygoal[(g['goal_tile'], g['goal_dir'])]
            out_ = [x for x in same_goal if T[x]['run'] != e['run']]
            out_.sort(key=lambda x: (T[x]['phase'] != g['phase'], rng.random()))
            extra['partner_out'].append({'task': 'partner_out', 'anchor': k, 'right': p_, 'wrong': out_[0], 'swap': rng.random() < 0.5, 'run': e['run'], 'phase': e['phase']})
            in_ = [x for x in same_goal if T[x]['run'] == e['run'] and T[x]['s'] not in (e['s'], e['r'])]
            if in_:
                extra['partner_in'].append({'task': 'partner_in', 'anchor': k, 'right': p_, 'wrong': rng.choice(in_), 'swap': rng.random() < 0.5, 'run': e['run'], 'phase': e['phase']})
        nb = [x for x in bt if T[x]['round'] == e['round'] + 1]  # partner sends in the very next round
        if nb:
            g = T[nb[0]]
            dec = [x for x in bygoal[(g['goal_tile'], g['goal_dir'])] if T[x]['run'] != e['run']]
            dec.sort(key=lambda x: (T[x]['phase'] != g['phase'], rng.random()))
            extra['pairnext'].append({'task': 'pairnext', 'anchor': k, 'right': nb[0], 'wrong': dec[0], 'swap': rng.random() < 0.5, 'run': e['run'], 'phase': e['phase'],
                                      'ctx': ctx_round(k)})
    def rec_hist(x, n=3):
        e = T[x]
        prev = sorted([y for y in T if T[y]['run'] == e['run'] and T[y]['r'] == e['r'] and T[y]['s'] == e['s'] and T[y]['round'] < e['round']],
                      key=lambda y: T[y]['round'])[-n:]
        return '\n'.join(f"  [earlier round] {ctx_round(y).replace(chr(10), ' | ')}" for y in prev) or '  (no earlier rounds with this partner)'
    for q in succ:
        extra['succ_ctx'].append({**q, 'task': 'succ_ctx', 'hist': {q['right']: rec_hist(q['right']), q['wrong']: rec_hist(q['wrong'])}})
    return same, succ, group, extra


def render(T, q, with_answer=False):
    a, b = (q['wrong'], q['right']) if q['swap'] else (q['right'], q['wrong'])
    if q['task'] == 'pairnext':
        s = f"PREVIOUS ROUND of this pair:\n{q['ctx']}\n\nCandidate A:\n{A.item(T[a])}\n\nCandidate B:\n{A.item(T[b])}"
        return s + (f"\nAnswer: {'B' if q['swap'] else 'A'}" if with_answer else '')
    if q['task'] == 'succ_ctx':
        s = (f"Candidate A:\n{A.item(T[a])}\nits Receiver's earlier rounds:\n{q['hist'][a]}\n\n"
             f"Candidate B:\n{A.item(T[b])}\nits Receiver's earlier rounds:\n{q['hist'][b]}")
        return s + (f"\nAnswer: {'B' if q['swap'] else 'A'}" if with_answer else '')
    s = (f"ANCHOR turn:\n{A.item(T[q['anchor']])}\n\n" if q['task'] == 'same' else '') + \
        (f"ANCHOR turn:\n{A.item(T[q['anchor']])}\n\n" if q['task'] in ('group', 'next', 'partner_out', 'partner_in') else '') + \
        f"Candidate A:\n{A.item(T[a])}\n\nCandidate B:\n{A.item(T[b])}"
    if with_answer:
        s += f"\nAnswer: {'B' if q['swap'] else 'A'}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-6.1-terra')
    ap.add_argument('--shots', default='0,10')
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--tasks', default='same,success,group')
    a = ap.parse_args()
    import openai
    client = openai.OpenAI(max_retries=6, timeout=60)
    usage, lock = {}, threading.Lock()
    T, _, _ = A.build()
    same, succ, group, extra = build(T)
    tasks = {'same': same, 'success': succ, 'group': group, **extra}
    cache_path = os.path.join(OUT, f'judge_tasks_cache_{a.model}.jsonl')
    cache = {}
    if os.path.exists(cache_path):
        for l in open(cache_path):
            r = json.loads(l); cache[r['key']] = r['out']

    def system(task, held, k, query=None):
        s = BASE + TASK[task]
        if k:
            players = lambda q: {(T[x]['run'], T[x]['s']) for x in (q['anchor'], q['right'], q['wrong']) if x}
            runs = lambda q: {T[x]['run'] for x in (q['anchor'], q['right'], q['wrong']) if x}
            if task in extra:  # examples from runs not involved in the query -> same examples per run set (prompt-cacheable)
                pool = [q for q in tasks[task] if not runs(query) & runs(q)]
                ex = random.Random(f"{task}:{sorted(runs(query))}:{k}").sample(pool, min(k, len(pool)))
                return s + '\n\nSOLVED EXAMPLES (other groups of players):\n\n' + '\n\n---\n\n'.join(render(T, q, True) for q in ex)
            elif task == 'same':
                pool = [q for q in same if q['run'] != held]
            elif task == 'group':  # no player in common with the query
                pl = {(T[x]['run'], T[x]['s']) for x in (query['anchor'], query['right'], query['wrong'])}
                pool = [q for q in group if not pl & {(T[x]['run'], T[x]['s']) for x in (q['anchor'], q['right'], q['wrong'])}]
            else:  # no player in common with the query pair
                pl = {(T[x]['run'], T[x]['s']) for x in (query['right'], query['wrong'])}
                pool = [q for q in succ if not pl & {(T[x]['run'], T[x]['s']) for x in (q['right'], q['wrong'])}]
            ex = random.Random(f"{held}:{k}:{query['right'] if task != 'same' else ''}").sample(pool, min(k, len(pool)))
            s += '\n\nSOLVED EXAMPLES (other groups of players):\n\n' + '\n\n---\n\n'.join(render(T, q, True) for q in ex)
        return s

    res = {}
    rows = []
    for k in [int(x) for x in a.shots.split(',')]:
        for task in a.tasks.split(','):
            qs = tasks[task]
            jobs = []
            for q in qs:
                held = q['run']
                jobs.append(q)

            def one(q):
                key = f"{task}:{k}:{q['anchor']}:{q['right']}:{q['wrong']}:{q['swap']}"
                if key not in cache:
                    out = F.ask_openai(client, a.model, system(task, q['run'], k, q), render(T, q) + '\n\nAnswer A or B.', SCHEMA, usage, lock)
                    with lock:
                        cache[key] = out
                        with open(cache_path, 'a') as f:
                            f.write(json.dumps({'key': key, 'out': out}) + '\n')
                out = cache[key]
                return None if not out else int(out['answer'] == ('B' if q['swap'] else 'A'))
            with ThreadPoolExecutor(a.workers) as ex:
                got = list(ex.map(one, jobs))
            ok = [(q, g) for q, g in zip(jobs, got) if g is not None]
            acc = sum(g for _, g in ok) / len(ok)
            byph = {ph: sum(g for q, g in ok if q['phase'] == ph) / max(1, sum(1 for q, _ in ok if q['phase'] == ph)) for ph in (1, 2, 3)}
            byrun = {r: sum(g for q, g in ok if q['run'] == r) / max(1, sum(1 for q, _ in ok if q['run'] == r)) for r in (1, 2, 3)}
            res[f'{task}_{k}shot'] = {'acc': acc, 'n': len(ok), 'by_phase': byph, 'by_run': byrun}
            rows += [{'task': task, 'shots': k, **{x: q[x] for x in ('anchor', 'right', 'wrong', 'run', 'phase')}, 'correct': g} for q, g in ok]
            print(f'{task:8s} {k:2d}-shot: {acc:.0%} correct (n={len(ok)}, chance 50%) | by phase ' +
                  ' '.join(f'{p}:{v:.0%}' for p, v in byph.items()) + ' | by run ' + ' '.join(f'{r}:{v:.0%}' for r, v in byrun.items()), flush=True)
    if usage:
        res['cost'] = (usage.get('input_tokens', 0) * 1 + usage.get('cache_read_input_tokens', 0) * 0.05 + usage.get('output_tokens', 0) * 5) / 1e6
        print(f"~${res['cost']:.2f}")
    json.dump(res, open(os.path.join(OUT, f'judge_tasks_{a.model}_{a.tasks.replace(",", "-")}.json'), 'w'), indent=1)
    with open(os.path.join(OUT, f'judge_tasks_rows_{a.model}_{a.tasks.replace(",", "-")}.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
