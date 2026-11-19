"""
Shared pieces for every predictor of the tokenized games (API few-shot LLM, SFT LLM, Decision Transformer, n-gram):
data loading, leave-one-run-out folds, text renderings of a turn, replaying tokens on the board, and the metrics.

Tasks
  receiver  Given what the Sender did (tokens) and this pair's history so far, where does the Receiver put the
            triangle (tile + direction)? Scored against what the HUMAN did (behaviour cloning, mistakes included):
            exact match (tile and direction), tile match, direction match; plus how often the prediction would
            have succeeded vs how often the human did.
  sender    Given the goal (and the pair's history), produce a Sender turn (tokens). Scored as a distribution
            against the held-out humans: strategy mix (same coding as analyze_tournament.py), mean length,
            TIMEOUT rate, share ending on the goal tile, wiggle count.
  nexttok   (models with probabilities) bits/token and accuracy, as in baseline_ngram.py.
"""
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # analysis/
import analyze_tournament as AT  # noqa: E402  (strategy coding)

TOKENS_PATH = os.path.join(os.path.dirname(HERE), 'tokens', 'turns.jsonl')
DIRS = ['up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left']
XY_TILE = {(-1, 1): 'top-left', (0, 1): 'top', (1, 1): 'top-right', (-1, 0): 'left', (0, 0): 'center',
           (1, 0): 'right', (-1, -1): 'bottom-left', (0, -1): 'bottom', (1, -1): 'bottom-right'}
TILES = list(XY_TILE.values())
VOCAB = ['U', 'D', 'L', 'R', 'RL', 'RR', 'W1', 'W2', 'W3', 'END', 'TIMEOUT']
WAIT_MS = {'W1': 700, 'W2': 1500, 'W3': 2600}


def load_turns(path=TOKENS_PATH):
    return [json.loads(l) for l in open(path)]


def folds(turns):
    """yield (held_out_run, train_turns, test_turns)"""
    for r in sorted({t['run'] for t in turns}):
        yield r, [t for t in turns if t['run'] != r], [t for t in turns if t['run'] == r]


def replay(tokens):
    """final (tile, direction) of a key that starts in the centre pointing up"""
    x = y = 0
    a = 0
    for t in tokens:
        if t == 'U': y = min(1, y + 1)
        elif t == 'D': y = max(-1, y - 1)
        elif t == 'L': x = max(-1, x - 1)
        elif t == 'R': x = min(1, x + 1)
        elif t == 'RL': a = (a - 1) % 8
        elif t == 'RR': a = (a + 1) % 8
    return XY_TILE[(x, y)], DIRS[a]


def tokens_to_moves(tokens):
    """rebuild a rounds.csv-style sender_moves string (synthetic times) so the strategy coder can run on it"""
    x = y = 0
    t = 0
    out = ['start(0,0)@0']
    pending = 250
    for tok in tokens:
        if tok in WAIT_MS:
            pending = WAIT_MS[tok]
            continue
        t += pending
        pending = 250
        if tok in ('U', 'D', 'L', 'R'):
            dx, dy = {'U': (0, 1), 'D': (0, -1), 'L': (-1, 0), 'R': (1, 0)}[tok]
            x, y = max(-1, min(1, x + dx)), max(-1, min(1, y + dy))
            out.append(f"{ {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}[tok] }({x},{y})@{t}")
        elif tok == 'END':
            out.append(f'stop({x},{y})@{t}')
        elif tok == 'TIMEOUT':
            out.append(f'timeout({x},{y})@{t}')
    return ' '.join(out)


def sender_strategy(tokens, goal_tile, goal_dir):
    code, _, _ = AT.code_turn({'sender_moves': tokens_to_moves(tokens), 'goal_tile': goal_tile, 'goal_direction': goal_dir})
    return code


# ---------------------------------------------------------------- text renderings (shared by prompts and SFT data)

def raw(tokens):
    return ' '.join(tokens)


def pair_history(turn, all_turns, role, max_items=6, render=raw):
    """earlier rounds of the same pair in the same run, as short lines from this player's point of view"""
    me, partner = turn['player'], turn['partner']
    prev = [t for t in all_turns if t['run'] == turn['run'] and t['round'] < turn['round'] and t['role'] == 'sender'
            and {t['player'], t['partner']} == {me, partner}]
    lines = []
    for t in prev[-max_items:]:
        who = 'you sent' if t['player'] == me else 'partner sent'
        lines.append(f"round {t['round']}: {who}: {render(t['tokens'])} -> receiver placed {t['receiver_tile']}/{t['receiver_direction']} "
                     f"({'success' if t['success'] else 'fail'})" + (f" [goal was {t['goal_tile']}/{t['goal_direction']}]" if t['player'] == me else ''))
    return lines


def receiver_query(turn, all_turns, render=raw):
    h = pair_history(turn, all_turns, 'receiver', render=render)
    return (f"Round {turn['round']} (phase {turn['phase']}, {turn['partner_games']} games already played with this partner).\n"
            + ("History with this partner:\n" + '\n'.join(h) + '\n' if h else 'First game with this partner.\n')
            + f"The Sender just did: {render(turn['saw_sender_tokens'])}")


def sender_query(turn, all_turns, render=raw):
    h = pair_history(turn, all_turns, 'sender', render=render)
    return (f"Round {turn['round']} (phase {turn['phase']}, {turn['partner_games']} games already played with this partner).\n"
            + ("History with this partner:\n" + '\n'.join(h) + '\n' if h else 'First game with this partner.\n')
            + f"Goal: tile {turn['goal_tile']}, triangle pointing {turn['goal_direction']}.")


# ---------------------------------------------------------------- metrics

def receiver_scores(rows):
    """rows: [{'turn': human turn, 'pred_tile', 'pred_dir'}]"""
    n = len(rows)
    tile = sum(r['pred_tile'] == r['turn']['receiver_tile'] for r in rows)
    dirn = sum(r['pred_dir'] == r['turn']['receiver_direction'] for r in rows)
    both = sum(r['pred_tile'] == r['turn']['receiver_tile'] and r['pred_dir'] == r['turn']['receiver_direction'] for r in rows)
    pred_succ = sum(r['pred_tile'] == r['turn']['goal_tile'] and r['pred_dir'] == r['turn']['goal_direction'] for r in rows)
    human_succ = sum(r['turn']['success'] for r in rows)
    return {'n': n, 'match_both': both / n, 'match_tile': tile / n, 'match_dir': dirn / n,
            'pred_success': pred_succ / n, 'human_success': human_succ / n}


def sender_profile(seqs):
    """seqs: [(tokens, goal_tile, goal_dir)] -> distribution summary"""
    codes = Counter(sender_strategy(t, g, d) for t, g, d in seqs)
    n = len(seqs)
    ends_goal = sum(replay([x for x in t if x in 'UDLR'])[0] == g for t, g, _ in seqs)
    return {'n': n, 'mean_len': sum(len(t) for t, _, _ in seqs) / n,
            'timeout_rate': sum(bool(t) and t[-1] == 'TIMEOUT' for t, _, _ in seqs) / n,
            'ends_on_goal': ends_goal / n,
            'strategy': {k: v / n for k, v in sorted(codes.items())}}


def js_divergence(p, q):
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}
    def kl(a):
        return sum(a.get(k, 0) * math.log2(a.get(k, 0) / m[k]) for k in keys if a.get(k, 0) > 0)
    return (kl(p) + kl(q)) / 2


def parse_tokens(text):
    toks = [t for t in text.replace(',', ' ').split() if t in VOCAB]
    if not toks or toks[-1] not in ('END', 'TIMEOUT'):
        toks.append('END')
    return toks
