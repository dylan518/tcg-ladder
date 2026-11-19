"""
Whole-game, strategy-level player transcripts: each player's full session (all rounds, all partners) as one line per
round, written at the level of strategies (what the signal meant), so a model sees conventions carry across partners.

  R12 | phase 2 | partner P4 (game 3 with them) | you SEND | goal: top-right, pointing up-right | you sent: wiggle toward the goal direction | partner read it: toward | SUCCESS
  R13 | phase 2 | partner P4 (game 4 with them) | you RECEIVE | partner signal: wiggle on top, arm pointing left | you read it: away | FAIL

Decisions to predict: the Sender's "you sent: <strategy>" and the Receiver's "you read it: <reading>".
Success then follows from the pair (strategy_pair outcome table, e.g. toward+toward 97%, any mismatch 0%).
Players are anonymised P1..P6 per run. Receivers see only SUCCESS/FAIL (never the goal), as in the game.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import analyze_tournament as AT  # noqa: E402
import strategy_pair as SP  # noqa: E402

S_TEXT = {'WF': 'wiggle toward the goal direction', 'WB': 'wiggle away from the goal direction', 'WS': 'wiggle sideways to the goal direction',
          'CD': 'mark the goal direction from the centre', 'GO': 'just go to the goal tile', 'X': 'something else'}
R_TEXT = {'F': 'toward', 'B': 'away', 'S': 'sideways', 'up': 'left it pointing up', 'guess': 'turned it somewhere'}
DIRS = C.DIRS

INTRO = ("Communication game on a 3x3 board, played by real people. Each round one player is the Sender (sees a goal: a tile "
         "and a direction for the Receiver's triangle) and signals only by moving a circle; the Receiver then places and turns "
         "the triangle. Players switch partners each phase. Below is one player's whole game, one line per round, written as "
         "strategies.\nSender strategies: " + '; '.join(S_TEXT.values()) + ".\nReceiver readings of a wiggle or mark: toward "
         "(point the triangle the way the arm/mark points), away (the opposite way), sideways; with no clear signal: "
         "left it pointing up, turned it somewhere.\n\n")


def dir_name(ang):
    return DIRS[int(round(ang / 45)) % 8]


def load_players():
    """-> {(run, player): [event dicts in round order]} with both roles; event has 'line_prefix', 'target', 'line_suffix'"""
    T = C.load_turns()
    rec = {(t['run'], t['round'], t['player']): t for t in T if t['role'] == 'receiver'}
    rounds = {(x['run'], x['round'], x['s']): x for x in SP.load()}
    names = {}
    for t in sorted(T, key=lambda t: (t['run'], t['player'])):
        names.setdefault(t['run'], {})
        names[t['run']].setdefault(t['player'], f'P{len(names[t["run"]]) + 1}')
    events = {}
    for t in sorted(T, key=lambda t: (t['run'], t['round'])):
        if t['role'] != 'sender':
            continue
        x = rounds[(t['run'], t['round'], t['player'])]
        code, ang, _ = AT.code_turn({'sender_moves': C.tokens_to_moves(t['tokens']), 'goal_tile': t['goal_tile'], 'goal_direction': t['goal_direction']})
        end = C.replay([k for k in t['tokens'] if k in 'UDLR'])[0]
        ev = dict(run=t['run'], round=t['round'], phase=t['phase'], g=x['g'], S=x['S'], R=x['R'], succ=x['succ'], tile_end=x['tile_end'],
                  goal_tile=t['goal_tile'], goal_dir=t['goal_direction'], end=end, sig_dir=dir_name(ang) if ang is not None else None,
                  pg=t['partner_games'], s=t['player'], r=t['partner'])
        events.setdefault((t['run'], t['player']), []).append(('sender', ev))
        events.setdefault((t['run'], t['partner']), []).append(('receiver', ev))
    return events, names


def signal_text(S, end, sig_dir):
    if S in ('WF', 'WB', 'WS'):
        return f'wiggle on {end}, arm pointing {sig_dir}'
    if S == 'CD':
        return f'marked the {sig_dir} side from the centre, ended on {end}'
    if S == 'GO':
        return f'just went to {end}'
    return f'unclear moves, ended on {end}'


def render(role, ev, names, S=None, R=None, succ=None):
    """(prefix, target, suffix) for one round from this player's view; S/R/succ override the real ones (simulation)"""
    S = ev['S'] if S is None else S
    R = ev['R'] if R is None else R
    succ = ev['succ'] if succ is None else succ
    nm = names[ev['run']]
    me, other = (ev['s'], ev['r']) if role == 'sender' else (ev['r'], ev['s'])
    head = f"R{ev['round']} | phase {ev['phase']} | partner {nm[other]} (game {ev['pg'] + 1} with them" + (', new partner' if ev['pg'] == 0 else '') + ') | '
    res = ' | ' + ('SUCCESS' if succ else 'FAIL') + '\n'
    if role == 'sender':
        return head + f"you SEND | goal: {ev['goal_tile']}, pointing {ev['goal_dir']} | you sent: ", S_TEXT[S], f" | partner read it: {R_TEXT[R]}" + res
    return head + f"you RECEIVE | partner signal: {signal_text(S, ev['end'], ev['sig_dir'])} | you read it: ", R_TEXT[R], res


def options(role, ev, S=None):
    if role == 'sender':
        return SP.SMENU[ev['g']]
    S = ev['S'] if S is None else S
    return SP.RMENU[int(S in SP.SIGNAL)]


def option_text(role, k):
    return S_TEXT[k] if role == 'sender' else R_TEXT[k]


def full_text(evlist, names):
    return INTRO + ''.join(''.join(render(role, ev, names)) for role, ev in evlist)


def sim_signal(S, ev, rng):
    """signal the Receiver sees for a simulated Sender strategy (Sender assumed to end on the goal tile)"""
    gi = DIRS.index(ev['goal_dir'])
    if S in ('WF', 'CD'):
        d = DIRS[gi]
    elif S == 'WB':
        d = DIRS[(gi + 4) % 8]
    elif S == 'WS':
        d = DIRS[(gi + rng.choice([-2, -1, 1, 2])) % 8]
    else:
        d = None
    return dict(ev, S=S, end=ev['goal_tile'], sig_dir=d)


def match(S, R):
    return (S, R) in {('WF', 'F'), ('WB', 'B'), ('CD', 'F'), ('WS', 'S')}


def softmax_logp(lps):
    m = max(lps)
    e = [math.exp(v - m) for v in lps]
    z = sum(e)
    return [v / z for v in e]
