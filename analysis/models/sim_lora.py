#!/usr/bin/env python3
"""
User simulator: LoRA fine-tune of a small LLM on per-action player transcripts (actions.py), trained to predict every
one of the player's own actions (time since previous action + key) across their whole game. Leave-one-run-out.

Usage (GPU): python3 analysis/models/sim_lora.py [--model Qwen/Qwen2.5-1.5B] [--epochs 20] [--sims 1] [--out DIR]

Generation is one action at a time with the game rules applied by us: the model writes "<dt> <A>", we compute the new
state (tile / pointing), enforce the turn's time limit (the turn becomes TIMEOUT when the clock runs out) and stop at
END. Senders cannot rotate (resampled).

Evaluation per held-out run:
  open loop    every turn generated from the REAL history before it; Receiver placement vs the human, Sender
               strategy mix / ends-on-goal, timing (first-action latency, median gap, timeout rate, actions per turn)
  closed loop  all players of the held-out run replaced by simulated users playing its real schedule and goals; the
               simulated turns feed each other's histories. Compared with the real run: success overall and by phase,
               strategy mix, timing. Synthetic sessions are written as JSONL for the usual analyses.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import actions as AC  # noqa: E402
import common as C  # noqa: E402

INTRO = ("Transcript of one player in a two-player communication game on a 3x3 board. The Sender sees a goal (tile + one of "
         "8 directions) and can only move a circle; the Receiver then moves and rotates a triangle (starts in the centre "
         "pointing up). Action lines: seconds since the player's previous action, then U/D/L/R (move; tile after it), "
         "RL/RR (rotate 45 degrees; direction after it), END (pressed Space) or TIMEOUT.\n")
MOVES = {'U': (0, 1), 'D': (0, -1), 'L': (-1, 0), 'R': (1, 0)}
ACT_RE = re.compile(r'^\s*\|?\s*(\d+(?:\.\d)?)\s+(RL|RR|U|D|L|R|END|TIMEOUT)(?=[\s|]|$)')


def load(path):
    return [json.loads(l) for l in open(path)]


def transcript_text(p):
    return INTRO + ''.join(AC.render_round(rd) for rd in p['rounds'])


def own_spans(text):
    """char spans of the player's own action lines (after 'you: ' up to end of line, newline included)"""
    return [(m.start(1), m.end(1) + 1) for m in re.finditer(r'^you: (.*)$', text, flags=re.M)]


def mirror_text(body):
    """left-right mirror of rendered round blocks: tile/direction words, L<->R moves, RL<->RR rotations"""
    body = re.sub(r'\b(left|right)\b', lambda m: 'right' if m.group(1) == 'left' else 'left', body)
    return re.sub(r'(?<=\d )(RL|RR|L|R)(?=[ \n|]|$)', lambda m: {'L': 'R', 'R': 'L', 'RL': 'RR', 'RR': 'RL'}[m.group(1)], body, flags=re.M)


def encode_train(tok, p, max_len, mirror=False):
    text = transcript_text(p)
    if mirror:
        text = INTRO + mirror_text(text[len(INTRO):])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    spans = own_spans(text)
    lab = [-100] * len(ids)
    j = 0
    for i, (a, b) in enumerate(offs):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and spans[j][0] <= a < spans[j][1]:
            lab[i] = ids[i]
    return ids[:max_len], lab[:max_len]


def to_tokens(acts):
    """[(dt, A, state)] -> game tokens with wait bins, for the strategy coder / replay"""
    out = []
    for dt, A, _ in acts:
        if dt >= 2.0: out.append('W3')
        elif dt >= 1.0: out.append('W2')
        elif dt >= 0.4: out.append('W1')
        out.append(A)
    return out


class Gen:
    """incremental generation with a KV cache; crop() rolls back to a saved length"""

    def __init__(self, model, tok, device):
        self.m, self.tok, self.dev = model, tok, device

    @torch.no_grad()
    def start(self, text):
        ids = self.tok(text, add_special_tokens=False)['input_ids']
        out = self.m(input_ids=torch.tensor([ids], device=self.dev), use_cache=True)
        self.cache, self.logits, self.len = out.past_key_values, out.logits[0, -1].float(), len(ids)

    @torch.no_grad()
    def feed_ids(self, ids):
        out = self.m(input_ids=torch.tensor([ids], device=self.dev), past_key_values=self.cache, use_cache=True)
        self.cache, self.logits, self.len = out.past_key_values, out.logits[0, -1].float(), self.len + len(ids)

    def feed(self, text):
        self.feed_ids(self.tok(text, add_special_tokens=False)['input_ids'])

    def crop(self, n):
        self.cache.crop(n)
        self.len = n

    def sample_tok(self, temp=1.0):
        return int(torch.multinomial(torch.softmax(self.logits / temp, -1), 1))


def gen_turn(g, role, limit_s, rng_seed=None, max_actions=60):
    """generate the player's own action line after 'you:' -> [(dt, A, state)] (rules applied)"""
    x = y = 0
    ang = 0
    elapsed = 0.0
    acts = []
    for _ in range(max_actions):
        base, logits0 = g.len, g.logits.clone()
        parsed = None
        for attempt in range(4):
            g.crop(base); g.logits = logits0.clone()
            text = ''
            for _k in range(12):
                t = g.sample_tok()
                text += g.tok.decode([t])
                g.feed_ids([t])
                m = ACT_RE.match(text)
                if m and len(text) > m.end():
                    break
                if '\n' in text:
                    break
            m = ACT_RE.match(text)
            if m and not (role == 'sender' and m.group(2) in ('RL', 'RR')):
                parsed = (float(m.group(1)), m.group(2))
                break
        g.crop(base); g.logits = logits0
        if parsed is None:
            parsed = (0.5, 'END')
        dt, A = parsed
        if A == 'TIMEOUT' or elapsed + dt >= limit_s:
            acts.append((round(max(0.0, limit_s - elapsed), 1), 'TIMEOUT', ''))
            break
        elapsed += dt
        if A in MOVES:
            x, y = max(-1, min(1, x + MOVES[A][0])), max(-1, min(1, y + MOVES[A][1]))
            state = AC.TILE[(x, y)]
        elif A in ('RL', 'RR'):
            ang = (ang + (1 if A == 'RR' else -1)) % 8
            state = 'pointing ' + AC.DIRS[ang]
        else:
            state = ''
        acts.append((dt, A, state))
        g.feed((' | ' if len(acts) > 1 else ' ') + f'{dt:.1f} {A}' + (f' {state}' if state else ''))
        if A == 'END':
            break
    g.feed('\n')
    return acts


def final_placement(acts):
    tile, d = 'center', 'up'
    for _, A, s in acts:
        if A in MOVES: tile = s
        elif A in ('RL', 'RR'): d = s.replace('pointing ', '')
    return tile, d


def timing(turns):
    """turns: list of action lists -> timing summary"""
    first = [a[0][0] for a in turns if a]
    gaps = [dt for a in turns for dt, A, _ in a[1:] if A not in ('TIMEOUT',)]
    med = lambda v: sorted(v)[len(v) // 2] if v else float('nan')
    return {'first_action_s': med(first), 'median_gap_s': med(gaps), 'timeout_rate': sum(bool(a) and a[-1][1] == 'TIMEOUT' for a in turns) / max(1, len(turns)),
            'actions_per_turn': sum(len(a) for a in turns) / max(1, len(turns))}


def round_block_prefix(rd, partner_acts=None):
    """text of a round up to and including 'you:' (Receivers see the partner's line first)"""
    if rd['role'] == 'sender':
        return f"{rd['header']}\nyou:"
    return f"{rd['header']}\npartner: {AC.fmt_actions(partner_acts if partner_acts is not None else rd['partner_actions'])}\nyou:"


def train(model_name, train_players, args, seed):
    tok = AutoTokenizer.from_pretrained(model_name)
    torch.manual_seed(seed); rng = random.Random(seed)
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to('cuda')
    base.gradient_checkpointing_enable(); base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
                                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    data = [encode_train(tok, p, args.max_len) for p in train_players]
    if args.mirror:
        data += [encode_train(tok, p, args.max_len, mirror=True) for p in train_players]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = args.epochs * len(data)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * max(0.05, 1 - s / steps))
    model.train(); step = 0; t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(data); tot = 0
        for ids, lab in data:
            loss = model(input_ids=torch.tensor([ids], device='cuda'), labels=torch.tensor([lab], device='cuda')).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); step += 1
        print(f'  epoch {ep} loss {tot / len(data):.3f} ({time.time() - t0:.0f}s)', flush=True)
    model.eval(); base.gradient_checkpointing_disable(); base.config.use_cache = True
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'transcripts.jsonl'))
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--mirror', action='store_true', help='add left-right mirrored copies of training transcripts')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--max_len', type=int, default=12000)
    ap.add_argument('--sims', type=int, default=1, help='closed-loop simulated sessions per held-out run')
    ap.add_argument('--limit_players', type=int, default=0, help='smoke test: open-loop players per fold')
    ap.add_argument('--limit_rounds', type=int, default=0, help='smoke test: rounds per player / per simulation')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'sim'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    players = load(a.data)
    runs = sorted({p['run'] for p in players})
    rec_rows, snd_model, snd_human, t_model, t_human = [], [], [], [], []
    closed = []
    for held in runs:
        train_p = [p for p in players if p['run'] != held]
        test_p = [p for p in players if p['run'] == held]
        print(f'=== fold {held}: train {len(train_p)} players, test {len(test_p)}', flush=True)
        model, tok = train(a.model, train_p, a, seed=held)
        g = Gen(model, tok, 'cuda')
        # ---- open loop
        for p in (test_p[:a.limit_players] if a.limit_players else test_p):
            rounds = p['rounds'][:a.limit_rounds] if a.limit_rounds else p['rounds']
            for i, rd in enumerate(rounds):
                prefix = INTRO + ''.join(AC.render_round(r) for r in p['rounds'][:i]) + round_block_prefix(rd)
                g.start(prefix)
                acts = gen_turn(g, rd['role'], rd['limit_s'])
                t_model.append(acts); t_human.append(rd['you'])
                toks = to_tokens(acts)
                if rd['role'] == 'receiver':
                    tile, d = final_placement(acts)
                    rec_rows.append({'turn': {'receiver_tile': rd['receiver_tile'], 'receiver_direction': rd['receiver_direction'],
                                              'goal_tile': rd['goal_tile'], 'goal_direction': rd['goal_direction'], 'success': rd['success']},
                                     'pred_tile': tile, 'pred_dir': d})
                else:
                    snd_model.append((toks, rd['goal_tile'], rd['goal_direction']))
                    snd_human.append((to_tokens(rd['you']), rd['goal_tile'], rd['goal_direction']))
            print(f'  open loop {p["player"]} done', flush=True)
        # ---- closed loop: the whole held-out session played by simulated users
        by_round = {}
        for p in test_p:
            for rd in p['rounds']:
                by_round.setdefault(rd['round'], []).append((p['player'], rd))
        for sim in range(a.sims):
            hist = {p['player']: [] for p in test_p}
            out_rounds = []
            for r in sorted(by_round)[: (a.limit_rounds or None)]:
                senders = [(pl, rd) for pl, rd in by_round[r] if rd['role'] == 'sender']
                for spl, srd in senders:
                    rpl, rrd = next((pl, rd) for pl, rd in by_round[r] if rd['role'] == 'receiver' and pl == srd['partner'])
                    g.start(INTRO + ''.join(hist[spl]) + round_block_prefix(srd))
                    s_acts = gen_turn(g, 'sender', srd['limit_s'])
                    g.start(INTRO + ''.join(hist[rpl]) + round_block_prefix(rrd, partner_acts=s_acts))
                    r_acts = gen_turn(g, 'receiver', rrd['limit_s'])
                    tile, d = final_placement(r_acts)
                    ok = int(tile == srd['goal_tile'] and d == srd['goal_direction'])
                    hist[spl].append(f"{srd['header']}\nyou: {AC.fmt_actions(s_acts)}\npartner: {AC.fmt_actions(r_acts)}\n"
                                     f"result: partner placed {tile} pointing {d}: {'SUCCESS' if ok else 'FAIL'}\n")
                    hist[rpl].append(f"{rrd['header']}\npartner: {AC.fmt_actions(s_acts)}\nyou: {AC.fmt_actions(r_acts)}\n"
                                     f"result: you placed {tile} pointing {d}: {'SUCCESS' if ok else 'FAIL'}\n")
                    out_rounds.append({'run': held, 'sim': sim, 'round': r, 'phase': srd['phase'], 'sender': spl, 'receiver': rpl,
                                       'goal_tile': srd['goal_tile'], 'goal_direction': srd['goal_direction'],
                                       'receiver_tile': tile, 'receiver_direction': d, 'success': ok,
                                       'sender_actions': s_acts, 'receiver_actions': r_acts, 'real_success': srd['success']})
            closed.extend(out_rounds)
            print(f'  closed-loop sim {sim}: success {sum(x["success"] for x in out_rounds)}/{len(out_rounds)} '
                  f'(real {sum(x["real_success"] for x in out_rounds)}/{len(out_rounds)})', flush=True)
        del model, g; torch.cuda.empty_cache()

    with open(os.path.join(a.out, 'synthetic_rounds.jsonl'), 'w') as f:
        for x in closed:
            f.write(json.dumps(x) + '\n')
    res = {'model': a.model, 'epochs': a.epochs, 'mirror': a.mirror}
    if rec_rows:
        res['open_receiver'] = C.receiver_scores(rec_rows)
    if snd_model:
        mp, hp = C.sender_profile(snd_model), C.sender_profile(snd_human)
        res['open_sender'] = {'model': mp, 'human': hp, 'strategy_js': C.js_divergence(mp['strategy'], hp['strategy'])}
    res['open_timing'] = {'model': timing(t_model), 'human': timing(t_human)}
    if closed:
        ph = {}
        for x in closed:
            ph.setdefault(x['phase'], [0, 0, 0]); ph[x['phase']][0] += x['success']; ph[x['phase']][1] += x['real_success']; ph[x['phase']][2] += 1
        res['closed_success_by_phase'] = {k: {'sim': v[0] / v[2], 'real': v[1] / v[2], 'n': v[2]} for k, v in sorted(ph.items())}
        res['closed_success'] = {'sim': sum(x['success'] for x in closed) / len(closed), 'real': sum(x['real_success'] for x in closed) / len(closed)}
        cs = C.sender_profile([(to_tokens(x['sender_actions']), x['goal_tile'], x['goal_direction']) for x in closed])
        res['closed_sender_strategy'] = cs['strategy']
        res['closed_timing'] = timing([x['sender_actions'] for x in closed] + [x['receiver_actions'] for x in closed])
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
