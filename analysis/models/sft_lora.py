#!/usr/bin/env python3
"""
Approach B: supervised fine-tuning (LoRA) of a small open LLM on the human turns, leave-one-run-out.

Usage (GPU): python3 analysis/models/sft_lora.py --model Qwen/Qwen2.5-1.5B [--epochs 4] [--out analysis/models/out/sft]

Each training example: prompt = the same text the API approach sees (C.receiver_query / C.sender_query, with a task
header), completion = the human's token sequence + EOS. Loss on the completion only.
Evaluation per held-out run:
  bits/token   -log2 p(completion) summed over the completion's sub-word pieces, divided by the number of GAME tokens
               (so it is comparable with the n-gram / transformer numbers despite the different tokenizer; EOS excluded)
  receiver     one sampled completion (temperature 1) -> replayed on the board -> tile/direction vs the human
  sender       one sampled completion -> strategy mix etc. vs the held-out humans
"""
import argparse
import json
import math
import os
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import readable as RD  # noqa: E402

FMT = {'format': 'raw'}


def render(tokens):
    return RD.describe(RD.steps_from_tokens(tokens)) if FMT['format'] == 'readable' else C.raw(tokens)


def decode(text):
    return RD.realize(RD.parse(text)) if FMT['format'] == 'readable' else C.parse_tokens(text)


def prompt_text(t, turns):
    if t['role'] == 'receiver':
        return 'TASK: predict the human Receiver\'s turn as game tokens.\n' + C.receiver_query(t, turns, render) + '\nReceiver turn:'
    return 'TASK: predict the human Sender\'s turn as game tokens.\n' + C.sender_query(t, turns, render) + '\nSender turn:'


def build(tok, t, turns):
    p = tok(prompt_text(t, turns), add_special_tokens=False)['input_ids']
    c = tok(' ' + render(t['tokens']), add_special_tokens=False)['input_ids'] + [tok.eos_token_id]
    return p, c


@torch.no_grad()
def completion_bits(model, tok, p, c, device):
    ids = torch.tensor([p + c], device=device)
    logits = model(input_ids=ids).logits[0, len(p) - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, torch.tensor(c, device=device).unsqueeze(-1)).squeeze(-1)
    return -(lp[:-1].sum().item()) / math.log(2)  # exclude EOS


@torch.no_grad()
def sample(model, tok, p, device, max_new=160):
    ids = torch.tensor([p], device=device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=True, temperature=1.0,
                         top_p=1.0, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, len(p):], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'sft'))
    ap.add_argument('--selfplay', action='store_true', help='also: Qwen Receiver decodes Qwen Sender turns')
    ap.add_argument('--limit_eval', type=int, default=0)
    ap.add_argument('--limit_train', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ap.add_argument('--format', default='raw', choices=['raw', 'readable'])
    a = ap.parse_args()
    FMT['format'] = a.format
    os.makedirs(a.out, exist_ok=True)
    device = a.device
    turns = C.load_turns()
    tok = AutoTokenizer.from_pretrained(a.model)
    results = {'model': a.model, 'format': a.format, 'folds': [], 'bits_unit': 'per game token' if a.format == 'raw' else 'per turn'}
    stats = {'sender': [0, 0.0], 'receiver': [0, 0.0]}
    rec_rows, snd_model, snd_human, preds, selfplay = [], [], [], [], []
    for held, train, test in C.folds(turns):
        torch.manual_seed(held); rng = random.Random(held)
        base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32).to(device)
        model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
                                                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        data = [build(tok, t, turns) for t in (train if not a.limit_train else train[:a.limit_train])]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
        steps = a.epochs * math.ceil(len(data) / a.bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * max(0.0, 1 - s / steps))
        model.train()
        step = 0
        for ep in range(a.epochs):
            rng.shuffle(data)
            for i in range(0, len(data), a.bs):
                batch = data[i:i + a.bs]
                L = max(len(p) + len(c) for p, c in batch)
                x = torch.full((len(batch), L), tok.eos_token_id); y = torch.full((len(batch), L), -100); att = torch.zeros((len(batch), L), dtype=torch.long)
                for j, (p, c) in enumerate(batch):
                    ids = p + c
                    x[j, :len(ids)] = torch.tensor(ids); att[j, :len(ids)] = 1
                    y[j, len(p):len(ids)] = torch.tensor(c)
                loss = model(input_ids=x.to(device), attention_mask=att.to(device), labels=y.to(device)).loss
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
                if step % 20 == 0:
                    print(f'fold {held} epoch {ep} step {step}/{steps} loss {loss.item():.3f}', flush=True)
        model.eval()
        evalset = test if not a.limit_eval else test[:a.limit_eval]
        for t in evalset:
            p, c = build(tok, t, turns)
            n_game = len(t['tokens']) if FMT['format'] == 'raw' else 1  # readable: bits per TURN
            stats[t['role']][0] += n_game
            stats[t['role']][1] += completion_bits(model, tok, p, c, device)
            text = sample(model, tok, p, device)
            gen = decode(text)
            preds.append({'run': held, 'round': t['round'], 'role': t['role'], 'player': t['player'], 'text': text, 'gen': gen})
            if t['role'] == 'receiver':
                tile, dirn = C.replay(gen)
                rec_rows.append({'turn': t, 'pred_tile': tile, 'pred_dir': dirn})
            else:
                snd_model.append((gen, t['goal_tile'], t['goal_direction']))
                snd_human.append((t['tokens'], t['goal_tile'], t['goal_direction']))
        print(f'fold {held} evaluated', flush=True)
        if a.selfplay:  # Qwen Receiver reads the Qwen Sender's generated turn for the same round
            sgen = {(p['round'], p['player']): p['gen'] for p in preds if p['run'] == held and p['role'] == 'sender'}
            for t in evalset:
                if t['role'] != 'receiver' or (t['round'], t['partner']) not in sgen:
                    continue
                q = dict(t, saw_sender_tokens=sgen[(t['round'], t['partner'])])
                p, _ = build(tok, q, turns)
                tile, dirn = C.replay(decode(sample(model, tok, p, device)))
                selfplay.append({'run': held, 'round': t['round'], 'phase': t['phase'], 'goal_tile': t['goal_tile'], 'goal_direction': t['goal_direction'],
                                 'tile': tile, 'direction': dirn, 'success': int(tile == t['goal_tile'] and dirn == t['goal_direction']), 'real_success': t['success']})
            print(f'fold {held} selfplay done', flush=True)
        del model, base
        if device == 'cuda':
            torch.cuda.empty_cache()
    sc = C.receiver_scores(rec_rows)
    mp, hp = C.sender_profile(snd_model), C.sender_profile(snd_human)
    results.update({
        'bits_per_token': {r: s[1] / s[0] for r, s in stats.items()},
        'receiver': sc, 'sender_model': mp, 'sender_human': hp,
        'sender_strategy_js': C.js_divergence(mp['strategy'], hp['strategy']),
        'selfplay': ({'n': len(selfplay), 'success': sum(x['success'] for x in selfplay) / len(selfplay),
                      'tile': sum(x['tile'] == x['goal_tile'] for x in selfplay) / len(selfplay),
                      'direction': sum(x['direction'] == x['goal_direction'] for x in selfplay) / len(selfplay),
                      'real_success': sum(x['real_success'] for x in selfplay) / len(selfplay),
                      'by_phase': {ph: sum(x['success'] for x in selfplay if x['phase'] == ph) / max(1, sum(x['phase'] == ph for x in selfplay)) for ph in (1, 2, 3)}}
                     if selfplay else None),
    })
    json.dump(results, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'predictions.jsonl'), 'w') as f:
        for p in preds:
            f.write(json.dumps(p) + '\n')
    print(json.dumps({k: results[k] for k in ('bits_per_token', 'receiver', 'sender_strategy_js')}, indent=1))
    print(f"sender: len {mp['mean_len']:.1f} vs {hp['mean_len']:.1f}, timeout {mp['timeout_rate']:.0%} vs {hp['timeout_rate']:.0%}, "
          f"ends on goal {mp['ends_on_goal']:.0%} vs {hp['ends_on_goal']:.0%}")
    print('strategy (model/human): ' + ', '.join(f"{k} {mp['strategy'].get(k, 0):.0%}/{hp['strategy'].get(k, 0):.0%}" for k in sorted(set(mp['strategy']) | set(hp['strategy']))))


if __name__ == '__main__':
    main()
