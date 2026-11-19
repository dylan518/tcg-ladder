#!/usr/bin/env python3
"""
Approach C: a small transformer trained from scratch on the game tokens (GPT-style behaviour model), plus a Decision
Transformer variant that is also told the outcome ("return") to aim for. Leave-one-run-out, same tasks as common.py.

Usage: python3 analysis/models/dt_train.py [--variant gpt|dt|both] [--steps 3000] [--device mps|cpu|cuda]

Sequence = context tokens, then the turn's game tokens, then <eos>; loss only on the game tokens + <eos>.
  Sender:   <snd> <pg_k> [<rtg_win>|<rtg_fail>] <gt_TILE> <gd_DIR>  canon tokens...  <eos>
            (goal-relative frame, so one learned pattern covers all goals; strategy codes are rotation-invariant)
  Receiver: <rcv> <pg_k> [<rtg_*>]  sender tokens...  <sep>  receiver tokens...  <eos>
  pg_k = games already played with this partner, bucketed 0 / 1-4 / 5-9 / 10+ (reunions).
Variant dt: the <rtg_*> token is the turn's real outcome during training; at test time it is set to <rtg_win>
("act like a human who succeeded") for generation, and to the true outcome for bits/token (reported separately,
since that leaks the outcome).
"""
import argparse
import math
import os
import random
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SPECIAL = ['<pad>', '<snd>', '<rcv>', '<sep>', '<eos>', '<rtg_win>', '<rtg_fail>', '<pg_0>', '<pg_1>', '<pg_5>', '<pg_10>']
GOAL = [f'<gt_{t}>' for t in C.TILES] + [f'<gd_{d}>' for d in C.DIRS]
ITOS = SPECIAL + GOAL + C.VOCAB
STOI = {s: i for i, s in enumerate(ITOS)}
GAME = set(C.VOCAB)


def pg_tok(n):
    return '<pg_0>' if n == 0 else '<pg_1>' if n < 5 else '<pg_5>' if n < 10 else '<pg_10>'


def encode(t, variant, rtg=None):
    """-> (ids, n_ctx): context length so the loss can skip it"""
    r = [] if variant == 'gpt' else [rtg or ('<rtg_win>' if t['success'] else '<rtg_fail>')]
    if t['role'] == 'sender':
        ctx = ['<snd>', pg_tok(t['partner_games'])] + r + [f"<gt_{t['canon_goal_tile']}>", f"<gd_{t['canon_goal_direction']}>"]
        tgt = t['canon_tokens']
    else:
        ctx = ['<rcv>', pg_tok(t['partner_games'])] + r + t['saw_sender_tokens'] + ['<sep>']
        tgt = t['tokens']
    return [STOI[x] for x in ctx + tgt + ['<eos>']], len(ctx)


def batches(data, bs, rng):
    idx = list(range(len(data)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx) - bs + 1, bs):
            yield [data[j] for j in idx[i:i + bs]]


def collate(items, device):
    L = max(len(ids) for ids, _ in items)
    x = torch.full((len(items), L), STOI['<pad>'])
    y = torch.full((len(items), L), -100)
    for i, (ids, n_ctx) in enumerate(items):
        x[i, :len(ids)] = torch.tensor(ids)
        lab = torch.tensor(ids)
        lab[:n_ctx] = -100
        y[i, :len(ids)] = lab
    att = (x != STOI['<pad>']).long()
    return x.to(device), y.to(device), att.to(device)


# Left-right mirror: the only board symmetry that leaves the game unchanged (the triangle always starts pointing up,
# which a mirror keeps; a rotation would not). Moves L<->R swap, rotations RL<->RR swap, U/D/waits/END stay.
MIRROR = {'L': 'R', 'R': 'L', 'RL': 'RR', 'RR': 'RL'}


def mirrored(t):
    """a Receiver turn with the board mirrored left-right (training augmentation)"""
    return [{**t, 'saw_sender_tokens': [MIRROR.get(x, x) for x in t['saw_sender_tokens']], 'tokens': [MIRROR.get(x, x) for x in t['tokens']]}]


def train_fold(train, variant, args, seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    if args.augment:
        train = train + [r for t in train if t['role'] == 'receiver' for r in mirrored(t)]
    data = [encode(t, variant) for t in train]
    rng.shuffle(data)
    nval = max(8, len(data) // 10)
    val, tr = data[:nval], data[nval:]
    cfg = GPT2Config(vocab_size=len(ITOS), n_positions=512, n_embd=args.dim, n_layer=args.layers, n_head=4,
                     resid_pdrop=0.2, embd_pdrop=0.2, attn_pdrop=0.2, bos_token_id=STOI['<eos>'], eos_token_id=STOI['<eos>'])
    model = GPT2LMHeadModel(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    best, best_state, gen = float('inf'), None, batches(tr, args.bs, rng)
    for step in range(1, args.steps + 1):
        model.train()
        x, y, att = collate(next(gen), args.device)
        loss = model(input_ids=x, attention_mask=att, labels=y).loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            vl = nll(model, val, args.device)[0]
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def nll(model, data, device):
    """mean nats/token and accuracy over target tokens"""
    model.eval()
    tot, n, correct = 0.0, 0, 0
    for i in range(0, len(data), 32):
        x, y, att = collate(data[i:i + 32], device)
        logits = model(input_ids=x, attention_mask=att).logits[:, :-1]
        lab = y[:, 1:]
        mask = lab != -100
        lp = torch.log_softmax(logits.float(), -1)
        tot += -lp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)[mask].sum().item()
        correct += (logits.argmax(-1)[mask] == lab[mask]).sum().item()
        n += mask.sum().item()
    return tot / n, correct / n


@torch.no_grad()
def generate(model, ctx_ids, device, greedy, max_new=80, temp=1.0):
    model.eval()
    ids = list(ctx_ids)
    out = []
    for _ in range(max_new):
        logits = model(input_ids=torch.tensor([ids[-511:]], device=device)).logits[0, -1].float()
        allowed = torch.full_like(logits, float('-inf'))
        for s in list(GAME) + ['<eos>']:
            allowed[STOI[s]] = 0
        logits = logits + allowed
        nxt = int(logits.argmax()) if greedy else int(torch.multinomial(torch.softmax(logits / temp, -1), 1))
        tok = ITOS[nxt]
        if tok == '<eos>':
            break
        out.append(tok); ids.append(nxt)
        if tok in ('END', 'TIMEOUT'):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='both', choices=['gpt', 'dt', 'both'])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--augment', action='store_true', help='add left-right mirrored copies of Receiver turns')
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    turns = C.load_turns()
    for variant in (['gpt', 'dt'] if args.variant == 'both' else [args.variant]):
        stats = {'sender': [0, 0, 0.0], 'receiver': [0, 0, 0.0]}
        rec_rows, snd_model, snd_human = [], [], []
        for held, train, test in C.folds(turns):
            model = train_fold(train, variant, args, seed=held)
            for role in ('sender', 'receiver'):
                d = [encode(t, variant) for t in test if t['role'] == role]
                nats, acc = nll(model, d, args.device)
                n = sum(len(i) - c for i, c in d)
                s = stats[role]; s[0] += n; s[1] += acc * n; s[2] += nats * n
            rng = random.Random(held)
            for t in test:
                if t['role'] == 'receiver':
                    ids, n_ctx = encode(t, variant, rtg='<rtg_win>')
                    toks = generate(model, ids[:n_ctx], args.device, greedy=False)
                    tile, dirn = C.replay(toks)
                    rec_rows.append({'turn': t, 'pred_tile': tile, 'pred_dir': dirn})
                else:
                    ids, n_ctx = encode(t, variant, rtg='<rtg_win>')
                    toks = generate(model, ids[:n_ctx], args.device, greedy=False)
                    snd_model.append((toks, t['canon_goal_tile'], t['canon_goal_direction']))
                    snd_human.append((t['canon_tokens'], t['canon_goal_tile'], t['canon_goal_direction']))
            print(f'  [{variant}] fold {held} done', flush=True)
        label = ('transformer (no outcome)' if variant == 'gpt' else 'Decision Transformer (told "win")') + (' + mirror augmentation' if args.augment else '')
        print(f'\n=== {label} ===')
        for role, (n, c, nats) in stats.items():
            extra = '  (bits use the TRUE outcome token: leaky)' if variant == 'dt' else ''
            print(f'  next-token {role:8s}: acc {c / n:.0%}  bits/token {nats / n / math.log(2):.2f}{extra}')
        sc = C.receiver_scores(rec_rows)
        print(f"  receiver placement (sampled): match human tile+dir {sc['match_both']:.0%}, tile {sc['match_tile']:.0%}, direction {sc['match_dir']:.0%} | "
              f"model success {sc['pred_success']:.0%} vs human {sc['human_success']:.0%}")
        mp, hp = C.sender_profile(snd_model), C.sender_profile(snd_human)
        print(f"  sender generation: len {mp['mean_len']:.1f} vs {hp['mean_len']:.1f}, timeout {mp['timeout_rate']:.0%} vs {hp['timeout_rate']:.0%}, "
              f"ends on goal {mp['ends_on_goal']:.0%} vs {hp['ends_on_goal']:.0%}, strategy JS {C.js_divergence(mp['strategy'], hp['strategy']):.3f} bits")
        print('    strategy (model vs human): ' + ', '.join(f"{k} {mp['strategy'].get(k, 0):.0%}/{hp['strategy'].get(k, 0):.0%}" for k in sorted(set(mp['strategy']) | set(hp['strategy']))))


if __name__ == '__main__':
    main()
