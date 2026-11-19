#!/usr/bin/env python3
"""
Qwen + LoRA on whole-game strategy-level transcripts (strat_ctx.py). Loss only on the decisions ("you sent: X",
"you read it: Y"). Leave-one-run-out.

Evaluation
  per-turn   held-out players, true history: probability of each option = normalised likelihood of its text;
             accuracy and bits for Sender strategy and Receiver reading (vs strategy_pair's history-free baseline)
  closed     the held-out run's schedule played by simulated players: sample the Sender strategy, show the Receiver the
             implied signal, sample the reading, success from the outcome table (fit on the training runs); both players'
             transcripts grow with the simulated rounds. Success and Sender–Receiver match by phase.

Usage: python3 analysis/models/strat_ctx_sft.py [--epochs 6] [--sims 5] [--out analysis/models/out/strat_ctx_sft]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strat_ctx as X  # noqa: E402
import strategy_pair as SP  # noqa: E402


def encode(tok, evlist, names):
    """ids + labels (loss only on target spans)"""
    text, spans = X.INTRO, []
    for role, ev in evlist:
        p, t, s = X.render(role, ev, names)
        text += p; spans.append((len(text), len(text) + len(t))); text += t + s
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    lab = [-100] * len(enc['input_ids'])
    j = 0
    for i, (a, b) in enumerate(enc['offset_mapping']):
        while j < len(spans) and spans[j][1] <= a:
            j += 1
        if j < len(spans) and spans[j][0] <= a < spans[j][1]:
            lab[i] = enc['input_ids'][i]
    return enc['input_ids'], lab


def train(a, players, names, seed):
    tok = AutoTokenizer.from_pretrained(a.model)
    torch.manual_seed(seed); rng = random.Random(seed)
    base = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to('cuda')
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
                                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    data = [encode(tok, ev, names) for ev in players]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = a.epochs * len(data)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 5) * max(0.05, 1 - s / steps))
    model.train(); t0 = time.time()
    for ep in range(a.epochs):
        rng.shuffle(data); tot = 0
        for ids, lab in data:
            loss = model(input_ids=torch.tensor([ids], device='cuda'), labels=torch.tensor([lab], device='cuda')).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        print(f'  epoch {ep} loss {tot / len(data):.3f} ({time.time() - t0:.0f}s)', flush=True)
    model.eval()
    return model, tok


@torch.no_grad()
def option_probs(model, tok, prefix, opts_text):
    """P(option) ∝ P(option text + ' |' | prefix); prefix KV computed once"""
    pids = tok(prefix, add_special_tokens=False)['input_ids']
    out = model(input_ids=torch.tensor([pids], device='cuda'), use_cache=True)
    past, last = out.past_key_values, out.logits[0, -1].float().log_softmax(-1)
    lps = []
    for o in opts_text:
        full = tok(prefix + o + ' |', add_special_tokens=False)['input_ids']
        cont = full[len(pids):] if full[:len(pids)] == pids else tok(o + ' |', add_special_tokens=False)['input_ids']
        lp = last[cont[0]].item()
        if len(cont) > 1:
            import copy
            pk = copy.deepcopy(past)
            o2 = model(input_ids=torch.tensor([cont[:-1]], device='cuda'), past_key_values=pk, use_cache=True)
            ls = o2.logits[0].float().log_softmax(-1)
            lp += sum(ls[i, cont[i + 1]].item() for i in range(len(cont) - 1))
        lps.append(lp)
    return X.softmax_logp(lps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--sims', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0, help='smoke: rounds per player / per simulation')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'strat_ctx_sft'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    events, names = X.load_players()
    runs = sorted({r for r, _ in events})
    allr = SP.load()
    pt = defaultdict(lambda: [0, 0.0, 0])  # role -> [n, bits, correct]
    closed, rows = [], []
    rng = np.random.default_rng(0)
    for held in runs:
        tr = [ev for (r, _), ev in events.items() if r != held]
        te = {p: ev for (r, p), ev in events.items() if r == held}
        print(f'=== fold {held}: train {len(tr)} players, test {len(te)}', flush=True)
        model, tok = train(a, tr, names, held)
        # ---- per-turn, true history
        for p, evl in te.items():
            ctx = X.INTRO
            for i, (role, ev) in enumerate(evl[:a.limit or None]):
                pre, tgt, suf = X.render(role, ev, names)
                opts = X.options(role, ev)
                pr = option_probs(model, tok, ctx + pre, [X.option_text(role, k) for k in opts])
                truth = ev['S'] if role == 'sender' else ev['R']
                j = opts.index(truth)
                pt[role][0] += 1; pt[role][1] -= math.log2(max(pr[j], 1e-9)); pt[role][2] += int(np.argmax(pr) == j)
                rows.append({'run': held, 'round': ev['round'], 'role': role, 'opts': opts, 'probs': pr, 'truth': truth})
                ctx += pre + tgt + suf
        print('  per-turn done', flush=True)
        # ---- closed loop
        tab, p_end = SP.outcome_table([x for x in allr if x['run'] != held])
        sched = sorted({(ev['round'], ev['s']): ev for evl in te.values() for role, ev in evl if role == 'sender'}.items())
        for sim in range(a.sims):
            ctx = {p: X.INTRO for p in te}
            out = []
            for (_, _), ev in sched[: (a.limit * len(te) // 2) or None]:
                pre, _, _ = X.render('sender', ev, names)
                so = X.options('sender', ev)
                ps = option_probs(model, tok, ctx[ev['s']] + pre, [X.option_text('sender', k) for k in so])
                S = so[rng.choice(len(so), p=np.array(ps) / sum(ps))]
                sev = X.sim_signal(S, ev, rng)
                rpre, _, _ = X.render('receiver', sev, names, S=S, R='F')
                ro = X.options('receiver', sev, S)
                pr = option_probs(model, tok, ctx[ev['r']] + rpre, [X.option_text('receiver', k) for k in ro])
                R = ro[rng.choice(len(ro), p=np.array(pr) / sum(pr))]
                succ = int(rng.random() < p_end and rng.random() < tab.get((S, R), 0.0))
                ctx[ev['s']] += ''.join(X.render('sender', sev, names, S=S, R=R, succ=succ))
                ctx[ev['r']] += ''.join(X.render('receiver', sev, names, S=S, R=R, succ=succ))
                out.append({'run': held, 'sim': sim, 'round': ev['round'], 'phase': ev['phase'], 'S': S, 'R': R, 'succ': succ,
                            'match': int(X.match(S, R)), 'signal': int(S in SP.SIGNAL), 'real_succ': ev['succ']})
            closed += out
            print(f"  closed sim {sim}: success {sum(o['succ'] for o in out)}/{len(out)} (real {sum(o['real_succ'] for o in out)})", flush=True)
        del model; torch.cuda.empty_cache()

    res = {'model': a.model, 'epochs': a.epochs,
           'per_turn': {r: {'n': v[0], 'bits': v[1] / v[0], 'acc': v[2] / v[0]} for r, v in pt.items()}}
    if closed:
        by = defaultdict(list)
        for o in closed:
            by['s', o['phase']].append(o['succ']); by['r', o['phase']].append(o['real_succ'])
            if o['signal']:
                by['m', o['phase']].append(o['match'])
        res['closed'] = {ph: {'sim_success': float(np.mean(by['s', ph])), 'real_success': float(np.mean(by['r', ph])),
                              'sim_match': float(np.mean(by['m', ph])) if by['m', ph] else None} for ph in (1, 2, 3) if by['s', ph]}
        per = defaultdict(list)
        for o in closed:
            per[o['run'], o['sim'], o['phase']].append(o['succ'])
        res['closed_band'] = {ph: np.percentile([np.mean(v) for (r, s, p), v in per.items() if p == ph], [5, 95]).tolist() for ph in (1, 2, 3)
                              if any(p == ph for (_, _, p) in per)}
        mix = defaultdict(lambda: defaultdict(int))
        for o in closed:
            mix[o['phase']]['S:' + o['S']] += 1; mix[o['phase']]['R:' + o['R']] += 1
        res['closed_mix'] = {ph: dict(m) for ph, m in mix.items()}
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'per_turn.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    with open(os.path.join(a.out, 'closed.jsonl'), 'w') as f:
        for r in closed:
            f.write(json.dumps(r) + '\n')
    print(json.dumps(res, indent=1))
    print('=== DONE ===')


if __name__ == '__main__':
    main()
