#!/usr/bin/env python3
"""
Baseline next-token predictors for the tokenized games — the numbers an LLM / decision model has to beat.

Usage: python3 analysis/baseline_ngram.py [analysis/tokens/turns.jsonl]

Evaluation: leave-one-run-out (train on the other runs, test on the held-out run), every token of every turn
predicted from what came before it in that turn (plus the context below). Reported per role:
  accuracy      share of tokens where the model's most likely next token was right
  bits/token    cross-entropy (lower = better; log2 of vocab size = uniform guessing)

Models (each adds information):
  uniform       every token equally likely
  unigram       token frequencies for this role
  bigram        previous token
  trigram       previous two tokens (backs off to bigram/unigram; add-k smoothing)
  trigram+ctx   trigram, Senders in the goal-relative frame with the goal type (cardinal/diagonal) as context;
                Receivers conditioned on whether the Sender ended with a wiggle-like pattern is NOT included here
                (that's the job of a richer model) — they get the plain trigram.
"""
import json
import math
import sys
from collections import Counter, defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else 'analysis/tokens/turns.jsonl'
turns = [json.loads(l) for l in open(PATH)]
VOCAB = sorted({t for x in turns for t in x['tokens'] + x.get('canon_tokens', [])})
V = len(VOCAB)
K = 0.1  # add-k smoothing


def seq(t, canon):
    return t['canon_tokens'] if canon and t['role'] == 'sender' else t['tokens']


def ctx_key(t, use_ctx):
    if not use_ctx or t['role'] != 'sender':
        return t['role']
    return t['role'] + ':' + ('diag' if '-' in t['goal_direction'] else 'card')


class NGram:
    def __init__(self, n, canon=False, use_ctx=False):
        self.n, self.canon, self.use_ctx = n, canon, use_ctx
        self.c = defaultdict(Counter)  # (ctx, history tuple) -> next-token counts

    def fit(self, data):
        for t in data:
            s = ['<s>'] * 2 + seq(t, self.canon)
            c = ctx_key(t, self.use_ctx)
            for i in range(2, len(s)):
                for k in range(self.n):  # store every order 0..n-1 for back-off
                    self.c[(c, tuple(s[i - k:i]))][s[i]] += 1
        return self

    def dist(self, c, hist):
        # interpolate orders with back-off weights (simple Jelinek-Mercer, equal weights over available orders)
        ps = []
        for k in range(self.n - 1, -1, -1):
            cnt = self.c.get((c, tuple(hist[len(hist) - k:])) if k else (c, ()))
            if cnt:
                tot = sum(cnt.values())
                ps.append({v: (cnt[v] + K) / (tot + K * V) for v in VOCAB})
        if not ps:
            return {v: 1 / V for v in VOCAB}
        w = [2 ** -i for i in range(len(ps))]
        z = sum(w)
        return {v: sum(wi * p[v] for wi, p in zip(w, ps)) / z for v in VOCAB}

    def score(self, data):
        res = defaultdict(lambda: [0, 0, 0.0])  # role -> [n, correct, bits]
        for t in data:
            s = ['<s>'] * 2 + seq(t, self.canon)
            c = ctx_key(t, self.use_ctx)
            for i in range(2, len(s)):
                d = self.dist(c, s[:i])
                r = res[t['role']]
                r[0] += 1
                r[1] += max(d, key=d.get) == s[i]
                r[2] += -math.log2(d[s[i]])
        return res


def uniform_score(data):
    res = defaultdict(lambda: [0, 0, 0.0])
    for t in data:
        for _ in seq(t, False):
            r = res[t['role']]
            r[0] += 1; r[1] += 1 / V; r[2] += math.log2(V)  # expected accuracy of a random guess
    return res


runs = sorted({t['run'] for t in turns})
models = {
    'uniform': None,
    'unigram': lambda: NGram(1),
    'bigram': lambda: NGram(2),
    'trigram': lambda: NGram(3),
    'trigram+ctx': lambda: NGram(3, canon=True, use_ctx=True),
}
print(f'{len(turns)} turns, vocab {V} tokens (uniform = {math.log2(V):.2f} bits/token); leave-one-run-out over runs {runs}\n')
print(f"{'model':13s} {'Sender acc':>10s} {'bits':>6s}   {'Receiver acc':>12s} {'bits':>6s}")
for name, mk in models.items():
    tot = defaultdict(lambda: [0, 0, 0.0])
    for held in runs:
        train = [t for t in turns if t['run'] != held]
        test = [t for t in turns if t['run'] == held]
        res = uniform_score(test) if mk is None else mk().fit(train).score(test)
        for role, (n, c, b) in res.items():
            tot[role][0] += n; tot[role][1] += c; tot[role][2] += b
    s, r = tot['sender'], tot['receiver']
    print(f"{name:13s} {s[1] / s[0]:10.0%} {s[2] / s[0]:6.2f}   {r[1] / r[0]:12.0%} {r[2] / r[0]:6.2f}")
print('\n(acc = most-likely next token correct; bits = cross-entropy per token, lower is better)')
