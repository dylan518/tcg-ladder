# Model comparison — predicting human play (3 runs, 480 turns, leave-one-run-out)
2026-11-11. Server job 376430 (WashU A6000) for the trained models; API run locally ($1.64).

| Model | Next-token bits/token (S / R, lower=better) | Next-token acc (S / R) | Receiver: matches human tile+dir | tile | dir | Receiver success (human 34%) | Sender strategy JS (0 = human mix) | Sender ends on goal (human 92%) |
|---|---|---|---|---|---|---|---|---|
| Uniform guess | 3.46 / 3.46 | 9% / 9% | – | – | – | – | – | – |
| Trigram + goal-relative | 2.33 / 2.38 | 40% / 36% | – | – | – | – | – | – |
| Rule: sender end tile, point up | – | – | 16% | 82% | 20% | 13% | – | – |
| Rule: perfect player (true goal) | – | – | 34% | 83% | 38% | 100% | – | – |
| Transformer from scratch | 2.11 / 2.39 | 47% / 40% | 3% | 17% | 17% | 2% | 0.174 | 42% |
| Decision Transformer ("win") | (2.11 / 2.38, leaky) | 47% / 40% | 2% | 12% | 16% | 2% | 0.127 | 42% |
| Transformer + mirror aug | 2.58 / 3.73 | 46% / 39% | 8% | 34% | 20% | 8% | 0.069 | 56% |
| Decision Transformer + mirror aug | (2.67 / 4.04, leaky) | 46% / 38% | 7% | 28% | 18% | 7% | 0.088 | 58% |
| SFT Qwen2.5-1.5B (LoRA) | **1.95 / 2.11** | – | 8% | 36% | 16% | 6% | 0.199 | 27% |
| Claude Opus 5.5, 40-shot, raw keys | – | – | **36%** | 74% | **43%** | 36% | 0.065 | 61% |
| Claude Sonnet 5.5, 40-shot, raw keys | – | – | 28% | 67% | 37% | 25% | **0.056** | 63% |
| Claude Sonnet 5.5, 40-shot, readable moves v1 | – | – | 30% | 82% | 35% | 40% | 0.064 | 97% |
| **Claude Sonnet 5.5, 40-shot, readable v2 (wiggle directions in words)** | – | – | **38%** | **84%** | **45%** | 37% | **0.028** | 97% |
| SFT Qwen2.5-1.5B, readable v1 | – | – | 13% | 70% | 18% | 10% | 0.050 | 90% |
| SFT Qwen2.5-1.5B, readable v2 | – | – | 10% | 73% | 14% | 12% | 0.045 | 88% |

S = Sender, R = Receiver. SFT bits are per game token (sub-word pieces summed). Decision Transformer bits use the true outcome token, so they leak.

Readable moves (analysis/models/readable.py): turns grouped into steps (went to X / wiggled A<->B xN / looped ... xN /
paused / turned ... / done|time ran out). Round trip keeps end tile 240/240, receiver placement 240/240, sender strategy
224/240. Sonnet readable vs raw ($1.69 for both full runs): tile tracking fixed (receiver tile 82% = the simple rule;
sender ends on goal 97% vs 63%); direction reading unchanged (35% vs 37%); readable over-produces goal-only Senders
(42% vs human 22%) and over-succeeds as Receiver (40% vs 34%).
