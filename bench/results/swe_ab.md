# Glimmer 30B as the worker: whole-file reads vs CortexNET vs CortexNET + fade

SWE-bench Lite (public), Muse Glimmer 30B on one H100 per arm, HF generate (no KV reuse across turns: every turn re-reads the whole context, so prefill tokens are the compute). Episodes per arm: {'whole': 40, 'cortex': 40, 'fade': 39}; paired over the 39 instances every arm finished. Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests are run). KV counts Glimmer's layout: 13 full-attention layers grow with context, 39 sliding-window layers stop at 2,048 tokens.

## 1. Cost and outcome (paired)

| metric | whole (mean) | cortex (mean) | fade (mean) | cortex / whole (95% CI) | fade / whole (95% CI) |
|---|---|---|---|---|---|
| prefill tokens (sum over turns) | 194,598 | 112,348 | 109,451 | 0.58 [0.47, 0.71] | 0.56 [0.45, 0.72] |
| prefill tokens with a prefix cache (fade: lower bound) | 18,695 | 7,765 | 6,677 | 0.42 [0.35, 0.50] | 0.36 [0.30, 0.43] |
| peak context tokens | 20,428 | 9,629 | 8,598 | 0.47 [0.40, 0.56] | 0.42 [0.36, 0.50] |
| peak KV cache (MB) | 337 | 200 | 187 | 0.59 [0.53, 0.67] | 0.56 [0.49, 0.63] |
| turns | 17 | 19 | 20 | 1.16 [1.02, 1.32] | 1.19 [1.03, 1.38] |
| generation seconds | 165 | 162 | 177 | 0.98 [0.85, 1.16] | 1.07 [0.86, 1.40] |
| submitted file in gold patch | 82% | 85% | 82% | | |
| submitted symbol in gold patch | 41% | 46% | 41% | | |
| submitted at all | 90% | 95% | 95% | | |
| hit the 60k-token context limit | 3% | 0% | 0% | | |
| ran out of GPU memory (80 GB H100) | 0% | 0% | 0% | | |

File tokens put into context (read_file + read_lines, summed): whole 636,184, cortex 208,319 (33%), fade 196,787. Reads faded in the fade arm: 42.

Prefix-cache row: final context minus generated tokens, i.e. each token prefilled once. Exact for whole and cortex (their history is append-only). Fade rewrites earlier reads, which forces a re-prefill from that point on, so its row is a lower bound; its real cost under caching is not recoverable from these records.

## 2. What Glimmer actually used of a whole file (attention, layer 28)

- 65 whole-file reads. Share of later attention on the block that lands on symbol bodies (vs outline/header): median 85%.
- 80% of the attention on symbol bodies falls on **56%** of the shown symbol tokens (median over reads; mean 56%).
- Needed symbols ranked by attention usage: MRR 0.845 vs random 0.205 (40 labelled reads).

## 3. Logit lens at the read decision (Glimmer's own vocabulary, no training)

First token: each name scored by its first token only. Word pieces: max log-prob over the pieces of the identifier (split on _ and case), recomputed from the saved states (`modal_glimmer.py::lens`).

| scoring | layer | MRR (needed symbols) | random | reads | names tied |
|---|---|---|---|---|---|
| first token | 13 | 0.306 | 0.300 | 91 | 54% |
| first token | 26 | 0.297 | 0.307 | 91 | 53% |
| first token | 39 | 0.282 | 0.300 | 91 | 53% |
| word pieces | 13 | 0.291 | 0.300 | 91 | 35% |
| word pieces | 26 | 0.319 | 0.301 | 91 | 33% |
| word pieces | 39 | 0.321 | 0.300 | 91 | 31% |
