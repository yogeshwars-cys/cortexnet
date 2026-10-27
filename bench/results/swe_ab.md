# Glimmer 30B as the worker: whole-file reads vs CortexNET vs CortexNET + fade

SWE-bench Lite (public), Muse Glimmer 30B on one H100 per arm, HF generate (no KV reuse across turns: every turn re-reads the whole context, so prefill tokens are the compute). Episodes per arm: {'whole': 18, 'cortex': 25, 'fade': 27}; paired over the 18 instances every arm finished. Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests are run). KV counts Glimmer's layout: 13 full-attention layers grow with context, 39 sliding-window layers stop at 2,048 tokens.

## 1. Cost and outcome (paired)

| metric | whole (mean) | cortex (mean) | fade (mean) | cortex / whole (95% CI) | fade / whole (95% CI) |
|---|---|---|---|---|---|
| prefill tokens (sum over turns) | 183,953 | 107,555 | 95,615 | 0.58 [0.44, 0.82] | 0.52 [0.40, 0.71] |
| peak context tokens | 18,276 | 8,870 | 8,265 | 0.49 [0.39, 0.61] | 0.45 [0.37, 0.57] |
| peak KV cache (MB) | 310 | 191 | 183 | 0.61 [0.52, 0.73] | 0.59 [0.50, 0.70] |
| turns | 17 | 20 | 18 | 1.19 [1.01, 1.39] | 1.09 [0.89, 1.31] |
| generation seconds | 177 | 150 | 151 | 0.85 [0.70, 1.04] | 0.85 [0.72, 1.01] |
| submitted file in gold patch | 94% | 89% | 89% | | |
| submitted symbol in gold patch | 56% | 56% | 56% | | |
| submitted at all | 94% | 89% | 89% | | |
| hit the 60k-token context limit | 0% | 0% | 0% | | |
| ran out of GPU memory (80 GB H100) | 0% | 0% | 0% | | |

File tokens put into context (read_file + read_lines, summed): whole 261,539, cortex 90,826 (35%), fade 88,408. Reads faded in the fade arm: 18.

## 2. What Glimmer actually used of a whole file (attention, layer 28)

- 30 whole-file reads. Share of later attention on the block that lands on symbol bodies (vs outline/header): median 85%.
- 80% of the attention on symbol bodies falls on **56%** of the shown symbol tokens (median over reads; mean 57%).
- Needed symbols ranked by attention usage: MRR 0.814 vs random 0.203 (17 labelled reads).

## 3. Logit lens at the read decision (Glimmer's own vocabulary, no training)

First token: each name scored by its first token only. Word pieces: max log-prob over the pieces of the identifier (split on _ and case), recomputed from the saved states (`modal_glimmer.py::lens`).

| scoring | layer | MRR (needed symbols) | random | reads | names tied |
|---|---|---|---|---|---|
| first token | 13 | 0.304 | 0.328 | 48 | 54% |
| first token | 26 | 0.340 | 0.326 | 48 | 52% |
| first token | 39 | 0.326 | 0.335 | 48 | 52% |
| word pieces | 13 | 0.382 | 0.324 | 18 | 34% |
| word pieces | 26 | 0.383 | 0.318 | 18 | 34% |
| word pieces | 39 | 0.389 | 0.312 | 18 | 33% |
