# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: Muse Glimmer 30B layer 26 (6656 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (854 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (1057s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.280 | 0.280 +- 0.024 | - | 0.118 | 0.356 | 0.549 |
| smallest first | 0.214 | 0.214 +- 0.028 | - | 0.104 | 0.390 | 0.556 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.313 | 0.313 +- 0.028 | - | 0.158 | 0.387 | 0.566 |
| v1 include features, logistic | 0.309 | 0.309 +- 0.037 | - | 0.163 | 0.330 | 0.498 |
| frozen zero-shot (max sub-span cosine) | 0.320 | 0.320 +- 0.025 | - | 0.165 | 0.334 | 0.530 |
| full Cortex-SubNN | 0.372 | 0.372 +- 0.064 | 0.354 +- 0.009 | 0.214 | 0.391 | 0.583 |
| - gate (uniform over present views) | 0.369 | 0.369 +- 0.052 | 0.353 +- 0.011 | 0.205 | 0.409 | 0.593 |
| - multi-view (prompt only) | 0.365 | 0.365 +- 0.061 | 0.357 +- 0.012 | 0.205 | 0.411 | 0.589 |
| - multi-view (single stream prompt+snapshot) | 0.384 | 0.384 +- 0.060 | 0.361 +- 0.005 | 0.224 | 0.404 | 0.582 |
| - late interaction (M = 1, mean pooled) | 0.382 | 0.382 +- 0.047 | 0.372 +- 0.004 | 0.224 | 0.412 | 0.588 |
| - cost weighting (alpha = beta = 0) | 0.386 | 0.386 +- 0.059 | 0.371 +- 0.008 | 0.220 | 0.394 | 0.573 |
| - structural prior | 0.361 | 0.361 +- 0.058 | 0.344 +- 0.003 | 0.201 | 0.406 | 0.588 |
| - cross-file negatives | 0.396 | 0.396 +- 0.061 | 0.381 +- 0.008 | 0.238 | 0.400 | 0.577 |
| - diagonal re-weighting (low-rank only) | 0.388 | 0.388 +- 0.054 | 0.363 +- 0.003 | 0.237 | 0.421 | 0.608 |
| - low-rank residual (diagonal only) | 0.332 | 0.332 +- 0.031 | 0.333 +- 0.001 | 0.166 | 0.343 | 0.542 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.379 | 0.379 +- 0.056 | 0.362 +- 0.005 | 0.217 | 0.416 | 0.597 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.052 [+0.030, +0.075]
- full Cortex-SubNN vs v1 include features, logistic: +0.063 [+0.037, +0.090]
- full Cortex-SubNN vs random: +0.093 [+0.067, +0.119]
- full vs - gate (uniform over present views): +0.003 [-0.015, +0.022]
- full vs - multi-view (prompt only): +0.007 [-0.013, +0.027]
- full vs - multi-view (single stream prompt+snapshot): -0.012 [-0.032, +0.008]
- full vs - late interaction (M = 1, mean pooled): -0.010 [-0.030, +0.011]
- full vs - cost weighting (alpha = beta = 0): -0.013 [-0.032, +0.006]
- full vs - structural prior: +0.011 [-0.006, +0.028]
- full vs - cross-file negatives: -0.024 [-0.044, -0.003]
- full vs - diagonal re-weighting (low-rank only): -0.016 [-0.033, +0.002]
- full vs - low-rank residual (diagonal only): +0.040 [+0.020, +0.062]
- full vs v2 bottleneck head (k = 32, int8, cosine skip): -0.007 [-0.031, +0.016]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (511) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.236 | 0.199 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.337 | 0.298 |
| v1 include features, logistic | 0.293 | 0.320 |
| frozen zero-shot (max sub-span cosine) | 0.345 | 0.304 |
| full Cortex-SubNN | 0.393 | 0.358 |
| - gate (uniform over present views) | 0.398 | 0.349 |
| - multi-view (prompt only) | 0.400 | 0.342 |
| - multi-view (single stream prompt+snapshot) | 0.420 | 0.360 |
| - late interaction (M = 1, mean pooled) | 0.395 | 0.373 |
| - cost weighting (alpha = beta = 0) | 0.416 | 0.365 |
| - structural prior | 0.378 | 0.350 |
| - cross-file negatives | 0.422 | 0.379 |
| - diagonal re-weighting (low-rank only) | 0.408 | 0.375 |
| - low-rank residual (diagonal only) | 0.347 | 0.322 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.407 | 0.361 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 275

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.292 | 0.091 |
| smallest first | 0.184 | 0.047 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.411 | 0.233 |
| frozen zero-shot (max sub-span cosine) | 0.340 | 0.171 |
| full Cortex-SubNN | 0.452 | 0.269 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.20999999344348907, 0.7599999904632568, 0.03999999910593033], without [0.8199999928474426, 0.0, 0.18000000715255737].

## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)

Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown up front, plus one extra round trip and the missing bodies if a needed symbol was not shown. Rule A returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the Reader's current policy with that ranker. Prefix-cache regimes: an extra round trip re-reads the cached history at 0.1x (L_hist = logged context at the read) plus 400 tokens of tool-step overhead; with compounding, every token put into context also costs 0.1x on each later API call until the session ends or is compacted (H_rem from the session logs for 1190 cases, median 72; median on test cases 84; missing logs use the median). Cache writes are priced at 1x, not 1.25x.

### C_turn = 500 (no cache accounting; ~ local KV cache)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 21% | 42% | 21% | 1% | 15% |
| v1 include features, logistic | 21% | 43% | 21% | 1% | 15% |
| frozen zero-shot (max sub-span cosine) | 21% | 43% | 21% | 1% | 15% |
| smallest first | 21% | 42% | 21% | 1% | 15% |

### C_turn = 2,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 38% | 55% | 36% | 15% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen zero-shot (max sub-span cosine) | 38% | 57% | 36% | 15% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 488% | 407% | 95% | 91% | 15% |
| v1 include features, logistic | 488% | 412% | 95% | 92% | 15% |
| frozen zero-shot (max sub-span cosine) | 488% | 428% | 95% | 92% | 15% |
| smallest first | 488% | 412% | 95% | 91% | 15% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 414% | 349% | 94% | 90% | 15% |
| v1 include features, logistic | 414% | 353% | 94% | 91% | 15% |
| frozen zero-shot (max sub-span cosine) | 414% | 367% | 94% | 91% | 15% |
| smallest first | 414% | 353% | 94% | 90% | 15% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 52% | 66% | 39% | 41% | 16% |
| v1 include features, logistic | 52% | 67% | 39% | 41% | 16% |
| frozen zero-shot (max sub-span cosine) | 52% | 68% | 38% | 40% | 16% |
| smallest first | 52% | 67% | 38% | 40% | 16% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.287 |
| smallest first | 0.253 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.310 |
| v1 include features, logistic | 0.298 |
| frozen zero-shot (max sub-span cosine) | 0.330 |
| full Cortex-SubNN | 0.260 +- 0.023 (3 seeds) |
