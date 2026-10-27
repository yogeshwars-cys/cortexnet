# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: Muse Glimmer 30B layer 39 (6656 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (854 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (990s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.280 | 0.280 +- 0.024 | - | 0.118 | 0.356 | 0.549 |
| smallest first | 0.214 | 0.214 +- 0.028 | - | 0.104 | 0.390 | 0.556 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.313 | 0.313 +- 0.028 | - | 0.158 | 0.387 | 0.566 |
| v1 include features, logistic | 0.309 | 0.309 +- 0.037 | - | 0.163 | 0.330 | 0.498 |
| frozen zero-shot (max sub-span cosine) | 0.311 | 0.311 +- 0.035 | - | 0.145 | 0.331 | 0.537 |
| full Cortex-SubNN | 0.367 | 0.367 +- 0.047 | 0.361 +- 0.013 | 0.193 | 0.381 | 0.574 |
| - gate (uniform over present views) | 0.364 | 0.364 +- 0.049 | 0.356 +- 0.010 | 0.203 | 0.384 | 0.568 |
| - multi-view (prompt only) | 0.364 | 0.364 +- 0.042 | 0.341 +- 0.007 | 0.197 | 0.410 | 0.595 |
| - multi-view (single stream prompt+snapshot) | 0.368 | 0.368 +- 0.052 | 0.358 +- 0.004 | 0.200 | 0.411 | 0.593 |
| - late interaction (M = 1, mean pooled) | 0.387 | 0.387 +- 0.035 | 0.362 +- 0.009 | 0.222 | 0.422 | 0.598 |
| - cost weighting (alpha = beta = 0) | 0.370 | 0.370 +- 0.031 | 0.366 +- 0.014 | 0.192 | 0.401 | 0.581 |
| - structural prior | 0.369 | 0.369 +- 0.044 | 0.350 +- 0.004 | 0.211 | 0.405 | 0.584 |
| - cross-file negatives | 0.404 | 0.404 +- 0.052 | 0.385 +- 0.001 | 0.235 | 0.403 | 0.583 |
| - diagonal re-weighting (low-rank only) | 0.367 | 0.367 +- 0.038 | 0.357 +- 0.001 | 0.197 | 0.402 | 0.590 |
| - low-rank residual (diagonal only) | 0.350 | 0.350 +- 0.035 | 0.349 +- 0.004 | 0.185 | 0.373 | 0.566 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.354 | 0.354 +- 0.052 | 0.355 +- 0.005 | 0.189 | 0.392 | 0.573 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.056 [+0.032, +0.079]
- full Cortex-SubNN vs v1 include features, logistic: +0.058 [+0.032, +0.084]
- full Cortex-SubNN vs random: +0.087 [+0.061, +0.114]
- full vs - gate (uniform over present views): +0.002 [-0.011, +0.016]
- full vs - multi-view (prompt only): +0.003 [-0.016, +0.023]
- full vs - multi-view (single stream prompt+snapshot): -0.001 [-0.016, +0.014]
- full vs - late interaction (M = 1, mean pooled): -0.021 [-0.041, -0.000]
- full vs - cost weighting (alpha = beta = 0): -0.003 [-0.022, +0.015]
- full vs - structural prior: -0.002 [-0.019, +0.015]
- full vs - cross-file negatives: -0.037 [-0.056, -0.018]
- full vs - diagonal re-weighting (low-rank only): -0.001 [-0.017, +0.016]
- full vs - low-rank residual (diagonal only): +0.017 [-0.004, +0.038]
- full vs v2 bottleneck head (k = 32, int8, cosine skip): +0.013 [-0.010, +0.035]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (511) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.236 | 0.199 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.337 | 0.298 |
| v1 include features, logistic | 0.293 | 0.320 |
| frozen zero-shot (max sub-span cosine) | 0.331 | 0.297 |
| full Cortex-SubNN | 0.405 | 0.341 |
| - gate (uniform over present views) | 0.401 | 0.340 |
| - multi-view (prompt only) | 0.383 | 0.351 |
| - multi-view (single stream prompt+snapshot) | 0.408 | 0.341 |
| - late interaction (M = 1, mean pooled) | 0.402 | 0.378 |
| - cost weighting (alpha = beta = 0) | 0.391 | 0.356 |
| - structural prior | 0.427 | 0.330 |
| - cross-file negatives | 0.448 | 0.374 |
| - diagonal re-weighting (low-rank only) | 0.386 | 0.355 |
| - low-rank residual (diagonal only) | 0.374 | 0.334 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.370 | 0.343 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 275

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.292 | 0.091 |
| smallest first | 0.184 | 0.047 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.411 | 0.233 |
| frozen zero-shot (max sub-span cosine) | 0.360 | 0.193 |
| full Cortex-SubNN | 0.493 | 0.324 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.5299999713897705, 0.4300000071525574, 0.03999999910593033], without [0.7200000286102295, 0.0, 0.2800000011920929].

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
| full Cortex-SubNN | 38% | 55% | 36% | 16% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen zero-shot (max sub-span cosine) | 38% | 57% | 36% | 16% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 488% | 412% | 95% | 92% | 15% |
| v1 include features, logistic | 488% | 412% | 95% | 92% | 15% |
| frozen zero-shot (max sub-span cosine) | 488% | 430% | 95% | 92% | 15% |
| smallest first | 488% | 412% | 95% | 91% | 15% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 414% | 353% | 94% | 91% | 15% |
| v1 include features, logistic | 414% | 353% | 94% | 91% | 15% |
| frozen zero-shot (max sub-span cosine) | 414% | 369% | 94% | 91% | 15% |
| smallest first | 414% | 353% | 94% | 90% | 15% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 52% | 66% | 38% | 40% | 16% |
| v1 include features, logistic | 52% | 67% | 39% | 41% | 16% |
| frozen zero-shot (max sub-span cosine) | 52% | 68% | 38% | 41% | 16% |
| smallest first | 52% | 67% | 38% | 40% | 16% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.287 |
| smallest first | 0.253 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.310 |
| v1 include features, logistic | 0.298 |
| frozen zero-shot (max sub-span cosine) | 0.340 |
| full Cortex-SubNN | 0.286 +- 0.011 (3 seeds) |
