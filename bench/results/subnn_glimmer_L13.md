# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: Muse Glimmer 30B layer 13 (6656 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (854 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (705s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.280 | 0.280 +- 0.024 | - | 0.118 | 0.356 | 0.549 |
| smallest first | 0.214 | 0.214 +- 0.028 | - | 0.104 | 0.390 | 0.556 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.313 | 0.313 +- 0.028 | - | 0.158 | 0.387 | 0.566 |
| v1 include features, logistic | 0.309 | 0.309 +- 0.037 | - | 0.163 | 0.330 | 0.498 |
| frozen zero-shot (max sub-span cosine) | 0.334 | 0.334 +- 0.038 | - | 0.171 | 0.346 | 0.539 |
| full Cortex-SubNN | 0.361 | 0.361 +- 0.061 | 0.348 +- 0.011 | 0.200 | 0.417 | 0.589 |
| - gate (uniform over present views) | 0.357 | 0.357 +- 0.057 | 0.341 +- 0.003 | 0.200 | 0.379 | 0.570 |
| - multi-view (prompt only) | 0.341 | 0.341 +- 0.042 | 0.333 +- 0.006 | 0.171 | 0.412 | 0.594 |
| - multi-view (single stream prompt+snapshot) | 0.354 | 0.354 +- 0.060 | 0.346 +- 0.009 | 0.189 | 0.391 | 0.576 |
| - late interaction (M = 1, mean pooled) | 0.353 | 0.353 +- 0.056 | 0.341 +- 0.007 | 0.196 | 0.395 | 0.569 |
| - cost weighting (alpha = beta = 0) | 0.377 | 0.377 +- 0.051 | 0.366 +- 0.003 | 0.207 | 0.414 | 0.583 |
| - structural prior | 0.344 | 0.344 +- 0.053 | 0.340 +- 0.006 | 0.174 | 0.387 | 0.576 |
| - cross-file negatives | 0.408 | 0.408 +- 0.045 | 0.387 +- 0.008 | 0.237 | 0.384 | 0.560 |
| - diagonal re-weighting (low-rank only) | 0.364 | 0.364 +- 0.042 | 0.346 +- 0.006 | 0.207 | 0.414 | 0.585 |
| - low-rank residual (diagonal only) | 0.331 | 0.331 +- 0.034 | 0.330 +- 0.003 | 0.167 | 0.369 | 0.563 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.350 | 0.350 +- 0.056 | 0.337 +- 0.008 | 0.186 | 0.382 | 0.575 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.027 [+0.003, +0.050]
- full Cortex-SubNN vs v1 include features, logistic: +0.052 [+0.026, +0.079]
- full Cortex-SubNN vs random: +0.081 [+0.054, +0.109]
- full vs - gate (uniform over present views): +0.003 [-0.014, +0.021]
- full vs - multi-view (prompt only): +0.019 [-0.000, +0.041]
- full vs - multi-view (single stream prompt+snapshot): +0.007 [-0.009, +0.023]
- full vs - late interaction (M = 1, mean pooled): +0.008 [-0.013, +0.029]
- full vs - cost weighting (alpha = beta = 0): -0.016 [-0.034, +0.005]
- full vs - structural prior: +0.017 [-0.001, +0.036]
- full vs - cross-file negatives: -0.047 [-0.069, -0.025]
- full vs - diagonal re-weighting (low-rank only): -0.003 [-0.022, +0.017]
- full vs - low-rank residual (diagonal only): +0.030 [+0.008, +0.054]
- full vs v2 bottleneck head (k = 32, int8, cosine skip): +0.011 [-0.011, +0.034]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (511) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.236 | 0.199 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.337 | 0.298 |
| v1 include features, logistic | 0.293 | 0.320 |
| frozen zero-shot (max sub-span cosine) | 0.371 | 0.309 |
| full Cortex-SubNN | 0.408 | 0.329 |
| - gate (uniform over present views) | 0.384 | 0.340 |
| - multi-view (prompt only) | 0.372 | 0.321 |
| - multi-view (single stream prompt+snapshot) | 0.396 | 0.326 |
| - late interaction (M = 1, mean pooled) | 0.385 | 0.331 |
| - cost weighting (alpha = beta = 0) | 0.414 | 0.351 |
| - structural prior | 0.379 | 0.320 |
| - cross-file negatives | 0.448 | 0.381 |
| - diagonal re-weighting (low-rank only) | 0.403 | 0.337 |
| - low-rank residual (diagonal only) | 0.367 | 0.307 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.362 | 0.341 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 275

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.292 | 0.091 |
| smallest first | 0.184 | 0.047 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.411 | 0.233 |
| frozen zero-shot (max sub-span cosine) | 0.357 | 0.185 |
| full Cortex-SubNN | 0.453 | 0.280 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.36000001430511475, 0.5899999737739563, 0.05000000074505806], without [0.8799999952316284, 0.0, 0.11999999731779099].

## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)

Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown up front, plus one extra round trip and the missing bodies if a needed symbol was not shown. Rule A returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the Reader's current policy with that ranker. Prefix-cache regimes: an extra round trip re-reads the cached history at 0.1x (L_hist = logged context at the read) plus 400 tokens of tool-step overhead; with compounding, every token put into context also costs 0.1x on each later API call until the session ends or is compacted (H_rem from the session logs for 1190 cases, median 72; median on test cases 84; missing logs use the median). Cache writes are priced at 1x, not 1.25x.

### C_turn = 500 (no cache accounting; ~ local KV cache)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 21% | 42% | 21% | 1% | 15% |
| v1 include features, logistic | 21% | 43% | 21% | 1% | 15% |
| frozen zero-shot (max sub-span cosine) | 21% | 42% | 21% | 1% | 15% |
| smallest first | 21% | 42% | 21% | 1% | 15% |

### C_turn = 2,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 38% | 54% | 36% | 16% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen zero-shot (max sub-span cosine) | 38% | 56% | 36% | 15% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 488% | 383% | 95% | 92% | 15% |
| v1 include features, logistic | 488% | 412% | 95% | 92% | 15% |
| frozen zero-shot (max sub-span cosine) | 488% | 414% | 95% | 92% | 15% |
| smallest first | 488% | 412% | 95% | 91% | 15% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 414% | 329% | 94% | 90% | 15% |
| v1 include features, logistic | 414% | 353% | 94% | 91% | 15% |
| frozen zero-shot (max sub-span cosine) | 414% | 355% | 94% | 91% | 15% |
| smallest first | 414% | 353% | 94% | 90% | 15% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 52% | 64% | 38% | 41% | 16% |
| v1 include features, logistic | 52% | 67% | 39% | 41% | 16% |
| frozen zero-shot (max sub-span cosine) | 52% | 67% | 38% | 41% | 16% |
| smallest first | 52% | 67% | 38% | 40% | 16% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.287 |
| smallest first | 0.253 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.310 |
| v1 include features, logistic | 0.298 |
| frozen zero-shot (max sub-span cosine) | 0.346 |
| full Cortex-SubNN | 0.299 +- 0.024 (3 seeds) |
