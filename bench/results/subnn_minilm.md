# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: MiniLM-L6 (384 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (854 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (722s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.280 | 0.280 +- 0.024 | - | 0.118 | 0.356 | 0.549 |
| smallest first | 0.214 | 0.214 +- 0.028 | - | 0.104 | 0.390 | 0.556 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.313 | 0.313 +- 0.028 | - | 0.158 | 0.387 | 0.566 |
| v1 include features, logistic | 0.309 | 0.309 +- 0.037 | - | 0.163 | 0.330 | 0.498 |
| frozen zero-shot (max sub-span cosine) | 0.348 | 0.348 +- 0.041 | - | 0.184 | 0.391 | 0.573 |
| full Cortex-SubNN | 0.374 | 0.374 +- 0.070 | 0.370 +- 0.002 | 0.207 | 0.413 | 0.584 |
| - gate (uniform over present views) | 0.385 | 0.385 +- 0.072 | 0.370 +- 0.002 | 0.221 | 0.425 | 0.598 |
| - multi-view (prompt only) | 0.353 | 0.353 +- 0.058 | 0.337 +- 0.004 | 0.190 | 0.399 | 0.568 |
| - multi-view (single stream prompt+snapshot) | 0.378 | 0.378 +- 0.070 | 0.375 +- 0.006 | 0.217 | 0.416 | 0.585 |
| - late interaction (M = 1, mean pooled) | 0.380 | 0.380 +- 0.042 | 0.364 +- 0.010 | 0.208 | 0.417 | 0.598 |
| - cost weighting (alpha = beta = 0) | 0.389 | 0.389 +- 0.070 | 0.373 +- 0.006 | 0.221 | 0.410 | 0.584 |
| - structural prior | 0.374 | 0.374 +- 0.057 | 0.362 +- 0.004 | 0.208 | 0.434 | 0.609 |
| - cross-file negatives | 0.416 | 0.416 +- 0.054 | 0.402 +- 0.004 | 0.248 | 0.411 | 0.596 |
| - diagonal re-weighting (low-rank only) | 0.383 | 0.383 +- 0.072 | 0.373 +- 0.007 | 0.215 | 0.420 | 0.588 |
| - low-rank residual (diagonal only) | 0.368 | 0.368 +- 0.051 | 0.368 +- 0.000 | 0.194 | 0.396 | 0.588 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.407 | 0.407 +- 0.051 | 0.379 +- 0.007 | 0.251 | 0.442 | 0.614 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.026 [+0.006, +0.047]
- full Cortex-SubNN vs v1 include features, logistic: +0.065 [+0.038, +0.092]
- full Cortex-SubNN vs random: +0.094 [+0.066, +0.122]
- full vs - gate (uniform over present views): -0.011 [-0.019, -0.002]
- full vs - multi-view (prompt only): +0.021 [-0.001, +0.043]
- full vs - multi-view (single stream prompt+snapshot): -0.004 [-0.013, +0.005]
- full vs - late interaction (M = 1, mean pooled): -0.007 [-0.027, +0.013]
- full vs - cost weighting (alpha = beta = 0): -0.015 [-0.027, -0.004]
- full vs - structural prior: +0.000 [-0.011, +0.011]
- full vs - cross-file negatives: -0.042 [-0.061, -0.023]
- full vs - diagonal re-weighting (low-rank only): -0.009 [-0.019, +0.001]
- full vs - low-rank residual (diagonal only): +0.006 [-0.016, +0.027]
- full vs v2 bottleneck head (k = 32, int8, cosine skip): -0.033 [-0.056, -0.010]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (511) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.236 | 0.199 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.337 | 0.298 |
| v1 include features, logistic | 0.293 | 0.320 |
| frozen zero-shot (max sub-span cosine) | 0.383 | 0.325 |
| full Cortex-SubNN | 0.416 | 0.346 |
| - gate (uniform over present views) | 0.428 | 0.356 |
| - multi-view (prompt only) | 0.350 | 0.355 |
| - multi-view (single stream prompt+snapshot) | 0.424 | 0.347 |
| - late interaction (M = 1, mean pooled) | 0.436 | 0.343 |
| - cost weighting (alpha = beta = 0) | 0.424 | 0.366 |
| - structural prior | 0.410 | 0.350 |
| - cross-file negatives | 0.439 | 0.400 |
| - diagonal re-weighting (low-rank only) | 0.419 | 0.358 |
| - low-rank residual (diagonal only) | 0.393 | 0.352 |
| v2 bottleneck head (k = 32, int8, cosine skip) | 0.447 | 0.380 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 275

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.292 | 0.091 |
| smallest first | 0.184 | 0.047 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.411 | 0.233 |
| frozen zero-shot (max sub-span cosine) | 0.367 | 0.160 |
| full Cortex-SubNN | 0.447 | 0.269 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.15000000596046448, 0.7900000214576721, 0.07000000029802322], without [0.5, 0.0, 0.5].

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
| full Cortex-SubNN | 38% | 54% | 36% | 15% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen zero-shot (max sub-span cosine) | 38% | 55% | 36% | 16% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 488% | 392% | 95% | 91% | 15% |
| v1 include features, logistic | 488% | 412% | 95% | 92% | 15% |
| frozen zero-shot (max sub-span cosine) | 488% | 392% | 95% | 92% | 15% |
| smallest first | 488% | 412% | 95% | 91% | 15% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 414% | 336% | 94% | 90% | 15% |
| v1 include features, logistic | 414% | 353% | 94% | 91% | 15% |
| frozen zero-shot (max sub-span cosine) | 414% | 336% | 94% | 91% | 15% |
| smallest first | 414% | 353% | 94% | 90% | 15% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 52% | 65% | 38% | 41% | 16% |
| v1 include features, logistic | 52% | 67% | 39% | 41% | 16% |
| frozen zero-shot (max sub-span cosine) | 52% | 65% | 38% | 41% | 16% |
| smallest first | 52% | 67% | 38% | 40% | 16% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.287 |
| smallest first | 0.253 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.310 |
| v1 include features, logistic | 0.298 |
| frozen zero-shot (max sub-span cosine) | 0.329 |
| full Cortex-SubNN | 0.343 +- 0.006 (3 seeds) |
