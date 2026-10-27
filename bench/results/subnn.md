# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). 2-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (854 cases). Early stopping and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (760s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies.

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.280 | 0.280 +- 0.004 | - | 0.118 | 0.356 | 0.549 |
| smallest first | 0.214 | 0.214 +- 0.009 | - | 0.104 | 0.390 | 0.556 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.313 | 0.313 +- 0.008 | - | 0.158 | 0.387 | 0.566 |
| v1 include features, logistic | 0.309 | 0.309 +- 0.001 | - | 0.164 | 0.323 | 0.485 |
| frozen MiniLM zero-shot (max sub-span cosine) | 0.348 | 0.348 +- 0.015 | - | 0.184 | 0.391 | 0.573 |
| full Cortex-SubNN | 0.359 | 0.359 +- 0.015 | 0.353 +- 0.010 | 0.190 | 0.401 | 0.591 |
| full Cortex-SubNN + LoRA (top 2 MiniLM layers, r = 8) | 0.363 | 0.363 +- 0.013 | - | 0.203 | 0.406 | 0.602 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen MiniLM zero-shot (max sub-span cosine): +0.011 [-0.012, +0.033]
- full Cortex-SubNN vs v1 include features, logistic: +0.050 [+0.023, +0.076]
- full Cortex-SubNN vs random: +0.079 [+0.053, +0.105]
- full vs full Cortex-SubNN + LoRA (top 2 MiniLM layers, r = 8): -0.004 [-0.020, +0.012]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (511) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.236 | 0.199 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.337 | 0.298 |
| v1 include features, logistic | 0.294 | 0.319 |
| frozen MiniLM zero-shot (max sub-span cosine) | 0.383 | 0.325 |
| full Cortex-SubNN | 0.371 | 0.350 |
| full Cortex-SubNN + LoRA (top 2 MiniLM layers, r = 8) | 0.381 | 0.351 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 275

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.292 | 0.091 |
| smallest first | 0.184 | 0.047 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.416 | 0.236 |
| frozen MiniLM zero-shot (max sub-span cosine) | 0.367 | 0.160 |
| full Cortex-SubNN | 0.398 | 0.200 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.33000001311302185, 0.5199999809265137, 0.1599999964237213], without [0.6399999856948853, 0.0, 0.36000001430511475].

## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)

Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown up front, plus one extra round trip (C_turn) and the missing bodies if a needed symbol was not shown. Rule A returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the Reader's current policy with that ranker.

### C_turn = 500

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 21% | 42% | 21% | 1% | 15% |
| v1 include features, logistic | 21% | 43% | 21% | 1% | 15% |
| frozen MiniLM zero-shot (max sub-span cosine) | 21% | 42% | 21% | 1% | 15% |
| smallest first | 21% | 42% | 21% | 1% | 15% |

### C_turn = 2,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 38% | 54% | 36% | 16% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen MiniLM zero-shot (max sub-span cosine) | 38% | 55% | 36% | 16% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### C_turn = 8,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 106% | 104% | 65% | 75% | 15% |
| v1 include features, logistic | 106% | 111% | 65% | 75% | 15% |
| frozen MiniLM zero-shot (max sub-span cosine) | 106% | 106% | 65% | 74% | 15% |
| smallest first | 106% | 110% | 65% | 71% | 15% |

### C_turn = logged context x 0.12 (cached price)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 488% | 388% | 95% | 92% | 15% |
| v1 include features, logistic | 488% | 415% | 95% | 92% | 15% |
| frozen MiniLM zero-shot (max sub-span cosine) | 488% | 392% | 95% | 92% | 15% |
| smallest first | 488% | 412% | 95% | 91% | 15% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.287 |
| smallest first | 0.253 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.310 |
| v1 include features, logistic | 0.298 |
| frozen MiniLM zero-shot (max sub-span cosine) | 0.329 |
| full Cortex-SubNN | 0.338 +- 0.000 (1 seeds) |

LoRA validation MRR per fold by epoch: [[0.45], [0.32]]
