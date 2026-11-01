# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: MiniLM-L6, symbols as 64-byte codes (259 coords, planes [259, 176, 64, 11, 2], mean decoded cosine 0.954) (384 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (853 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (283s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.279 | 0.280 +- 0.022 | - | 0.121 | 0.348 | 0.545 |
| smallest first | 0.211 | 0.211 +- 0.032 | - | 0.103 | 0.390 | 0.558 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.312 | 0.311 +- 0.027 | - | 0.156 | 0.394 | 0.576 |
| v1 include features, logistic | 0.307 | 0.307 +- 0.047 | - | 0.164 | 0.334 | 0.495 |
| frozen zero-shot (max sub-span cosine) | 0.349 | 0.349 +- 0.040 | - | 0.190 | 0.394 | 0.572 |
| full Cortex-SubNN | 0.392 | 0.392 +- 0.037 | 0.392 +- 0.006 | 0.217 | 0.412 | 0.600 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.043 [+0.024, +0.062]
- full Cortex-SubNN vs v1 include features, logistic: +0.085 [+0.060, +0.110]
- full Cortex-SubNN vs random: +0.113 [+0.085, +0.139]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (510) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.233 | 0.196 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.336 | 0.295 |
| v1 include features, logistic | 0.292 | 0.317 |
| frozen zero-shot (max sub-span cosine) | 0.390 | 0.322 |
| full Cortex-SubNN | 0.420 | 0.374 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 274

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.302 | 0.106 |
| smallest first | 0.201 | 0.066 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.405 | 0.226 |
| frozen zero-shot (max sub-span cosine) | 0.360 | 0.150 |
| full Cortex-SubNN | 0.452 | 0.248 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.38999998569488525, 0.5099999904632568, 0.10999999940395355], without [0.6200000047683716, 0.0, 0.3799999952316284].

## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)

Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown up front, plus one extra round trip and the missing bodies if a needed symbol was not shown. Rule A returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the Reader's current policy with that ranker. Prefix-cache regimes: an extra round trip re-reads the cached history at 0.1x (L_hist = logged context at the read) plus 400 tokens of tool-step overhead; with compounding, every token put into context also costs 0.1x on each later API call until the session ends or is compacted (H_rem from the session logs for 1188 cases, median 72; median on test cases 84; missing logs use the median). Cache writes are priced at 1x, not 1.25x.

### C_turn = 500 (no cache accounting; ~ local KV cache)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 21% | 42% | 21% | 0% | 15% |
| v1 include features, logistic | 21% | 43% | 21% | 0% | 15% |
| frozen zero-shot (max sub-span cosine) | 21% | 42% | 21% | 0% | 15% |
| smallest first | 21% | 42% | 21% | 0% | 15% |

### C_turn = 2,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 38% | 54% | 36% | 14% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen zero-shot (max sub-span cosine) | 38% | 55% | 36% | 15% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 487% | 390% | 95% | 91% | 15% |
| v1 include features, logistic | 487% | 408% | 95% | 92% | 15% |
| frozen zero-shot (max sub-span cosine) | 487% | 390% | 95% | 92% | 15% |
| smallest first | 487% | 411% | 95% | 92% | 15% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 413% | 334% | 94% | 90% | 15% |
| v1 include features, logistic | 413% | 350% | 94% | 91% | 15% |
| frozen zero-shot (max sub-span cosine) | 413% | 335% | 94% | 91% | 15% |
| smallest first | 413% | 353% | 94% | 90% | 15% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 52% | 64% | 39% | 39% | 16% |
| v1 include features, logistic | 52% | 67% | 39% | 40% | 16% |
| frozen zero-shot (max sub-span cosine) | 52% | 65% | 39% | 39% | 16% |
| smallest first | 52% | 67% | 39% | 38% | 16% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.288 |
| smallest first | 0.247 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.306 |
| v1 include features, logistic | 0.299 |
| frozen zero-shot (max sub-span cosine) | 0.326 |
| full Cortex-SubNN | 0.356 +- 0.000 (1 seeds) |
