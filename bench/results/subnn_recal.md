# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: MiniLM-L6, symbols as 64-byte codes (259 coords, planes [259, 176, 64, 11, 2], mean decoded cosine 0.954) (384 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (854 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (453s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.280 | 0.280 +- 0.024 | - | 0.118 | 0.356 | 0.549 |
| smallest first | 0.214 | 0.214 +- 0.028 | - | 0.104 | 0.390 | 0.556 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.313 | 0.313 +- 0.028 | - | 0.158 | 0.387 | 0.566 |
| v1 include features, logistic | 0.309 | 0.309 +- 0.037 | - | 0.163 | 0.330 | 0.498 |
| frozen zero-shot (max sub-span cosine) | 0.349 | 0.349 +- 0.040 | - | 0.187 | 0.404 | 0.596 |
| full Cortex-SubNN | 0.414 | 0.414 +- 0.052 | 0.403 +- 0.011 | 0.251 | 0.415 | 0.601 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.065 [+0.043, +0.087]
- full Cortex-SubNN vs v1 include features, logistic: +0.105 [+0.079, +0.130]
- full Cortex-SubNN vs random: +0.134 [+0.106, +0.163]

## By snapshot presence (MRR)

| ranker | with snapshot (343) | without (511) |
|---|---|---|
| random | 0.294 | 0.270 |
| smallest first | 0.236 | 0.199 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.337 | 0.298 |
| v1 include features, logistic | 0.293 | 0.320 |
| frozen zero-shot (max sub-span cosine) | 0.390 | 0.321 |
| full Cortex-SubNN | 0.437 | 0.398 |

## Edit-time cases (no read before the edit; views taken at the edit, so optimistic): 275

| ranker | MRR | top-1 |
|---|---|---|
| random | 0.292 | 0.091 |
| smallest first | 0.184 | 0.047 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.341 | 0.142 |
| v1 include features, logistic | 0.411 | 0.233 |
| frozen zero-shot (max sub-span cosine) | 0.356 | 0.149 |
| full Cortex-SubNN | 0.479 | 0.295 |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.25999999046325684, 0.6600000262260437, 0.07999999821186066], without [0.5799999833106995, 0.0, 0.41999998688697815].

## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)

Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown up front, plus one extra round trip and the missing bodies if a needed symbol was not shown. Rule A returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the Reader's current policy with that ranker. Prefix-cache regimes: an extra round trip re-reads the cached history at 0.1x (L_hist = logged context at the read) plus 400 tokens of tool-step overhead; with compounding, every token put into context also costs 0.1x on each later API call until the session ends or is compacted (H_rem from the session logs for 1190 cases, median 72; median on test cases 84; missing logs use the median). Cache writes are priced at 1x, not 1.25x.

### C_turn = 500 (no cache accounting; ~ local KV cache)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 21% | 41% | 21% | 1% | 15% |
| v1 include features, logistic | 21% | 43% | 21% | 1% | 15% |
| frozen zero-shot (max sub-span cosine) | 21% | 42% | 21% | 1% | 15% |
| smallest first | 21% | 42% | 21% | 1% | 15% |

### C_turn = 2,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 38% | 54% | 36% | 14% | 15% |
| v1 include features, logistic | 38% | 56% | 36% | 15% | 15% |
| frozen zero-shot (max sub-span cosine) | 38% | 54% | 36% | 15% | 15% |
| smallest first | 38% | 56% | 36% | 15% | 15% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 488% | 390% | 95% | 91% | 15% |
| v1 include features, logistic | 488% | 412% | 95% | 92% | 15% |
| frozen zero-shot (max sub-span cosine) | 488% | 394% | 95% | 92% | 15% |
| smallest first | 488% | 412% | 95% | 91% | 15% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 414% | 335% | 94% | 89% | 15% |
| v1 include features, logistic | 414% | 353% | 94% | 91% | 15% |
| frozen zero-shot (max sub-span cosine) | 414% | 338% | 94% | 91% | 15% |
| smallest first | 414% | 353% | 94% | 89% | 15% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 52% | 64% | 39% | 38% | 16% |
| v1 include features, logistic | 52% | 67% | 38% | 39% | 16% |
| frozen zero-shot (max sub-span cosine) | 52% | 65% | 38% | 39% | 16% |
| smallest first | 52% | 67% | 39% | 38% | 16% |

## Secondary: time split (train on the past, test on the most recent 171 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.287 |
| smallest first | 0.253 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.310 |
| v1 include features, logistic | 0.298 |
| frozen zero-shot (max sub-span cosine) | 0.335 |
| full Cortex-SubNN | 0.359 +- 0.006 (3 seeds) |
