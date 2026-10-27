# Cortex-SubNN: which symbol bodies to show up front

Symbol-level cases from 91 local agent sessions plus git history (private data; only these aggregates are kept). Frozen encoder: Muse Glimmer 30B layer 26 (6656 dims). 5-fold cross-validation grouped by session: every read-time case is tested once, by a model that never saw its session (55 cases). Early stopping (which also scores the untrained step-0 model) and calibration use 15% of the training sessions. Rankings are within the file that was read (median ~20 leaf symbols); budget = the Reader's include budget, max(400, 25% of file tokens). (19s)

## Ranking (pooled over folds)

Learned rows are the mean of a seed ensemble per fold; the single-seed column shows how much one run varies. The full model starts as the frozen zero-shot ranker exactly (identity init).

| ranker | MRR | MRR per fold (mean +- sd) | single seed MRR (mean +- sd) | top-1 | recall in budget | >= 1 needed shown |
|---|---|---|---|---|---|---|
| random | 0.344 | 0.344 +- 0.056 | - | 0.200 | 0.365 | 0.800 |
| smallest first | 0.262 | 0.262 +- 0.025 | - | 0.109 | 0.498 | 0.800 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.560 | 0.560 +- 0.128 | - | 0.400 | 0.588 | 0.945 |
| v1 include features, logistic | 0.581 | 0.581 +- 0.090 | - | 0.418 | 0.457 | 0.855 |
| frozen zero-shot (max sub-span cosine) | 0.582 | 0.582 +- 0.096 | - | 0.436 | 0.362 | 0.836 |
| full Cortex-SubNN | 0.640 | 0.640 +- 0.076 | 0.616 +- 0.016 | 0.491 | 0.460 | 0.891 |

Paired bootstrap over cases (MRR difference, 95% interval):

- full Cortex-SubNN vs frozen zero-shot (max sub-span cosine): +0.057 [-0.022, +0.140]
- full Cortex-SubNN vs v1 include features, logistic: +0.058 [-0.051, +0.168]
- full Cortex-SubNN vs random: +0.296 [+0.194, +0.404]

## By snapshot presence (MRR)

| ranker | with snapshot (55) | without (0) |
|---|---|---|
| random | 0.344 | nan |
| smallest first | 0.262 | nan |
| lexical (name + identifier overlap with prompt/snapshot) | 0.560 | nan |
| v1 include features, logistic | 0.581 | nan |
| frozen zero-shot (max sub-span cosine) | 0.582 | nan |
| full Cortex-SubNN | 0.640 | nan |

## Gate (full model, held-out cases)

Mean weight on (prompt, snapshot, trail): with a snapshot [0.5400000214576721, 0.33000001311302185, 0.14000000059604645], without [nan, nan, nan].

## Decision head: realised tokens on held-out cases (whole-file reads = 100%; lower is better)

Each ranker is Platt-calibrated per fold on the validation sessions. A case costs the outline + bodies shown up front, plus one extra round trip and the missing bodies if a needed symbol was not shown. Rule A returns the whole file when that is cheaper in expectation; Rule B picks the up-front set; budget fill is the Reader's current policy with that ranker. Prefix-cache regimes: an extra round trip re-reads the cached history at 0.1x (L_hist = logged context at the read) plus 400 tokens of tool-step overhead; with compounding, every token put into context also costs 0.1x on each later API call until the session ends or is compacted (H_rem from the session logs for 55 cases, median 7; median on test cases 7; missing logs use the median). Cache writes are priced at 1x, not 1.25x.

### C_turn = 500 (no cache accounting; ~ local KV cache)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 27% | 48% | 27% | 2% | 23% |
| v1 include features, logistic | 27% | 48% | 27% | 2% | 23% |
| frozen zero-shot (max sub-span cosine) | 27% | 48% | 27% | 2% | 23% |
| smallest first | 27% | 49% | 27% | 2% | 23% |

### C_turn = 2,000

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 40% | 59% | 37% | 22% | 23% |
| v1 include features, logistic | 40% | 58% | 37% | 22% | 23% |
| frozen zero-shot (max sub-span cosine) | 40% | 59% | 37% | 22% | 23% |
| smallest first | 40% | 61% | 37% | 22% | 23% |

### previous: C_turn = logged context x 0.12, no compounding

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 27% | 47% | 26% | 5% | 23% |
| v1 include features, logistic | 27% | 47% | 26% | 5% | 23% |
| frozen zero-shot (max sub-span cosine) | 27% | 48% | 26% | 5% | 23% |
| smallest first | 27% | 49% | 26% | 5% | 23% |

### prefix cache, no compounding: T = 0.1 L_hist + 400

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 30% | 50% | 29% | 9% | 23% |
| v1 include features, logistic | 30% | 49% | 29% | 7% | 23% |
| frozen zero-shot (max sub-span cosine) | 30% | 50% | 29% | 9% | 23% |
| smallest first | 30% | 51% | 29% | 9% | 23% |

### prefix cache + compounding: T = 0.1 L_hist + 400, context tokens x (1 + 0.1 H_rem)

| ranker | outline only | budget fill | Rule A/B | whole-file bypass | oracle |
|---|---|---|---|---|---|
| full Cortex-SubNN | 27% | 48% | 27% | 4% | 24% |
| v1 include features, logistic | 27% | 48% | 27% | 4% | 24% |
| frozen zero-shot (max sub-span cosine) | 27% | 48% | 27% | 4% | 24% |
| smallest first | 27% | 49% | 27% | 4% | 24% |

## Secondary: time split (train on the past, test on the most recent 11 read-time cases)

| ranker | MRR |
|---|---|
| random | 0.339 |
| smallest first | 0.310 |
| lexical (name + identifier overlap with prompt/snapshot) | 0.612 |
| v1 include features, logistic | 0.547 |
| frozen zero-shot (max sub-span cosine) | 0.534 |
| full Cortex-SubNN | 0.746 +- 0.000 (1 seeds) |
