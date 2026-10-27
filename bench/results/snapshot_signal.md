# Snapshot signal: does the model's reasoning before a call predict the next file it opens?

3651 file-to-file transitions in 99 sessions. Candidates per transition: median 76 files. Snapshot (thinking + text since the previous call) non-empty in 58%. First-time opens: 25%. Logistic fits on the first half of sessions by time (2028), tested on the second (1623).

## all test transitions (1623)

| ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| random | 0.02 | 0.10 | 0.09 |
| recency (LRU) | 0.18 | 0.48 | 0.32 |
| snapshot names a symbol defined in the file | 0.17 | 0.46 | 0.31 |
| snapshot names the file | 0.18 | 0.50 | 0.33 |
| logistic: recency + frequency + same dir | 0.19 | 0.53 | 0.34 |
| logistic: + snapshot (symbol + file mention) | 0.19 | 0.54 | 0.35 |

## re-opens (1184)

| ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| random | 0.02 | 0.09 | 0.08 |
| recency (LRU) | 0.24 | 0.63 | 0.42 |
| snapshot names a symbol defined in the file | 0.23 | 0.59 | 0.39 |
| snapshot names the file | 0.24 | 0.63 | 0.41 |
| logistic: recency + frequency + same dir | 0.26 | 0.68 | 0.44 |
| logistic: + snapshot (symbol + file mention) | 0.26 | 0.70 | 0.45 |

## first-time opens (439)

| ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| random | 0.04 | 0.14 | 0.11 |
| recency (LRU) | 0.00 | 0.07 | 0.06 |
| snapshot names a symbol defined in the file | 0.02 | 0.10 | 0.08 |
| snapshot names the file | 0.03 | 0.12 | 0.10 |
| logistic: recency + frequency + same dir | 0.00 | 0.10 | 0.07 |
| logistic: + snapshot (symbol + file mention) | 0.00 | 0.10 | 0.07 |

The snapshot names the opened file or one of its symbols in 11% of test transitions (23% of those are first-time opens).
Logistic weights (standardized; recency, frequency, symbol mention, file mention, same dir): -0.92, +0.20, +0.02, +0.10, +0.43
