# Dense snapshot similarity (zero-shot proxy for a learned intent encoder)

Transitions with a non-empty snapshot: train 1257, test 874; 3291 distinct candidate files.

## all (874)

| ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| random | 0.03 | 0.10 | 0.09 |
| recency (LRU) | 0.17 | 0.47 | 0.31 |
| dense snapshot-file similarity alone | 0.09 | 0.29 | 0.19 |
| logistic: recency + frequency + same dir | 0.19 | 0.54 | 0.35 |
| logistic: + dense similarity | 0.22 | 0.55 | 0.36 |

## re-opens (657)

| ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| random | 0.03 | 0.09 | 0.08 |
| recency (LRU) | 0.22 | 0.61 | 0.40 |
| dense snapshot-file similarity alone | 0.08 | 0.28 | 0.18 |
| logistic: recency + frequency + same dir | 0.26 | 0.70 | 0.44 |
| logistic: + dense similarity | 0.29 | 0.71 | 0.46 |

## first-time opens (217)

| ranker | top-1 | top-5 | MRR |
|---|---|---|---|
| random | 0.04 | 0.12 | 0.11 |
| recency (LRU) | 0.01 | 0.04 | 0.05 |
| dense snapshot-file similarity alone | 0.11 | 0.31 | 0.21 |
| logistic: recency + frequency + same dir | 0.01 | 0.06 | 0.06 |
| logistic: + dense similarity | 0.01 | 0.06 | 0.06 |

Weights (standardized; recency, frequency, same dir, dense): -0.91, +0.27, +0.39, +0.32
