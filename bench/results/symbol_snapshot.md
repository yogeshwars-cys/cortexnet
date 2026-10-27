# Symbol-level snapshot signal: which symbols of a file will the model need?

8 full reads of outline-size files with a later use of specific lines (edit or ranged re-read) within 30 tool calls; 5 have a non-empty snapshot. Median 12 leaf symbols per file, 16% of them needed on average. Test = later half of the cases with a snapshot (3); budget = max(400, 25% of file tokens).

| ranker | recall of needed symbols in budget | cases with >= 1 needed symbol shown | MRR |
|---|---|---|---|
| random | 67% | 67% | 0.57 |
| smallest first (size only) | 33% | 33% | 0.13 |
| lexical overlap with snapshot | 33% | 33% | 0.56 |
| snapshot names the symbol | 33% | 33% | 0.13 |
| dense similarity to snapshot | 33% | 33% | 0.36 |
| dense similarity per sqrt(token) | 33% | 33% | 0.14 |
| logistic: size + lexical + name | 33% | 33% | 0.70 |
| logistic: + dense | 33% | 33% | 0.70 |

Logistic weights (standardized; log size, lexical, name, dense): +0.68, -0.03, +0.00, +0.00
