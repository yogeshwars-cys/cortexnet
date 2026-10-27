# Git-history include test: does intent text pick the symbols that get changed?

482 (commit, file) cases from 16 repos (files > 800 tokens; median 10 leaf symbols per file, 30% changed on average). The commit message names a changed symbol in 7% of cases (leakage: messages are written after the change). Budget = max(400, 25% of file tokens). Repo-level split, two folds, weighted mean.

| ranker | recall of changed symbols in budget | cases with >= 1 changed symbol shown | MRR |
|---|---|---|---|
| random | 34% | 60% | 0.49 |
| smallest first (size only) | 36% | 59% | 0.33 |
| largest first | 32% | 61% | 0.74 |
| lexical overlap with message | 45% | 71% | 0.56 |
| message names the symbol | 36% | 60% | 0.35 |
| dense similarity to message | 44% | 72% | 0.69 |
| logistic: size + lexical + name | 40% | 68% | 0.76 |
| logistic: + dense | 40% | 69% | 0.77 |
