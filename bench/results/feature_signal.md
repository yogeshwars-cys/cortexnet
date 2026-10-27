# Feature signal: do the trail and the symbol graph predict what gets edited?

20 outline-mode reads followed by an edit in the same file; 320 leaf symbols, 64 later edited before the file was read again. Features as computed at read time.

## Single-feature AUC (pooled)

| feature | AUC | share of rows non-zero |
|---|---|---|
| query_name | 0.490 | 2% |
| query_overlap | 0.615 | 74% |
| centrality | 0.424 | 26% |
| rarity | 0.441 | 46% |
| log_tokens | 0.735 | 100% |
| kind_def | 0.402 | 72% |
| kind_class | 0.498 | 0% |
| kind_const | 0.482 | 4% |
| kind_imports | 0.492 | 1% |
| kind_section | 0.625 | 22% |
| depth | 0.650 | 40% |
| trail | 0.661 | 78% |
| conn | 0.496 | 16% |

## Fitted on one half of the sessions (by time), evaluated on the other

| features | held-out AUC (A->B, B->A) | edited symbols shown at the Reader's budget |
|---|---|---|
| v1 (no trail, no graph) | 0.780, 0.753 | 18% |
| all (with trail + graph) | 0.783, 0.685 | 19% |
| prior weights (shipped) | 0.392, 0.421 | 49% |
| smallest symbols first (size only) | - | 45% |
| random symbols (mean of 20) | - | 33% |
