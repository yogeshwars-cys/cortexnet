# SWE-bench Lite, baseten_muse_spark_open_wide: whole vs cortex vs coproc

Paired over the 39 instances every arm finished (of 40; the Glimmer A/B's set). Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.

| metric | whole | cortex | coproc | cortex / whole | coproc / whole |
|---|---|---|---|---|---|
| input tokens (sum over turns) (mean) | 702,442 | 645,695 | 523,837 | 0.92 [0.73, 1.18] | 0.75 [0.54, 1.03] |
| of which cached (mean) | 610,307 | 596,697 | 469,067 | 0.98 [0.75, 1.29] | 0.77 [0.54, 1.11] |
| output tokens (incl. reasoning) (mean) | 32,151 | 46,911 | 43,888 | 1.46 [1.20, 1.80] | 1.37 [1.06, 1.74] |
| cost (USD) (mean) | 0.0000 | 0.0000 | 0.0000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| peak context (mean) | 85,070 | 39,472 | 37,031 | 0.46 [0.40, 0.53] | 0.44 [0.38, 0.50] |
| file tokens put into context (mean) | 74,317 | 29,739 | 27,776 | 0.40 [0.34, 0.47] | 0.37 [0.32, 0.44] |
| turns (mean) | 14 | 25 | 21 | 1.85 [1.53, 2.27] | 1.51 [1.18, 1.93] |
| tool calls (mean) | 18 | 30 | 26 | 1.70 [1.45, 2.00] | 1.47 [1.19, 1.77] |
| submitted the gold file | 54% | 15% | 18% | | |
| submitted the gold symbol | 38% | 10% | 10% | | |
| hit the context cap | 23% | 0% | 0% | | |

- gold symbol, cortex vs whole: 0 instances only cortex solved, 11 only whole (exact McNemar p = 0.00)
- gold symbol, coproc vs whole: 1 instances only coproc solved, 12 only whole (exact McNemar p = 0.00)
