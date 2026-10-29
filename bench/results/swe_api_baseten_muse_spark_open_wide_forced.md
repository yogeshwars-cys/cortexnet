# SWE-bench Lite, baseten_muse_spark_open_wide_forced: whole vs cortex vs coproc

Paired over the 40 instances every arm finished (of 40; the Glimmer A/B's set). Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.

| metric | whole | cortex | coproc | cortex / whole | coproc / whole |
|---|---|---|---|---|---|
| input tokens (sum over turns) (mean) | 542,399 | 425,489 | 490,926 | 0.78 [0.63, 0.98] | 0.91 [0.71, 1.13] |
| of which cached (mean) | 464,544 | 387,286 | 443,581 | 0.83 [0.66, 1.07] | 0.95 [0.73, 1.22] |
| output tokens (incl. reasoning) (mean) | 25,788 | 32,549 | 37,838 | 1.26 [1.03, 1.58] | 1.47 [1.15, 1.88] |
| cost (USD) (mean) | 0.0000 | 0.0000 | 0.0000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| peak context (mean) | 75,511 | 33,091 | 38,161 | 0.44 [0.38, 0.51] | 0.51 [0.44, 0.59] |
| file tokens put into context (mean) | 65,698 | 24,392 | 29,086 | 0.37 [0.31, 0.44] | 0.44 [0.37, 0.53] |
| turns (mean) | 12 | 19 | 19 | 1.61 [1.36, 1.93] | 1.61 [1.33, 1.92] |
| tool calls (mean) | 16 | 24 | 25 | 1.54 [1.33, 1.81] | 1.59 [1.34, 1.86] |
| submitted the gold file | 70% | 78% | 70% | | |
| submitted the gold symbol | 30% | 48% | 40% | | |
| hit the context cap | 18% | 0% | 0% | | |

- gold symbol, cortex vs whole: 7 instances only cortex solved, 0 only whole (exact McNemar p = 0.02)
- gold symbol, coproc vs whole: 6 instances only coproc solved, 2 only whole (exact McNemar p = 0.29)
