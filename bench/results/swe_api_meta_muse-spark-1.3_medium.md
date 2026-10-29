# SWE-bench Lite, meta_muse-spark-1.3_medium: whole vs cortex vs coproc

Paired over the 40 instances every arm finished (of 40; the Glimmer A/B's set). Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.

| metric | whole | cortex | coproc | cortex / whole | coproc / whole |
|---|---|---|---|---|---|
| input tokens (sum over turns) (mean) | 231,318 | 135,187 | 140,688 | 0.58 [0.49, 0.71] | 0.61 [0.52, 0.73] |
| of which cached (mean) | 98,778 | 65,841 | 69,121 | 0.67 [0.55, 0.85] | 0.70 [0.56, 0.93] |
| output tokens (incl. reasoning) (mean) | 8,643 | 8,349 | 9,155 | 0.97 [0.78, 1.19] | 1.06 [0.81, 1.42] |
| cost (USD) (mean) | 0.2172 | 0.1320 | 0.1387 | 0.61 [0.51, 0.73] | 0.64 [0.54, 0.77] |
| peak context (mean) | 33,327 | 17,964 | 19,292 | 0.54 [0.46, 0.64] | 0.58 [0.49, 0.68] |
| file tokens put into context (mean) | 27,069 | 12,778 | 13,613 | 0.47 [0.39, 0.58] | 0.50 [0.41, 0.62] |
| turns (mean) | 11 | 12 | 12 | 1.09 [0.99, 1.20] | 1.09 [0.98, 1.23] |
| tool calls (mean) | 13 | 15 | 15 | 1.12 [1.04, 1.23] | 1.12 [1.03, 1.22] |
| submitted the gold file | 68% | 55% | 57% | | |
| submitted the gold symbol | 35% | 38% | 32% | | |
| hit the context cap | 10% | 0% | 0% | | |

- gold symbol, cortex vs whole: 2 instances only cortex solved, 1 only whole (exact McNemar p = 1.00)
- gold symbol, coproc vs whole: 1 instances only coproc solved, 2 only whole (exact McNemar p = 1.00)
