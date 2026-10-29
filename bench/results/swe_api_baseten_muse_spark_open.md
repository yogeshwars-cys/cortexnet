# SWE-bench Lite, baseten_muse_spark_open: whole vs cortex vs coproc

Paired over the 40 instances every arm finished (of 40; the Glimmer A/B's set). Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.

| metric | whole | cortex | coproc | cortex / whole | coproc / whole |
|---|---|---|---|---|---|
| input tokens (sum over turns) (mean) | 257,374 | 173,666 | 206,476 | 0.67 [0.57, 0.79] | 0.80 [0.68, 0.94] |
| of which cached (mean) | 186,550 | 141,827 | 171,418 | 0.76 [0.62, 0.93] | 0.92 [0.76, 1.11] |
| output tokens (incl. reasoning) (mean) | 17,559 | 22,379 | 19,426 | 1.27 [0.76, 2.08] | 1.11 [0.72, 1.70] |
| cost (USD) (mean) | 0.0000 | 0.0000 | 0.0000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| peak context (mean) | 58,935 | 25,871 | 28,809 | 0.44 [0.39, 0.50] | 0.49 [0.43, 0.56] |
| file tokens put into context (mean) | 53,034 | 20,160 | 22,257 | 0.38 [0.32, 0.46] | 0.42 [0.36, 0.49] |
| turns (mean) | 8 | 12 | 12 | 1.41 [1.26, 1.57] | 1.50 [1.36, 1.67] |
| tool calls (mean) | 12 | 17 | 17 | 1.38 [1.24, 1.53] | 1.41 [1.28, 1.56] |
| submitted the gold file | 32% | 15% | 18% | | |
| submitted the gold symbol | 12% | 10% | 8% | | |
| hit the context cap | 52% | 2% | 2% | | |

- gold symbol, cortex vs whole: 2 instances only cortex solved, 3 only whole (exact McNemar p = 1.00)
- gold symbol, coproc vs whole: 1 instances only coproc solved, 3 only whole (exact McNemar p = 0.62)
