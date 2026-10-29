# SWE-bench Lite, openai_llama_4_maverick_v2: whole vs cortex

Paired over the 28 instances every arm finished (of 40; the Glimmer A/B's set). Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.

| metric | whole | cortex | cortex / whole |
|---|---|---|---|
| input tokens (sum over turns) (mean) | 35,984 | 26,588 | 0.74 [0.48, 1.09] |
| of which cached (mean) | 11,900 | 18,095 | 1.52 [0.99, 2.62] |
| output tokens (incl. reasoning) (mean) | 470 | 566 | 1.20 [1.06, 1.38] |
| cost (USD) (mean) | 0.0000 | 0.0000 | 0.00 [0.00, 0.00] |
| peak context (mean) | 25,243 | 9,991 | 0.40 [0.26, 0.60] |
| file tokens put into context (mean) | 28,704 | 9,006 | 0.31 [0.21, 0.46] |
| turns (mean) | 4 | 6 | 1.43 [1.18, 1.74] |
| tool calls (mean) | 8 | 11 | 1.33 [1.11, 1.59] |
| submitted the gold file | 61% | 61% | |
| submitted the gold symbol | 21% | 32% | |
| hit the context cap | 0% | 0% | |

- gold symbol, cortex vs whole: 5 instances only cortex solved, 2 only whole (exact McNemar p = 0.45)
