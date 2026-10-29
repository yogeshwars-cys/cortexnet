# SWE-bench Lite, openai_llama_3_3_70b_read: whole vs cortex vs coproc

Paired over the 40 instances every arm finished (of 40; the Glimmer A/B's set). Success = the submitted file is in the gold patch / the submitted symbol overlaps its changed lines (no tests run). Ratios against whole-file reads with 95% intervals from a paired bootstrap over instances.

| metric | whole | cortex | coproc | cortex / whole | coproc / whole |
|---|---|---|---|---|---|
| input tokens (sum over turns) (mean) | 16,218 | 9,357 | 9,818 | 0.58 [0.38, 0.90] | 0.61 [0.44, 0.83] |
| of which cached (mean) | 6,603 | 6,875 | 7,778 | 1.04 [0.61, 1.74] | 1.18 [0.75, 1.73] |
| output tokens (incl. reasoning) (mean) | 179 | 166 | 178 | 0.93 [0.81, 1.06] | 0.99 [0.84, 1.19] |
| cost (USD) (mean) | 0.0000 | 0.0000 | 0.0000 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| peak context (mean) | 10,326 | 3,267 | 3,695 | 0.32 [0.25, 0.41] | 0.36 [0.28, 0.46] |
| file tokens put into context (mean) | 7,033 | 1,323 | 1,648 | 0.19 [0.12, 0.26] | 0.23 [0.16, 0.32] |
| turns (mean) | 4 | 4 | 5 | 1.01 [0.87, 1.19] | 1.09 [0.91, 1.34] |
| tool calls (mean) | 4 | 4 | 5 | 1.02 [0.87, 1.22] | 1.12 [0.93, 1.39] |
| submitted the gold file | 40% | 42% | 42% | | |
| submitted the gold symbol | 12% | 8% | 10% | | |
| hit the context cap | 0% | 0% | 0% | | |

- gold symbol, cortex vs whole: 0 instances only cortex solved, 2 only whole (exact McNemar p = 0.50)
- gold symbol, coproc vs whole: 1 instances only coproc solved, 2 only whole (exact McNemar p = 1.00)
