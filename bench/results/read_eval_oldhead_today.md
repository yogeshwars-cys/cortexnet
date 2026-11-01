# Offline read evaluation: SWE-bench Lite read cases

139 read_file cases from the Glimmer A/B episodes (107 with gold-patch lines in the read file), one read each, query = issue text, snapshot = the agent's reasoning before the read. Coprocessor head `head_minilm_c64_v2.npz` (trained without SWE data). (395s)

| configuration | gold-line recall (mean) | reads with all gold lines visible | needed symbol visible | tokens per read (mean) | share of the whole file (median) |
|---|---|---|---|---|---|
| v1 | 0.273 | 21% | 45% | 2,214 | 21% |
| coprocessor | 0.329 | 26% | 53% | 2,305 | 23% |
| coprocessor top-3 | 0.403 | 29% | 71% | 2,529 | 28% |

Paired differences against `v1`, 95% intervals from a bootstrap over the 40 tasks (reads of one task resampled together, since they are not independent):

- coprocessor, gold-line recall: +0.056 [+0.005, +0.133]
- coprocessor, tokens per read: +91.504 [+59.144, +127.613]
- coprocessor top-3, gold-line recall: +0.129 [+0.046, +0.221]
- coprocessor top-3, tokens per read: +315.662 [+244.496, +393.960]
