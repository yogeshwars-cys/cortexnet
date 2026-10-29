# Offline read evaluation: SWE-bench Lite read cases

139 read_file cases from the Glimmer A/B episodes (107 with gold-patch lines in the read file), one read each, query = issue text, snapshot = the agent's reasoning before the read. Coprocessor head `head_minilm_c64_v2.npz` (trained without SWE data). (475s)

| configuration | gold-line recall (mean) | reads with all gold lines visible | needed symbol visible | tokens per read (mean) | share of the whole file (median) |
|---|---|---|---|---|---|
| v1 | 0.312 | 23% | 38% | 2,385 | 19% |
| v1 + x-ray | 0.341 | 23% | 66% | 2,694 | 26% |
| coprocessor | 0.396 | 32% | 55% | 2,498 | 23% |
| coprocessor + x-ray | 0.444 | 34% | 87% | 2,806 | 29% |
| coprocessor top-3 | 0.497 | 37% | 74% | 2,724 | 27% |
| coprocessor top-3 + MMR | 0.476 | 35% | 73% | 2,722 | 27% |

Paired differences against `v1`, 95% intervals from a bootstrap over the 40 tasks (reads of one task resampled together, since they are not independent):

- v1 + x-ray, gold-line recall: +0.029 [+0.006, +0.060]
- v1 + x-ray, tokens per read: +308.331 [+291.113, +324.421]
- coprocessor, gold-line recall: +0.084 [+0.014, +0.170]
- coprocessor, tokens per read: +112.511 [+78.225, +150.908]
- coprocessor + x-ray, gold-line recall: +0.133 [+0.050, +0.232]
- coprocessor + x-ray, tokens per read: +420.871 [+379.165, +466.231]
- coprocessor top-3, gold-line recall: +0.185 [+0.081, +0.307]
- coprocessor top-3, tokens per read: +338.770 [+265.500, +417.992]
- coprocessor top-3 + MMR, gold-line recall: +0.164 [+0.069, +0.279]
- coprocessor top-3 + MMR, tokens per read: +336.317 [+268.848, +412.936]
