# Offline read evaluation: SWE-bench Lite read cases

139 read_file cases from the Glimmer A/B episodes (107 with gold-patch lines in the read file), one read each, query = issue text, snapshot = the agent's reasoning before the read. Coprocessor head `head_base_c64.npz` (trained without SWE data). (412s)

| configuration | gold-line recall (mean) | reads with all gold lines visible | needed symbol visible | tokens per read (mean) | share of the whole file (median) |
|---|---|---|---|---|---|
| v1 | 0.273 | 21% | 45% | 2,214 | 21% |
| coprocessor | 0.324 | 23% | 59% | 2,298 | 23% |
| coprocessor top-3 | 0.324 | 23% | 67% | 2,497 | 27% |

Paired differences against `v1`, 95% intervals from a bootstrap over the 40 tasks (reads of one task resampled together, since they are not independent):

- coprocessor, gold-line recall: +0.051 [+0.005, +0.123]
- coprocessor, tokens per read: +84.604 [+54.639, +115.511]
- coprocessor top-3, gold-line recall: +0.051 [+0.005, +0.118]
- coprocessor top-3, tokens per read: +283.734 [+225.308, +342.684]
