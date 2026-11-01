# Offline read evaluation: SWE-bench Lite read cases

139 read_file cases from the Glimmer A/B episodes (107 with gold-patch lines in the read file), one read each, query = issue text, snapshot = the agent's reasoning before the read. Coprocessor head `head_minilm_c64_v2.npz` (trained without SWE data). (42s)

| configuration | gold-line recall (mean) | reads with all gold lines visible | needed symbol visible | tokens per read (mean) | share of the whole file (median) |
|---|---|---|---|---|---|
| v1 | 0.273 | 21% | 45% | 2,214 | 21% |
| v1 no anchors | 0.312 | 23% | 40% | 2,199 | 19% |

Paired differences against `v1`, 95% intervals from a bootstrap over the 40 tasks (reads of one task resampled together, since they are not independent):

- v1 no anchors, gold-line recall: +0.038 [-0.055, +0.143]
- v1 no anchors, tokens per read: -14.381 [-88.304, +62.914]
