# Offline read evaluation: SWE-bench Lite read cases

139 read_file cases from the Glimmer A/B episodes (107 with gold-patch lines in the read file), one read each, query = issue text, snapshot = the agent's reasoning before the read. Coprocessor head `head_swe_c64.npz` (trained without SWE data). (387s)

| configuration | gold-line recall (mean) | reads with all gold lines visible | needed symbol visible | tokens per read (mean) | share of the whole file (median) |
|---|---|---|---|---|---|
| v1 | 0.273 | 21% | 45% | 2,214 | 21% |
| coprocessor | 0.306 | 22% | 50% | 2,278 | 23% |
| coprocessor top-3 | 0.376 | 28% | 71% | 2,439 | 28% |

Paired differences against `v1`, 95% intervals from a bootstrap over the 40 tasks (reads of one task resampled together, since they are not independent):

- coprocessor, gold-line recall: +0.033 [+0.000, +0.084]
- coprocessor, tokens per read: +64.734 [+40.785, +91.224]
- coprocessor top-3, gold-line recall: +0.103 [+0.023, +0.198]
- coprocessor top-3, tokens per read: +225.633 [+174.866, +273.610]
