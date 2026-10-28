# Proxy A/B: agy, task `audit`

2026-10-28 01:18; arms alternate order per trial. Input (weighted) is in the provider's units (see bench/proxy_ab.py `cost`). Check the saved answers before reading token savings as a win.

| run | arm | exit | requests | rewrites | saved at read | input (weighted) | output | seconds | top tools |
|---|---|---|---|---|---|---|---|---|---|
| 1 | cortex | 0 | 73 | 0 | 0 | 608,014 | 74,912 | 745.5 | context-mode_context-mode.ctx_execute 36, run_command 1, context-mode.ctx_execute 1 |
| 2 | passthrough | 0 | 78 | 0 | 0 | 766,164 | 75,084 | 824.9 | context-mode_context-mode.ctx_execute 41, context-mode.ctx_execute 1, run_command 1 |
| 3 | passthrough | 0 | 85 | 0 | 0 | 774,344 | 53,412 | 653.2 | context-mode_context-mode.ctx_execute 41, read_file 9, run_command 1, context-mode.ctx_execute 1 |
| 4 | cortex | 0 | 72 | 0 | 0 | 648,406 | 44,978 | 549.7 | context-mode_context-mode.ctx_execute 42, context-mode_context-mode.ctx_batch_execute 2 |

| arm | runs | mean requests | mean input (weighted) | mean output | mean seconds |
|---|---|---|---|---|---|
| cortex | 2 | 72.5 | 628,210 | 59,945 | 648 |
| passthrough | 2 | 81.5 | 770,254 | 64,248 | 739 |
