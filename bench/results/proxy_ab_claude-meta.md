# Proxy A/B: claude-meta, task `audit`

2026-10-28 01:47; arms alternate order per trial. Input (weighted) is in the provider's units (see bench/proxy_ab.py `cost`). Check the saved answers before reading token savings as a win.

| run | arm | exit | requests | rewrites | saved at read | input (weighted) | output | seconds | top tools |
|---|---|---|---|---|---|---|---|---|---|
| 1 | aware | 0 | 8 | 5 | 10,367 | 26,791 | 15,552 | 135.8 | Read 26 |
| 2 | cortex | 0 | 13 | 9 | 20,824 | 47,689 | 13,672 | 125.4 | Read 26 |
| 3 | passthrough | 0 | 11 | 0 | 0 | 82,285 | 18,508 | 152.7 | Read 22 |
| 4 | passthrough | 0 | 11 | 0 | 0 | 90,437 | 20,285 | 187.4 | Read 24 |
| 5 | cortex | 0 | 11 | 7 | 18,352 | 38,923 | 16,770 | 99.9 | Read 21 |
| 6 | aware | 0 | 10 | 10 | 21,214 | 48,729 | 15,999 | 88.7 | Read 26 |

| arm | runs | mean requests | mean input (weighted) | mean output | mean seconds |
|---|---|---|---|---|---|
| aware | 2 | 9.0 | 37,760 | 15,776 | 112 |
| cortex | 2 | 12.0 | 43,306 | 15,221 | 113 |
| passthrough | 2 | 11.0 | 86,361 | 19,396 | 170 |
