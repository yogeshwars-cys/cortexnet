# Working set of coding sessions

59 sessions, 3,032 file operations (reads, edits, writes), median 18 distinct files per session (max 74). Source: local agent logs (aggregates only).

| window (file ops) | distinct files in the window (mean) | share of the session's files |
|---|---|---|
| 4 | 2.7 | 20% |
| 8 | 4.8 | 32% |
| 16 | 8.5 | 50% |
| 32 | 14.5 | 58% |
| 64 | 23.8 | 70% |

Returns to a file used before: 1,773 (58% of operations).

| gap since the file was last used (ops) | share of returns |
|---|---|
| 1-1 | 71% |
| 2-4 | 9% |
| 5-8 | 7% |
| 9-16 | 5% |
| 17-32 | 5% |
| 33-64 | 2% |
| 65-more | 1% |

50% of returns come within 1 operations.
80% of returns come within 5 operations.
90% of returns come within 12 operations.
95% of returns come within 24 operations.
