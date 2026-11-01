# CortexNET engine stress test

`bench/stress_engine.py`, Python 3.13.13, 2026-11-01. No model: the reader alone, on generated pathological files, whole repositories and the stdio MCP server.

## Pathological files

| file | MB | ok | read s | re-read after edit s | tokens shown / whole | peak MB | problems |
|---|---|---|---|---|---|---|---|
| empty.py | 0.00 | yes | 0.00 | 0.00 | 1 / 1 | 0 |  |
| blank_lines.py | 0.20 | yes | 0.37 | 0.46 | 4036 / 50000 | 8 |  |
| one_line_minified.js | 5.40 | yes | 4.82 | 7.77 | 58 / 1350000 | 49 |  |
| long_line_no_newline.txt | 5.00 | yes | 0.40 | 0.06 | 59 / 1250000 | 15 |  |
| binary_random.py | 2.00 | yes | 0.23 | 0.40 | 1964 / 473657 | 29 |  |
| png_header.png | 0.50 | yes | 0.07 | 0.12 | 2673 / 118394 | 5 |  |
| nul_bytes.py | 0.50 | yes | 0.39 | 0.56 | 49 / 125000 | 7 |  |
| latin1.py | 0.12 | yes | 0.50 | 0.62 | 4019 / 30001 | 25 |  |
| utf16_bom.py | 0.22 | yes | 0.12 | 0.19 | 1833 / 55000 | 3 |  |
| crlf.py | 0.22 | yes | 0.70 | 1.06 | 4014 / 48000 | 51 |  |
| tabs.py | 0.20 | yes | 0.98 | 1.78 | 3930 / 51000 | 49 |  |
| syntax_error.py | 0.20 | yes | 0.71 | 0.96 | 4033 / 50003 | 105 |  |
| deep_nesting.py | 0.02 | yes | 0.04 | 0.04 | 33 / 4341 | 0 |  |
| deep_expression.py | 0.08 | yes | 0.16 | 0.28 | 4040 / 19075 | 3 |  |
| many_functions.py | 0.72 | yes | 2.36 | 3.71 | 4025 / 179444 | 152 |  |
| many_consts.py | 0.43 | yes | 1.63 | 2.63 | 4011 / 106944 | 84 |  |
| one_huge_function.py | 0.74 | yes | 1.44 | 1.70 | 51 / 184447 | 115 |  |
| c_prototypes.h | 0.41 | yes | 0.28 | 0.42 | 3666 / 102222 | 8 |  |
| regex_bomb.c | 0.20 | yes | 0.07 | 0.09 | 55 / 48755 | 1 |  |
| regex_bomb2.java | 0.18 | yes | 0.05 | 0.06 | 56 / 45011 | 1 |  |
| markdown_headings.md | 0.48 | yes | 4.63 | 6.15 | 3971 / 120972 | 49 |  |
| only_comments.c | 1.60 | yes | 0.79 | 1.22 | 4036 / 400000 | 16 |  |
| unicode_names.py | 0.21 | yes | 0.67 | 0.75 | 4037 / 49445 | 36 |  |
| surrogates.py | 0.13 | yes | 0.50 | 0.58 | 4019 / 32500 | 25 |  |

## Whole repositories

Roots: `Lib`, `django__django-11001`, `sympy__sympy-11400`, `matplotlib__matplotlib-22711`, `llama.cpp`

- files read: 3,000 (1,397 above the whole-file threshold, outlined) in 627 s; errors / invariant failures: 0
- read latency: median 102.8 ms, p95 220.0 ms, p99 302.3 ms, max 1561 ms
- outlined files, tokens shown / whole file: median 0.14, p90 0.29, total 0.12; reads that showed at least the whole file: 0 (0.0%)

| ext | outlined files | tokens shown / whole |
|---|---|---|
| .py | 787 | 0.14 |
| .h | 463 | 0.10 |
| .txt | 25 | 0.03 |
| .cpp | 22 | 0.11 |
| .zig | 20 | 0.07 |
| .hpp | 16 | 0.11 |
| .c | 16 | 0.13 |
| .js | 15 | 0.01 |
| .json | 10 | 0.15 |
| .md | 4 | 0.08 |
| .rst | 3 | 0.06 |
| .yaml | 2 | 0.11 |
| .ts | 2 | 0.20 |
| .metal | 2 | 0.16 |

Slowest reads: `schema_py_generated.py` 1561 ms (649 kB); `numpy_test.py` 1167 ms (448 kB); `test_distributions.py` 717 ms (436 kB); `hlo.pb.h` 665 ms (914 kB); `test_solvers.py` 556 ms (69 kB)

## MCP server

2018 messages in one pipe (18 malformed or with bad arguments, then a burst of 2000 tool calls): 2017 replies in 23.9 s, 0 burst calls unanswered, 7 JSON-RPC errors, 7 tool errors, 0 unparsable lines, exit code 0.

