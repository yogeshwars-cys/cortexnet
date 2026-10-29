# Session replay: trail memory, symbol graph and connected-symbol menu

67 sessions (with both reads and edits), replayed in time order; include model carried across sessions. Tokens = chars/4. Baseline = tokens the real Read calls returned. Upper bound: the simulated agent fetches the right symbol after a miss and approves exactly the menu items it needs.

| arm | read tokens vs baseline | edits with lines already shown | first half / second half of sessions | misses served by menu | misses by range | menu items offered | menu tokens | with prefix-cache compounding |
|---|---|---|---|---|---|---|---|---|
| outline only (no up-front includes) | 400,022 / 547,018 (73%) | 39% of 716 | 34% / 43% | 0 | 435 | 0 | 0 | 119% |
| outline only, big symbols split at 600 tok | 363,946 / 547,018 (67%) | 37% of 716 | 32% / 40% | 0 | 453 | 0 | 0 | 114% |
| outline only, big symbols split at 300 tok | 356,260 / 547,018 (65%) | 35% of 716 | 31% / 38% | 0 | 464 | 0 | 0 | 114% |
| v1 (query, changed, asked, structure) | 404,970 / 547,018 (74%) | 40% of 716 | 34% / 43% | 0 | 431 | 0 | 0 | 119% |
| + trail | 416,512 / 547,018 (76%) | 40% of 716 | 35% / 43% | 0 | 433 | 0 | 0 | 122% |
| + trail + graph | 419,423 / 547,018 (77%) | 40% of 716 | 35% / 43% | 0 | 431 | 0 | 0 | 122% |
| + trail + graph + menu | 412,175 / 547,018 (75%) | 40% of 716 | 34% / 43% | 0 | 432 | 32 | 545 | 121% |
| Rule A/B policy (prefix-cache cost), split 600 | 367,346 / 547,018 (67%) | 37% of 716 | 32% / 40% | 0 | 452 | 0 | 0 | 115% |
| + homeostatic threshold (beta 1) | 412,554 / 547,018 (75%) | 40% of 716 | 34% / 43% | 0 | 432 | 32 | 545 | 121% |
| + homeostatic threshold (beta 3) | 412,579 / 547,018 (75%) | 40% of 716 | 34% / 43% | 0 | 432 | 32 | 545 | 121% |
| + refractory shown bodies | 412,372 / 547,018 (75%) | 40% of 716 | 34% / 43% | 0 | 432 | 32 | 545 | 121% |
| + normalised online learning (nlms) | 408,886 / 547,018 (75%) | 40% of 716 | 34% / 43% | 0 | 432 | 29 | 491 | 120% |
| + winner-take-all menu (floor 0.25) | 419,993 / 547,018 (77%) | 40% of 716 | 34% / 43% | 0 | 432 | 9 | 116 | 122% |
| budget fill + coprocessor | 408,007 / 547,018 (75%) | 37% of 716 | 33% / 40% | 0 | 450 | 0 | 0 | 122% |
| Rule A/B + coprocessor | 384,166 / 547,018 (70%) | 37% of 716 | 33% / 40% | 0 | 450 | 0 | 0 | 117% |

Compounding column: every token served costs 8.2x (re-read from the prefix cache on the median 72 later calls at 0.1x) and every extra round trip (menu approval or range after a miss) 4,700 tokens (0.1 x the median 43k logged context + 400 tool overhead); medians, not per-event values. Coprocessor arms pass the replay query (user prompt + agent's last words) as the prompt view.


Reads replayed: 193 full reads, 115 served outline-first (the rest are small files returned whole).

Learned weights after the last session (query_name, query_overlap, changed, asked_before, centrality, rarity, log_tokens, kind_def, kind_class, kind_const, kind_imports, kind_section, depth, trail, conn):

- outline only (no up-front includes): [4.92, 2.0, 4.0, 1.0, 0.11, -0.03, 0.62, 0.05, 0.01, 0.03, 0.0, 0.03, 0.16, 2.0, 1.5]
- outline only, big symbols split at 600 tok: [4.81, 2.0, 4.0, 1.0, 0.09, 0.02, 0.47, 0.02, 0.01, 0.03, 0.01, 0.01, 0.12, 2.0, 1.5]
- outline only, big symbols split at 300 tok: [4.81, 2.0, 4.0, 1.0, 0.08, 0.01, 0.49, 0.02, 0.01, 0.02, 0.01, 0.04, 0.28, 2.0, 1.5]
- v1 (query, changed, asked, structure): [4.94, 2.01, 4.0, 1.0, 0.1, -0.03, 0.56, 0.05, 0.01, 0.03, -0.0, 0.02, 0.15, 2.0, 1.5]
- + trail: [4.94, 1.99, 4.0, 1.0, 0.1, -0.03, 0.53, 0.05, 0.01, 0.01, -0.0, 0.01, 0.15, 2.01, 1.5]
- + trail + graph: [4.94, 1.99, 4.0, 1.0, 0.06, -0.04, 0.29, 0.02, 0.0, 0.01, -0.0, 0.02, 0.13, 1.99, 1.54]
- + trail + graph + menu: [4.94, 1.98, 4.0, 0.98, 0.14, -0.08, 0.28, -0.07, 0.01, 0.01, 0.01, 0.02, 0.25, 1.79, 1.31]
- Rule A/B policy (prefix-cache cost), split 600: [4.81, 2.0, 4.0, 1.0, 0.03, 0.01, 0.11, -0.02, 0.01, 0.02, 0.0, -0.02, 0.03, 1.93, 1.48]
- + homeostatic threshold (beta 1): [4.94, 1.98, 4.0, 0.98, 0.14, -0.08, 0.28, -0.07, 0.01, 0.01, 0.01, 0.02, 0.25, 1.79, 1.31]
- + homeostatic threshold (beta 3): [4.94, 1.98, 4.0, 0.98, 0.14, -0.08, 0.28, -0.07, 0.01, 0.01, 0.01, 0.02, 0.25, 1.79, 1.31]
- + refractory shown bodies: [4.94, 1.98, 4.0, 0.98, 0.14, -0.08, 0.28, -0.07, 0.01, 0.01, 0.01, 0.02, 0.25, 1.79, 1.31]
- + normalised online learning (nlms): [4.9, 1.93, 4.0, 0.77, 0.43, -0.07, -0.1, -0.05, 0.11, 0.23, 0.15, 0.13, 0.46, 1.56, 1.08]
- + winner-take-all menu (floor 0.25): [4.94, 1.99, 4.0, 1.0, 0.07, -0.03, 0.29, 0.02, 0.0, 0.01, 0.0, -0.0, 0.14, 1.94, 1.52]
- budget fill + coprocessor: [4.9, 2.0, 4.0, 1.0, 0.02, 0.02, 0.08, -0.04, 0.01, 0.01, 0.01, 0.05, 0.08, 1.99, 1.51]
- Rule A/B + coprocessor: [4.82, 2.0, 4.0, 1.0, 0.02, 0.01, 0.12, -0.03, 0.01, 0.02, 0.0, 0.0, 0.04, 1.95, 1.49]
