# Session replay: trail memory, symbol graph and connected-symbol menu

63 sessions (with both reads and edits), replayed in time order; include model carried across sessions. Tokens = chars/4. Baseline = tokens the real Read calls returned. Upper bound: the simulated agent fetches the right symbol after a miss and approves exactly the menu items it needs.

| arm | read tokens vs baseline | edits with lines already shown | first half / second half of sessions | misses served by menu | misses by range | menu items offered | menu tokens | with prefix-cache compounding |
|---|---|---|---|---|---|---|---|---|
| outline only (no up-front includes) | 347,310 / 544,583 (64%) | 40% of 585 | 35% / 45% | 0 | 351 | 0 | 0 | 101% |
| outline only, big symbols split at 600 tok | 314,242 / 544,583 (58%) | 37% of 585 | 33% / 41% | 0 | 369 | 0 | 0 | 97% |
| outline only, big symbols split at 300 tok | 310,825 / 544,583 (57%) | 35% of 585 | 31% / 39% | 0 | 380 | 0 | 0 | 97% |
| v1 (query, changed, asked, structure) | 351,478 / 544,583 (65%) | 40% of 585 | 35% / 45% | 0 | 351 | 0 | 0 | 101% |
| + trail | 361,787 / 544,583 (66%) | 40% of 585 | 36% / 45% | 0 | 350 | 0 | 0 | 103% |
| + trail + graph | 363,415 / 544,583 (67%) | 40% of 585 | 36% / 45% | 0 | 350 | 0 | 0 | 104% |
| + trail + graph + menu | 357,500 / 544,583 (66%) | 40% of 585 | 35% / 45% | 0 | 351 | 25 | 469 | 103% |
| Rule A/B policy (prefix-cache cost), split 600 | 317,697 / 544,583 (58%) | 37% of 585 | 33% / 41% | 0 | 369 | 0 | 0 | 97% |
| budget fill + coprocessor | 313,713 / 544,583 (58%) | 37% of 585 | 33% / 41% | 0 | 369 | 0 | 0 | 96% |
| Rule A/B + coprocessor | 314,333 / 544,583 (58%) | 37% of 585 | 33% / 41% | 0 | 369 | 0 | 0 | 97% |

Compounding column: every token served costs 8.2x (re-read from the prefix cache on the median 72 later calls at 0.1x) and every extra round trip (menu approval or range after a miss) 4,700 tokens (0.1 x the median 43k logged context + 400 tool overhead); medians, not per-event values. Coprocessor arms pass the replay query (user prompt + agent's last words) as the prompt view.


Reads replayed: 199 full reads, 118 served outline-first (the rest are small files returned whole).

Learned weights after the last session (query_name, query_overlap, changed, asked_before, centrality, rarity, log_tokens, kind_def, kind_class, kind_const, kind_imports, kind_section, depth, trail, conn):

- outline only (no up-front includes): [4.92, 2.0, 4.0, 1.0, 0.03, -0.06, 0.15, 0.0, 0.0, -0.0, -0.01, 0.03, 0.12, 2.0, 1.5]
- outline only, big symbols split at 600 tok: [4.81, 2.0, 4.0, 1.0, 0.04, 0.01, 0.17, 0.01, 0.0, 0.01, 0.0, -0.0, 0.1, 2.0, 1.5]
- outline only, big symbols split at 300 tok: [4.81, 2.0, 4.0, 1.0, 0.04, -0.0, 0.13, 0.02, 0.0, 0.01, 0.0, -0.0, 0.19, 2.0, 1.5]
- v1 (query, changed, asked, structure): [4.94, 2.0, 4.0, 1.0, 0.03, -0.06, 0.15, 0.02, 0.0, -0.0, -0.01, 0.02, 0.12, 2.0, 1.5]
- + trail: [4.94, 1.99, 4.0, 1.0, 0.03, -0.05, 0.13, 0.01, 0.0, -0.0, -0.01, 0.01, 0.11, 1.99, 1.5]
- + trail + graph: [4.94, 1.99, 4.0, 1.0, 0.04, -0.05, 0.13, -0.0, 0.0, -0.0, -0.01, 0.01, 0.12, 1.99, 1.53]
- + trail + graph + menu: [4.94, 1.98, 4.0, 0.98, 0.08, -0.11, -0.04, -0.11, 0.01, 0.0, -0.01, 0.01, 0.22, 1.78, 1.26]
- Rule A/B policy (prefix-cache cost), split 600: [4.81, 2.0, 4.0, 1.0, 0.05, 0.0, 0.14, 0.0, 0.0, 0.02, -0.01, -0.01, 0.07, 1.95, 1.52]
- budget fill + coprocessor: [4.8, 1.99, 4.0, 1.0, 0.04, 0.0, 0.14, -0.01, 0.0, 0.01, 0.0, -0.0, 0.08, 1.94, 1.51]
- Rule A/B + coprocessor: [4.74, 1.99, 4.0, 1.0, 0.05, 0.01, 0.14, -0.0, 0.0, 0.01, -0.0, -0.0, 0.08, 1.92, 1.5]
