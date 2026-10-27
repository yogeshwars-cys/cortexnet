# Session replay: trail memory, symbol graph and connected-symbol menu

62 sessions (with both reads and edits), replayed in time order; include model carried across sessions. Tokens = chars/4. Baseline = tokens the real Read calls returned. Upper bound: the simulated agent fetches the right symbol after a miss and approves exactly the menu items it needs.

| arm | read tokens vs baseline | edits with lines already shown | first half / second half of sessions | misses served by menu | misses by range | menu items offered | menu tokens |
|---|---|---|---|---|---|---|---|
| outline only (no up-front includes) | 346,072 / 555,913 (62%) | 40% of 575 | 35% / 46% | 0 | 345 | 0 | 0 |
| outline only, big symbols split at 600 tok | 313,658 / 555,913 (56%) | 37% of 575 | 33% / 42% | 0 | 363 | 0 | 0 |
| outline only, big symbols split at 300 tok | 310,566 / 555,913 (56%) | 35% of 575 | 31% / 39% | 0 | 374 | 0 | 0 |
| v1 (query, changed, asked, structure) | 350,239 / 555,913 (63%) | 40% of 575 | 35% / 46% | 0 | 345 | 0 | 0 |
| + trail | 360,521 / 555,913 (65%) | 40% of 575 | 35% / 46% | 0 | 344 | 0 | 0 |
| + trail + graph | 362,155 / 555,913 (65%) | 40% of 575 | 35% / 46% | 0 | 344 | 0 | 0 |
| + trail + graph + menu | 356,245 / 555,913 (64%) | 40% of 575 | 35% / 46% | 0 | 345 | 25 | 469 |

Reads replayed: 204 full reads, 122 served outline-first (the rest are small files returned whole).

Learned weights after the last session (query_name, query_overlap, changed, asked_before, centrality, rarity, log_tokens, kind_def, kind_class, kind_const, kind_imports, kind_section, depth, trail, conn):

- outline only (no up-front includes): [4.92, 2.0, 4.0, 1.0, 0.03, -0.06, 0.15, 0.0, 0.0, -0.0, -0.01, 0.03, 0.12, 2.0, 1.5]
- outline only, big symbols split at 600 tok: [4.81, 2.0, 4.0, 1.0, 0.04, 0.01, 0.17, 0.01, 0.0, 0.01, 0.0, -0.0, 0.1, 2.0, 1.5]
- outline only, big symbols split at 300 tok: [4.81, 2.0, 4.0, 1.0, 0.04, -0.0, 0.13, 0.02, 0.0, 0.01, 0.0, -0.0, 0.19, 2.0, 1.5]
- v1 (query, changed, asked, structure): [4.94, 2.0, 4.0, 1.0, 0.03, -0.06, 0.15, 0.02, 0.0, -0.0, -0.01, 0.02, 0.12, 2.0, 1.5]
- + trail: [4.94, 1.99, 4.0, 1.0, 0.03, -0.05, 0.13, 0.01, 0.0, -0.0, -0.01, 0.01, 0.11, 1.99, 1.5]
- + trail + graph: [4.94, 1.99, 4.0, 1.0, 0.04, -0.05, 0.13, -0.0, 0.0, -0.0, -0.01, 0.01, 0.12, 1.99, 1.53]
- + trail + graph + menu: [4.94, 1.98, 4.0, 0.98, 0.08, -0.11, -0.04, -0.11, 0.01, 0.0, -0.01, 0.01, 0.22, 1.78, 1.26]
