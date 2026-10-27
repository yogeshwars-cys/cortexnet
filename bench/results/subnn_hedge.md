# Librarian / Professor switch (online Hedge over two rankers)

Out-of-fold per-symbol scores of the full Cortex-SubNN (3-seed ensembles, 5-fold CV grouped by session) for MiniLM-L6 (Librarian, default) and Muse Glimmer 30B layer 26 (Professor, scored silently). 854 read-time cases in 80 sessions (854 reads). Scores are z-normalised within the file. Cases of one read share alpha; the scoreboard updates after the read with both experts' reciprocal ranks. Knobs (gamma, eta1, eta2, tau, b0) and static alpha are picked by 5-fold nested CV over sessions (a different session split from the training CV); CIs are a session-clustered paired bootstrap vs the Librarian.

| arm | MRR | vs Librarian (95% CI) |
|---|---|---|
| Librarian only (MiniLM full SubNN) | 0.416 | +0.000 [+0.000, +0.000] |
| Professor only (Glimmer L26 full SubNN) | 0.396 | -0.020 [-0.049, +0.010] |
| static 50/50 z-score blend | 0.422 | +0.007 [-0.013, +0.025] |
| static blend, alpha picked by nested CV | 0.413 | -0.003 [-0.022, +0.015] |
| margin-only gate (no scoreboard, eta1 = 0) | 0.424 | +0.008 [-0.008, +0.024] |
| Hedge scoreboard only (eta2 = 0) | 0.427 | +0.012 [-0.004, +0.027] |
| Hedge scoreboard + margin (the proposal) | 0.428 | +0.012 [-0.002, +0.026] |
| oracle per case (upper bound) | 0.520 | +0.104 [+0.085, +0.123] |

## Where each expert wins

Per-case reciprocal ranks: correlation 0.51; Professor strictly better on 36% of cases, Librarian on 44%, tied on 20%.

| read index in session | cases | Librarian MRR | Professor MRR | Professor - Librarian (95% CI) |
|---|---|---|---|---|
| 1st read | 80 | 0.479 | 0.457 | -0.022 [-0.109, +0.058] |
| 2nd-3rd | 119 | 0.425 | 0.421 | -0.003 [-0.080, +0.071] |
| 4th-7th | 169 | 0.399 | 0.393 | -0.006 [-0.071, +0.057] |
| 8th+ | 486 | 0.409 | 0.381 | -0.028 [-0.067, +0.010] |

Static blend by alpha (in-sample, for shape only): 0.0: 0.416, 0.1: 0.419, 0.2: 0.424, 0.3: 0.422, 0.4: 0.426, 0.5: 0.422, 0.6: 0.425, 0.7: 0.423, 0.8: 0.413, 0.9: 0.406, 1.0: 0.396

Knobs picked per outer fold (gamma, eta1, eta2, tau, b0): [(0.7, 2.0, 0.5, 0.0, 0.5), (0.5, 4.0, 0.5, 0.0, 0.5), (0.5, 2.0, 0.5, 0.0, 0.5), (0.5, 2.0, 0.5, 0.0, 0.5), (0.8, 2.0, 1.0, 2.0, 0.0)]
Static alpha per fold: [0.6, 0.6, 0.6, 0.4, 0.2]
