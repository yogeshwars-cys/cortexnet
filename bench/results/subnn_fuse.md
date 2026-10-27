# Combined report: stacked fusion of every ranker with a per-read gate

Out-of-fold per-symbol scores (854 read-time cases, 80 sessions). Clerks: MiniLM full, Glimmer L26 full, MiniLM zero-shot, Glimmer L26 zero-shot, lexical, v1 include, smallest first, each z-scored within the file. Read context: log symbol count, MiniLM/Glimmer Spearman and top-1 agreement, both margins, lexical peak. Listwise loss; L2 by inner CV; 5-fold CV over sessions; session-clustered paired bootstrap vs MiniLM full.

| arm | MRR | vs Librarian (95% CI) |
|---|---|---|
| Librarian only (MiniLM full) | 0.416 | +0.000 [+0.000, +0.000] |
| Glimmer L26 full | 0.396 | -0.020 [-0.049, +0.010] |
| static 50/50 blend (MiniLM full + Glimmer full) | 0.422 | +0.007 [-0.013, +0.025] |
| combined report, linear | 0.428 | +0.013 [-0.007, +0.032] |
| combined report, gated | 0.429 | +0.014 [-0.007, +0.036] |
| combined report, bilinear | 0.427 | +0.011 [-0.007, +0.030] |
| oracle per case, MiniLM vs Glimmer (upper bound) | 0.520 | +0.104 [+0.085, +0.123] |
| oracle per case, any of the 7 clerks | 0.697 | +0.281 [+0.258, +0.304] |

Mean learned base weights per clerk (linear): MiniLM full +0.70, Glimmer L26 full +0.24, MiniLM zero-shot -0.09, Glimmer L26 zero-shot -0.00, lexical +0.07, v1 include -0.14, smallest first -0.22

Single clerks: MiniLM full 0.416, Glimmer L26 full 0.396, MiniLM zero-shot 0.348, Glimmer L26 zero-shot 0.321, lexical 0.307, v1 include 0.309, smallest first 0.212

L2 picked per fold: linear [0.001, 0.1, 0.1, 0.1, 0.1]; gated [0.1, 0.1, 0.1, 0.1, 0.1]; bilinear [0.1, 0.1, 0.1, 0.1, 0.1]
