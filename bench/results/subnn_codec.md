# Cortex-SubNN with symbol vectors stored as bit-plane codes

Codec: `cortexnet/codec.py`, a port of the Wormhole BPF64 final codec (uncentred PCA basis, greedy rate-distortion
bit allocation of at most 5 bits per coordinate, per-coordinate Lloyd-Max levels, bit-plane packing, asymmetric LUT
scoring with the query left unquantised). Fitted on the symbol sub-span vectors (unsupervised, all rows); symbols are
scored on their decoded codes, queries stay fp32.

Frozen MiniLM-L6, 5-fold CV grouped by session (854 read-time cases), full model (identity-init bilinear head, no
cross-file negatives), 3-seed ensembles. Zero-shot rows use the same coded symbols.

| symbol storage per sub-span | coords | bit planes | mean decoded cosine | full SubNN MRR | per fold (mean +- sd) | full vs zero-shot (95% CI) | time split MRR |
|---|---|---|---|---|---|---|---|
| fp32, 1,536 B | 384 | - | 1.000 | 0.416 | 0.416 +- 0.061 | +0.068 [+0.046, +0.091] | 0.360 |
| **64 B** (Wormhole's size) | 259 | 259/176/64/11/2 | 0.954 | **0.414** | 0.414 +- 0.052 | +0.065 [+0.043, +0.087] | 0.359 |
| 128 B | 353 | 353/299/237/113/22 | 0.991 | 0.421 | 0.421 +- 0.046 | +0.073 [+0.051, +0.096] | 0.356 |
| 240 B (5 bits x 384) | 384 | 384 x 5 | 0.998 | 0.403 | 0.403 +- 0.071 | +0.050 [+0.028, +0.071] | 0.359 |

All four are within fold noise. 64 bytes per sub-span (24x smaller than fp32, 256 B per symbol) costs no measurable
MRR, so it is the default code size for the coprocessor's symbol cache. With MiniLM's 384 dimensions, 240 B is the
largest code at 5 bits per coordinate; larger codes only make sense for wider encoders (Glimmer, 6,656 dims).
