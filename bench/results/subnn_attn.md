# Cortex-SubNN with attention-distilled labels

5-fold CV (sessions grouped as in subnn_train.py, SWE instances grouped separately), 2 seeds averaged per condition and fold, 40 epochs max, early stopping on session cases only. 57 SWE reads carry Glimmer layer-28 attention (whole-file arm). Encoder MiniLM-L6.

## Test set A sessions (854 read cases)

| ranker | MRR | top-1 | recall@budget | any hit | MRR diff vs C0 (95% CI) |
|---|---|---|---|---|---|
| C0 sessions + git | 0.413 | 0.248 | 0.413 | 0.593 |  |
| C1 + SWE hard labels | 0.406 | 0.241 | 0.402 | 0.582 | -0.007 [-0.023, +0.009] |
| C2 + SWE hard labels + attention KL (lam 1) | 0.403 | 0.241 | 0.394 | 0.576 | -0.010 [-0.024, +0.003] |
| frozen zero-shot | 0.348 | 0.184 | 0.391 | 0.573 | -0.065 [-0.087, -0.044] |

## Test set B SWE (139 read cases)

| ranker | MRR | top-1 | recall@budget | any hit | MRR diff vs C0 (95% CI) |
|---|---|---|---|---|---|
| C0 sessions + git | 0.689 | 0.532 | 0.579 | 0.899 |  |
| C1 + SWE hard labels | 0.694 | 0.540 | 0.615 | 0.914 | +0.005 [-0.027, +0.036] |
| C2 + SWE hard labels + attention KL (lam 1) | 0.696 | 0.532 | 0.616 | 0.914 | +0.008 [-0.018, +0.034] |
| frozen zero-shot | 0.611 | 0.468 | 0.585 | 0.906 | -0.078 [-0.119, -0.038] |

Run time 299s on CPU.
