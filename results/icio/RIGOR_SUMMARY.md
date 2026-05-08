# rigor checks: icio

seeds: [0, 1, 2, 3, 4]

## bootstrap 95% CI on Δ (test split)

| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0475 | 0.0188 | 0.0876 | 1.000 | 27 |
| 1 | 0.0352 | 0.0058 | 0.0714 | 1.000 | 27 |
| 2 | 0.0226 | 0.0051 | 0.0453 | 0.926 | 27 |
| 3 | 0.0326 | 0.0127 | 0.0610 | 1.000 | 27 |
| 4 | 0.0503 | 0.0209 | 0.0917 | 1.000 | 27 |

## train-split diagnostic (does not generalize → train >> test)

| seed | train graph | train seq | train Δ | test Δ | train-test gap |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9818 | 0.9295 | 0.0522 | 0.0475 | +0.0047 |
| 1 | 0.9814 | 0.9590 | 0.0223 | 0.0352 | -0.0129 |
| 2 | 0.9883 | 0.9750 | 0.0133 | 0.0226 | -0.0093 |
| 3 | 0.9677 | 0.9287 | 0.0390 | 0.0326 | +0.0064 |
| 4 | 0.9909 | 0.9498 | 0.0411 | 0.0503 | -0.0092 |

a small (or negative) train-test gap means the model is generalizing, not memorizing. a large positive gap (train Δ >> test Δ) suggests graph-jepa's advantage relies on memorizing training graphs.
