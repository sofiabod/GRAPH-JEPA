# rigor checks: metrla

seeds: [0, 1, 2, 3, 4]

## bootstrap 95% CI on Δ (test split)

| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |
|---|---:|---:|---:|---:|---:|
| 0 | -0.0031 | -0.0034 | -0.0028 | 0.119 | 1476 |
| 1 | -0.0047 | -0.0053 | -0.0042 | 0.149 | 1476 |
| 2 | -0.0034 | -0.0038 | -0.0031 | 0.124 | 1476 |
| 3 | -0.0048 | -0.0053 | -0.0044 | 0.007 | 1476 |
| 4 | -0.0040 | -0.0047 | -0.0035 | 0.018 | 1476 |

## train-split diagnostic (does not generalize → train >> test)

| seed | train graph | train seq | train Δ | test Δ | train-test gap |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9952 | 0.9979 | -0.0027 | -0.0031 | +0.0004 |
| 1 | 0.9928 | 0.9965 | -0.0037 | -0.0047 | +0.0010 |
| 2 | 0.9947 | 0.9977 | -0.0030 | -0.0034 | +0.0004 |
| 3 | 0.9948 | 0.9987 | -0.0039 | -0.0048 | +0.0009 |
| 4 | 0.9954 | 0.9988 | -0.0034 | -0.0040 | +0.0007 |

a small (or negative) train-test gap means the model is generalizing, not memorizing. a large positive gap (train Δ >> test Δ) suggests graph-jepa's advantage relies on memorizing training graphs.
