# rigor checks: dblp

seeds: [0, 1, 2, 3, 4]

## bootstrap 95% CI on Δ (test split)

| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |
|---|---:|---:|---:|---:|---:|
| 0 | -0.1295 | -0.1354 | -0.1237 | 0.016 | 800 |
| 1 | -0.1760 | -0.1811 | -0.1710 | 0.000 | 800 |
| 2 | -0.0762 | -0.0802 | -0.0722 | 0.043 | 800 |
| 3 | 0.2380 | 0.2220 | 0.2543 | 0.978 | 800 |
| 4 | 0.0636 | 0.0590 | 0.0681 | 0.879 | 800 |

## train-split diagnostic (does not generalize → train >> test)

| seed | train graph | train seq | train Δ | test Δ | train-test gap |
|---|---:|---:|---:|---:|---:|
| 0 | 0.3688 | 0.5354 | -0.1665 | -0.1295 | -0.0370 |
| 1 | 0.3210 | 0.5142 | -0.1932 | -0.1760 | -0.0172 |
| 2 | 0.4551 | 0.5498 | -0.0947 | -0.0762 | -0.0184 |
| 3 | 0.5333 | 0.1753 | 0.3580 | 0.2380 | +0.1200 |
| 4 | 0.4527 | 0.3609 | 0.0918 | 0.0636 | +0.0282 |

a small (or negative) train-test gap means the model is generalizing, not memorizing. a large positive gap (train Δ >> test Δ) suggests graph-jepa's advantage relies on memorizing training graphs.
