# rigor checks: enron

seeds: [0, 1, 2, 3, 4]

## bootstrap 95% CI on Δ (test split)

| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |
|---|---:|---:|---:|---:|---:|
| 0 | -0.0872 | -0.1291 | -0.0481 | 0.337 | 190 |
| 1 | -0.0819 | -0.1068 | -0.0562 | 0.311 | 190 |
| 2 | -0.0088 | -0.0466 | 0.0289 | 0.542 | 190 |
| 3 | -0.0400 | -0.0699 | -0.0097 | 0.568 | 190 |
| 4 | -0.0338 | -0.0641 | -0.0043 | 0.458 | 190 |

## train-split diagnostic (does not generalize → train >> test)

| seed | train graph | train seq | train Δ | test Δ | train-test gap |
|---|---:|---:|---:|---:|---:|
| 0 | 0.6632 | 0.7364 | -0.0733 | -0.0872 | +0.0140 |
| 1 | 0.6680 | 0.6972 | -0.0292 | -0.0819 | +0.0527 |
| 2 | 0.6905 | 0.6971 | -0.0066 | -0.0088 | +0.0022 |
| 3 | 0.6262 | 0.6487 | -0.0225 | -0.0400 | +0.0174 |
| 4 | 0.6419 | 0.6696 | -0.0277 | -0.0338 | +0.0061 |

a small (or negative) train-test gap means the model is generalizing, not memorizing. a large positive gap (train Δ >> test Δ) suggests graph-jepa's advantage relies on memorizing training graphs.
