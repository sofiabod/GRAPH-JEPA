# rigor checks: tgbn_trade

seeds: [0, 1, 2, 3, 4]

## bootstrap 95% CI on Δ (test split)

| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |
|---|---:|---:|---:|---:|---:|
| 0 | 0.0754 | 0.0548 | 0.0968 | 0.753 | 255 |
| 1 | 0.0982 | 0.0810 | 0.1157 | 0.839 | 255 |
| 2 | 0.1362 | 0.1194 | 0.1537 | 0.984 | 255 |
| 3 | 0.0566 | 0.0429 | 0.0707 | 0.714 | 255 |
| 4 | 0.0836 | 0.0676 | 0.1003 | 0.757 | 255 |

## train-split diagnostic (does not generalize → train >> test)

| seed | train graph | train seq | train Δ | test Δ | train-test gap |
|---|---:|---:|---:|---:|---:|
| 0 | 0.8312 | 0.7572 | 0.0740 | 0.0754 | -0.0014 |
| 1 | 0.8560 | 0.7520 | 0.1041 | 0.0982 | +0.0059 |
| 2 | 0.8804 | 0.7498 | 0.1306 | 0.1362 | -0.0057 |
| 3 | 0.8419 | 0.7890 | 0.0529 | 0.0566 | -0.0038 |
| 4 | 0.8286 | 0.7563 | 0.0723 | 0.0836 | -0.0113 |

a small (or negative) train-test gap means the model is generalizing, not memorizing. a large positive gap (train Δ >> test Δ) suggests graph-jepa's advantage relies on memorizing training graphs.
