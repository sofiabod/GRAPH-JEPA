# rigor checks: baci_gravity

seeds: [0, 1, 2, 3, 4]

## bootstrap 95% CI on Δ (test split)

| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |
|---|---:|---:|---:|---:|---:|
| 0 | 0.2436 | 0.2212 | 0.2655 | 1.000 | 180 |
| 1 | 0.1631 | 0.1435 | 0.1826 | 0.928 | 180 |
| 2 | 0.1887 | 0.1705 | 0.2068 | 0.928 | 180 |
| 3 | 0.1990 | 0.1788 | 0.2194 | 0.972 | 180 |
| 4 | 0.1963 | 0.1783 | 0.2144 | 0.961 | 180 |

## train-split diagnostic (does not generalize → train >> test)

| seed | train graph | train seq | train Δ | test Δ | train-test gap |
|---|---:|---:|---:|---:|---:|
| 0 | 0.9471 | 0.7221 | 0.2251 | 0.2436 | -0.0185 |
| 1 | 0.9360 | 0.7671 | 0.1689 | 0.1631 | +0.0059 |
| 2 | 0.9087 | 0.7220 | 0.1868 | 0.1887 | -0.0019 |
| 3 | 0.9357 | 0.7521 | 0.1836 | 0.1990 | -0.0153 |
| 4 | 0.9140 | 0.7237 | 0.1903 | 0.1963 | -0.0060 |

a small (or negative) train-test gap means the model is generalizing, not memorizing. a large positive gap (train Δ >> test Δ) suggests graph-jepa's advantage relies on memorizing training graphs.
