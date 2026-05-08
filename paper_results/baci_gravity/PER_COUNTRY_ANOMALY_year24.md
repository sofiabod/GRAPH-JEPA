# per-country anomaly: baci_gravity year 2020

target year: 2020 (snap_idx 24)
reference year: 2019 (snap_idx 23)
n_active countries: 224 of 227
n_seeds: 5

## Spearman correlation: predicted deviation vs actual trade decline

| condition | Spearman ρ | bootstrap CI95 |
|---|---:|---|
| graph-JEPA | 0.017 | [-0.118, 0.150] |
| sequential | -0.027 | [-0.165, 0.111] |

**Δρ (graph − sequential): +0.044**

interpretation: a positive Spearman ρ means countries flagged as more anomalous by the model also experienced larger actual trade declines. ρ > 0.3 with CI excluding 0 supports the claim that the model recovered the country-level pattern of the shock.