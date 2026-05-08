# RG-1 + RG-10 + RG-4: random baselines for baci_gravity

5-seed median effective rank for **untrained** encoders compared to JEPA-trained.

| condition | median eff_rank | min | max |
|---|---:|---:|---:|
| **random graph encoder** (RG-1a) | 2.40 | 2.35 | 2.55 |
| **random sequential encoder** (RG-1b) | 3.43 | 3.11 | 3.54 |
| **random 6d→256 projection** (RG-10) | 2.45 | 2.41 | 2.54 |

Compare to JEPA-trained eff_rank (from main eval). If random encoders produce eff_rank ≈ 8,
the 'JEPA training induces d≈8 attractor' claim is wrong — geometry is from architecture
or from the input dim, not from training.

## RG-4: parameter counts

- graph encoder: **399,616** parameters
- sequential encoder: **398,080** parameters

predictor + target encoder shared across both conditions; the difference between
encoder param counts is the 'capacity-matched' margin.