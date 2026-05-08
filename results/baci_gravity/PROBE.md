# linear probe: baci_gravity

frozen-encoder 5-fold ridge regression. target: log(trade_vol[t+1]) − log(trade_vol[t]).
seeds: [0, 1, 2, 3, 4].

## R² (5-fold CV, median across seeds)

| condition | R² | MAE | per-seed R² |
|---|---:|---:|---|
| graph | 0.006 | 0.159 | -0.004, -0.013, 0.009, 0.013, 0.006 |
| sequential | -0.003 | 0.163 | -0.016, -0.020, 0.003, -0.002, -0.003 |
| raw_features | -0.006 | 0.163 | -0.018, -0.023, 0.001, -0.003, -0.006 |

**R² gap (graph − sequential): +0.009**
**R² gap (graph − raw features): +0.012**

a positive gap (especially > +0.05) means graph-JEPA's frozen embeddings linearly carry information about next-period growth that the corresponding baseline does not.