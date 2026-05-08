# bloc discovery: baci_gravity

unsupervised k-means clustering of frozen country embeddings, scored against UN M49 subregion ground truth (17 clusters, 227 countries). seeds: [0, 1, 2, 3, 4].

## results (median across seeds)

| condition | ARI | NMI | purity |
|---|---:|---:|---:|
| graph | 0.032 | 0.291 | 0.322 |
| sequential | 0.024 | 0.283 | 0.295 |
| raw_features | 0.039 | 0.290 | 0.330 |

## per-seed breakdown

### graph

| seed | ARI | NMI | purity |
|---|---:|---:|---:|
| 0 | 0.032 | 0.291 | 0.330 |
| 1 | 0.030 | 0.288 | 0.322 |
| 2 | 0.030 | 0.283 | 0.317 |
| 3 | 0.034 | 0.294 | 0.317 |
| 4 | 0.039 | 0.299 | 0.335 |

### sequential

| seed | ARI | NMI | purity |
|---|---:|---:|---:|
| 0 | 0.025 | 0.283 | 0.300 |
| 1 | 0.024 | 0.287 | 0.291 |
| 2 | 0.023 | 0.276 | 0.291 |
| 3 | 0.025 | 0.291 | 0.295 |
| 4 | 0.021 | 0.282 | 0.300 |

### raw_features

| seed | ARI | NMI | purity |
|---|---:|---:|---:|
| 0 | 0.035 | 0.280 | 0.322 |
| 1 | 0.038 | 0.290 | 0.326 |
| 2 | 0.039 | 0.288 | 0.330 |
| 3 | 0.041 | 0.295 | 0.330 |
| 4 | 0.053 | 0.298 | 0.335 |
