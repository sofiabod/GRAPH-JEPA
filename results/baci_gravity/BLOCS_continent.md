# bloc discovery: baci_gravity

unsupervised k-means clustering of frozen country embeddings, scored against UN M49 subregion ground truth (5 clusters, 227 countries). seeds: [0, 1, 2, 3, 4].

## results (median across seeds)

| condition | ARI | NMI | purity |
|---|---:|---:|---:|
| graph | 0.119 | 0.172 | 0.427 |
| sequential | 0.077 | 0.130 | 0.370 |
| raw_features | 0.120 | 0.165 | 0.396 |

## per-seed breakdown

### graph

| seed | ARI | NMI | purity |
|---|---:|---:|---:|
| 0 | 0.120 | 0.178 | 0.427 |
| 1 | 0.086 | 0.158 | 0.379 |
| 2 | 0.119 | 0.172 | 0.427 |
| 3 | 0.119 | 0.172 | 0.427 |
| 4 | 0.081 | 0.155 | 0.370 |

### sequential

| seed | ARI | NMI | purity |
|---|---:|---:|---:|
| 0 | 0.077 | 0.130 | 0.374 |
| 1 | 0.079 | 0.132 | 0.379 |
| 2 | 0.076 | 0.131 | 0.366 |
| 3 | 0.078 | 0.130 | 0.370 |
| 4 | 0.077 | 0.126 | 0.357 |

### raw_features

| seed | ARI | NMI | purity |
|---|---:|---:|---:|
| 0 | 0.120 | 0.165 | 0.396 |
| 1 | 0.120 | 0.165 | 0.396 |
| 2 | 0.120 | 0.165 | 0.396 |
| 3 | 0.120 | 0.165 | 0.396 |
| 4 | 0.120 | 0.165 | 0.396 |
