# enron hidden-role recovery: enron

test: do frozen embeddings cluster Enron executives by FUNCTIONAL ROLE
(legal, govaffairs, trading, exec, admin, research, operations) without
any role labels appearing in training? compare pairwise cosine similarity
for within-role pairs vs between-role pairs.

seeds: [0, 1, 2, 3, 4].  n_labeled_people: 38
n_within_pairs: 135, n_between_pairs: 568

## median(within) − median(between), median across seeds

| condition | median within | median between | gap | min MWU p | max MWU p |
|---|---:|---:|---:|---:|---:|
| graph | 0.793 | -0.085 | **+0.859** | 8.67e-03 | 1.70e-02 |
| sequential | 0.627 | -0.067 | **+0.688** | 3.87e-03 | 8.41e-03 |
| raw_features | 0.876 | 0.877 | **-0.001** | 6.32e-01 | 6.32e-01 |

**gap delta (graph − sequential): +0.171**
**gap delta (graph − raw features): +0.860**

interpretation: a positive gap means within-role pairs are MORE similar than between-role pairs. graph >> sequential (and graph >> raw_features) means the graph encoder discovered role structure that the per-node baseline could not.