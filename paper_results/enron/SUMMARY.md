# eval summary: enron

seeds: [0, 1, 2, 3, 4]

## enron
- seed 0: pred_cos=0.3888  copy_cos=0.6407  graph_avg_cos=0.4815  eff_rank=9.26  mean_pair_cos=0.026
- seed 1: pred_cos=0.2849  copy_cos=0.6720  graph_avg_cos=0.5418  eff_rank=8.56  mean_pair_cos=0.023
- seed 2: pred_cos=0.4579  copy_cos=0.5720  graph_avg_cos=0.4718  eff_rank=7.70  mean_pair_cos=0.024
- seed 3: pred_cos=0.3646  copy_cos=0.5881  graph_avg_cos=0.5076  eff_rank=8.91  mean_pair_cos=0.025
- seed 4: pred_cos=0.4018  copy_cos=0.5547  graph_avg_cos=0.4830  eff_rank=9.56  mean_pair_cos=0.024

## sequential-ablation
- seed 0: pred_cos=0.4550  copy_cos=0.5995  graph_avg_cos=0.4240  eff_rank=5.58  mean_pair_cos=0.022
- seed 1: pred_cos=0.3959  copy_cos=0.6093  graph_avg_cos=0.4737  eff_rank=5.15  mean_pair_cos=0.021
- seed 2: pred_cos=0.4348  copy_cos=0.5386  graph_avg_cos=0.4262  eff_rank=5.46  mean_pair_cos=0.021
- seed 3: pred_cos=0.4138  copy_cos=0.5228  graph_avg_cos=0.4410  eff_rank=5.57  mean_pair_cos=0.024
- seed 4: pred_cos=0.4272  copy_cos=0.5419  graph_avg_cos=0.4541  eff_rank=5.13  mean_pair_cos=0.020

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.3888  seq_cos=0.4550  wilcoxon_p=1.000e+00  win_rate=0.405  n=190
- seed 1: graph_cos=0.2849  seq_cos=0.3959  wilcoxon_p=1.000e+00  win_rate=0.358  n=190
- seed 2: graph_cos=0.4579  seq_cos=0.4348  wilcoxon_p=1.154e-01  win_rate=0.437  n=190
- seed 3: graph_cos=0.3646  seq_cos=0.4138  wilcoxon_p=1.000e+00  win_rate=0.300  n=190
- seed 4: graph_cos=0.4018  seq_cos=0.4272  wilcoxon_p=9.724e-01  win_rate=0.405  n=190
