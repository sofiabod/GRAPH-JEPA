# eval summary: enron

seeds: [0, 1, 2, 3, 4]

## enron
- seed 0: pred_cos=0.3503  copy_cos=0.5608  graph_avg_cos=0.4753  eff_rank=9.55  mean_pair_cos=0.027
- seed 1: pred_cos=0.3185  copy_cos=0.6332  graph_avg_cos=0.5563  eff_rank=8.33  mean_pair_cos=0.020
- seed 2: pred_cos=0.4229  copy_cos=0.5730  graph_avg_cos=0.3889  eff_rank=7.67  mean_pair_cos=0.024
- seed 3: pred_cos=0.4445  copy_cos=0.5472  graph_avg_cos=0.5072  eff_rank=8.83  mean_pair_cos=0.024
- seed 4: pred_cos=0.3082  copy_cos=0.5619  graph_avg_cos=0.4991  eff_rank=9.28  mean_pair_cos=0.025

## sequential-ablation
- seed 0: pred_cos=0.4375  copy_cos=0.5338  graph_avg_cos=0.4298  eff_rank=5.29  mean_pair_cos=0.019
- seed 1: pred_cos=0.4004  copy_cos=0.5648  graph_avg_cos=0.4857  eff_rank=4.81  mean_pair_cos=0.018
- seed 2: pred_cos=0.4317  copy_cos=0.5392  graph_avg_cos=0.3655  eff_rank=5.31  mean_pair_cos=0.020
- seed 3: pred_cos=0.4845  copy_cos=0.4964  graph_avg_cos=0.4573  eff_rank=5.39  mean_pair_cos=0.023
- seed 4: pred_cos=0.3420  copy_cos=0.5560  graph_avg_cos=0.4912  eff_rank=5.18  mean_pair_cos=0.019

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.3503  seq_cos=0.4375  wilcoxon_p=1.000e+00  win_rate=0.337  n=190
- seed 1: graph_cos=0.3185  seq_cos=0.4004  wilcoxon_p=1.000e+00  win_rate=0.311  n=190
- seed 2: graph_cos=0.4229  seq_cos=0.4317  wilcoxon_p=7.943e-01  win_rate=0.542  n=190
- seed 3: graph_cos=0.4445  seq_cos=0.4845  wilcoxon_p=9.513e-01  win_rate=0.568  n=190
- seed 4: graph_cos=0.3082  seq_cos=0.3420  wilcoxon_p=9.058e-01  win_rate=0.458  n=190
