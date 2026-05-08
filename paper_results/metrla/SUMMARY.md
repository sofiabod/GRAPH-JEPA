# eval summary: metrla

seeds: [0, 1, 2, 3, 4]

## metrla
- seed 0: pred_cos=0.9945  copy_cos=0.9863  graph_avg_cos=0.6830  eff_rank=21.62  mean_pair_cos=0.011
- seed 1: pred_cos=0.9920  copy_cos=0.9759  graph_avg_cos=0.6591  eff_rank=20.42  mean_pair_cos=0.015
- seed 2: pred_cos=0.9944  copy_cos=0.9848  graph_avg_cos=0.6391  eff_rank=17.51  mean_pair_cos=0.009
- seed 3: pred_cos=0.9938  copy_cos=0.9851  graph_avg_cos=0.6576  eff_rank=18.19  mean_pair_cos=0.014
- seed 4: pred_cos=0.9946  copy_cos=0.9795  graph_avg_cos=0.6594  eff_rank=15.97  mean_pair_cos=0.010

## sequential-ablation
- seed 0: pred_cos=0.9976  copy_cos=0.9638  graph_avg_cos=0.2219  eff_rank=7.07  mean_pair_cos=0.000
- seed 1: pred_cos=0.9975  copy_cos=0.9306  graph_avg_cos=0.1590  eff_rank=7.40  mean_pair_cos=0.000
- seed 2: pred_cos=0.9976  copy_cos=0.9714  graph_avg_cos=0.1681  eff_rank=7.63  mean_pair_cos=0.000
- seed 3: pred_cos=0.9987  copy_cos=0.9765  graph_avg_cos=0.1099  eff_rank=6.40  mean_pair_cos=-0.000
- seed 4: pred_cos=0.9988  copy_cos=0.9874  graph_avg_cos=0.2116  eff_rank=6.74  mean_pair_cos=-0.000

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.9945  seq_cos=0.9976  wilcoxon_p=1.000e+00  win_rate=0.153  n=1476
- seed 1: graph_cos=0.9920  seq_cos=0.9975  wilcoxon_p=1.000e+00  win_rate=0.062  n=1476
- seed 2: graph_cos=0.9944  seq_cos=0.9976  wilcoxon_p=1.000e+00  win_rate=0.149  n=1476
- seed 3: graph_cos=0.9938  seq_cos=0.9987  wilcoxon_p=1.000e+00  win_rate=0.012  n=1476
- seed 4: graph_cos=0.9946  seq_cos=0.9988  wilcoxon_p=1.000e+00  win_rate=0.012  n=1476
