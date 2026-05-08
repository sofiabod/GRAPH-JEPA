# eval summary: tgbn_trade

seeds: [0, 1, 2, 3, 4]

## tgbn_trade
- seed 0: pred_cos=0.8164  copy_cos=0.5397  graph_avg_cos=0.2852  eff_rank=8.29  mean_pair_cos=0.366
- seed 1: pred_cos=0.8103  copy_cos=0.5966  graph_avg_cos=0.3734  eff_rank=7.76  mean_pair_cos=0.350
- seed 2: pred_cos=0.8109  copy_cos=0.4971  graph_avg_cos=0.1848  eff_rank=7.97  mean_pair_cos=0.328
- seed 3: pred_cos=0.8353  copy_cos=0.4949  graph_avg_cos=0.2337  eff_rank=8.10  mean_pair_cos=0.347
- seed 4: pred_cos=0.8317  copy_cos=0.5129  graph_avg_cos=0.2601  eff_rank=7.88  mean_pair_cos=0.382

## sequential-ablation
- seed 0: pred_cos=0.7399  copy_cos=0.5002  graph_avg_cos=0.1274  eff_rank=6.28  mean_pair_cos=0.073
- seed 1: pred_cos=0.7453  copy_cos=0.5159  graph_avg_cos=0.1114  eff_rank=5.94  mean_pair_cos=0.073
- seed 2: pred_cos=0.6909  copy_cos=0.5370  graph_avg_cos=0.0895  eff_rank=6.22  mean_pair_cos=0.082
- seed 3: pred_cos=0.7670  copy_cos=0.5019  graph_avg_cos=0.1329  eff_rank=6.50  mean_pair_cos=0.076
- seed 4: pred_cos=0.7401  copy_cos=0.5005  graph_avg_cos=0.1188  eff_rank=6.55  mean_pair_cos=0.088

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.8164  seq_cos=0.7399  wilcoxon_p=9.627e-16  win_rate=0.757  n=255
- seed 1: graph_cos=0.8103  seq_cos=0.7453  wilcoxon_p=3.009e-15  win_rate=0.769  n=255
- seed 2: graph_cos=0.8109  seq_cos=0.6909  wilcoxon_p=4.719e-27  win_rate=0.835  n=255
- seed 3: graph_cos=0.8353  seq_cos=0.7670  wilcoxon_p=9.396e-24  win_rate=0.725  n=255
- seed 4: graph_cos=0.8317  seq_cos=0.7401  wilcoxon_p=6.983e-23  win_rate=0.776  n=255
