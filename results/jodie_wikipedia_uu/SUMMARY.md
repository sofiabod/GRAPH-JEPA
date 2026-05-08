# eval summary: jodie_wikipedia_uu

seeds: [0, 1, 2, 3, 4]

## jodie_wikipedia_uu
- seed 0: pred_cos=0.6892  copy_cos=0.3134  graph_avg_cos=0.4515  eff_rank=3.60  mean_pair_cos=0.314
- seed 1: pred_cos=0.6889  copy_cos=0.3323  graph_avg_cos=0.4639  eff_rank=3.48  mean_pair_cos=0.317
- seed 2: pred_cos=0.7318  copy_cos=0.3110  graph_avg_cos=0.4317  eff_rank=3.45  mean_pair_cos=0.319
- seed 3: pred_cos=0.7200  copy_cos=0.3256  graph_avg_cos=0.4512  eff_rank=3.25  mean_pair_cos=0.309
- seed 4: pred_cos=0.6942  copy_cos=0.3093  graph_avg_cos=0.4199  eff_rank=3.41  mean_pair_cos=0.322

## sequential-ablation
- seed 0: pred_cos=0.6499  copy_cos=0.4311  graph_avg_cos=0.6948  eff_rank=3.23  mean_pair_cos=0.205
- seed 1: pred_cos=0.6262  copy_cos=0.4438  graph_avg_cos=0.6960  eff_rank=3.14  mean_pair_cos=0.207
- seed 2: pred_cos=0.6426  copy_cos=0.4286  graph_avg_cos=0.6610  eff_rank=3.19  mean_pair_cos=0.213
- seed 3: pred_cos=0.6875  copy_cos=0.4230  graph_avg_cos=0.6403  eff_rank=3.16  mean_pair_cos=0.218
- seed 4: pred_cos=0.6299  copy_cos=0.4328  graph_avg_cos=0.6371  eff_rank=3.09  mean_pair_cos=0.229

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.6892  seq_cos=0.6499  wilcoxon_p=9.444e-01  win_rate=0.286  n=500
- seed 1: graph_cos=0.6889  seq_cos=0.6262  wilcoxon_p=6.421e-01  win_rate=0.358  n=500
- seed 2: graph_cos=0.7318  seq_cos=0.6426  wilcoxon_p=7.521e-59  win_rate=0.898  n=500
- seed 3: graph_cos=0.7200  seq_cos=0.6875  wilcoxon_p=3.259e-58  win_rate=0.814  n=500
- seed 4: graph_cos=0.6942  seq_cos=0.6299  wilcoxon_p=9.517e-01  win_rate=0.274  n=500
