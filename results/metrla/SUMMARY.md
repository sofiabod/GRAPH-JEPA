# eval summary: metrla

seeds: [1, 2, 3, 4]

## metrla
- seed 1: pred_cos=0.9918  copy_cos=0.9764  graph_avg_cos=0.6608  eff_rank=20.48  mean_pair_cos=0.015
- seed 2: pred_cos=0.9942  copy_cos=0.9849  graph_avg_cos=0.6395  eff_rank=17.55  mean_pair_cos=0.009
- seed 3: pred_cos=0.9939  copy_cos=0.9845  graph_avg_cos=0.6571  eff_rank=18.28  mean_pair_cos=0.014
- seed 4: pred_cos=0.9947  copy_cos=0.9796  graph_avg_cos=0.6586  eff_rank=15.77  mean_pair_cos=0.010

## sequential-ablation
- seed 1: pred_cos=0.9965  copy_cos=0.9346  graph_avg_cos=0.2360  eff_rank=7.97  mean_pair_cos=0.001
- seed 2: pred_cos=0.9977  copy_cos=0.9713  graph_avg_cos=0.1675  eff_rank=7.65  mean_pair_cos=0.000
- seed 3: pred_cos=0.9987  copy_cos=0.9764  graph_avg_cos=0.1094  eff_rank=6.39  mean_pair_cos=-0.000
- seed 4: pred_cos=0.9988  copy_cos=0.9872  graph_avg_cos=0.2123  eff_rank=6.85  mean_pair_cos=-0.000

## eval 2 (paired graph vs sequential)
- seed 1: graph_cos=0.9918  seq_cos=0.9965  wilcoxon_p=1.000e+00  win_rate=0.149  n=1476
- seed 2: graph_cos=0.9942  seq_cos=0.9977  wilcoxon_p=1.000e+00  win_rate=0.124  n=1476
- seed 3: graph_cos=0.9939  seq_cos=0.9987  wilcoxon_p=1.000e+00  win_rate=0.007  n=1476
- seed 4: graph_cos=0.9947  seq_cos=0.9988  wilcoxon_p=1.000e+00  win_rate=0.018  n=1476
