# eval summary: icio

seeds: [0, 1, 2, 3, 4]

## icio
- seed 0: pred_cos=0.9829  copy_cos=0.1501  graph_avg_cos=0.0825  eff_rank=2.43  mean_pair_cos=0.667
- seed 1: pred_cos=0.9657  copy_cos=0.1127  graph_avg_cos=-0.0273  eff_rank=2.54  mean_pair_cos=0.608
- seed 2: pred_cos=0.9795  copy_cos=0.1905  graph_avg_cos=0.1119  eff_rank=2.54  mean_pair_cos=0.569
- seed 3: pred_cos=0.9630  copy_cos=0.2696  graph_avg_cos=0.1945  eff_rank=2.54  mean_pair_cos=0.567
- seed 4: pred_cos=0.9868  copy_cos=0.0801  graph_avg_cos=0.0253  eff_rank=2.36  mean_pair_cos=0.695

## sequential-ablation
- seed 0: pred_cos=0.9354  copy_cos=0.3335  graph_avg_cos=0.4611  eff_rank=3.32  mean_pair_cos=0.061
- seed 1: pred_cos=0.9305  copy_cos=0.3047  graph_avg_cos=0.3160  eff_rank=3.23  mean_pair_cos=0.098
- seed 2: pred_cos=0.9569  copy_cos=0.1618  graph_avg_cos=0.3429  eff_rank=3.17  mean_pair_cos=0.040
- seed 3: pred_cos=0.9303  copy_cos=0.2463  graph_avg_cos=0.3672  eff_rank=3.22  mean_pair_cos=0.029
- seed 4: pred_cos=0.9366  copy_cos=0.2515  graph_avg_cos=0.2964  eff_rank=3.18  mean_pair_cos=0.046

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.9829  seq_cos=0.9354  wilcoxon_p=7.451e-09  win_rate=1.000  n=27
- seed 1: graph_cos=0.9657  seq_cos=0.9305  wilcoxon_p=7.451e-09  win_rate=1.000  n=27
- seed 2: graph_cos=0.9795  seq_cos=0.9569  wilcoxon_p=7.451e-08  win_rate=0.926  n=27
- seed 3: graph_cos=0.9630  seq_cos=0.9303  wilcoxon_p=7.451e-09  win_rate=1.000  n=27
- seed 4: graph_cos=0.9868  seq_cos=0.9366  wilcoxon_p=7.451e-09  win_rate=1.000  n=27
