# eval summary: baci_gravity

seeds: [0, 1, 2, 3, 4]

## baci_gravity
- seed 0: pred_cos=0.9752  copy_cos=0.2935  graph_avg_cos=0.2018  eff_rank=7.23  mean_pair_cos=0.283
- seed 1: pred_cos=0.9445  copy_cos=0.3019  graph_avg_cos=0.1314  eff_rank=6.93  mean_pair_cos=0.234
- seed 2: pred_cos=0.9139  copy_cos=0.3879  graph_avg_cos=0.0941  eff_rank=6.97  mean_pair_cos=0.258
- seed 3: pred_cos=0.9569  copy_cos=0.4045  graph_avg_cos=0.2649  eff_rank=7.16  mean_pair_cos=0.246
- seed 4: pred_cos=0.9295  copy_cos=0.3838  graph_avg_cos=0.1293  eff_rank=7.03  mean_pair_cos=0.229

## sequential-ablation
- seed 0: pred_cos=0.7316  copy_cos=0.5728  graph_avg_cos=0.1474  eff_rank=5.28  mean_pair_cos=0.015
- seed 1: pred_cos=0.7815  copy_cos=0.5344  graph_avg_cos=0.1645  eff_rank=4.85  mean_pair_cos=0.016
- seed 2: pred_cos=0.7252  copy_cos=0.5822  graph_avg_cos=0.1272  eff_rank=4.94  mean_pair_cos=0.019
- seed 3: pred_cos=0.7580  copy_cos=0.5660  graph_avg_cos=0.1570  eff_rank=5.10  mean_pair_cos=0.010
- seed 4: pred_cos=0.7332  copy_cos=0.5857  graph_avg_cos=0.2050  eff_rank=5.43  mean_pair_cos=0.011

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.9752  seq_cos=0.7316  wilcoxon_p=1.367e-31  win_rate=1.000  n=180
- seed 1: graph_cos=0.9445  seq_cos=0.7815  wilcoxon_p=9.148e-30  win_rate=0.928  n=180
- seed 2: graph_cos=0.9139  seq_cos=0.7252  wilcoxon_p=3.225e-30  win_rate=0.928  n=180
- seed 3: graph_cos=0.9569  seq_cos=0.7580  wilcoxon_p=1.879e-31  win_rate=0.972  n=180
- seed 4: graph_cos=0.9295  seq_cos=0.7332  wilcoxon_p=5.633e-31  win_rate=0.961  n=180
