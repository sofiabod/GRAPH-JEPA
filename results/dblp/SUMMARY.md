# eval summary: dblp

seeds: [0, 1, 2, 3, 4]

## dblp
- seed 0: pred_cos=0.4809  copy_cos=0.5312  graph_avg_cos=0.3439  eff_rank=9.48  mean_pair_cos=0.171
- seed 1: pred_cos=0.3986  copy_cos=0.5206  graph_avg_cos=0.3439  eff_rank=10.09  mean_pair_cos=0.178
- seed 2: pred_cos=0.5296  copy_cos=0.4842  graph_avg_cos=0.2679  eff_rank=7.72  mean_pair_cos=0.212
- seed 3: pred_cos=0.5704  copy_cos=0.4264  graph_avg_cos=0.2494  eff_rank=6.30  mean_pair_cos=0.243
- seed 4: pred_cos=0.5194  copy_cos=0.5107  graph_avg_cos=0.3067  eff_rank=8.31  mean_pair_cos=0.208

## sequential-ablation
- seed 0: pred_cos=0.6104  copy_cos=0.5046  graph_avg_cos=0.3312  eff_rank=3.48  mean_pair_cos=0.257
- seed 1: pred_cos=0.5746  copy_cos=0.4954  graph_avg_cos=0.3394  eff_rank=4.20  mean_pair_cos=0.245
- seed 2: pred_cos=0.6058  copy_cos=0.4961  graph_avg_cos=0.3124  eff_rank=4.38  mean_pair_cos=0.242
- seed 3: pred_cos=0.3324  copy_cos=0.5570  graph_avg_cos=0.3449  eff_rank=5.42  mean_pair_cos=0.147
- seed 4: pred_cos=0.4559  copy_cos=0.5474  graph_avg_cos=0.3390  eff_rank=5.40  mean_pair_cos=0.193

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.4809  seq_cos=0.6104  wilcoxon_p=1.000e+00  win_rate=0.016  n=800
- seed 1: graph_cos=0.3986  seq_cos=0.5746  wilcoxon_p=1.000e+00  win_rate=0.000  n=800
- seed 2: graph_cos=0.5296  seq_cos=0.6058  wilcoxon_p=1.000e+00  win_rate=0.043  n=800
- seed 3: graph_cos=0.5704  seq_cos=0.3324  wilcoxon_p=4.297e-129  win_rate=0.978  n=800
- seed 4: graph_cos=0.5194  seq_cos=0.4559  wilcoxon_p=7.539e-100  win_rate=0.879  n=800
