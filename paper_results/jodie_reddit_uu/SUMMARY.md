# eval summary: jodie_reddit_uu

seeds: [0, 1, 2, 3, 4]

## jodie_reddit_uu
- seed 0: pred_cos=0.6348  copy_cos=0.4516  graph_avg_cos=0.3321  eff_rank=13.18  mean_pair_cos=0.028
- seed 1: pred_cos=0.6852  copy_cos=0.4272  graph_avg_cos=0.2782  eff_rank=12.51  mean_pair_cos=0.041
- seed 2: pred_cos=0.7260  copy_cos=0.3799  graph_avg_cos=0.2812  eff_rank=11.00  mean_pair_cos=0.054
- seed 3: pred_cos=0.7041  copy_cos=0.3892  graph_avg_cos=0.2321  eff_rank=10.07  mean_pair_cos=0.092
- seed 4: pred_cos=0.7086  copy_cos=0.3532  graph_avg_cos=0.2194  eff_rank=10.44  mean_pair_cos=0.069

## sequential-ablation
- seed 0: pred_cos=0.5848  copy_cos=0.4721  graph_avg_cos=0.3290  eff_rank=5.36  mean_pair_cos=0.003
- seed 1: pred_cos=0.5699  copy_cos=0.4863  graph_avg_cos=0.3276  eff_rank=5.30  mean_pair_cos=0.003
- seed 2: pred_cos=0.5815  copy_cos=0.4703  graph_avg_cos=0.3256  eff_rank=5.33  mean_pair_cos=0.003
- seed 3: pred_cos=0.5885  copy_cos=0.4464  graph_avg_cos=0.3466  eff_rank=5.39  mean_pair_cos=0.003
- seed 4: pred_cos=0.6073  copy_cos=0.4268  graph_avg_cos=0.3092  eff_rank=5.47  mean_pair_cos=0.002

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.6348  seq_cos=0.5848  wilcoxon_p=1.923e-03  win_rate=0.488  n=500
- seed 1: graph_cos=0.6852  seq_cos=0.5699  wilcoxon_p=4.096e-37  win_rate=0.668  n=500
- seed 2: graph_cos=0.7260  seq_cos=0.5815  wilcoxon_p=1.041e-65  win_rate=0.890  n=500
- seed 3: graph_cos=0.7041  seq_cos=0.5885  wilcoxon_p=3.038e-42  win_rate=0.706  n=500
- seed 4: graph_cos=0.7086  seq_cos=0.6073  wilcoxon_p=5.451e-57  win_rate=0.834  n=500
