# eval summary: tgbn_trade

seeds: [0, 1, 2, 3, 4]

## tgbn_trade
- seed 0: pred_cos=0.8153  copy_cos=0.5418  graph_avg_cos=0.2888  eff_rank=8.36  mean_pair_cos=0.362
- seed 1: pred_cos=0.8434  copy_cos=0.5201  graph_avg_cos=0.3390  eff_rank=7.96  mean_pair_cos=0.350
- seed 2: pred_cos=0.8272  copy_cos=0.4692  graph_avg_cos=0.1665  eff_rank=8.00  mean_pair_cos=0.333
- seed 3: pred_cos=0.8237  copy_cos=0.4969  graph_avg_cos=0.2178  eff_rank=7.97  mean_pair_cos=0.355
- seed 4: pred_cos=0.8237  copy_cos=0.5455  graph_avg_cos=0.2808  eff_rank=7.91  mean_pair_cos=0.368

## sequential-ablation
- seed 0: pred_cos=0.7399  copy_cos=0.5002  graph_avg_cos=0.1274  eff_rank=6.28  mean_pair_cos=0.073
- seed 1: pred_cos=0.7453  copy_cos=0.5159  graph_avg_cos=0.1114  eff_rank=5.94  mean_pair_cos=0.073
- seed 2: pred_cos=0.6909  copy_cos=0.5370  graph_avg_cos=0.0895  eff_rank=6.22  mean_pair_cos=0.082
- seed 3: pred_cos=0.7670  copy_cos=0.5019  graph_avg_cos=0.1329  eff_rank=6.50  mean_pair_cos=0.076
- seed 4: pred_cos=0.7401  copy_cos=0.5005  graph_avg_cos=0.1188  eff_rank=6.55  mean_pair_cos=0.088

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.8153  seq_cos=0.7399  wilcoxon_p=1.584e-15  win_rate=0.753  n=255
- seed 1: graph_cos=0.8434  seq_cos=0.7453  wilcoxon_p=1.691e-24  win_rate=0.839  n=255
- seed 2: graph_cos=0.8272  seq_cos=0.6909  wilcoxon_p=9.326e-44  win_rate=0.984  n=255
- seed 3: graph_cos=0.8237  seq_cos=0.7670  wilcoxon_p=1.333e-14  win_rate=0.714  n=255
- seed 4: graph_cos=0.8237  seq_cos=0.7401  wilcoxon_p=8.184e-19  win_rate=0.757  n=255
