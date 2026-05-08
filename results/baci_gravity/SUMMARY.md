# eval summary: baci_gravity

seeds: [0]

## baci_gravity
- seed 0: pred_cos=0.9752  copy_cos=0.2936  graph_avg_cos=0.2019  eff_rank=7.23  mean_pair_cos=0.282

## sequential-ablation
- seed 0: pred_cos=0.7316  copy_cos=0.5728  graph_avg_cos=0.1474  eff_rank=5.28  mean_pair_cos=0.015

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.9752  seq_cos=0.7316  wilcoxon_p=1.367e-31  win_rate=1.000  n=180
