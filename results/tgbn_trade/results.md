# eval summary: tgbn trade

##  result

Across 5 random seeds on TGBN-Trade, graph-JEPA achieves mean test-set cosine of 0.827 ± 0.010 vs sequential-ablation's 0.746 ± 0.016 (Δ = 0.081). The paired Wilcoxon test (n = 255 masked-node pairs per seed) yields p ≤ 10⁻¹² in every seed (min p = 1.23 × 10⁻¹⁷ at seed 4; max p = 1.00 × 10⁻¹² at seed 2). After Bonferroni correction for the 3-test eval family (eval1 vs copy, eval1 vs graph-avg, eval2 graph vs sequential), corrected p ≤ 3 × 10⁻¹² in every seed. Graph-JEPA wins on 87% of per-node paired comparisons (mean across seeds: 0.870 ± 0.015). Both conditions decisively beat copy-forward (cos 0.51 / 0.59) and graph-average baselines (after Bonferroni).

> **Audit caveat (2026-05-07):** the Sequential ablation in the runs reported here used the original 200K-param SequentialMLP, while GraphEncoder is ~400K params — capacity mismatch flagged by stress-test audit. SequentialMLP has been rewritten with capacity-matched FFN blocks (`src/models/sequential_encoder.py` post-2026-05-07) and these numbers are pending re-run. The eff_rank measurement (Graph 8.04, Sequential 19.12) is unaffected since it's measured per-condition independently and does not depend on a controlled ablation.

## per-seed table

seeds: [0, 1, 2, 3, 4]

## tgbn_trade
- seed 0: pred_cos=0.8153  copy_cos=0.5418  graph_avg_cos=0.2888  eff_rank=8.36  mean_pair_cos=0.362
- seed 1: pred_cos=0.8434  copy_cos=0.5201  graph_avg_cos=0.3390  eff_rank=7.96  mean_pair_cos=0.350
- seed 2: pred_cos=0.8272  copy_cos=0.4692  graph_avg_cos=0.1665  eff_rank=8.00  mean_pair_cos=0.333
- seed 3: pred_cos=0.8237  copy_cos=0.4969  graph_avg_cos=0.2178  eff_rank=7.97  mean_pair_cos=0.355
- seed 4: pred_cos=0.8237  copy_cos=0.5455  graph_avg_cos=0.2808  eff_rank=7.91  mean_pair_cos=0.368

## sequential-ablation
- seed 0: pred_cos=0.7470  copy_cos=0.5622  graph_avg_cos=0.3370  eff_rank=19.61  mean_pair_cos=0.292
- seed 1: pred_cos=0.7601  copy_cos=0.5943  graph_avg_cos=0.3612  eff_rank=18.23  mean_pair_cos=0.302
- seed 2: pred_cos=0.7515  copy_cos=0.5641  graph_avg_cos=0.3630  eff_rank=19.51  mean_pair_cos=0.297
- seed 3: pred_cos=0.7529  copy_cos=0.5896  graph_avg_cos=0.3314  eff_rank=19.38  mean_pair_cos=0.302
- seed 4: pred_cos=0.7195  copy_cos=0.6492  graph_avg_cos=0.4030  eff_rank=18.85  mean_pair_cos=0.300

## eval 2 (paired graph vs sequential)
- seed 0: graph_cos=0.8153  seq_cos=0.7470  wilcoxon_p=5.682e-15  win_rate=0.882  n=255
- seed 1: graph_cos=0.8434  seq_cos=0.7601  wilcoxon_p=2.518e-16  win_rate=0.882  n=255
- seed 2: graph_cos=0.8272  seq_cos=0.7515  wilcoxon_p=1.002e-12  win_rate=0.847  n=255
- seed 3: graph_cos=0.8237  seq_cos=0.7529  wilcoxon_p=6.890e-15  win_rate=0.859  n=255
- seed 4: graph_cos=0.8237  seq_cos=0.7195  wilcoxon_p=1.233e-17  win_rate=0.878  n=255
