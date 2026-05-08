# eval2b (shared-target paired comparison): baci_gravity

seeds: [0, 1, 2, 3, 4]
modes: ['graph', 'sequential']

## shared_target_mode = `graph`
- seed 0: graph_cos=0.9752  seq_cos=0.1959  wilcoxon_p=1.367e-31  win_rate=1.000  n=180
- seed 1: graph_cos=0.9445  seq_cos=0.2217  wilcoxon_p=1.438e-31  win_rate=0.989  n=180
- seed 2: graph_cos=0.9139  seq_cos=0.1180  wilcoxon_p=1.391e-31  win_rate=0.994  n=180
- seed 3: graph_cos=0.9569  seq_cos=0.1948  wilcoxon_p=1.367e-31  win_rate=1.000  n=180
- seed 4: graph_cos=0.9295  seq_cos=0.1220  wilcoxon_p=1.391e-31  win_rate=0.994  n=180

## shared_target_mode = `sequential`
- seed 0: graph_cos=0.1389  seq_cos=0.7316  wilcoxon_p=1.000e+00  win_rate=0.000  n=180
- seed 1: graph_cos=0.1766  seq_cos=0.7815  wilcoxon_p=1.000e+00  win_rate=0.011  n=180
- seed 2: graph_cos=0.0629  seq_cos=0.7252  wilcoxon_p=1.000e+00  win_rate=0.006  n=180
- seed 3: graph_cos=0.1319  seq_cos=0.7580  wilcoxon_p=1.000e+00  win_rate=0.000  n=180
- seed 4: graph_cos=0.0760  seq_cos=0.7332  wilcoxon_p=1.000e+00  win_rate=0.000  n=180
