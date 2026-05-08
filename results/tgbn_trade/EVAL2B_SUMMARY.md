# eval2b (shared-target paired comparison): tgbn_trade

seeds: [0, 1, 2, 3, 4]
modes: ['graph', 'sequential']

## shared_target_mode = `graph`
- seed 0: graph_cos=0.8153  seq_cos=0.1844  wilcoxon_p=1.573e-41  win_rate=0.882  n=255
- seed 1: graph_cos=0.8434  seq_cos=0.2494  wilcoxon_p=1.573e-41  win_rate=0.882  n=255
- seed 2: graph_cos=0.8272  seq_cos=0.1200  wilcoxon_p=7.027e-44  win_rate=1.000  n=255
- seed 3: graph_cos=0.8237  seq_cos=0.1992  wilcoxon_p=7.027e-44  win_rate=1.000  n=255
- seed 4: graph_cos=0.8237  seq_cos=0.0529  wilcoxon_p=2.243e-41  win_rate=0.878  n=255

## shared_target_mode = `sequential`
- seed 0: graph_cos=0.1472  seq_cos=0.7399  wilcoxon_p=1.000e+00  win_rate=0.067  n=255
- seed 1: graph_cos=0.2160  seq_cos=0.7453  wilcoxon_p=1.000e+00  win_rate=0.000  n=255
- seed 2: graph_cos=0.0805  seq_cos=0.6909  wilcoxon_p=1.000e+00  win_rate=0.141  n=255
- seed 3: graph_cos=0.1754  seq_cos=0.7670  wilcoxon_p=1.000e+00  win_rate=0.000  n=255
- seed 4: graph_cos=0.0297  seq_cos=0.7401  wilcoxon_p=1.000e+00  win_rate=0.110  n=255
