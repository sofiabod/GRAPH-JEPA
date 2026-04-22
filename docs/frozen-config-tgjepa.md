# frozen config: temporal graph jepa

these values are frozen before any eval run. any change requires a new test split.

| parameter | value |
|---|---|
| optimizer | adamw |
| lr | 3e-4 |
| weight_decay | 0.01 |
| lr schedule | cosine anneal to 1e-5 |
| batch size | 16 |
| max epochs | 200 |
| early stopping patience | 30 |
| mask ratio | 0.20 |
| context window k | 4 |
| ema momentum start | 0.996 |
| ema momentum end | 1.0 |
| gradient clip max norm | 1.0 |
| lambda_reg | 0.01 |
| bcs num slices | 1024 |
| bcs lmbd | 0.1 |
| inactive node features | zeros |
| encoder hidden dim | 256 |
| encoder layers | 3 |
| encoder heads | 4 |
| predictor layers | 2 |
| predictor heads | 4 |
| temporal stride | 1 |
