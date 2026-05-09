# Reproduction Guide — GRAPH-JEPA-2

This guide reproduces every number in the paper from source. All commands assume the repository root as the working directory and Python ≥ 3.11.

## 1. Environment

```bash
# local development
python -m venv .venv && . .venv/bin/activate
pip install -e '.[experiments,dev]'

# cloud (Modal, for the full sweep)
modal secret create graph-jepa-secret HF_TOKEN=...
```

`requirements.txt` ships floating version floors for quick install. For exact reproducibility, `pyproject.toml` pins the versions used in the paper.

## 2. Data pipeline

All datasets are public. Builders live under `src/data/`.

```bash
# TGB datasets (TGBN-Trade primary, also TGBN-Genre, TGBN-Genre-v2)
python experiments/download_benchmarks.py --dataset tgbn-trade
python experiments/download_benchmarks.py --dataset tgbn-genre

# METR-LA (flagship benchmark — via PyG Temporal)
python -c "from src.data.metrla_builder import build_and_save; build_and_save(out='data/metrla')"

# JODIE datasets
python experiments/download_jodie.py --dataset reddit
python experiments/download_jodie.py --dataset wikipedia

# EU-Email (SNAP)
python experiments/download_eu_email.py
```

Every builder writes `data/<dataset>/graphs.pt` (list of PyG Data objects) and `data/<dataset>/meta.json`.

## 3. Compute budget

Per-experiment estimates on H100 80GB:

| Experiment | Cells | Seeds | H100-hr |
|---|---|---|---|
| T1.0 METR-LA training | 1 | 5 | 10 |
| T1.1 TGBN-Trade (extend to 10 seeds) | 1 | 5 (additional) | 6 |
| T1.2 TGBN-Genre + EMA features | 3 (τ values) | 3 | 8 |
| T1.3 TGBN-Genre-v2 + bipartite head | 1 | 5 | 8 |
| T1.4 EU-Email monthly agg sweep | 3 (windows) | 3 | 6 |
| T1.5 JODIE-Reddit | 1 | 5 | 10 |
| T1.6 JODIE-Wikipedia | 1 | 5 | 10 |
| T2.1 Sequential-JEPA-rank-8 | 1 | 5 | 20 |
| T2.2 Random-edge control | 1 | 5 | 20 |
| T2.5 Multi-step rollout | 4 horizons | 5 | 4 |
| T3.x robustness (hparam sweep, encoder swap, etc.) | varies | varies | ~80 |
| T4.x baselines (contrastive, MAE, supervised) | varies | varies | ~50 |

**Total: ~232 H100-hr** for the full plan. Modal H100 concurrency 8 → ~30 hr wall-clock. Modal H100 concurrency 16 → ~15 hr wall-clock (request quota in advance).

## 4. Running the experiments

Each Tier 1 dataset has its own training script.

```bash
# TGBN-Trade (5 seeds, extend to 10)
for seed in 0 1 2 3 4 5 6 7 8 9; do
  python experiments/train_tgjepa.py --config configs/tgbn_trade.yaml --seed $seed
done

# METR-LA (5 seeds)
for seed in 0 1 2 3 4; do
  python experiments/train_metrla.py --config configs/metrla.yaml --seed $seed
done

# Modal cloud variants
modal run --detach experiments/train_tgjepa.py::run_seeds --config tgbn_trade --n-seeds 10
modal run --detach experiments/train_metrla.py::run_seeds --config metrla --n-seeds 5
```

Each training run writes `results/<dataset>/seed{N}/` with:
- `best_model.pt`
- `eval1_<dataset>.json` — one-step prediction metrics
- `eval2.json` — paired Graph-JEPA vs Sequential-JEPA
- `eval3_rollout.json` — multi-step rollout (after T2.5 lands)
- `train_log_graph.json`, `train_log_seq.json` — per-epoch losses

## 5. Mechanism ablations (Tier 2)

After Tier 1 datasets are trained:

```bash
# T2.1 — Sequential-JEPA with rank-8 bottleneck
python experiments/train_tgjepa.py \
    --config configs/tgbn_trade.yaml \
    --override "model.sequential.rank_bottleneck=8" \
    --seeds 0..4

# T2.2 — random-edge control (degree-preserving rewire)
python experiments/train_tgjepa.py \
    --config configs/tgbn_trade.yaml \
    --override "data.rewire_seed=42" \
    --seeds 0..4

# T2.4 — linear probe on frozen Graph-JEPA representations
python experiments/eval_probe.py \
    --checkpoint results/tgbn_trade/seed{0..4}/best_model.pt \
    --dataset tgbn_trade

# T2.5 — multi-step rollout (uses existing checkpoints, no retraining)
python experiments/eval_rollout.py \
    --checkpoint results/tgbn_trade/seed{0..4}/best_model.pt \
    --horizons 1,2,4,8

# T2.6 — sparsity metric (uses existing checkpoints, no retraining)
python experiments/eval_sparsity.py \
    --checkpoint results/tgbn_trade/seed{0..4}/best_model.pt
```

## 6. Aggregating and figure-making

After all experiments reduce to per-seed JSON:

```bash
# aggregate per-dataset summary tables
python scripts/aggregate.py --datasets tgbn_trade,metrla --out results/FINDINGS.md

# regenerate paper figures from results/
python scripts/make_figures.py --out paper/neurips/figs

# regenerate paper tables (LaTeX)
python scripts/make_tables.py --out paper/neurips/tables
```

## 7. Building the paper

```bash
cd paper/neurips
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supp && pdflatex supp
```

## 8. Verification — does my run match the paper?

After running TGBN-Trade for 5 seeds, compare against the published numbers in `results/tgbn_trade/results.md`:

| Metric | Expected | Tolerance |
|---|---|---|
| mean_pred_cos (Graph) | 0.827 ± 0.010 | ±0.02 |
| mean_pred_cos (Sequential) | 0.746 ± 0.016 | ±0.03 |
| effective_rank (Graph) | 8.04 ± 0.20 | ±0.5 |
| effective_rank (Sequential) | 19.12 ± 0.65 | ±1.0 |
| Wilcoxon p (paired) | < 10⁻¹² | (must be < 10⁻⁹) |
| win rate | 0.873 ± 0.015 | ±0.03 |

If your numbers fall outside these bands, see `SPEC.md` §7 (falsification summary) and check determinism (`src/utils/seed.py` — verify `cudnn.deterministic=True`).
