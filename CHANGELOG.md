# Changelog

All notable changes to GRAPH-JEPA-2 are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `SPEC.md` — pre-registered hypotheses (H1a–H7), experimental contract, frozen hyperparameters, falsification criteria, and audit log for the 7-dataset matrix.
- `LICENSE` (MIT).
- `CITATION.cff` for software citation.
- `REPRODUCE.md` — reproduction guide with per-experiment compute budget (~190 H100-hr total).
- `pyproject.toml` with optional-deps groups (`[experiments]`, `[modal]`, `[dev]`) and ruff lint config.
- `.github/workflows/test.yml` — CI for pytest + ruff on push and pull request.
- `scripts/aggregate.py` — canonical aggregator for per-seed JSON → `summary.json` + `FINDINGS.md`.
- `src/losses/rdmreg.py` — Rectified Distribution Matching Regularization, ported from RLpJEPA (Kuang et al. 2026, arXiv:2602.01456) under MIT license. Drop-in alternative to BCS for the T3.7 ablation.
- `src/data/rewire.py` — degree-preserving edge rewiring for the T2.2 random-edge control.
- `src/data/feature_smoothing.py` — EMA-smoothed node features for the T1.2 TGBN-Genre re-attempt.
- `rigour.md` — full operational experiment plan with the 7-dataset matrix (METR-LA, TGBN-Trade, TGBN-Genre, TGBN-Genre-v2, EU-Email, JODIE-Reddit, JODIE-Wikipedia).

### Fixed (post 2026-05-07 stress-test audit)
- **CRITICAL — capacity mismatch:** SequentialMLP rewritten to use 2-linear residual FFN blocks (`src/models/sequential_encoder.py`). Param count moved from ~200K (single-linear) to ~495K (2-linear blocks at n_layers=3) to match GraphEncoder ~400K. Existing 5-seed TGBN-Trade results require re-running.
- **HIGH — BCS lambda_reg double-scaling:** `src/losses/prediction.py:20` refactored from `lambda_reg * sigreg["loss"]` to explicit `inv_coeff * invariance + bcs_coeff * bcs`. Math is identical; structure is now transparent.
- **MEDIUM — phrasing:** `results/tgbn_trade/results.md` corrected "p < 10⁻¹²" → "p ≤ 10⁻¹²" (seed 2 is exactly 1.00e-12, not less than).
- **MEDIUM — Bonferroni reporting:** `results/tgbn_trade/results.md` now explicitly states the 3-test family Bonferroni-corrected p (≤ 3 × 10⁻¹²).
- **CRITICAL for Genre-v2 only — active-week filter:** `src/data/tgb_builder.py::build_tgbn_genre_v2_graphs_from_raw` docstring documents the inactive-node-mask degeneracy and points to the mitigation (`mask_active_only` flag in TemporalGraphDataset). TGBN-Trade unaffected.

### Notes
- `src/eval/eval_runner.py:_eval3_multistep_rollout` is implemented (lines 181–280) and wired into `run_all_with_eval2` at line 65; existing per-seed JSON files report `"status": "not_implemented"` because they pre-date the implementation. Re-running eval on existing checkpoints will populate `eval3` with horizon-{1,2,4} cosines.

## [0.1.0] - prior to 2026-05-07

Initial 5-seed TGBN-Trade result: Δcos = 0.081 over Sequential-JEPA, paired Wilcoxon p ≤ 10⁻¹², 87% per-node win rate, eff_rank = 8.04 ± 0.20. Pending validation post capacity-fix re-run.
