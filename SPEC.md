# SPEC: GRAPH-JEPA-2 — Temporal Graph as Input to a JEPA-based World Model

**Pre-registered specification.** This document fixes the hypotheses, experimental contract, frozen hyperparameters, and falsification criteria *before* the experiments run. Any deviation from this spec must be documented in `SPEC_AUDIT.md` with rationale.

**Author:** Sofia Bodnar (University of Waterloo, undergraduate thesis)
**Created:** 2026-05-07
**Last updated:** 2026-05-07
**Companion document:** `rigour.md` (operational experiment plan)

---

## 0. Executive summary

### 0.1 The paper in one paragraph

We present GRAPH-JEPA-2, the first joint-embedding predictive architecture trained on temporal graph data as a world model. The architecture (GATv2 encoder + bidirectional Transformer predictor + EMA target encoder + BCS / LeJEPA-family anti-collapse regularizer) consumes a sequence of past graph snapshots and predicts the latent state of the future graph, autoregressively rolled out to horizon h. We test the hypothesis that JEPA's straightening property (proven by Wang et al. 2026 for image-JEPA) transfers to graph-structured observations, manifesting as an emergent low-rank attractor in the latent space. On TGBN-Trade across 5 seeds, the model achieves Δcos = 0.081 over a capacity-matched non-graph ablation (paired Wilcoxon p < 10⁻¹²) with effective rank 8.04 ± 0.20 — quantitatively matching Wang's d=8 finding for image-JEPA. We extend to METR-LA (the standard temporal-graph forecasting benchmark) for the world-model rollout claim, and characterise empirical scope across a 7-dataset matrix.

### 0.2 The four contributions

1. **First node-level temporal Graph-JEPA.** Skenderi et al. 2025 did graph-level static; we extend to per-node temporal — explicitly named as future work in their conclusion (Skenderi et al. 2025, p. 12).

2. **Empirical evidence that JEPA-induced temporal straightening transfers from images to graphs.** Wang et al. 2026 prove for image-JEPA that channel dim d=8 is sufficient and that implicit straightening occurs in any JEPA training. We observe eff_rank = 8.04 ± 0.20 emergent on TGBN-Trade, a quantitative match.

3. **Causal mechanism for the Graph-JEPA advantage.** A rank-controlled Sequential-JEPA ablation (forced low-rank projection) and a degree-preserving randomised-edge control isolate graph-induced compression as the load-bearing mechanism, not architectural capacity or any-graph-structure inductive bias.

4. **Empirical scope of node-level temporal JEPA on graphs.** A 7-dataset matrix (METR-LA, TGBN-Trade, TGBN-Genre, TGBN-Genre-v2, EU-Email, JODIE-Reddit, JODIE-Wikipedia) characterises the autocorrelation prerequisite for JEPA on graphs, with documented architectural responses (EMA feature smoothing, type-conditioned predictor, longer aggregation windows) for non-stationary datasets.

---

## 1. Positioning vs prior work

GRAPH-JEPA-2 sits at the intersection of three lineages:

| Lineage | Representative work | What we share | What's new |
|---|---|---|---|
| Graph-JEPA | Skenderi et al. 2025 (TMLR) | JEPA + GNN encoder + EMA + stop-grad anti-collapse | temporal extension; node-level prediction; no hyperbolic 2D target |
| JEPA world models | I-JEPA (Assran et al. 2023), V-JEPA / V-JEPA 2 (Bardes et al. 2024, Assran et al. 2025), DINO-WM | latent prediction architecture; world-model framing | graph as input modality, not image / video |
| JEPA representation theory | Wang et al. 2026 (temporal straightening), Kuang et al. 2026 (RLpJEPA), Balestriero & LeCun 2025 (LeJEPA) | low-rank attractor / sparsity property; LeJEPA-family anti-collapse | first empirical test of cross-modality transfer of these properties to graphs |

Skenderi (TMLR 2025) and Wang (March 2026, NYU) are the two most direct anchors. Kuang et al. (RLpJEPA, March 2026) and Balestriero & LeCun (LeJEPA, Nov 2025) are concurrent JEPA-method work cited for the anti-collapse lineage that includes our BCS regularizer.

---

## 2. Pre-registered hypotheses

These are the load-bearing claims, with explicit falsification conditions. Each is fixed before the corresponding experiment runs.

### H1a — Performance, TGBN-Trade

**Claim:** Graph-JEPA outperforms capacity-matched Sequential-JEPA on TGBN-Trade with paired Δcos ≥ 0.05 across 10 seeds (Wilcoxon, Bonferroni-corrected, p < 10⁻⁶).

**Status:** Already supported by 5-seed result: Δcos = 0.081 ± 0.018, all per-seed p < 10⁻¹², 87% per-node win rate. Goal is to extend to 10 seeds for tighter CI.

**Falsification:** if Δcos < 0.05 across the additional 5 seeds, OR if the paired Wilcoxon fails to reject at α = 10⁻⁶ in any seed.

### H1b — Performance, METR-LA

**Claim:** On METR-LA horizon-{3, 6, 12} forecasting under linear-probe protocol, frozen Graph-JEPA representations achieve MAE within 5% of supervised DCRNN / Graph WaveNet baselines.

**Falsification:** if MAE gap exceeds 15% at h=12 against the DCRNN baseline (Li et al. 2018 numbers).

### H2 — Mechanism: temporal straightening

**Claim:** On TGBN-Trade, Graph-JEPA's predictor velocity vectors $v_t = \hat{z}_{t+1} - z_t$ satisfy $\overline{\cos(v_t, v_{t+1})} \ge 0.5$, versus ≤ 0.2 for Sequential-JEPA.

**Falsification:** if Sequential-JEPA's mean velocity cosine is at least as high as Graph-JEPA's, OR if Graph-JEPA's mean velocity cosine is < 0.4.

### H3 — Effective rank ↔ Wang d=8

**Claim:** $\text{eff\_rank}(\text{Graph-JEPA}) \in [5, 12]$ across all datasets where the method learns (TGBN-Trade, METR-LA, and any successful re-attempts).

**Status:** Already supported on TGBN-Trade: 8.04 ± 0.20 across 5 seeds.

**Falsification:** if eff_rank > 15 on any dataset where H1 still holds.

### H4 — Causality of low-rank compression

**Claim:** When Sequential-JEPA is forced to rank ≤ 8 via a low-rank projection bottleneck (dim 256 → 8 → 256), the Δcos gap on TGBN-Trade drops to < 0.02 (n.s.).

**Falsification:** if Sequential-JEPA-rank-8 still loses by Δcos > 0.05 (would indicate graph aggregation does work beyond compression).

### H5 — Empirical scope: temporal autocorrelation prerequisite

**Claim:** Node-level temporal JEPA requires datasets with strong temporal autocorrelation in the prediction target. Architectural responses (EMA feature smoothing, longer-window aggregation, type-conditioned predictor heads) can extend the method to non-stationary datasets by inducing the missing autocorrelation.

**Status:** Already partially supported by negative results on TGBN-Genre raw weekly, TGBN-Genre-v2 raw, EU-Email weekly.

**Falsification:** if a low-autocorrelation dataset learns under the *unmodified* architecture, OR if all architectural extensions (T1.2, T1.3, T1.4) fail.

### H6 — World-model rollout

**Claim:** Multi-step rollout MSE for Graph-JEPA grows sub-linearly with horizon h ∈ {1, 2, 4, 8} on both TGBN-Trade and METR-LA, with rollout-MSE@h=8 / h=1 ratio ≤ 0.7× Sequential-JEPA's ratio.

**Falsification:** if rollout MSE grows super-linearly OR matches Sequential-JEPA at h ≥ 4.

### H7 — Sparsity (RLpJEPA-style)

**Claim:** BCS-trained Graph-JEPA latents on TGBN-Trade exhibit a fraction of near-zero entries (|z_i| < 0.01) at least 1.5× higher than Sequential-JEPA latents under matched batch normalisation.

**Falsification:** if Graph-JEPA latents are not measurably sparser than Sequential-JEPA latents.

---

## 3. Experimental contract — what supports what

Each experiment is bound to specific hypotheses. Departures from this binding require an audit entry.

| Experiment | Hypotheses tested | Dataset(s) | Already done? |
|---|---|---|---|
| T1.0 — METR-LA train | H1b, H3 | METR-LA | no |
| T1.1 — TGBN-Trade extend to 10 seeds | H1a, H3 | TGBN-Trade | partial (5/10) |
| T1.2 — Genre + EMA features | H5 | TGBN-Genre | no |
| T1.3 — Genre-v2 + bipartite head | H5 | TGBN-Genre-v2 | no |
| T1.4 — EU-Email monthly aggregation sweep | H5 | EU-Email | no |
| T1.5 — JODIE-Reddit | H5 | JODIE-Reddit | no |
| T1.6 — JODIE-Wikipedia | H5 | JODIE-Wikipedia | no |
| T2.1 — Sequential-JEPA-rank-8 | H4 | TGBN-Trade | no |
| T2.2 — Random-edge control | C1 (corollary: real topology load-bearing) | TGBN-Trade | no |
| T2.3 — Curvature metric | H2 | TGBN-Trade, METR-LA | metric impl needed |
| T2.4 — Linear probe | C2 (representations transfer) | TGBN-Trade, METR-LA | no |
| T2.5 — Multi-step rollout | H6 | TGBN-Trade, METR-LA | impl needed (eval3 placeholder) |
| T2.6 — Sparsity metric | H7 | TGBN-Trade, METR-LA | metric impl needed |
| T3.7 — RDMReg ablation | not a hypothesis test, robustness check | TGBN-Trade | no |

Corollaries C1, C2 are not hypothesised in advance because they support stronger claims than they refute — they're *evidence-strengthening* experiments rather than load-bearing tests.

---

## 4. Frozen hyperparameters

The following hyperparameters are LOCKED before any experiment in this spec runs. Changing any of these constitutes a new experiment requiring its own pre-registration.

### 4.1 Architecture (TGBN-Trade primary)

```yaml
encoder:
  type: GATv2
  in_dim: 6  # 1d normalized volume + 5d structural
  hidden_dim: 256
  n_layers: 3
  n_heads: 4
  dropout: 0.1

predictor:
  type: bidirectional_transformer
  embed_dim: 256
  n_heads: 4
  n_layers: 2
  mlp_ratio: 2

target_encoder:
  identical_to: encoder  # EMA copy
```

### 4.2 Training

```yaml
training:
  optimizer: AdamW
  lr: 3e-4
  weight_decay: 0.01
  batch_size: 16
  max_epochs: 200
  context_k: 4
  mask_ratio: 0.20
  ema_momentum_start: 0.996
  ema_momentum_end: 1.0
  lambda_reg_bcs: <see configs/tgbn_trade.yaml>  # frozen value
  seed: <per-experiment, deterministic via src/utils/seed.py>
```

### 4.3 Evaluation

```yaml
eval:
  context_k: 4
  mask_ratio: 0.20  # same as training
  splits:
    train_frac: 0.70
    val_frac: 0.15
    test_frac: 0.15  # by snapshot index
  metrics:
    - mean_pred_cos  # cos(pred, target)
    - mean_copy_cos  # cos(z_{t-1}, target)
    - mean_graph_avg_cos  # cos(graph_avg(neighbors), target)
    - effective_rank  # exp(-Σ p_i log p_i) on test predictor outputs
    - mean_pairwise_cosine
    - mean_velocity_cos  # cos(v_t, v_{t+1}) — H2 metric (T2.3)
    - sparsity_at_threshold  # |z_i| < 0.01 fraction — H7 metric (T2.6)
    - rollout_mse_h{1,2,4,8}  # H6 metric (T2.5)
  statistical_tests:
    - paired_wilcoxon (alternative='greater')
    - bonferroni_correct (family_size: see eval_runner)
  family_size: see eval_runner.py — counts ALL pairwise tests in the eval family (eval1 vs copy, eval1 vs graph_avg, eval2 graph vs seq)
```

---

## 5. Statistical pre-registration

### 5.1 Primary test

Paired Wilcoxon signed-rank test, alternative='greater', on per-node-snapshot cosine differences (Graph-JEPA cosine − Sequential-JEPA cosine). Pairing axis: each test snapshot × each masked node = one paired observation.

### 5.2 Multiple-comparison correction

Bonferroni correction across the eval family. Family includes:
- eval1 vs copy-forward baseline
- eval1 vs graph-average baseline
- eval2 paired Graph-JEPA vs Sequential-JEPA

Correction: each raw p-value multiplied by `family_size`, clamped to 1.0.

### 5.3 Effect size reporting

Alongside p-values, report:
- Δcos = mean_pred_cos(Graph) − mean_pred_cos(Sequential), with 95% CI from 5 (or 10) seeds
- Cliff's δ (rank-biserial correlation) on per-node paired differences
- Win rate = fraction of paired comparisons where Graph > Sequential (with bootstrap 95% CI)

### 5.4 Seed protocol

Seeds 0–4 already trained for TGBN-Trade. Extension to 10 seeds: seeds 5–9 added independently, no per-seed model-selection. Mean and 95% CI across 10 seeds reported.

---

## 6. Compute budget

Total budget: ~190 H100-hr for full Tier 1 + Tier 2 + Tier 3 matrix. Detailed breakdown in `rigour.md` §3.10.

Key non-negotiables:
- T1.0 (METR-LA training): ~10 H100-hr
- T1.1 (TGBN-Trade ext. to 10 seeds): ~6 H100-hr
- T2.1 (Sequential-JEPA-rank-8): ~20 H100-hr
- T2.5 (rollout impl + run): ~4 H100-hr
- Tier 1 dataset re-attempts (T1.2-T1.6): ~50 H100-hr

If budget is constrained, the priority order is in `rigour.md` §10.

---

## 7. Falsification summary

The paper claim survives if:
- H1a holds (TGBN-Trade Δcos ≥ 0.05 across 10 seeds) ✓ already supported at 5 seeds
- H3 holds (eff_rank ∈ [5, 12]) ✓ already supported on TGBN-Trade
- H6 holds (rollout sub-linear) — requires T2.5 to test
- At least one of H2 (straightening), H4 (causality), H5 (scope) holds with the expected sign

The paper claim is partially weakened but recoverable if:
- H1b fails (METR-LA gap > 15%) — restructure as TGBN-Trade-only with H6 still validated
- H4 fails (rank-8 doesn't close the gap) — reframe contribution as "graph aggregation matters beyond compression," theoretically interesting different claim
- H7 fails (sparsity not higher) — drop H7 from paper, keep H1-H6

The paper claim collapses if:
- H1a fails on the additional 5 seeds (would contradict the existing strong result — would force investigating data pipeline correctness)
- H3 fails (eff_rank not in expected range) — Wang signature didn't transfer, paper becomes a negative result
- H6 fails (rollout super-linear) — world-model framing has to be dropped, paper reduces to representation-learning only

---

## 8. Audit log

| Date | Change | Reason | Author |
|---|---|---|---|
| 2026-05-07 | Initial spec written | Pre-registration before extending experiments | Sofia |
| 2026-05-07 | Stress-test audit conducted (4 parallel agents: data, loss/EMA, eval, architecture) | Pre-training due-diligence | audit subagents |
| 2026-05-07 | **Issue 1 (CRITICAL):** SequentialMLP capacity mismatch fixed (~200K → ~495K params via 2-linear FFN blocks) — `src/models/sequential_encoder.py`. **Requires re-running TGBN-Trade 5 seeds.** | Capacity-matched ablation claim | code fix |
| 2026-05-07 | **Issue 2 (CRITICAL):** Loss MSE confirmed in `src/losses/prediction.py:17`. Comment added; no behavioral change. | Loss audit | inspection |
| 2026-05-07 | **Issue 3 (HIGH):** eval3 multi-step rollout already wired in `run_all_with_eval2` (`src/eval/eval_runner.py:65`). Existing per-seed JSONs were generated by older runs; running `run_all_with_eval2` on existing checkpoints will populate `eval3_multistep_rollout` with real numbers (`h1`, `h2`, `h4`). No code change required. | Stale JSON outputs | inspection |
| 2026-05-07 | **Issue 4 (HIGH):** BCS lambda_reg double-scaling refactored in `src/losses/prediction.py:20`. Math is identical to prior version (`lambda_reg * sigreg["loss"]`), now split into transparent `inv_coeff * invariance_loss + bcs_coeff * bcs_loss`. **No retraining needed; effective coefficients unchanged.** | Audit clarity | code refactor |
| 2026-05-07 | **Issue 5 (CRITICAL for Genre-v2):** docstring warning added to `tgb_builder.py::build_tgbn_genre_v2_graphs_from_raw` flagging the inactive-node-mask degeneracy. Mitigation path: add `mask_active_only` flag to `TemporalGraphDataset` before training Genre-v2 (T1.3). TGBN-Trade unaffected. | Pre-T1.3 fix | docstring |
| 2026-05-07 | **Issue 6 (MEDIUM):** "p < 10⁻¹²" phrasing corrected to "p ≤ 10⁻¹²; min p = 1.23e-17, max p = 1.00e-12" in `results/tgbn_trade/results.md`. | Phrasing accuracy | text fix |
| 2026-05-07 | **Issue 7 (MEDIUM):** Bonferroni-corrected p (3-test family) added to results.md narrative. The `eval_runner.run_all_with_eval2` implementation already applies the correct correction at line 94; reporting now reflects it. | Reporting completeness | text fix |
| 2026-05-07 | New code added: `src/losses/rdmreg.py` (RLpJEPA port for T3.7), `src/data/rewire.py` (T2.2 random-edge control), `src/data/feature_smoothing.py` (T1.2 EMA features). License-compliant ports from upstream MIT-licensed code. | Implementation | code |
| 2026-05-07 | Repo-rigour scaffolding added: `LICENSE` (MIT), `CITATION.cff`, `REPRODUCE.md`. | NeurIPS submission readiness | scaffolding |

Future audits go below.

---

## References

- Skenderi, Li, Tang, Cristani — *Graph-level Representation Learning with Joint-Embedding Predictive Architectures*, TMLR 01/2025. https://github.com/geriskenderi/graph-jepa
- Wang et al. (LeCun, Ren) — *Temporal Straightening for Latent Planning*, arXiv:2603.12231, March 2026.
- Kuang, Dagade, Rudner, Balestriero, LeCun — *Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations*, arXiv:2602.01456, March 2026.
- Balestriero & LeCun — *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544, November 2025.
- Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas — *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), arXiv:2404.08471, 2024.
- Assran et al. — *V-JEPA 2*, arXiv:2506.09985, 2025.
- Li, Yu, Shahabi, Liu — *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting* (DCRNN, METR-LA), ICLR 2018.
- Huang et al. — *TGB: Temporal Graph Benchmark*, NeurIPS 2023.
- Kumar, Zhang, Leskovec — *Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks* (JODIE), KDD 2019.
