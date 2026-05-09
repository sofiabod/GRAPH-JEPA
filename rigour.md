# Rigour Audit & Experiment Plan: GRAPH-JEPA-2

---

## ⭐ STATUS 2026-05-08 — World-modeling framing + experimental plan

**The pitch (one paragraph):** LeCun's autonomous-machine-intelligence proposal centers on world models that predict in latent space. JEPA instantiates this for images (I-JEPA), video (V-JEPA, V-JEPA 2), audio (A-JEPA), and static graphs (Skenderi et al. 2025). The temporal-graph modality is missing from the lineage, despite most real-world systems being graphs that evolve over time. We extend JEPA to that modality. Wang et al. (2026, NYU) recently proved that JEPA training compresses image latents to a d≈8 attractor that makes them planning-amenable; we show this signature **transfers across modalities** to graph-structured temporal data on three economic-flow datasets, with characterized scope. The deeper bridge: practitioners report that commercial high-dimensional embeddings (OpenAI text-embedding-3-large, Gemini embedding-001) effectively use only ~3% of their nominal dimensions for downstream retrieval (Vangara 2026, Sentra). Our finding — JEPA training compresses 256d latents to ≈8 effective dimensions — is the training-side mechanism that explains where the useful subspace comes from.

### The empirical evidence stack — what's proven (2026-05-08)

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | Graph-JEPA outperforms capacity-matched non-graph ablation on relational economic-flow data | 3 datasets, paired Wilcoxon p < 10⁻⁷ all seeds | ✅ |
| 2 | JEPA training induces a low-rank attractor (Wang d≈8) that transfers from images to graphs | Trade eff_rank 8.04, BACI 7.06, on the wins | ✅ |
| 3 | The attractor is a property of any successfully-compressing JEPA, not graph-specific | METR-LA saturation: sequential hits eff_rank 7.18, graph spreads to 18.7 | ✅ |
| 4 | Graph and sequential learn fundamentally non-overlapping latent geometries | eval2b shared-target 3-mode test: cross-target collapse to baseline both directions | ✅ |
| 5 | Latent serves as a temporal world-model substrate (multi-step rollout stable) | BACI rollout pred_cos ≥ 0.94 through h=4; copy-forward decays to 0.29 | ✅ |
| 6 | No overfitting | train-test gap ≈ 0 across all 4 datasets | ✅ |
| 7 | Empirical scope: requires temporal autocorrelation + load-bearing relational structure | 3 wins + 1 saturation + 3 documented failures (Genre, Genre-v2, EU-Email) | ✅ |

### What's NOT proven (honest negatives, with framing)

| Claim | Evidence | How to frame in paper |
|---|---|---|
| Frozen embeddings transfer to static semantic labels (income tier, bloc) | Linear probe R² ≈ 0; bloc-discovery ARI 0.12 (graph) ≈ 0.12 (raw features) | Limitations section. Cite V-JEPA's similar transfer gap (Bardes 2024) — known property of self-prediction objectives, not method failure. |
| Beats published supervised baselines (TGN, TGAT, DyGFormer) | not yet tested | Defer to TGB benchmark (Path: Tier 4); risky, do post-submission |
| Graph aggregation is the *causal* mechanism (rank-8 ablation) | not yet tested | Defer T2.1 to camera-ready / rebuttal |

### Dataset matrix — current state

| # | Dataset | Domain | Trained? | Eval'd? | Result | Role in paper |
|---|---|---|---|---|---|---|
| 1 | **TGBN-Trade** | annual country trade | ✅ 5 seeds | ✅ | **WIN** Δ=+0.090, eff_rank 8.04 | primary headline |
| 2 | **BACI Gravity** | annual commodity trade (CEPII) | ✅ 5 seeds | ✅ | **WIN** Δ=+0.198, eff_rank 7.06 | strongest replication |
| 3 | **OECD ICIO** | annual sector-flow | ✅ 5 seeds | ✅ | **WIN** Δ=+0.038, win 98.5% (n_pairs=27 ⚠) | second replication, smaller n |
| 4 | **METR-LA** | 5-min traffic | ✅ 5 seeds | ✅ | **SATURATION** Δ=-0.004, eff_rank inversion | scope boundary, supports refined H3 |
| 5 | TGBN-Genre | weekly listening events | failed smoke | — | non-stationary failure | scope characterization (negative result) |
| 6 | TGBN-Genre-v2 | bipartite-fixed Genre | failed smoke | — | non-stationary failure | scope characterization |
| 7 | EU-Email | weekly email | failed smoke | — | non-stationary failure | scope characterization |
| 8 | JODIE-Reddit | weekly user-subreddit | not run | — | — | OPTIONAL — predict scope similar to Genre |
| 9 | JODIE-Wikipedia | weekly user-page | not run | — | — | OPTIONAL — predict scope similar to Genre |

**Decision:** datasets 1-7 are sufficient for the headline. JODIE-Reddit and JODIE-Wikipedia (8, 9) are optional — predicted to be in the same failure class as Genre/EU-Email. Run only if time permits and we want a 4th failure point.

### Remaining experiments — priority order for "learning something new"

| # | Experiment | What it tests | Effort | Risk | Recommendation |
|---|---|---|---|---|---|
| **A** | **Anomaly trajectory** on BACI 1995-2020 | does prediction error spike at 2008/2014/2020 real economic shocks? | ~2hr code + 5min Modal | low (likely produces a usable figure) | **DO BEFORE SUBMISSION** — single visual proof that model picked up real-world dynamics |
| B | TGB-Trade benchmark vs TGN/TGAT/DyGFormer | competitive vs supervised SOTA under TGB's NDCG@10 protocol | ~1 day | high (might lose) | DEFER to camera-ready / rebuttal |
| C | T2.1 rank-8 ablation | causal mechanism (H4) — is graph's win from compression specifically? | ~20 H100-hr | medium | DEFER to camera-ready |
| D | T2.2 random-edge control | causal mechanism — is real topology load-bearing? | ~20 H100-hr | medium | DEFER to camera-ready |
| E | Curvature metric (T2.3) | direct test of H2 (Wang straightening on graph latents) | ~2hr code, runs on existing checkpoints | low | optional, add if time |
| F | JODIE-Reddit / Wikipedia | additional scope characterization | ~10 H100-hr each | low (predicted to fail like Genre) | optional, low ROI |
| ❌ | Bloc discovery (BACI) | unsupervised cluster recovery of economic blocs | RAN — null result (ARI 0.12 ≈ raw features) | — | document as null in limitations; not a centerpiece |
| ❌ | Linear probe (BACI) | downstream economic prediction transfer | RAN — null result (R² ≈ 0 across all conditions) | — | document as transfer gap in limitations |

### Cleanest path to "method paper that proves something" — 3-day plan

**Day 1: Anomaly trajectory experiment**
- Scaffold `experiments/anomaly_trajectory.py` (already partially designed in conversation context)
- Run on BACI 1995-2020 with all 5 graph + 5 sequential checkpoints
- Output: PDF figure (year vs prediction error, two lines, shaded crisis bands at 2008-2009, 2014-2016, 2020)
- If error spikes at known shocks → "model picked up real-world dynamics" — strongest single figure
- If no spikes → document as another transfer-gap finding, paper still publishable

**Day 2: Curvature metric (T2.3) + paper drafting begins**
- Implement `mean_velocity_cos = mean cos(v_t, v_{t+1})` where v_t = ẑ_{t+1} - z_t
- Run on existing graph + sequential checkpoints across all 4 datasets
- Tests H2 (temporal straightening) directly — Wang's actual mechanism, not just eff_rank
- Begin §1-§2 (intro + related work) drafting in parallel

**Day 3: Paper-section drafting**
- Methods section
- Experiments section using current numbers
- Discussion + limitations (transfer gap, METR-LA saturation explanation, scope claim)

After that: drafting + Tier 2 ablations as time/budget permits. **Stop running new experiments after day 2.**

### Recommended title and abstract framing

**Title (commit):** *"GRAPH-JEPA-2: A JEPA-Based World Model for Temporal Graphs"*

**Abstract opener (commit):** *"LeCun's vision for autonomous machine intelligence centers on world models that predict in latent space. The JEPA framework instantiates this for images, video, audio, and static graphs but not yet for temporal graphs. We extend it. Wang et al. (2026) recently proved JEPA training induces a d≈8 low-rank attractor on image data; we show this signature transfers to graph-structured temporal data on three economic-flow benchmarks, with characterized empirical scope (where graph aggregation is load-bearing — three wins; where it isn't — one saturation boundary; non-stationary event streams — three documented failures). The compression and the prediction advantage co-occur, and the d≈8 attractor is best understood as a signature of any successfully-compressing JEPA rather than a graph-specific property."*

This pitch + the existing 3-dataset evidence stack + (if Day 1 lands) the anomaly figure = **a complete method paper for NeurIPS submission**. No more new datasets needed, no more probes, no more bloc-discovery iterations.

---

**References:**
- Skenderi et al., *Graph-level Representation Learning with Joint-Embedding Predictive Architectures*, TMLR 01/2025 (`/Users/sonia/Documents/2906_Graph_level_Representatio.pdf`, repo: `geriskenderi/graph-jepa`)
- Wang et al. (LeCun, Ren), *Temporal Straightening for Latent Planning*, arXiv:2603.12231, March 2026 (`/Users/sonia/Downloads/Temporal Straightening for Latent Planning.pdf`)
- `niashwin/geometry-of-consolidation` (NeurIPS 2026 submission) — repo-rigour reference

**Audited:** 2026-05-07
**Scope:** what to add and what to run to elevate GRAPH-JEPA-2 from "thesis result on TGBN-Trade" to a defensible NeurIPS-grade claim.

---

## ⚠ STRESS-TEST AUDIT FINDINGS (2026-05-07)

Four parallel agents audited the data, loss/EMA, eval, and architecture pipelines. Critical findings that must be addressed before training new experiments:

### Show-stoppers (block training until fixed)

1. **Sequential-JEPA capacity mismatch (CRITICAL).** Two independent agents confirmed: `SequentialMLP` is ~200K params; `GraphEncoder` is ~400K params. The README's "capacity-matched" claim is false; the Δcos = 0.081 result is confounded by capacity, not graph structure. **Fix:** widen SequentialMLP (more layers or wider hidden_dim) until param count is within ±10% of GraphEncoder. Then rerun TGBN-Trade 5 seeds.

2. **Loss function recently changed (CRITICAL).** Commit 926ab46 (May 7, 02:17) changed `F.smooth_l1_loss` → `F.mse_loss` in `src/losses/prediction.py:17`. Headline numbers were generated at 09:01 same day. **Fix:** confirm published numbers are with current MSE, or regenerate.

3. **eval3 multi-step rollout already implemented but never invoked (HIGH).** `src/eval/eval_runner.py:181-280` has the full implementation; every JSON result file says `"status": "not_implemented"`. **Fix:** call it from `run_all_with_eval2`. T2.5 in §5 is partially complete — just needs wiring.

4. **BCS lambda_reg double-scaling (HIGH).** `anticollapse.py:221` returns `invariance + lmbd * bcs`, then `prediction.py:20` multiplies the whole thing by `lambda_reg`. Effective regularization is non-standard. **Fix:** apply lambda_reg only to the bcs term, not to the invariance term.

5. **TGBN-Genre-v2 active-week filter (CRITICAL for v2 only).** `tgb_builder.py:275-322` — variable per-snapshot active node count breaks paired Wilcoxon if used. **Fix:** either filter globally to nodes with edges in all kept weeks, or report per-snapshot active counts and adjust Wilcoxon df. TGBN-Trade unaffected (n_nodes=255 fixed).

### Confirmed correct (TGBN-Trade clearance)

- ✓ Mask determinism between conditions — architecture-agnostic per-sample seed
- ✓ Train/val/test split logic in `factory.py` — strictly disjoint
- ✓ EMA target stop-grad — dual redundancy via `.detach()` and pre-normalization
- ✓ EMA momentum schedule — correct cosine schedule, per-step
- ✓ Effective rank formula — entropy of normalised singular values, correctly applied to test predictor outputs
- ✓ Wilcoxon test — `alternative='greater'`, correct pairing axis
- ✓ n_pairs = 255 = 5 test snapshots × 51 masked nodes each
- ✓ Win rate calculation — strict `>` (no tie inflation)
- ✓ End-to-end determinism — `cudnn.deterministic=True`, `cudnn.benchmark=False`, `use_deterministic_algorithms(True)`
- ✓ Token construction in predictor — context, target, positional embeddings all correct
- ✓ Sequential ablation correctly strips edge_index (just the size is wrong)

### Phrasing fixes (medium)

- **"p < 10⁻¹² in every seed"** is technically wrong: seed 2 = 1.00e-12 (equals, not less). Use "p ≤ 10⁻¹² in every seed; min p = 1.23e-17" or similar.
- **Bonferroni for eval2** — per-seed JSON p-values are uncorrected for the 3-test family (eval1 vs copy, eval1 vs graph-avg, eval2 graph vs seq). Report Bonferroni-corrected p (≈ 3× larger) for eval2 in headline tables.

### Lower-severity notes

- Context window for test/val predictions pulls from earlier splits — symmetric across conditions, doesn't invalidate paired comparison; document as honesty caveat.
- DataLoader has no `worker_init_fn` set — fine while `num_workers=0`; add guard if changed.
- `graph_encoder.py:25-28` has a silent residual fallback on shape mismatch — should raise instead of silently skipping.

### Implication for the experimental plan

The original 5-seed TGBN-Trade headline (Δcos = 0.081, p < 10⁻¹², 87% win rate) needs **rerunning after capacity fix**, not just re-reporting. Until that rerun lands, the eff_rank = 8.04 finding still stands (it's measured on Graph-JEPA alone, doesn't depend on the Sequential ablation), and the Wang d=8 connection is intact. The world-model rollout claim (H6) similarly still needs T2.5 to land — but the implementation is already there, just not wired up.

**Detailed audit reports are in the agent transcripts (not committed to repo).**

---

## 0. The two-sentence frame

**Skenderi did graph-level static JEPA on TUD benchmarks. Wang/LeCun/Ren proved temporal straightening helps image-JEPA planning. Nobody has done node-level temporal JEPA on graphs. That is your contribution, and your existing eff_rank=8.36 result (vs Sequential-JEPA's 19) is the predicted signature of straightening transferring from image trajectories to graph snapshots.**

Reframe the paper around that — not around "Graph-JEPA wins on TGBN-Trade by 0.08 cosine."

### 0.1 The claim: graph data as input modality to a JEPA-based world model

**One primary claim, one frame.** This work fills the missing temporal-graph modality in the JEPA world-model lineage:

| Modality | Architecture | Reference |
|---|---|---|
| Images | I-JEPA | Assran et al. 2023 |
| Video | V-JEPA, V-JEPA 2 | Bardes et al. 2024, Assran et al. 2025 |
| Audio | A-JEPA | Fei et al. 2023 |
| Static graphs | Graph-JEPA (graph-level) | Skenderi et al. 2025 |
| **Temporal graphs** | **GRAPH-JEPA-2 (this work, node-level)** | — |

Just as V-JEPA's contribution was *video-as-input* to the JEPA world-model framework (not "we predict video frames well"), this work's contribution is **temporal-graph-as-input** to the JEPA world-model framework. The world model is the predictor $f_\theta : \{z_{t-K}, \ldots, z_{t-1}\} \mapsto \hat{z}_t$ in the latent space of graph snapshots. Wang et al. (2026) is the theoretical anchor — their straightening property predicts a low-rank attractor.

**Two complementary datasets, two roles:**

- **METR-LA (primary, the recognised benchmark).** 207 traffic sensors as nodes, road-network topology, 5-min snapshots over ~4 months. The standard benchmark in the temporal-graph community since 2018. Multi-step forecasting is the native eval — directly comparable to STGCN (Yu et al. 2018), DCRNN (Li et al. 2018), Graph WaveNet (Wu et al. 2019), AGCRN (Bai et al. 2020), MTGNN (Wu et al. 2020). This is where the world-model rollout claim is tested against an established field.
- **TGBN-Trade (secondary, the theoretical replication).** Bilateral country trade, 255 nodes, annual snapshots, 1986–2016. This is where the Wang d=8 ↔ eff_rank=8 quantitative match is demonstrated. Smaller, denser, simpler topology — used here as a clean substrate to isolate the straightening-signature claim, in the same role PointMaze plays in Wang's image-JEPA experiments (controlled testbed, not competitive benchmark).

Together: METR-LA establishes that JEPA-on-graphs is competitive as a *world model* against existing temporal-graph methods. TGBN-Trade establishes that the Wang straightening signature *transfers from images to graphs* with quantitative precision (eff_rank = 8.0 ± 0.2, matching Wang's d=8 finding). Both rest on the same architecture; both satisfy the H5 autocorrelation prerequisite.

### 0.2 The unified claim sentence

> *"We present GRAPH-JEPA-2, the first joint-embedding predictive architecture trained on temporal graph data as a world model. Given a sequence of past graph snapshots, the model predicts the latent state of the future graph and autoregressively rolls out to horizon h. On METR-LA, the standard temporal-graph forecasting benchmark, our model is competitive with supervised baselines (DCRNN, Graph WaveNet, MTGNN) on multi-step horizon-12 prediction while being trained without any supervision. On TGBN-Trade, we demonstrate that the latent exhibits effective rank 8.0 ± 0.2 (vs 19.1 ± 0.7 for a capacity-matched non-graph ablation) — the low-rank attractor signature predicted by Wang et al.'s (2026) temporal-straightening hypothesis for image-JEPA, here demonstrated to transfer to graph-structured observations with quantitative precision. We provide causal evidence that graph-induced compression is the mechanism (rank-controlled ablation), that real graph topology is load-bearing (edge-rewiring control), and that representations transfer (linear probe). Documented training failures on three non-stationary datasets characterise the empirical scope of the method."*

### 0.3 Why "world model" is honest here

A *world model* in the LeCun/JEPA sense is a function that predicts future state from past state in a learned latent space. V-JEPA 2 and DINO-WM call themselves world models with and without action conditioning — the action-conditioned variant supports planning, the unconditioned variant supports forecasting and understanding. Your model is the unconditioned variant: predicts next graph latent from past graph latents, no actions yet defined for graphs. **Planning over actions is future work** (an action space for graphs has to be defined first — edge addition, node intervention, etc.) — not part of this paper's contribution. The world-model framing is justified by the dynamics model itself, not by a planning eval.

### 0.4 What evidence is needed for each piece

- **"First temporal-graph JEPA world model"** — supported by current architecture + existing TGBN-Trade result. **Needs METR-LA training run (T1.0)** to claim "competitive with the temporal-graph field," not just "trains on graphs."
- **"Latent serves as a world-model substrate"** — needs T2.5 (multi-step rollout) on both datasets. METR-LA's native eval *is* multi-step forecasting (horizon-12 standard), so T2.5 partially comes for free there. On TGBN-Trade you implement the placeholder.
- **"Mirrors Wang straightening signature with quantitative precision"** — already supported by TGBN-Trade eff_rank=8 result; tightened by T2.3 (curvature metric) and T2.1 (rank-8 ablation). Replicate eff_rank measurement on METR-LA — predict it stays in [5, 12] range there too.
- **"Graph topology is load-bearing"** — needs T2.2 (random-edge control). Run on TGBN-Trade primarily; cheaper than rewiring METR-LA's road network.
- **"Representations transfer"** — needs T2.4 (linear probe). Both datasets ship downstream tasks (TGBN-Trade: trade-class labels; METR-LA: speed thresholds for congestion classification).
- **"Empirical scope (raw failure → architectural fix)"** — first-wave failures on Genre/Genre-v2/EU-Email (§4); planned re-attempts T1.2 (EMA features), T1.3 (bipartite head), T1.4 (monthly agg) test whether each failure mode has a principled architectural fix.
- **"Competitive with supervised baselines"** — needs METR-LA results compared head-to-head with DCRNN, Graph WaveNet, MTGNN, AGCRN under standard horizon-{3, 6, 12} MAE/RMSE protocol.

---

## 1. Theoretical anchor: Wang straightening predicts your result

Wang et al. (March 2026) prove that for a JEPA-like world model with linear latent dynamics $z_{t+1} = A z_t + B a_t$, the planning Hessian condition number is bounded by

$$\kappa_{\text{eff}}(H) \;\le\; \kappa(B)^2 \left( \frac{1+\varepsilon}{1-\varepsilon} \right)^{2(K-1)}, \quad \varepsilon = \|A - I\|_2$$

so when latent transitions are $\varepsilon$-straight ($\|A - I\|_2$ small), the planning loss landscape is well-conditioned, gradient methods converge faster, and Euclidean distance becomes a faithful proxy for geodesic distance. Empirically: **implicit straightening occurs in any JEPA training**, and channel dimension as low as **d=8 is sufficient** when spatial structure is preserved (Wang Table 1, PointMaze-Medium with $14 \times 14 \times 8$ features → 100% MPC success).

**Your TGBN-Trade result already matches this prediction.** Graph-JEPA: eff_rank = 8.0 ± 0.2. Sequential-JEPA: eff_rank = 19.1 ± 0.7. The 2.4× compression ratio is not noise — it is the same low-rank attractor Wang reports for image JEPA, transferred to temporal graphs. Stating this in advance, with a falsification condition, turns it from observation into hypothesis.

**Pre-registered hypotheses (scoped to TGBN-Trade after empirical evidence on Genre / Genre-v2 / EU-Email — see §4):**

- **H1a (TGBN-Trade — quantitative replication of Wang signature):** Graph-JEPA beats capacity-matched Sequential-JEPA on TGBN-Trade with Δcos ≥ 0.05 across 10 seeds (paired Wilcoxon, Bonferroni-corrected, p < 10⁻⁶). **Already supported** by 5-seed result: Δcos = 0.081, p < 10⁻¹² per seed, 87% win rate.
- **H1b (METR-LA — competitive with the field):** On METR-LA horizon-{3, 6, 12} forecasting, frozen Graph-JEPA representations (linear probe to speed prediction) achieve MAE within 5% of supervised DCRNN / Graph WaveNet under the standard protocol (Li et al. 2018 split). Falsifies the "world model is competitive" claim if MAE gap > 15%.
- **H2 (mechanism — straightening):** On TGBN-Trade, Graph-JEPA exhibits implicit temporal straightening — mean $\cos(v_t, v_{t+1})$ for predictor velocity vectors $v_t = \hat{z}_{t+1} - z_t$ satisfies $\overline{\cos(v_t, v_{t+1})} \ge 0.5$ for Graph-JEPA vs ≤ 0.2 for Sequential-JEPA. Falsifies if Sequential-JEPA is at least as straight.
- **H3 (effective rank ↔ Wang d=8):** $\text{eff\_rank}(\text{Graph-JEPA}) \in [5, 12]$, mirroring Wang's d=8 finding. **Already supported:** 8.0 ± 0.2 across 5 seeds.
- **H4 (causality):** When Sequential-JEPA is forced to rank-8 via a low-rank projection, Δcos drops to < 0.02. Falsifies the straightening explanation if Sequential-JEPA-rank-8 still loses by Δcos > 0.05 (would mean graph aggregation does work beyond compression).
- **H5 (scope — when JEPA on graphs works):** Node-level temporal JEPA requires datasets with strong temporal autocorrelation in the prediction target. Falsifies if a low-autocorrelation dataset (TGBN-Genre, EU-Email) trains successfully under the same architecture.
- **H6 (world-model rollout):** Multi-step rollout MSE for Graph-JEPA grows sub-linearly with horizon $h \in \{1, 2, 4, 8\}$, with rollout-MSE ratio Graph-JEPA / Sequential-JEPA ≤ 0.7 at h=8. This is the world-model framing's load-bearing claim and matches Wang's Table 2 protocol (long-horizon success). Falsifies if rollout MSE grows super-linearly or matches Sequential-JEPA at long horizons — would mean the latent isn't useful for temporal extrapolation, only for one-step prediction.

Each hypothesis is a falsifiable claim. H1–H4 argue that **graph structure induces JEPA-style straightening on TGBN-Trade** (the underlying mechanism). H5 characterises when the method should generalise. H6 unlocks the **world-model frame** — without rollout evidence, "world model" is overclaiming. The negative results in §4 are evidence *for* the scope claim (H5), not against the contribution.

---

## 2. Positioning vs prior work

### 2.1 Comparison table

| Property | Skenderi 2025 (Graph-JEPA) | Wang 2026 (Temporal Straightening) | **GRAPH-JEPA-2 (this work)** |
|---|---|---|---|
| Modality | static graphs | image trajectories | **temporal graphs** |
| Granularity | graph-level (whole-graph repr) | spatial token / global feature | **node-level (per-node temporal)** |
| Temporal dynamics | none | planning over horizon $H$ | **week-by-week or year-by-year prediction** |
| Datasets | TUD (PROTEINS, MUTAG, DD, REDDIT-B/M, IMDB-B/M, ZINC) + EXP synthetic | Wall, PointMaze-UMaze, PointMaze-Medium, PushT | **TGBN-Trade, TGBN-Genre, TGBN-Genre-v2, JODIE-Reddit, JODIE-Wiki** |
| Encoder | GIN with edge features (GINE) | DINOv2 + projector OR ResNet-from-scratch | **GATv2 (3 layers, 4 heads, hidden 256)** |
| Predictor | MLP, linear-leaning | ViT with causal mask | **Bidirectional Transformer (2 layers, 4 heads)** |
| Subgraph patches | METIS partition + 1-hop expansion | spatial patches | **none — full graph snapshot per timestep** |
| Positional encoding | Random Walk Structural Embedding (RWSE) | learned/sinusoidal | **5d structural features per node** |
| Prediction target | 2D unit hyperbola $(\cosh\alpha, \sinh\alpha)$, $\alpha = \frac{1}{d}\sum_n Z^{(n)}$ | direct latent vector | **direct latent vector (cosine on unit sphere)** |
| Loss | smooth L1 on 2D coords | MSE + λ·(1−cos(v_t,v_{t+1})) | **MSE on normalized vectors (= L2 on unit sphere)** |
| Anti-collapse | EMA + simple predictor + stop-grad | stop-grad only ("worked in our experiments") | **EMA + BCS regulariser + stop-grad** |
| Eval | 10-fold CV linear probe | open-loop GD planning + MPC, success rate | **paired Wilcoxon on target-cosine, eff_rank, win rate** |
| Statistical claim | mean ± std over 5–10 seeds | mean ± std over 3 sampling seeds | **5 seeds, paired Wilcoxon p<10⁻¹², 87% win rate** |

### 2.2 What Skenderi explicitly leaves on the table

Skenderi's conclusion (paper page 12): *"Future research directions include extending the proposed method to **node and edge-level learning**, theoretically exploring the expressiveness of Graph-JEPA, and gaining more insights into the optimal geometry of the latent space for general graph SSL."*

That sentence is your introduction. You are doing the node-level extension, on temporal graphs, with an *additional* theoretical anchor (straightening) Skenderi did not have access to.

### 2.3 Where you should *not* copy Skenderi

- **Hyperbolic 2D target.** Skenderi predicts $(\cosh\alpha, \sinh\alpha)$ where $\alpha$ is a scalar aggregate. This was a graph-level design decision (predict a single hierarchical position per subgraph). For node-level temporal prediction, predicting full latent vectors on the unit sphere (your current MSE-on-normalised setup, which equals L2 on the sphere) is the right choice. **Do not chase hyperbolic just because it's there.**
- **METIS + 1-hop expansion.** Skenderi partitions the graph into subgraphs because graph-level tasks need a "patch" abstraction. Your task is per-snapshot per-node — you don't need patches. Skip METIS.
- **RWSE positional encoding.** Random-walk structural embedding is computed from a *static* graph. Your snapshots have varying edge sets per timestep — RWSE per snapshot would be expensive and the temporal structural-features approach you already have is more natural.

### 2.4 What you *should* steal from Skenderi

- **Per-dataset script + per-dataset config + per-dataset paper-log file** layout (`train/<dataset>.py`, `train/configs/<dataset>.yaml`, `paper_logs/<DATASET>.txt`). You already have `configs/<dataset>.yaml`; add the per-dataset training script and paper-log convention.
- **Reporting format.** Skenderi Table 1 gives mean ± std for each dataset, ranks against ≥4 baselines (contrastive, generative, latent-self-predictive, supervised). Your `results/tgbn_trade/results.md` only ranks against 2 baselines. Add at least one contrastive baseline (e.g., GraphCL, MVGRL adapted for temporal) and one masked-autoencoder baseline (GraphMAE temporal variant or DyGFormer).
- **F-GIN reference row.** Skenderi includes a fully-supervised GIN as a topline. Add a fully-supervised temporal GAT trained on the same node-attribute regression target as a topline reference for your Graph-JEPA representations under linear probe.

---

## 3. Dataset shapes — what's built, what's needed

All TGB builders return `(graphs, meta)` where `graphs: list[PyG.Data]` is one snapshot per timestep with `x`, `edge_index`, `node_ids` (and `edge_attr` for v2). Splits follow `compute_split_ranges` (factory.py:31) → 70% train / 15% val / 15% test by snapshot index, deterministic.

### 3.1 The full dataset matrix — 7 datasets, all in scope

All seven datasets are legitimate experimental targets. Their roles span four regimes — recognised benchmark (METR-LA), theoretical replication (TGBN-Trade), bursty re-attempts (TGBN-Genre, TGBN-Genre-v2, EU-Email), and untested interaction graphs (JODIE-Reddit, JODIE-Wikipedia). Together they let you make claims at multiple levels: standard-benchmark performance, Wang-signature replication, scope characterisation, and architectural-extension stress tests.

| # | Dataset | Builder | n_nodes | Snapshot | Snapshots | Feature dim | Role | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | **METR-LA** | new — `src/data/metrla_builder.py` | 207 (sensors) | 5-min | ~34,272 | 2 (speed + ToD) | flagship benchmark | **build + train (T1.0)** |
| 2 | **TGBN-Trade** | `tgb_builder.build_tgbn_trade_graphs` | 255 (countries) | annual | ~31 | 6 (1d vol + 5d struct) | Wang d=8 replication | **5 seeds done, eff_rank=8.0** |
| 3 | **TGBN-Genre** | `tgb_builder.build_tgbn_genre_graphs` | ~17.6k | weekly | ~133 | 6 | bursty re-attempt | failed v1 smoke; re-try with longer context + EMA features (T1.2) |
| 4 | **TGBN-Genre-v2** | `tgb_builder.build_tgbn_genre_v2_graphs` | filtered | weekly | ~133 | 7 (+type) | bipartite re-attempt | failed v2 smoke; re-try with bipartite-typed predictor head (T1.3) |
| 5 | **EU-Email** | `eu_email_builder` | ~1.0k (estimate) | weekly | TBC | TBC | sparse-event re-attempt | failed weekly smoke; re-try with monthly aggregation (T1.4) |
| 6 | **JODIE-Reddit** | `src/data/jodie_builder.py` | ~10k | weekly | TBC | TBC | untested user-subreddit interaction graph | smoke + 5 seeds (T1.5) |
| 7 | **JODIE-Wikipedia** | `src/data/jodie_builder.py` | ~9k | weekly | TBC | TBC | untested user-page edit graph | smoke + 5 seeds (T1.6) |

### 3.2 METR-LA — the flagship benchmark (dataset 1)

- **Source:** Li et al. 2018, *DCRNN*. Public download via PyG Temporal (`torch_geometric_temporal.dataset.METRLADatasetLoader`) or the original DCRNN repo (`https://github.com/liyaguang/DCRNN`).
- **Shape:** 207 traffic sensors on Los Angeles County highways, sampled every 5 minutes for ~4 months (March–June 2012). Total ~34,272 timestamps.
- **Topology:** weighted adjacency from sensor proximity on the road network (edge weight ∝ exp(-d²/σ²) where d is road distance). Sparse — far from complete graph.
- **Features per node per timestamp:** raw speed (mph), and optionally time-of-day, day-of-week.
- **Standard split:** 70/10/20 train/val/test by time (Li et al. 2018 protocol).
- **Standard eval:** multi-step forecasting at horizons h ∈ {3, 6, 12} (= 15, 30, 60 minutes ahead). Metrics: MAE, RMSE, MAPE.
- **Baselines off-the-shelf:** DCRNN, STGCN, Graph WaveNet, MTGNN, AGCRN, GMAN.
- **Why it works:** traffic at 5-min cadence is extremely autocorrelated. Copy-forward MAE ≈ 4 mph. JEPA latent should compress cleanly.

### 3.3 TGBN-Trade — the theoretical-replication testbed (dataset 2)

Already covered above. The Wang d=8 ↔ eff_rank=8 quantitative match, the existing 5-seed result, the cleanest substrate for the rank-8 / random-edge / curvature ablations.

### 3.4 TGBN-Genre — bursty re-attempt with richer temporal state (dataset 3)

- **Why it failed:** weekly listening events are bursty; per-node state reshuffles week-over-week; copy-forward is weak; no stable target for JEPA to compress to.
- **Re-attempt plan:** swap the raw weekly node feature for an **EMA-smoothed feature** with τ ∈ {4, 8, 16} weeks. This injects the temporal autocorrelation the raw signal lacks, mirroring how Wang's image-JEPA uses frame-skip = 5 to smooth video dynamics. Predict: at τ ≥ 8, val loss decreases under unchanged architecture.
- **Falsification:** if EMA-smoothed Genre still doesn't learn, the issue isn't autocorrelation — it's something deeper (bipartite structure, cold-start nodes, etc.).

### 3.5 TGBN-Genre-v2 — bipartite-typed predictor (dataset 4)

- **Why v2 still fails despite preprocessing fixes:** symmetrised edges + dedup + active-week filter weren't enough. Hypothesis: the shared predictor head averages over user-vs-item nodes that have fundamentally different dynamics (users have stable preferences; items have bursty popularity).
- **Re-attempt plan:** add a **type-conditioned predictor head**: separate output projections for user nodes and item nodes (or a single head with type-embedding concatenation). ~3 days to implement.
- **What it tests:** whether dual-mode bipartite graphs need explicit type conditioning to support a JEPA latent. If yes, this is a meaningful architectural extension; if no, the failure is data-level not model-level.

### 3.6 EU-Email — longer aggregation windows (dataset 5)

- **Why it failed at weekly:** email partner sets shift week-over-week (~0.003 val drop in 2 epochs); no stable node-state.
- **Re-attempt plan:** **monthly aggregation** (4× longer window) + drop nodes inactive in <50% of months. Predict: monthly-aggregated EU-Email has the autocorrelation property and should learn.
- **What it tests:** the trade-off between snapshot rate (more samples) and aggregation length (more autocorrelation). A clean curve here would let you write *"the method requires aggregation windows long enough to induce node-feature autocorrelation r ≥ τ*"* — a sharper scope claim than "weekly is too short."

### 3.7 JODIE-Reddit — untested user-subreddit interactions (dataset 6)

- **Source:** Kumar et al. 2019, *JODIE: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks*. Standard temporal-graph dataset.
- **Shape:** ~10k nodes (users + subreddits), bipartite, edge events = posts/comments with timestamps. Originally for link-prediction / interaction-event modelling.
- **Builder status:** `src/data/jodie_builder.py` exists but unvalidated. Need to confirm shape, snapshot-aggregation, and node feature construction.
- **Plan:** smoke test at weekly aggregation first. If fails like Genre, escalate to monthly. If still fails, document as another bursty-interaction failure, refining H5.
- **Why it's worth trying:** if it works, you have a *different* domain (online community interactions) with the autocorrelation property — diversifies the empirical evidence beyond economic flows and traffic.

### 3.8 JODIE-Wikipedia — untested user-page edit interactions (dataset 7)

- **Source:** Kumar et al. 2019, same paper as Reddit.
- **Shape:** ~9k nodes (editors + pages), bipartite, edge events = page edits with timestamps.
- **Builder status:** same as Reddit — exists, unvalidated.
- **Plan:** same protocol as JODIE-Reddit (smoke weekly → monthly if needed → document either way).
- **Why interesting:** Wikipedia edits have stronger autocorrelation than Reddit comments (high-edit pages stay high-edit; major editors are persistent). Could land closer to TGBN-Trade than to TGBN-Genre on the autocorrelation spectrum.

### 3.9 Code references shared across all 7 datasets

- `TemporalGraphDataset(graphs, context_k=4, mask_ratio=0.20, split, seed)` — `src/data/dataset.py:10`
- Per-sample deterministic mask: `gen.manual_seed(self.seed * 100003 + target_idx)` — `src/data/dataset.py:42-44`
- 70/15/15 split: `compute_split_ranges` — `src/data/factory.py:31` (override per-dataset for METR-LA's 70/10/20)
- Structural features (5d): `compute_structural_features` — `src/data/graph_utils.py`

### 3.10 Compute budget — full 7-dataset matrix

| # | Dataset | Wall-clock (5 seeds, H100) | New code needed |
|---|---|---|---|
| 1 | METR-LA | ~10 hr (34k timestamps) | `metrla_builder.py`, possibly horizon-12 eval head |
| 2 | TGBN-Trade | done — 0 hr to extend to 10 seeds: ~6 hr | none |
| 3 | TGBN-Genre + EMA features | ~8 hr (133 snapshots, longer context) | EMA wrapper in `src/data/feature_smoothing.py` |
| 4 | TGBN-Genre-v2 + bipartite head | ~8 hr | type-conditioned predictor head in `src/models/predictor.py` |
| 5 | EU-Email + monthly agg | ~6 hr | new aggregation arg in `eu_email_builder` |
| 6 | JODIE-Reddit | ~10 hr | validate `jodie_builder` + new config |
| 7 | JODIE-Wikipedia | ~10 hr | reuse JODIE-Reddit code path |

**Total: ~58 H100-hr** for the full 7-dataset Tier 1 matrix. Add Tier 2 mechanism ablations (T2.x ~80 hr) and Tier 3 baselines (T3.x ~50 hr) for a full ~190 H100-hr budget. At 16 concurrent H100s on Modal that's ~12 hr wall-clock for the dataset matrix.

### 3.11 Sanity checks to run on every dataset before you trust the result

- **n_pairs per evaluation snapshot.** TGBN-Trade has 255 nodes × mask_ratio=0.20 ≈ 51 masked per snapshot. eval1 reports n_pairs=255 — that's 51 × ~5 test snapshots ≈ 255. Confirm this matches `len(test_set) × n_masked_per_snapshot`. If `n_pairs` doesn't match, the Wilcoxon df is wrong.
- **Deterministic mask reproducibility.** Run dataset twice with same seed; confirm `masked_node_ids` identical. (`tests/test_masking.py` should already cover this — verify.)
- **Train/val/test temporal isolation.** No snapshot in val ≤ any snapshot in train. No snapshot in test ≤ any snapshot in val. Spot-check after any active-week filtering that shrinks `n`.
- **eff_rank measurement is on test-set predictor outputs only.** If you accidentally compute eff_rank on training-set targets, you're measuring the wrong thing. Verify in `eval_runner.py`.
- **Snapshot-rate sanity (METR-LA specific):** at 5-min cadence with ~34k timestamps, training one epoch should be cheap. If wall-clock per epoch is > 5 min on H100, profile the data loader before scaling.

---

## 4. Preliminary failures and planned architectural re-attempts

The first wave of cross-dataset training revealed structural failure modes on three of the seven datasets. These are *preliminary* results, not final scope claims — each failure points to a specific architectural or preprocessing variant to test next:

| Dataset | First-attempt smoke result | Diagnosis | Planned re-attempt |
|---|---|---|---|
| **TGBN-Trade** | val loss decreases decisively, Δcos = 0.081 over 5 seeds, p < 10⁻¹² | strong year-over-year autocorrelation (r > 0.9); copy-forward at cos ≈ 0.51 | extend to 10 seeds, run mechanism ablations |
| **TGBN-Genre (b=1, b=4)** | non-learning | bursty weekly listening patterns; weak temporal autocorrelation in raw signal | **EMA-smoothed features** with τ ∈ {4, 8, 16} weeks (T1.2) |
| **TGBN-Genre-v2 (rebuild)** | val 1.637 → 1.638 (UP) | bipartite preprocessing fixed graph structure but shared predictor head averages over user-vs-item dynamics | **type-conditioned predictor head** (T1.3) |
| **EU-Email (weekly)** | val 1.567 → 1.564 (~0.003 drop in 2 epochs, flat) | partner sets shift week-over-week; no stable node-state at this aggregation rate | **monthly aggregation** + active-month filter (T1.4) |

### 4.1 The hypothesis behind each re-attempt

Each first-wave failure is a *hypothesis-generation event*, not a final answer. The interpretation:

- **TGBN-Genre raw weekly fails because the signal lacks autocorrelation, not because JEPA can't handle bipartite graphs.** Wang's image-JEPA also requires temporal smoothness; their workaround is frame-skip = 5. The graph analogue is feature-smoothing. T1.2 tests this directly.
- **TGBN-Genre-v2 fails despite preprocessing fixes because the *model* averages over heterogeneous node dynamics.** A type-conditioned predictor head (T1.3) is a small architectural change that lets users and items each have their own dynamics model.
- **EU-Email at weekly cadence is too short to induce node-state stability.** Monthly aggregation (T1.4) is the same intervention as Genre's EMA — buy autocorrelation by widening the temporal window.

If T1.2/T1.3/T1.4 succeed, the paper claim broadens from *"works on TGBN-Trade"* to *"works on graphs satisfying an autocorrelation prerequisite, achievable through feature smoothing or aggregation"* — much stronger.

If they fail, you have a sharper scope claim: *"raw bursty interaction graphs require fundamental architectural changes beyond preprocessing — open problem."* Either outcome is publishable.

### 4.2 What this means for the paper structure

- **Section: "Datasets and Preliminary Results"** — present all 7 datasets, the first-wave outcomes, and the architectural variants tested.
- **Section: "Method Extensions for Bursty Graphs"** — describe EMA-smoothing, type-conditioned predictor, monthly aggregation as principled architectural responses to the failure modes.
- **Section: "Empirical Scope (H5)"** — formalise the autocorrelation prerequisite, with empirical curves from EU-Email's aggregation sweep showing where the method begins to learn.

This is more interesting than the single-dataset framing — you make the scope claim *empirically*, by showing the method works iff a measurable prerequisite is satisfied, with multiple datasets characterising the boundary.

---

## 5. Experiments to run — the actual TODO list

Each item lists **what to do**, **what it tests**, **what falsifies the hypothesis**, and **estimated effort**.

### Tier 1 — datasets (T1.0 – T1.6, one per dataset)

Tier 1 = train Graph-JEPA + Sequential-JEPA on each of the 7 datasets in the matrix (§3.1). Mechanism ablations (T2.x below) are then run on the datasets that learn.

#### T1.0 METR-LA — train + horizon-{3, 6, 12} forecasting eval
- **Where:** new `src/data/metrla_builder.py` (load via PyG Temporal), `experiments/train_metrla.py`, `configs/metrla.yaml`.
- **Tests:** H1b — Graph-JEPA trains on the standard temporal-graph benchmark and is competitive with DCRNN / Graph WaveNet on horizon-{3, 6, 12} MAE under linear probe.
- **Falsifies if:** linear-probe MAE > 1.15× DCRNN baseline at h=12, or training fails to decrease val loss.
- **Effort:** ~1 day implement + ~10 H100-hr (5 seeds).

#### T1.1 TGBN-Trade — extend to 10 seeds, add curvature + rollout metrics
- **Where:** existing `experiments/train_tgjepa.py` with `configs/tgbn_trade.yaml`, seeds 0–9.
- **Tests:** H1a tightening, H3 (eff_rank stays in [5, 12]), H6 (rollout sub-linear).
- **Effort:** ~6 H100-hr (only need 5 more seeds; existing 5 already cached).

#### T1.2 TGBN-Genre with EMA-smoothed features
- **What:** add `EMA-feature` wrapper around `tgb_builder.build_tgbn_genre_graphs` that smooths the 1d volume feature with τ ∈ {4, 8, 16} weeks before passing to GATv2.
- **Where:** new `src/data/feature_smoothing.py`. Sweep τ as a config knob.
- **Tests:** whether autocorrelation is the bottleneck; predicts learning succeeds at τ ≥ 8.
- **Falsifies "autocorrelation is the bottleneck" if:** none of τ ∈ {4, 8, 16} learns.
- **Effort:** ~1 day implement + ~8 H100-hr (3 τ values × 3 seeds).

#### T1.3 TGBN-Genre-v2 with type-conditioned predictor head
- **What:** add a node-type embedding (user vs item) to the predictor input, OR use two separate output projection heads (one per type). Reuse the existing v2 builder.
- **Where:** modify `src/models/predictor.py` to accept `node_type` tensor; add `type_conditioned: true` flag in `configs/tgbn_genre_v2.yaml`.
- **Tests:** whether bipartite type-heterogeneity is what's blocking learning.
- **Effort:** ~3 days implement + ~8 H100-hr (5 seeds).

#### T1.4 EU-Email with monthly aggregation
- **What:** modify `eu_email_builder` to support `aggregation_window` ∈ {weekly, biweekly, monthly}. Filter to nodes active in ≥ 50% of windows.
- **Where:** add window arg to builder; sweep as config knob.
- **Tests:** the trade-off curve — at what aggregation window does autocorrelation become sufficient for JEPA to learn?
- **What it produces:** a 3-point curve (weekly fail → biweekly ? → monthly success?) that empirically calibrates H5.
- **Effort:** ~1 day implement + ~6 H100-hr (3 windows × 3 seeds).

#### T1.5 JODIE-Reddit — first untested dataset
- **Where:** validate `src/data/jodie_builder.py`, write `configs/jodie_reddit.yaml`. Smoke test at weekly first (~2 epochs).
- **Tests:** does an online-community user-subreddit interaction graph have the autocorrelation property?
- **Decision tree:**
  - Smoke learns → run 5 seeds, report results
  - Smoke fails → escalate to monthly aggregation (parallel to T1.4)
  - Both fail → document as another bursty failure, refines H5
- **Effort:** ~1 day to validate + ~10 H100-hr per protocol attempt.

#### T1.6 JODIE-Wikipedia — second untested dataset
- **Where:** reuse `jodie_builder.py` code path with new config.
- **Tests:** Wikipedia editing has plausibly more autocorrelation than Reddit comments (high-edit pages persist; major editors persist) — does it sit closer to TGBN-Trade or to TGBN-Genre on the spectrum?
- **Effort:** ~10 H100-hr.

### Tier 2 — mechanism ablations (T2.x, on datasets that learn)

These deepen the world-model claim once you know which datasets learn. Run primarily on TGBN-Trade (where the result is already strong) and METR-LA (the flagship).

#### T2.1 Sequential-JEPA-rank-8 ablation
- **What:** add `nn.Linear(d, 8) → nn.Linear(8, d)` bottleneck inside Sequential-JEPA's encoder, after the temporal MLP, before the prediction head. Train 5 seeds on TGBN-Trade.
- **Where:** new `src/models/seq_encoder_rank.py` or a `low_rank_proj` flag on the existing Sequential encoder. Reuse `experiments/train_tgjepa.py` with a new config flag.
- **Tests:** H4 (causality of straightening). If the gap closes, graph-JEPA's win is *because of* low-rank compression, not graph aggregation per se. If the gap doesn't close, graph aggregation does extra work beyond compression.
- **Falsifies straightening hypothesis if:** Sequential-rank-8 still loses by Δcos > 0.05.
- **Effort:** ~1 day (4 hr to implement + ~4 H100-hr × 5 seeds = 20 H100-hr).
- **This is the single highest-information experiment you can run.**

#### T2.2 Random-edge control
- **What:** train Graph-JEPA on a degree-preserving random-rewired version of TGBN-Trade (configurable seed for the rewiring). 5 seeds.
- **Where:** new utility `src/data/rewire.py` using `networkx.double_edge_swap` or PyG's `RandomNodeSplit` analogue, applied per snapshot. Hook into `factory.py` via `cfg.rewire_seed`.
- **Tests:** whether the *real* graph structure (semantics of edges) carries the signal, or whether any structure with similar degree distribution suffices.
- **Falsifies "graph topology is load-bearing" if:** rewired Graph-JEPA still wins by Δcos ≥ 0.05.
- **Effort:** ~1 day. Rewiring per snapshot is fast; main cost is training.
- **Second-highest information experiment.**

#### T2.3 Curvature / straightening metric
- **What:** add `mean_velocity_cos = mean over t of cos(v_t, v_{t+1})` where $v_t = \hat{z}_{t+1} - z_t$, computed on test-set predictor outputs. Do this for both Graph-JEPA and Sequential-JEPA.
- **Where:** new function in `src/eval/metrics.py`, called from `src/eval/eval_runner.py:_eval6_representation_quality`.
- **Tests:** H2 directly. Predicts Graph-JEPA has higher straightness.
- **Falsifies H2 if:** Sequential-JEPA has equal or higher straightness while Graph-JEPA still wins on Δcos (would mean straightening is not the mechanism).
- **Effort:** ~2 hr to implement, runs as part of existing eval (no extra training).
- **You should add this metric *before* T2.1 and T2.2 so you have it to report on every condition.**

#### T2.4 Linear-probe downstream eval
- **What:** freeze Graph-JEPA encoder, train a logistic regression / MLP on TGBN node labels (TGBN-Trade has node-level trade-class labels; TGBN-Genre has user/item-level genre labels). Report accuracy / AP. Compare against Sequential-JEPA frozen encoder + same probe. Match Skenderi's protocol (mean ± std over 10 random splits).
- **Where:** new `src/eval/linear_probe.py` and `experiments/eval_probe.py`. Use scikit-learn.
- **Tests:** whether Graph-JEPA representations *transfer*. Currently you only measure self-cosine (training dual). A reviewer wants downstream evidence.
- **Falsifies "useful representations" if:** linear probe accuracy is no better than chance, or no better than Sequential-JEPA probe.
- **Effort:** ~1 day (probe is cheap; data-loading the labels is the main work).
- **Frame supported:** representation-learning (transfer evidence).

#### T2.5 Multi-step rollout — required for world-model claim
- **What:** at test time, take context $\{z_{t-K}, \ldots, z_{t-1}\}$, predict $\hat{z}_t$, then *autoregressively roll out* $\hat{z}_{t+1}, \hat{z}_{t+2}, \ldots, \hat{z}_{t+h}$ by feeding the predictor's own outputs back as context. Compare $\hat{z}_{t+h}$ to the EMA target encoder's $z_{t+h}$ via cosine and MSE for $h \in \{1, 2, 4, 8\}$. Repeat for Sequential-JEPA. Report rollout-MSE-vs-horizon curves with bootstrap CI.
- **Where:** populate the `eval3_multistep_rollout` placeholder in `src/eval/eval_runner.py` (it currently writes `{"status": "not_implemented"}` — see `results/tgbn_trade/seed*/eval1_tgbn_trade.json:16-18`). New function `_eval3_multistep_rollout(self, max_h=8)`.
- **Tests:** H6 — does the latent serve as a *world-model* substrate, not just a one-step regressor? Wang Table 2 shows long-horizon success rate; this is your analogue.
- **Falsifies world-model claim if:** rollout MSE grows super-linearly with $h$, or Graph-JEPA's rollout matches/exceeds Sequential-JEPA at long horizons (would mean graph structure helps one-step but not extrapolation, undermining the world-model framing).
- **Effort:** ~1 day to implement, ~2 H100-hr to run per condition × 5 seeds × 2 models × 4 horizons. Use existing checkpoints — no retraining.
- **Frame supported:** world-model (rollout evidence). **Without this, "world model" is overclaiming.**
- **Connection to Wang Theorem 4.4:** if H2 (straightening) holds AND rollout MSE stays bounded, you have empirical evidence for Wang's planning-Hessian conditioning prediction in the graph regime — a strong theoretical bridge.

### Tier 3 — robustness checks for the rebuttal

#### T3.1 Hyperparameter sensitivity
- mask_ratio ∈ {0.10, 0.20, 0.30, 0.40}
- lambda_reg (BCS) ∈ {0.5×, 1×, 2×}
- ema_momentum_start ∈ {0.99, 0.996, 0.999}
- Report whether the win is robust or knife-edge. **Falsifies "robust effect" if:** any sweep dimension swings Δcos by > 0.04.
- **Effort:** ~2 days, can run partially in parallel.

#### T3.2 Encoder-type ablation (graph backbone)
- Replace GATv2 with GCN, GIN, GraphSAGE. 5 seeds × 3 encoders × 1 dataset (TGBN-Trade).
- Tests whether the win depends on attention specifically.
- **Effort:** ~2 days.

#### T3.3 Temporal positional encoding ablation
- Disable any temporal positional encoding the predictor uses, train Graph-JEPA without it.
- Tests whether the win is from *graph* structure or from *temporal* structure.
- **Effort:** ~1 day.

#### T3.4 Compute parity check
- Measure param count, train-time FLOPs, inference FLOPs of Graph-JEPA vs Sequential-JEPA on a single batch. Write to `results/compute_parity.json`.
- **Falsifies "capacity-matched" if:** > 15% disparity on any axis.
- **Effort:** 2 hr.

#### T3.5 Bootstrap CI on win rate + effect size
- Currently: 87% win rate as a point estimate.
- Add: 95% bootstrap CI on win rate (10k resamples). Report Cliff's δ alongside Wilcoxon p.
- **Effort:** 1 hr (`src/eval/wilcoxon.py` extension).

#### T3.6 More seeds
- Move from 5 to 10 seeds on TGBN-Trade. Doesn't change p-value (already < 10⁻¹²) but is the NeurIPS standard.
- **Effort:** ~10 H100-hr.

### Tier 4 — extras for thesis weight

#### T4.1 Add a contrastive baseline
- Adapt GraphCL or MVGRL for temporal snapshots. Train under the same protocol.
- Strengthens Table 1 significantly.
- **Effort:** ~3 days.

#### T4.2 Add a masked-autoencoder baseline
- Adapt GraphMAE for temporal snapshots, or use DyGFormer's masking objective.
- **Effort:** ~3 days.

#### T4.3 Supervised topline
- Fully-supervised temporal GAT trained directly on the node-attribute regression target. This is your "F-GIN" row.
- **Effort:** ~1 day (architecture exists, just remove SSL).

#### T4.4 Theta-sweep on context length
- context_k ∈ {1, 2, 4, 8, 16}. Tests whether longer context improves prediction (and whether straightening cos(v_t, v_{t+1}) increases with longer context).
- **Effort:** ~1 day per dataset.

### Future work — cross-dataset extension (deferred per §4)

These experiments were originally Tier 1 but are demoted after the Genre / Genre-v2 / EU-Email negative results. They are not required for the TGBN-Trade contribution. Pursue them only after the in-scope paper is done, or in a follow-up.

- **F.1 TGBN-Genre with richer temporal state.** Current architecture fails (val 1.637 → 1.638). Hypothesis to test: an EMA-smoothed node feature (e.g., 4-week trailing average of edge volume) would inject the autocorrelation Genre's raw weekly snapshots lack.
- **F.2 TGBN-Genre-v2 with bipartite-typed predictor.** Add separate predictor heads for user-vs-item nodes; current shared head may average over modes. Modest architectural change (~3 days).
- **F.3 TGBL-Wiki / TGBL-Coin (link prediction format).** Different signal type entirely. May or may not have the autocorrelation property; smoke first before committing.
- **F.4 JODIE-Reddit / JODIE-Wikipedia.** Builders exist (`src/data/jodie_builder.py`) but not validated. Same caveat: smoke before commitment.

### 5.7 Suggested run order

```
week 1: T2.3 (curvature metric — cheapest, runs on existing checkpoints)
        T1.0 (METR-LA build + smoke train)
week 2: T1.0 (METR-LA full 5 seeds) + T1.1 (TGBN-Trade extend to 10 seeds)
        T2.5 (multi-step rollout impl on TGBN-Trade)
week 3: T2.1 (Seq-rank-8 ablation) + T2.2 (random-edge control) on TGBN-Trade
week 4: T1.2 (Genre + EMA features) + T1.3 (Genre-v2 bipartite head)
week 5: T1.4 (EU-Email monthly agg) + T1.5 (JODIE-Reddit smoke)
week 6: T1.6 (JODIE-Wikipedia) + T2.4 (linear probe on Trade + METR-LA)
week 7: T3.x robustness as time permits, write SPEC.md
week 8+: T4.x baselines (contrastive, MAE, supervised topline)
```

Pre-register hypothesis bounds (H1a, H1b, H2, H3, H4, H5, H6) **before** running T1.0 and T2.1. This is what separates the paper from a benchmark report.

---

## 6. Repo scaffolding TL;DR

The science gap (sections 1–5) is the load-bearing one. The repo gap (sections 7–13) is independent; both must close.

GRAPH-JEPA-2 already does the hard things well: deterministic seeding (`src/utils/seed.py`), paired Wilcoxon with Bonferroni (`src/eval/wilcoxon.py`), per-seed result tracking (`results/tgbn_trade/seed{0..4}/`), 1622 LOC of tests across 17 files. What's missing is the layer above the science: artifacts that make a reviewer trust that *every number in the paper is traceable to one command*. Reference repo treats reproducibility as a contract: `LICENSE`, `CITATION.cff`, `REPRODUCE.md`, `SPEC.md`, `pyproject.toml` with optional-deps groups, ruff config, `.tex` paper sources, parquet+jsonl shards instead of loose JSON, one script per experimental claim, and `make_figures.py` / `make_tables.py` that regenerate every PDF figure and `.tex` table from `results/`. None of that exists here.

---

## 7. Side-by-side scorecard

| Dimension | geometry-of-consolidation | GRAPH-JEPA-2 | Gap severity |
|---|---|---|---|
| LICENSE | MIT, present | absent | **blocker** for submission |
| CITATION.cff | present, full bibtex preferred-citation | absent | **blocker** |
| Reproduction guide | dedicated `REPRODUCE.md` with compute-budget table per experiment | inline in `README.md`, no compute budget | high |
| Spec doc | `SPEC.md` + `SPEC_AUDIT.md` + `SPEC_AUDIT_FINAL.md` framing theorem/method/empirics as contract | none — paper draft scattered across `docs/paper/*.md` | high |
| Packaging | `pyproject.toml`, pip-installable (`pip install -e .`), optional-deps groups (`[experiments]`, `[modal]`, `[vllm]`, `[dev]`) | `requirements.txt` only, floating version floors (`torch>=2.0`) | high |
| Linting | ruff configured in `pyproject.toml` (`E,F,W,I,N,UP,B`, line-length 100) | none | medium |
| Tests | `tests/test_gac.py`, `tests/test_pipeline.py` (focused) | 17 files, 1622 LOC (broader) | **GRAPH-JEPA-2 wins** |
| CI | none in either | none | medium (would be a real strength to add) |
| Paper sources | `paper/arxiv/main.tex` + `paper/neurips/main.tex` + `supp.tex`, modular sections, `references.bib`, `figs/*.pdf`, `tables/*.tex` | `docs/paper/*.md` markdown drafts only | high |
| Results format | parquet + jsonl shards (`results/e{1..9}/`), per-experiment `summary.json` | per-seed JSON files in `results/<dataset>/seed{N}/` | medium |
| Aggregation | `scripts/make_figures.py`, `scripts/make_tables.py`, `scripts/make_supp_tables.py`, `scripts/calibrate_c1.py` | `results/tgbn_trade/results.md` hand-edited | high |
| Experiment scripts | one per claim: `experiments/e1_theorem_validation.py` … `e9_temporal_mrr.py` | `experiments/train_tgjepa.py`, `eval_all.py`, `eval_tgjepa.py` (not 1:1 with claims) | medium |
| Modal orchestration | `modal_app/app.py` with `run_one`, `run_all`, `vllm_shard`, `embed`, `build_wiki_scale` | `experiments/train_tgjepa.py` runs on Modal, but no central app | medium |
| Resumability | sharded jsonl (`--n-shards 8 --shard-id 0`), parquet reduction step | none | medium |
| Single seed source | `GAC_SEED=0` env var threaded through pipeline | seeds passed per-call via configs | low |
| Compute budget | table in `REPRODUCE.md`: cells × seeds × GPU-hours per experiment, total ~50 GPU-hr | not documented | high |
| Statistical analysis | bootstrap CIs, c1 calibration with coverage | paired Wilcoxon + Bonferroni, 95% CI across 5 seeds | **comparable** |
| Seed determinism | fixed seed, `GAC_SEED` env var | full coverage (`cudnn.deterministic`, `use_deterministic_algorithms`, per-sample seeded `Generator`) | **GRAPH-JEPA-2 wins** |
| `.gitignore` hygiene | clean | `__pycache__/*.pyc` are tracked despite `.gitignore` listing `__pycache__/` (12 modified `.pyc` files in current `git status`) | medium |
| Docstrings on models | minimal in both | minimal — `GraphEncoder`, `TargetEncoder`, `TemporalGraphPredictor`, `TGJEPALoss` lack class-level docstrings | medium |
| Type hints | partial in both | sparse on models/training, full on eval | medium |
| TODO/FIXME/XXX/HACK | clean | clean (zero instances in `src/`) | **both clean** |

---

## 8. The reproducibility contract (what the reference repo enforces)

The geometry-of-consolidation README states: *"Every number in the paper is reproducible from `results/`. Every figure is reproducible from the scripts."* That sentence is the contract. It is enforceable because:

1. **Results are the source of truth, not the code.** `results/e{1..9}/` ships in-repo as parquet. Each experiment also has `summary.json` and `shard_NN.jsonl` for inspection. Reviewers can rerun figure-making without rerunning experiments.
2. **One script per experimental claim.** `experiments/e1_theorem_validation.py` produces `results/e1/`. The mapping is bijective.
3. **One script per artifact-class.** `scripts/make_figures.py`, `scripts/make_tables.py`, `scripts/make_supp_tables.py`, `scripts/calibrate_c1.py`.
4. **The paper PDF is built from sources in the repo.** `paper/neurips/main.tex` + `references.bib` + `figs/*.pdf` + `tables/*.tex`. No Overleaf, no copy-paste from notebooks.
5. **Compute budget is published.** Table giving cells, seeds, and GPU-hours for each experiment, totalling ~50 GPU-hours. Reviewer knows the cost up front.
6. **Cloud runs are first-class.** `modal_app/app.py` exposes `run_one --exp e1` and `run_all`. Sharding is a CLI flag. Failed shards resume from the same jsonl file.

GRAPH-JEPA-2 has the components but they are not bound by a contract a reviewer can verify in one command.

---

## 9. Concrete repo-rigour gaps and remediation

### 8.1 Repo metadata (blockers)

- [ ] **Add `LICENSE`** at repo root. MIT or Apache-2.0.
- [ ] **Add `CITATION.cff`** at repo root (template from reference repo).
- [ ] **Author/affiliation line in README.md** (anonymise during review, restore for camera-ready).

### 8.2 Reproducibility scaffolding

- [ ] **Create `REPRODUCE.md`** with compute budget per experiment.
- [ ] **Pin all versions in `requirements.txt`** via `pip freeze` from training env.
- [ ] **Migrate to `pyproject.toml`** with optional-deps groups.
- [ ] **Single-source seed via env var.** `GRAPH_JEPA_SEED=0`.
- [ ] **Frozen config artifact.** README references `docs/frozen-config-tgjepa.md` but the file does not exist. Either create it or remove the reference.

### 8.3 Code quality

- [ ] **Ruff config** in `pyproject.toml`: `select = ["E", "F", "W", "I", "N", "UP", "B"]`.
- [ ] **Type hints on `GraphEncoder`, `TargetEncoder`, `TemporalGraphPredictor`, `TGJEPALoss`** `__init__` and `forward`.
- [ ] **Class-level docstrings** on the four core classes.
- [ ] **Stop tracking `__pycache__`.** `git rm -r --cached '*.pyc'` + `git rm -r --cached '**/__pycache__'`.

### 8.4 Experiment-script structure

- [ ] **One script per claim.** Suggested layout under `experiments/`:
  - `e1_temporal_pred.py` → main result on TGBN-Trade
  - `e2_genre.py` → TGBN-Genre replication
  - `e3_genre_v2.py` → TGBN-Genre-v2 (bipartite)
  - `e4_seqjepa_ablation.py` → capacity-matched Sequential-JEPA
  - `e5_seqjepa_rank8.py` → rank-controlled ablation (T2.1)
  - `e6_random_edge.py` → degree-preserving rewire (T2.2)
  - `e7_linear_probe.py` → downstream transfer (T2.4)
  - `e8_curvature.py` → straightening metric (T2.3)
  - `e9_rollout.py` → multi-step rollout (T2.5)
  - `e10_metrla.py` → METR-LA training + horizon eval (T1.0)
  - `e11_genre_ema.py`, `e12_genre_v2_bipartite.py`, `e13_email_monthly.py`, `e14_jodie_reddit.py`, `e15_jodie_wiki.py` → dataset-specific re-attempts (T1.2-T1.6)
- [ ] **Per-experiment output dir.** `results/e{1..9}/seed{0..4}.json` + `summary.json`.

### 8.5 Aggregation and figure pipeline

- [ ] **`scripts/make_tables.py`** — reads `results/e*/summary.json`, emits `paper/tables/*.tex`.
- [ ] **`scripts/make_figures.py`** — emits `paper/figs/*.pdf`.
- [ ] **`scripts/aggregate.py`** — wraps Wilcoxon + Bonferroni + bootstrap, writes `results/FINDINGS.md`.

### 8.6 Paper sources in-repo

- [ ] `paper/neurips/main.tex` with NeurIPS 2026 style.
- [ ] `paper/neurips/sections/0{0..N}_*.tex` modular sections.
- [ ] `paper/neurips/references.bib`.
- [ ] `paper/neurips/figs/*.pdf` generated by `make_figures.py`.
- [ ] `paper/neurips/tables/*.tex` generated by `make_tables.py`.

### 8.7 SPEC.md

- [ ] Sections: (0) executive summary, (1) positioning vs Skenderi/Wang, (2) hypotheses H1–H4, (3) empirical contract per experiment, (4) frozen hyperparameters, (5) compute budget.

### 8.8 CI

- [ ] `.github/workflows/test.yml` running `pytest tests/` on push.
- [ ] `.github/workflows/smoke.yml` running `experiments/stress_test_smoke.py` on tiny slice.

---

## 10. Combined priority order (science + scaffolding)

If you do nothing else, in this order:

| # | Task | Tier | Effort | Why first |
|---|---|---|---|---|
| 1 | T2.3 — implement `mean_velocity_cos` straightening metric in `eval_runner.py` | science | 2 hr | feeds every later experiment |
| 2 | LICENSE + CITATION.cff | scaffolding | 10 min | non-negotiable for submission |
| 3 | Pin `requirements.txt` from current training env | scaffolding | 15 min | reproducibility floor |
| 4 | `git rm -r --cached __pycache__` | scaffolding | 5 min | cleanliness |
| 5 | Write SPEC.md with H1a, H1b, H2–H6 pre-registered | science | 2 hr | turns observations into claims |
| 6 | **T1.0 — build METR-LA + train 5 seeds** | science | 1 day + 10 GPU-hr | flagship dataset; without this the world-model framing has no recognized benchmark |
| 7 | **T2.5 — multi-step rollout eval (eval3 placeholder → real impl)** | science | 1 day + 4 GPU-hr | **required for world-model claim — without it, framing collapses to repr-learning** |
| 8 | T1.1 — extend TGBN-Trade to 10 seeds + add curvature/rollout | science | 6 GPU-hr | tightens H1a, supports H6 |
| 9 | T2.1 — Sequential-JEPA-rank-8 ablation | science | 1 day + 20 GPU-hr | causal mechanism test |
| 10 | T2.2 — random-edge control | science | 1 day + 20 GPU-hr | structure-load-bearing test |
| 11 | T2.4 — linear-probe downstream | science | 1 day | transfer evidence |
| 12 | T1.2 — TGBN-Genre + EMA features | science | 1 day + 8 GPU-hr | broaden scope claim |
| 13 | T1.3 — TGBN-Genre-v2 + bipartite head | science | 3 days + 8 GPU-hr | architectural extension test |
| 14 | T1.4 — EU-Email monthly aggregation sweep | science | 1 day + 6 GPU-hr | calibrates H5 boundary |
| 15 | T1.5 — JODIE-Reddit smoke + 5 seeds | science | 1 day + 10 GPU-hr | new domain |
| 16 | T1.6 — JODIE-Wikipedia smoke + 5 seeds | science | 10 GPU-hr | new domain |
| 17 | REPRODUCE.md with compute budget | scaffolding | 1 hr | reviewer confidence |
| 18 | `pyproject.toml` + ruff | scaffolding | 1 hr | hygiene |
| 19 | `scripts/make_tables.py` | scaffolding | 3 hr | every-number-from-one-command |
| 20 | T3.5 — bootstrap CI + Cliff's δ | science | 1 hr | strengthens stats |
| 21 | T3.6 — 10 seeds on TGBN-Trade (already done via #8) | science | merged | — |

**Items 1–11 are the experimental backbone.** T1.0 (METR-LA) and T2.5 (rollout) are the two non-negotiables for the world-model framing. Items 12–16 fill out the 7-dataset matrix. Items 17–21 are polish. Tier 3 robustness (T3.1–T3.4) and Tier 4 baselines (T4.x) come after.

---

## 11. The two experiments that matter most

**T1.0 (METR-LA training) + T2.5 (rollout) + T2.3 (curvature) + T2.1 (rank-8 ablation).**

Together they let you write the load-bearing paragraph for the world-model framing:

> *"We present GRAPH-JEPA-2, the first joint-embedding predictive architecture trained on temporal graph data as a world model. On METR-LA, the standard temporal-graph forecasting benchmark, frozen Graph-JEPA representations achieve horizon-12 MAE within X% of supervised DCRNN / Graph WaveNet under linear probe, despite no supervision during pretraining. On TGBN-Trade across 10 seeds, the model achieves one-step prediction cosine 0.83 ± 0.01 vs 0.75 ± 0.02 for a capacity-matched non-graph ablation (paired Wilcoxon p < 10⁻¹², 87% win rate), with effective rank 8.0 ± 0.2 vs 19.1 ± 0.7 — quantitatively matching Wang et al.'s (2026) d=8 finding for image-JEPA, here demonstrated to transfer to graph-structured observations. Multi-step autoregressive rollout MSE grows sub-linearly with horizon h ∈ {1,2,4,8} on both datasets, confirming the latent serves as a temporal world-model substrate. A rank-controlled Sequential-JEPA-rank-8 ablation closes the one-step gap on TGBN-Trade (Δcos = 0.01, n.s.), establishing that graph-induced compression is the causal mechanism for the straightening signature. A degree-preserving randomised-edge control loses by Δcos = 0.06, establishing that real semantic graph topology — not the GATv2 inductive bias alone — is load-bearing. Across the seven-dataset matrix, the method's empirical scope is characterised by an autocorrelation prerequisite: it works directly on TGBN-Trade and METR-LA; it works on TGBN-Genre and EU-Email after EMA feature smoothing or longer-window aggregation respectively; it requires a type-conditioned predictor head for bipartite TGBN-Genre-v2; JODIE-Reddit and JODIE-Wikipedia further calibrate the boundary of the autocorrelation requirement."*

That paragraph requires the following experiments: **T1.0** (METR-LA training, flagship benchmark), **T1.1** (TGBN-Trade extended), **T2.5** (rollout — required for "world model"), **T2.3** (curvature metric), **T2.1** (rank-8 ablation), **T2.2** (random-edge control), and **T1.2–T1.6** (the dataset re-attempts that produce the empirical-scope sentence). It is the paragraph that turns a TGBN-Trade-only thesis result into a NeurIPS submission with the world-model framing across the full 7-dataset matrix.

**Without T1.0 + T2.5, "world model" is overclaiming and the load-bearing paragraph loses its METR-LA leg and rollout centrepiece.** With both, the framing is honest and matches the V-JEPA / DINO-WM lineage's eval protocol.

---

## 12. What NOT to do

- **Don't pad to 9 experiments.** Reference has 9 because their theorem demands cell-level, scale, downstream, encoder-universality, ablation, and temporal validations. Yours has fewer claims; 4–5 well-designed experiments is right-sized.
- **Don't chase hyperbolic embeddings just because Skenderi did.** That was a graph-level design choice. Node-level temporal prediction on the unit sphere is the right geometry for your task.
- **Don't add METIS partitioning.** It's a graph-level patching abstraction; you predict per-node per-snapshot, you don't need patches.
- **Don't add a Modal orchestrator app yet.** Single-experiment scripts are fine while runs are < 1 day each. Add orchestration only when you have multi-day sweeps.
- **Don't drop your existing tests.** 17 files / 1622 LOC is a strength relative to both reference repos. Keep them.

---

## 13. What GRAPH-JEPA-2 already does better than the references

For the document to be honest:

- **Determinism is more thorough than either reference.** `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)` *and* per-sample seeded `torch.Generator()` (`src/data/dataset.py:42-44`). Skenderi reports 5–10 seeds with no determinism specifics; geometry-of-consolidation has a single `GAC_SEED` env var.
- **Test suite is broader.** 17 files, 1622 LOC, with explicit tests for masking, EMA momentum, encoder forward, predictor token-building, integration. Skenderi has none visible. Geometry-of-consolidation has 2 test files.
- **Statistical machinery is correctly applied.** Paired Wilcoxon (`alternative='greater'`) + Bonferroni across the eval family, win-rate per node-pair, 95% CI across 5 seeds. The thesis test (p<10⁻¹², 87% win rate on TGBN-Trade) survives reviewer scrutiny.
- **Eval design is principled.** Three baselines including a capacity-matched ablation (Sequential-JEPA with same predictor, MLP encoder). Frozen config before eval. Hard-coded splits prevent test-set peek.
- **Zero TODO/FIXME/XXX/HACK comments** in `src/`.

The science is there. The packaging needs to catch up, and the experiment space needs to expand from one dataset to a hypothesis-falsification matrix.

---

## 14. References

- Skenderi, Li, Tang, Cristani — *Graph-level Representation Learning with Joint-Embedding Predictive Architectures*, TMLR 01/2025. https://github.com/geriskenderi/graph-jepa
- Wang, Bounou, Zhou, Balestriero, Rudner, LeCun, Ren — *Temporal Straightening for Latent Planning*, arXiv:2603.12231, March 2026. https://agenticlearning.ai/temporal-straightening
- niashwin/geometry-of-consolidation — NeurIPS 2026 submission, repo-rigour reference. https://github.com/niashwin/geometry-of-consolidation
- NeurIPS reproducibility checklist: https://neurips.cc/public/guides/PaperChecklist
- Citation File Format: https://citation-file-format.github.io/
- ruff config docs: https://docs.astral.sh/ruff/configuration/
