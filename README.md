# Graph-JEPA

PyTorch codebase for **Graph-JEPA** (Temporal Graph Joint-Embedding Predictive Architecture), a method for self-supervised learning of node-level representations from temporal graphs.

[\[LeCun JEPA\]](https://openreview.net/pdf?id=BZ5a1r-kVsf)
[\[I-JEPA\]](https://arxiv.org/abs/2301.08243)
[\[V-JEPA\]](https://arxiv.org/abs/2404.08471)
[\[Graph-JEPA (Skenderi)\]](https://arxiv.org/abs/2410.06747)
[\[LeJEPA\]](https://arxiv.org/abs/2511.08544)

## Method

Graph-JEPA extends the joint-embedding predictive architecture to temporal graphs. A trained GATv2 encoder produces node embeddings from each weekly graph snapshot. An EMA target encoder provides stable prediction targets. A spatiotemporal transformer predictor takes the full visible context: all nodes across all history steps plus all other nodes at the target timestep via the target encoder, and predicts the masked node's representation at t+1. This is the direct V-JEPA translation to graphs.

**Research question:** Does relational context (who communicates with whom) improve node-level state prediction beyond sequential observation of a single node's history alone?

**Lineage:** I-JEPA (images) -> V-JEPA (video) -> Graph-JEPA static (Skenderi, 2025) -> **Temporal Graph-JEPA (this work)**

```
weekly graph snapshots [t-k, ..., t, t+1]
         |
         v
  online encoder (trained)             target encoder (EMA, stop-grad)
  GATv2, 3 layers, 256d                slow-moving copy of online encoder
  runs on full graph at t-k to t       runs on full graph at t+1
         |                                      |
         v                                      v
  node embeddings                      node embeddings at t+1
  (all nodes, all history steps)       (all nodes, stop-grad)
         |                                      |
         +------------------+------------------->
                            |
                            v
                   temporal predictor
                   bidirectional transformer, 2 layers
                   input tokens: [node_v history t-k:t]
                               + [all nodes u history t-k:t]
                               + [all nodes u at t+1 via EMA]
                               + [mask token at (v, t+1)]
                   output: predicted embedding at mask position
                            |
                            v
                   SigReg (LeJEPA / BCS loss)
                   isotropic Gaussian regularization
                   prevents representation collapse
```

## Approach

**Graph encoder (online, trained).** A 3-layer GATv2Conv network (input projection Linear(384, 256), 4 attention heads, dropout 0.1, LayerNorm after each layer). The encoder is trained jointly with the predictor under the JEPA objective. Frozen encoders learn retrieval features, not temporal dynamics. Approximately 1-2M parameters.

**Target encoder (EMA, stop-grad).** Exact architectural copy of the graph encoder. Updated via EMA (momentum cosine-scheduled 0.996 -> 1.0). All outputs are stop-gradded. Provides stable regression targets.

**Temporal predictor.** A 2-layer bidirectional transformer (4 heads, dim 256, MLP ratio 2). Takes the full spatiotemporal context: k history steps for all nodes via the online encoder, all nodes at t+1 via the target encoder (stop-grad), and a learnable mask token at position (v, t+1) carrying temporal and node-identity embeddings. For N=49 visible nodes and k=4: 250 total input tokens. Approximately 500K-1M parameters.

Why bidirectional and not causal: every visible token at every timestep should inform the prediction of the masked node. The only withheld information is the masked node's embedding at t+1. The attention pattern is otherwise unrestricted.

**Mask token identity.** The mask token carries the identity of the missing node and its target timestep: `mask = temporal_pos_emb[t+1] + node_id_emb[v]`. The predictor knows who and when it is predicting.

**Loss.**
```
L = SmoothL1(z_pred, sg(z_target)) + lambda_reg * L_SigReg
```
lambda_reg is frozen in `docs/frozen-config-tgjepa.md` before any evaluation run.

**Masking.** 15-30% of nodes are masked at timestep t+1. Both encoders always process the full graph without masking. Masking is applied at the predictor stage only: masked nodes at t+1 have their target encoder embedding replaced by the learnable mask token.

## Benchmarks

| Dataset | Domain | Nodes | Timesteps | Key Feature |
|---|---|---|---|---|
| Enron Email | Organizational communication | ~150 executives | ~180 weeks | Natural collapse event (Oct 2001) for anomaly detection |
| EU Email (SNAP) | Academic institution | ~986 people | ~3 years | Same structure as Enron, different context |
| JODIE — Reddit | User-subreddit interaction | Users + subreddits | Continuous | Large-scale, widely cited temporal graph benchmark |
| JODIE — Wikipedia | User-page edit network | Users + pages | Continuous | Temporal node feature dynamics |
| TGBN-Trade (TGB) | Country trade flows | Countries | Annual | Non-communication domain, tests generalization |

## Evaluations

5 random seeds. Paired Wilcoxon signed-rank test for all comparisons. Bonferroni correction across the full evaluation family. Mean +/- 95% CI for all metrics.

| Eval | Metric | Baseline | Pass Criterion |
|---|---|---|---|
| Node state prediction | cos(z_pred, z_target) | copy-forward, graph-average | model > copy-forward, p < 0.05 after Bonferroni |
| Graph context ablation | Graph-JEPA vs Sequential-JEPA (see below) | Sequential-JEPA, copy-forward | Graph-JEPA > Sequential-JEPA, p < 0.05 |
| Multi-step rollout | cos at k=1,2,4 weeks ahead (autoregressive) | copy-forward at each horizon | stays above copy-forward at k >= 2 |
| Downstream probes | linear probe accuracy on 4 tasks | frozen BGE-small encoder | beats frozen BGE on >= 2/4 tasks, p < 0.05 |
| Anomaly detection | per-node prediction error over time (Enron) | n/a | qualitative: error spike before Oct 2001 collapse? |
| Representation quality | effective rank, mean pairwise cosine, UMAP | n/a | rank > 30, mean cos < 0.5 |

Downstream probe tasks: (1) communication volume next week (regression), (2) primary topic next week (K=20 clusters), (3) new contact binary classification, (4) hub vs periphery role classification.

**Primary metric: Eval 2 (graph context ablation). This is the thesis experiment.**

## Ablation Design

| Model | Encoder | Predictor Input | Purpose |
|---|---|---|---|
| Graph-JEPA (full) | GATv2, trained | masked node history + all other nodes at all timesteps | main model |
| Sequential-JEPA | 3-layer MLP, matched params | masked node history only (k=4 tokens) | graph context ablation |
| Copy-forward | none | last known node embedding | trivial baseline |
| Graph-average | none | mean neighbor embedding at t | graph-aware non-learned baseline |

The MLP encoder in the sequential ablation is matched in parameter count to GATv2. This controls for model capacity and isolates relational context as the independent variable.

## Code Structure

```
.
├── configs/                            # experiment config files (.yaml)
├── src/
│   ├── data/
│   │   ├── enron_loader.py             #   download, parse, deduplicate Enron emails
│   │   ├── graph_builder.py            #   emails to weekly PyG snapshots
│   │   └── dataset.py                  #   temporal window sampling, masking, batching
│   ├── model/
│   │   ├── graph_encoder.py            #   GATv2 online encoder
│   │   ├── target_encoder.py           #   EMA copy + stop-gradient
│   │   ├── predictor.py                #   spatiotemporal transformer predictor
│   │   └── ema.py                      #   EMA update logic
│   ├── loss/
│   │   ├── prediction.py               #   SmoothL1
│   │   ├── sigreg.py                   #   SigReg / BCS (from eb_jepa/losses.py)
│   │   └── total.py                    #   combined loss
│   ├── train.py
│   ├── eval.py
│   └── utils/
├── experiments/
│   ├── train_tgjepa.py                 #   Modal entrypoint for training
│   ├── train_sequential_ablation.py    #   ablation training
│   ├── eval_all.py                     #   run all 6 evals
│   └── download_enron.py               #   download and preprocess Enron data
├── tests/
├── docs/
│   └── frozen-config-tgjepa.md        #   frozen hyperparameters (do not edit after eval)
├── results/
├── data/                               #   gitignored, populated by download script
├── ijepa/                              #   reference: facebookresearch/ijepa
├── jepa/                               #   reference: facebookresearch/jepa (V-JEPA)
├── graph-jepa/                         #   reference: geriskenderi/graph-jepa
├── jodie/                              #   reference: srijankr/jodie
├── TGB/                                #   reference: shenyangHuang/TGB
└── DyGLib/                             #   reference: yule-BUAA/DyGLib
```

## Getting Started

```bash
conda create -n graph-jepa python=3.10 pip
conda activate graph-jepa
pip install -r requirements.txt
```

**Download and preprocess Enron data:**
```bash
python experiments/download_enron.py
```

**Training (local):**
```bash
python src/main.py --fname configs/tgjepa_base.yaml --devices cuda:0
```

**Training (Modal, GPU):**
```bash
modal run experiments/train_tgjepa.py
```

**Run all evaluations:**
```bash
python experiments/eval_all.py --checkpoint results/tgjepa/seed0/best_model.pt
```

## Requirements

```
torch >= 2.0
torch-geometric >= 2.4
sentence-transformers >= 2.2
py-tgb
numpy
scikit-learn
scipy
matplotlib
umap-learn
pytest
modal
```

## License

See [LICENSE](./LICENSE).

## Citation

```bibtex
@misc{bodnar2026graphjepa,
  title={Temporal Graph-JEPA: Node-Level State Prediction from Relational Context},
  author={Bodnar, Sofia},
  year={2026}
}
```

**Related work:**
```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and others},
  journal={CVPR},
  year={2023}
}

@article{bardes2024vjepa,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and others},
  journal={arXiv:2404.08471},
  year={2024}
}

@inproceedings{skenderi2025graphjepa,
  title={Graph-JEPA: Graph-Level Representations via Joint Embedding Predictive Architecture},
  author={Skenderi, Geri},
  booktitle={GSP Workshop, ICASSP},
  year={2025}
}

@article{huang2023tgb,
  title={Towards Better Evaluation for Dynamic Link Prediction},
  author={Huang, Shenyang and others},
  booktitle={NeurIPS},
  year={2023}
}
```
