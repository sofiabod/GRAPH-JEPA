"""eval 14: synthetic shock injection — controlled OOD perturbation.

claim under test: in a controlled setting where we manufacture an OOD graph
state, does the model register the perturbation? specifically, does dropping
a structurally important country (one with many edges) produce a larger
prediction-error increase than dropping a structurally peripheral country?

setup:
  - take a trained encoder + predictor (frozen)
  - pick a year_idx in the training range (model has seen it)
  - for each candidate country to "knock out":
      a. zero out all edges incident to that country in graphs[year_idx]
      b. predict graphs[year_idx + 1] using context including the modified snapshot
      c. measure prediction error increase vs the un-perturbed baseline
  - report (knockout_country, error_delta) pairs
  - correlate error_delta with the country's actual graph centrality
    (degree centrality, total trade volume) → if positive, the model has
    internalized which countries matter

unlike anomaly detection on real shocks, this test is fully controlled:
we manufacture the perturbation, so any signal is direct evidence the model
has learned graph structure (not coincidence with external events).
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
import numpy as np

from src.eval.metrics import cosine_sim


def _predict_next_snapshot_cosines(online, target, predictor, context_graphs,
                                    target_graph, masked_ids, device):
    """forward graph-jepa once: encoder on context, predictor on tokens,
    return cosine sims of predicted vs target latents at masked positions."""
    from src.train import _encode_context, _build_tokens_for_sample

    masked_ids = masked_ids.to(device)
    visible_ids = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        ctx_embs = _encode_context(online, context_graphs)
        tgt_emb = target(target_graph.to(device))
        tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
            ctx_embs, tgt_emb, masked_ids, visible_ids, predictor
        )
        out = predictor(
            tokens.unsqueeze(0),
            time_indices.unsqueeze(0),
            node_ids_seq.unsqueeze(0),
        ).squeeze(0)
        z_pred = F.normalize(out[mask_positions], dim=-1)
        z_true = tgt_emb[masked_ids]
        return cosine_sim(z_pred, z_true).cpu().numpy()


def _knockout_country(graph, country_idx: int):
    """return a copy of `graph` with all edges incident to country_idx removed."""
    new_g = copy.copy(graph)
    ei = graph.edge_index.cpu()
    # keep edges that don't touch country_idx
    keep = (ei[0] != country_idx) & (ei[1] != country_idx)
    new_g.edge_index = ei[:, keep].to(graph.edge_index.device)
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        new_g.edge_attr = graph.edge_attr[keep]
    # x stays the same
    return new_g


def _country_degree(graph, n_nodes: int) -> np.ndarray:
    """compute degree centrality (in + out edge count) per country."""
    ei = graph.edge_index.cpu().numpy()
    deg = np.zeros(n_nodes)
    for k in range(ei.shape[1]):
        deg[int(ei[0, k])] += 1
        deg[int(ei[1, k])] += 1
    return deg


def _country_trade_volume(graph, n_nodes: int) -> np.ndarray:
    """sum of incident edge weights per country (trade volume centrality)."""
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return _country_degree(graph, n_nodes).astype(float)
    ei = graph.edge_index.cpu().numpy()
    ew = graph.edge_attr.cpu().numpy()
    if ew.ndim > 1:
        ew = ew[:, 0]
    vol = np.zeros(n_nodes)
    for k in range(ei.shape[1]):
        vol[int(ei[0, k])] += float(ew[k])
        vol[int(ei[1, k])] += float(ew[k])
    return vol


def run_synthetic_shock(online, target, predictor, graphs, cfg, *,
                        year_idx: int, n_top_countries: int = 30,
                        seed: int = 0, device=None) -> dict:
    """run synthetic shock for top-N countries by trade volume.

    args:
      year_idx: snapshot index to perturb (must have year_idx + 1 available)
      n_top_countries: pick the top-N highest-volume countries to test
        (so we get a meaningful range of centrality magnitudes)
    """
    from src.data.dataset import TemporalGraphDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online.eval()
    if hasattr(target, "encoder"):
        target.encoder.eval()
    predictor.eval()

    K = cfg.training.context_k
    if year_idx < K or year_idx + 1 >= len(graphs):
        raise ValueError(f"year_idx={year_idx} invalid for K={K} and {len(graphs)} snapshots")

    n_nodes = graphs[0].x.shape[0]

    # select top-n countries by trade volume on the perturbation snapshot
    target_graph_unperturbed = graphs[year_idx]
    next_graph = graphs[year_idx + 1]
    volumes = _country_trade_volume(target_graph_unperturbed, n_nodes)
    degrees = _country_degree(target_graph_unperturbed, n_nodes)
    top_country_indices = np.argsort(volumes)[::-1][:n_top_countries].tolist()

    # baseline: prediction error WITHOUT perturbation
    # mask all nodes in next_graph (predict every country) for comparable measurement
    masked_ids = torch.arange(n_nodes, device=device)
    context_graphs = graphs[year_idx + 1 - K:year_idx + 1]  # K snapshots before next_graph
    baseline_cos = _predict_next_snapshot_cosines(
        online, target, predictor, context_graphs, next_graph, masked_ids, device,
    )
    baseline_dev_mean = float(1.0 - np.mean(baseline_cos))

    # perturbed predictions: for each candidate country, knock it out at year_idx
    # context_graphs[-1] IS graphs[year_idx]. we replace it with the perturbed copy.
    results = []
    for ci in top_country_indices:
        perturbed = _knockout_country(target_graph_unperturbed, ci)
        # build perturbed context: same K-1 snapshots + perturbed year_idx
        perturbed_ctx = list(context_graphs[:-1]) + [perturbed]
        perturbed_cos = _predict_next_snapshot_cosines(
            online, target, predictor, perturbed_ctx, next_graph, masked_ids, device,
        )
        # change in prediction quality
        perturbed_dev_mean = float(1.0 - np.mean(perturbed_cos))
        delta_dev = perturbed_dev_mean - baseline_dev_mean
        # also compute deviation specifically for the OTHER N-1 countries
        # (not the knocked-out one — its self-prediction is corrupted by definition)
        other_mask = np.ones(n_nodes, dtype=bool)
        other_mask[ci] = False
        delta_others = float(np.mean(1.0 - perturbed_cos[other_mask]) -
                              np.mean(1.0 - baseline_cos[other_mask]))
        results.append({
            "country_idx": int(ci),
            "knocked_out_volume": float(volumes[ci]),
            "knocked_out_degree": int(degrees[ci]),
            "delta_dev_mean": delta_dev,
            "delta_dev_excluding_self": delta_others,
            "baseline_dev_at_country": float(1.0 - baseline_cos[ci]),
            "perturbed_dev_at_country": float(1.0 - perturbed_cos[ci]),
        })

    # spearman correlation: does delta_dev correlate with knocked-out country's volume/degree?
    deltas_excl_self = np.array([r["delta_dev_excluding_self"] for r in results])
    knocked_volumes = np.array([r["knocked_out_volume"] for r in results])
    knocked_degrees = np.array([r["knocked_out_degree"] for r in results])

    def _spearman(a, b):
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    return {
        "year_idx": int(year_idx),
        "n_countries_tested": int(n_top_countries),
        "baseline_dev_mean": baseline_dev_mean,
        "per_country_results": results,
        "spearman_dev_vs_volume": _spearman(deltas_excl_self, knocked_volumes),
        "spearman_dev_vs_degree": _spearman(deltas_excl_self, knocked_degrees.astype(float)),
        "mean_delta_dev_excl_self": float(deltas_excl_self.mean()),
        "max_delta_dev_excl_self": float(deltas_excl_self.max()),
    }
