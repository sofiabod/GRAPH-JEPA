"""eval 10: does the encoder preserve bilateral trade topology?

claim: graph-JEPA's frozen embeddings encode pairwise relational structure such
that cosine similarity between country embeddings correlates with their actual
bilateral trade volume. a non-graph encoder, with no access to edge weights,
should produce embeddings whose pairwise similarities are uncorrelated with
trade weight.

method:
  1. for each model (graph / sequential / raw features), produce one embedding
     per country averaged across test snapshots.
  2. compute the upper-triangular pairwise cosine similarity matrix [N*(N-1)/2].
  3. compute the upper-triangular bilateral trade weight matrix from the same
     test snapshots (sum over snapshots of edge weights between each pair).
  4. score: Spearman rank correlation between embedding similarity and trade
     weight. report bootstrap 95% CI.

a positive Spearman with CI excluding 0 means the encoder has learned to
embed countries such that "neighbors in embedding space" are "trading partners
in the real world." sequential is the natural counterfactual: it has the same
JEPA objective and parameter count but never sees edge weights.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def _country_embeddings(online, graphs, context_k: int, device,
                        eval_indices: Optional[list[int]] = None) -> np.ndarray:
    """produce one L2-normalized embedding per country averaged across
    eval_indices snapshots. shape [N, D]."""
    online.eval()
    n_nodes = graphs[0].x.shape[0]
    if eval_indices is None:
        eval_indices = list(range(context_k, len(graphs)))

    # initialize accum with the first eval call so dtype/shape match without guessing
    assert eval_indices, "eval_indices must be non-empty"
    accum = torch.zeros(0, device=device)  # placeholder; replaced on first iter
    count = 0
    with torch.no_grad():
        for i, t_idx in enumerate(eval_indices):
            g = graphs[t_idx].to(device)
            z = online(g)
            if i == 0:
                accum = torch.zeros_like(z)
            accum = accum + z
            count += 1
    accum = accum / max(count, 1)
    accum = F.normalize(accum, dim=-1)
    return accum.cpu().numpy()


def _aggregate_trade_weights(graphs, eval_indices: list[int],
                              n_nodes: int, symmetric: bool = True) -> np.ndarray:
    """sum bilateral edge weights across test snapshots into an [N, N] matrix.

    if symmetric=True (default), w[i,j] = total i→j + j→i flow across snapshots —
    this captures the "intensity of bilateral relationship" regardless of direction.
    """
    W = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for t in eval_indices:
        g = graphs[t]
        ei = g.edge_index.cpu().numpy()  # [2, E]
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            ew = g.edge_attr.cpu().numpy()
            if ew.ndim > 1:
                ew = ew[:, 0]  # use first edge-attr column (trade volume)
        else:
            ew = np.ones(ei.shape[1])
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            W[i, j] += float(ew[k])
            if symmetric and i != j:
                W[j, i] += float(ew[k])
    if symmetric:
        # collapse to undirected: the [i,j] entry already includes both directions
        pass
    return W


def _pairwise_cosine(emb: np.ndarray) -> np.ndarray:
    """upper-triangular pairwise cosine similarity. emb is L2-normalized."""
    return emb @ emb.T


def _upper_triangle_pairs(M: np.ndarray) -> np.ndarray:
    """flatten upper-triangular entries (excluding diagonal). returns [N*(N-1)/2]."""
    n = M.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return M[iu, ju]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via numpy rank+pearson (avoids scipy
    typing annoyances and works in any environment)."""
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _bootstrap_spearman_ci(a: np.ndarray, b: np.ndarray,
                            n_resamples: int = 5000, ci: float = 0.95,
                            seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = a.size
    rhos = np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        rhos[r] = _spearman(a[idx], b[idx])
    alpha = (1.0 - ci) / 2.0
    return {
        "rho": _spearman(a, b),
        "ci_low": float(np.quantile(rhos, alpha)),
        "ci_high": float(np.quantile(rhos, 1.0 - alpha)),
        "n_pairs": int(n),
        "n_resamples": int(n_resamples),
    }


def run_topology_test(online, graphs, cfg, *, kind: str = "graph",
                      raw_features_idx: Optional[list[int]] = None,
                      seed: int = 0, device=None) -> dict:
    """run topology test for one model.

    kind ∈ {"graph", "sequential", "raw_features"}. for raw_features, online may
    be None and raw_features_idx selects which feature dims to use as the
    "embedding" (default: all dims).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_nodes = graphs[0].x.shape[0]
    K = cfg.training.context_k
    # determine test snapshots (bilateral trade aggregation + embedding averaging)
    import json
    try:
        test_lo, test_hi = cfg.data.test_weeks
    except Exception:
        with open(f"/app/{cfg.data.meta_path}") as f:
            meta = json.load(f)
        test_lo, test_hi = meta["test_range"]
    eval_indices = list(range(test_lo, test_hi + 1))

    if kind == "raw_features":
        # mean of raw features across test snapshots
        accum = torch.zeros(n_nodes, cfg.encoder.in_dim)
        for t in eval_indices:
            accum = accum + graphs[t].x.cpu()
        emb = (accum / len(eval_indices)).numpy()
        if raw_features_idx is not None:
            emb = emb[:, raw_features_idx]
        # L2 normalize for fair cosine comparison
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
        emb = emb / norms
    else:
        emb = _country_embeddings(online, graphs, K, device, eval_indices=eval_indices)

    cos_M = _pairwise_cosine(emb)
    W = _aggregate_trade_weights(graphs, eval_indices, n_nodes=n_nodes, symmetric=True)

    cos_pairs = _upper_triangle_pairs(cos_M)
    trade_pairs = _upper_triangle_pairs(W)

    # log-transform trade (heavy-tailed) for Spearman robustness; rank-order
    # is invariant to monotonic transform but bootstrap CI is more stable on log
    log_trade = np.log1p(trade_pairs)

    boot = _bootstrap_spearman_ci(cos_pairs, log_trade, n_resamples=5000, seed=seed)
    return {
        "kind": kind,
        "n_countries": int(n_nodes),
        "n_pairs": boot["n_pairs"],
        "spearman_rho": boot["rho"],
        "spearman_ci95_low": boot["ci_low"],
        "spearman_ci95_high": boot["ci_high"],
        "test_snapshots": eval_indices,
    }
