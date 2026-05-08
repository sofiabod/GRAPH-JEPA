"""eval 9: linear probe for transferable representations.

claim under test: graph-JEPA's frozen embeddings encode dynamics that
linearly predict next-period trade growth. a probe trained on (embedding[i,t],
growth[i,t→t+1]) pairs should:
  - achieve meaningful R² for graph
  - underperform with sequential embeddings (less context)
  - underperform with raw features (growth is a derivative, not a feature)

a clean "graph wins, sequential ~baseline, raw features ~baseline" result
is the thesis-grade "learned something transferable" demonstration.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _all_snapshot_embeddings(online, graphs, device, from_t: int, to_t: int) -> np.ndarray:
    """produce per-snapshot embeddings via the encoder. returns array of
    shape [T, N, D] where T = to_t - from_t + 1."""
    online.eval()
    embs = []
    with torch.no_grad():
        for t in range(from_t, to_t + 1):
            g = graphs[t].to(device)
            z = online(g)
            # encoder may already L2-normalize; preserve consistent geometry
            z = F.normalize(z, dim=-1)
            embs.append(z.cpu().numpy())
    return np.stack(embs, axis=0)


def _trade_volume_per_country(graphs, from_t: int, to_t: int, feature_idx: int = 0) -> np.ndarray:
    """extract per-country trade volume signal across snapshots.

    feature_idx 0 = normalized volume in our 6d node feature convention.
    returns [T, N].
    """
    out = []
    for t in range(from_t, to_t + 1):
        x = graphs[t].x.cpu().numpy()
        out.append(x[:, feature_idx])
    return np.stack(out, axis=0)


def _growth_targets(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """compute log-growth Y[t] = log(vol[t+1]) - log(vol[t]) for t = 0..T-2.
    returns [T-1, N]. clipped to avoid log of non-positive volumes."""
    safe = np.clip(volume, eps, None)
    log_v = np.log(safe)
    return log_v[1:] - log_v[:-1]


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """coefficient of determination."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _kfold_linear_probe(X: np.ndarray, Y: np.ndarray, k: int = 5, seed: int = 0) -> dict:
    """k-fold cross-validated linear probe. returns mean R², MAE, per-fold lists."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold
    except ImportError as e:
        raise RuntimeError("sklearn required for linear probe") from e

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    r2s, maes, n_test_total = [], [], 0
    for train_idx, test_idx in kf.split(X):
        # ridge regularization is necessary because X may have D > 100; OLS overfits
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X[train_idx], Y[train_idx])
        Y_pred = model.predict(X[test_idx])
        r2s.append(_r2_score(Y[test_idx], Y_pred))
        maes.append(float(np.mean(np.abs(Y[test_idx] - Y_pred))))
        n_test_total += len(test_idx)
    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "r2_per_fold": r2s,
        "mae_mean": float(np.mean(maes)),
        "mae_per_fold": maes,
        "n_test_total": int(n_test_total),
        "n_folds": int(k),
    }


def run_probe(
    online,
    graphs,
    cfg,
    splits,
    *,
    kind: str = "graph",
    feature_idx: int = 0,
    folds: int = 5,
    seed: int = 0,
    device=None,
) -> dict:
    """run linear probe for one model.

    args:
      online: encoder (graph or sequential)
      kind: label string ("graph" / "sequential" / "raw_features") for output
      feature_idx: which scalar in node features is "trade volume" (default 0)
      folds: k-fold CV
      seed: RNG for fold split

    returns r2/mae over k-fold CV against next-period log-growth target.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_range, val_range, test_range = splits
    # use ALL valid (context_k <= t < last_snapshot) pairs across train+val+test
    K = cfg.training.context_k
    T = len(graphs)
    from_t = K
    to_t = T - 2  # need t+1 to exist for growth target

    # Y[t-from_t, n] = log-growth from snapshot t to t+1 for country n
    volume = _trade_volume_per_country(graphs, from_t, to_t + 1, feature_idx=feature_idx)
    Y_full = _growth_targets(volume)  # [to_t - from_t + 1, N]

    if kind == "raw_features":
        # raw features at t (full feature vector, all dims) used to predict growth t→t+1
        X_full = np.stack(
            [graphs[t].x.cpu().numpy() for t in range(from_t, to_t + 1)],
            axis=0,
        )  # [T-K-1, N, F]
    else:
        X_full = _all_snapshot_embeddings(online, graphs, device, from_t, to_t)
        # X_full: [T-K-1, N, D]

    # flatten (t, n) → row of dataset; one prediction per (country, time) pair
    n_t, n_nodes, d = X_full.shape
    X_flat = X_full.reshape(n_t * n_nodes, d)
    Y_flat = Y_full.reshape(n_t * n_nodes)

    # drop any rows with NaN/inf in Y (shouldn't happen but defensive)
    finite = np.isfinite(Y_flat) & np.all(np.isfinite(X_flat), axis=1)
    X_flat = X_flat[finite]
    Y_flat = Y_flat[finite]

    metrics = _kfold_linear_probe(X_flat, Y_flat, k=folds, seed=seed)
    metrics.update(
        {
            "kind": kind,
            "n_samples": int(X_flat.shape[0]),
            "embedding_dim": int(X_flat.shape[1]),
            "snapshots_used": [int(from_t), int(to_t)],
            "feature_idx_used_for_volume": int(feature_idx),
        }
    )
    return metrics
