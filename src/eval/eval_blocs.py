"""eval 8: bloc / region discovery from frozen graph-JEPA embeddings.

claim under test: graph-JEPA's encoder produces country embeddings that
cluster by real-world geopolitical region without any region label being
provided in training. a non-graph baseline should fail this because the
bloc concept lives in graph adjacency, not in per-node temporal signal.

method:
  1. for each model (graph, sequential, raw-features), produce one embedding
     per country by averaging encoder output across all valid context windows
     in the test split.
  2. k-means cluster the embeddings (k = number of UN M49 subregions present).
  3. score the clustering against the region ground truth via:
     - Adjusted Rand Index (ARI) — chance-corrected
     - Normalized Mutual Information (NMI)
     - Cluster purity (fraction of nodes in their majority-region cluster)
  4. compare graph vs sequential vs raw-features baseline.

a clean win for graph means it discovers the bloc structure without
supervision; sequential and raw-features should be noticeably worse.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def _country_embeddings(
    online, graphs, context_k: int, device, eval_indices: list[int] | None = None
) -> np.ndarray:
    """produce one embedding per node by averaging the online encoder's
    output across all valid context windows in eval_indices.

    eval_indices defaults to all snapshots that have a full context window
    available (idx >= context_k).
    """
    online.eval()
    n_nodes = graphs[0].x.shape[0]
    if eval_indices is None:
        eval_indices = list(range(context_k, len(graphs)))

    accum = torch.zeros(
        n_nodes,
        online.out_dim if hasattr(online, "out_dim") else _infer_out_dim(online, graphs[0], device),
        device=device,
    )
    count = 0
    with torch.no_grad():
        for t_idx in eval_indices:
            g = graphs[t_idx].to(device)
            z = online(g)  # [N, D]
            # encoder may already L2-normalize; we average raw embeddings.
            accum = accum + z
            count += 1
    accum = accum / max(count, 1)
    # final L2 normalize so cosine-style geometry is preserved
    accum = F.normalize(accum, dim=-1)
    return accum.cpu().numpy()


def _infer_out_dim(online, sample_graph, device) -> int:
    g = sample_graph.to(device)
    with torch.no_grad():
        z = online(g)
    return z.shape[-1]


def _kmeans_clustering(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    """k-means clustering with deterministic init via seed. returns
    cluster assignments [N]. uses sklearn if available; falls back to
    a simple numpy implementation if not."""
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        return km.fit_predict(embeddings)
    except ImportError:
        return _kmeans_numpy(embeddings, k, seed)


def _kmeans_numpy(X: np.ndarray, k: int, seed: int, n_iter: int = 100) -> np.ndarray:
    """fallback k-means for environments without sklearn."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=-1)
        labels = d.argmin(axis=1)
        new_centers = np.zeros_like(centers)
        for c in range(k):
            mask = labels == c
            if mask.any():
                new_centers[c] = X[mask].mean(axis=0)
            else:
                new_centers[c] = X[rng.integers(0, n)]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def _adjusted_rand_index(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(labels_true, labels_pred))
    except ImportError:
        return _ari_numpy(labels_true, labels_pred)


def _normalized_mutual_info(labels_true, labels_pred) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score

        return float(normalized_mutual_info_score(labels_true, labels_pred))
    except ImportError:
        return _nmi_numpy(labels_true, labels_pred)


def _ari_numpy(t, p) -> float:
    t = np.asarray(t)
    p = np.asarray(p)
    n = len(t)
    contingency: dict = {}
    for a, b in zip(t, p, strict=False):
        contingency[(a, b)] = contingency.get((a, b), 0) + 1
    a_marg: dict = {}
    b_marg: dict = {}
    for (a, b), v in contingency.items():
        a_marg[a] = a_marg.get(a, 0) + v
        b_marg[b] = b_marg.get(b, 0) + v

    def comb2(x):
        return x * (x - 1) // 2

    sum_comb = sum(comb2(v) for v in contingency.values())
    sum_a = sum(comb2(v) for v in a_marg.values())
    sum_b = sum(comb2(v) for v in b_marg.values())
    expected = sum_a * sum_b / max(comb2(n), 1)
    max_index = (sum_a + sum_b) / 2
    if max_index == expected:
        return 0.0
    return float((sum_comb - expected) / (max_index - expected))


def _nmi_numpy(t, p) -> float:
    t = np.asarray(t)
    p = np.asarray(p)
    n = len(t)

    def entropy(x):
        _, c = np.unique(x, return_counts=True)
        pr = c / n
        return float(-np.sum(pr * np.log(pr + 1e-12)))

    h_t = entropy(t)
    h_p = entropy(p)
    pairs: dict = {}
    for a, b in zip(t, p, strict=False):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), v in pairs.items():
        pa = (t == a).sum() / n
        pb = (p == b).sum() / n
        pab = v / n
        mi += pab * np.log(pab / (pa * pb) + 1e-12)
    if h_t + h_p == 0:
        return 0.0
    return float(2 * mi / (h_t + h_p))


def _cluster_purity(labels_true, labels_pred) -> float:
    """fraction of nodes in their majority-region cluster.
    purity ~ accuracy under best label permutation."""
    pred = np.asarray(labels_pred)
    truth = np.asarray(labels_true)
    n = len(pred)
    total = 0
    for c in np.unique(pred):
        mask = pred == c
        if mask.any():
            most_common = Counter(truth[mask]).most_common(1)[0][1]
            total += most_common
    return float(total / n)


def run_bloc_discovery(
    online,
    graphs,
    cfg,
    iso3_codes: list[str],
    seed: int = 0,
    device=None,
    partition: str = "subregion",
    k_override: int | None = None,
) -> dict:
    """run bloc discovery for one model. returns embeddings, cluster labels,
    and scores against ground-truth partition.

    args:
        partition: "subregion" (UN M49 subregions, 17 categories) or
                   "continent" (5 buckets — coarser, more recoverable).
        k_override: override the cluster count. defaults to # distinct labels
                    in the ground-truth partition.
    """
    from src.eval.iso3_regions import label_iso3_list, label_iso3_list_continent

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if partition == "continent":
        region_labels, missing = label_iso3_list_continent(iso3_codes)
    else:
        region_labels, missing = label_iso3_list(iso3_codes)
    region_to_int = {r: i for i, r in enumerate(sorted(set(region_labels)))}
    labels_true = np.array([region_to_int[r] for r in region_labels])

    k = k_override if k_override is not None else len(region_to_int)

    # extract embeddings averaged across test snapshots
    test_lo, test_hi = (
        cfg.data.test_weeks
        if hasattr(cfg.data, "test_weeks") and cfg.data.test_weeks is not None
        else (None, None)
    )
    if test_lo is None:
        with open(f"/app/{cfg.data.meta_path}") as f:
            meta = json.load(f)
        test_lo, test_hi = meta["test_range"]
    eval_indices = list(range(test_lo, test_hi + 1))

    emb = _country_embeddings(
        online, graphs, cfg.training.context_k, device, eval_indices=eval_indices
    )

    # k-means with deterministic seed
    labels_pred = _kmeans_clustering(emb, k=k, seed=seed)

    return {
        "n_clusters": int(k),
        "n_countries": int(len(iso3_codes)),
        "n_unmapped": int(len(missing)),
        "partition": partition,
        "ari": _adjusted_rand_index(labels_true, labels_pred),
        "nmi": _normalized_mutual_info(labels_true, labels_pred),
        "purity": _cluster_purity(labels_true, labels_pred),
        "embeddings": emb.tolist(),  # for downstream visualization
        "iso3": list(iso3_codes),
        "region_labels": region_labels,
        "cluster_assignments": labels_pred.tolist(),
        "test_snapshots_used": eval_indices,
    }
