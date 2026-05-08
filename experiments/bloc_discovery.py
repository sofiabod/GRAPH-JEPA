"""bloc-discovery experiment: does graph-jepa's frozen encoder recover known
economic blocs without supervision?

usage:
    python experiments/bloc_discovery.py \\
        --checkpoint results/baci_gravity/seed0/checkpoint.pt \\
        --config configs/baci_gravity.yaml \\
        --condition graph

protocol:
    1. load encoder checkpoint (graph or sequential)
    2. embed every country at every test snapshot, mean-pool across time
    3. k-means cluster (k=10)
    4. compare clusters to known economic-bloc labels (src/eval/blocs.py)
    5. report ari, nmi, purity vs the bloc taxonomy
    6. baseline: same metrics on raw node features (no learned encoder)

a "wow" result: graph-jepa clusters recover blocs (high ARI/NMI/purity);
sequential-ablation does not. demonstrates the model learned graph-aware
structure consistent with the geopolitical organization of trade.

requires: scikit-learn, numpy, omegaconf, torch, torch_geometric (for the
encoder forward; data is loaded from cfg.data.graphs_path).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.eval.blocs import bloc_distribution, labels_for_isos


def _purity(cluster_labels: np.ndarray, true_labels: list[str]) -> float:
    """fraction of points in their cluster's most common true label."""
    n = len(cluster_labels)
    correct = 0
    for c in set(cluster_labels.tolist()):
        cluster_true = [true_labels[i] for i in range(n) if cluster_labels[i] == c]
        if cluster_true:
            correct += Counter(cluster_true).most_common(1)[0][1]
    return correct / n if n > 0 else 0.0


def _build_encoder(condition: str, cfg, device: torch.device):
    if condition == "graph":
        from src.builders import build_graph_encoder
        return build_graph_encoder(cfg.encoder).to(device)
    elif condition == "sequential":
        from src.models.sequential_encoder import SequentialMLP
        return SequentialMLP(
            in_dim=cfg.encoder.in_dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        ).to(device)
    else:
        raise ValueError(f"unknown condition {condition!r} (expected 'graph' or 'sequential')")


def embed_test_mean(encoder: torch.nn.Module, graphs, test_range: tuple[int, int],
                    device: torch.device) -> np.ndarray:
    """encode every test snapshot, return mean-per-country embedding [N, D]."""
    encoder.eval()
    stack = []
    with torch.no_grad():
        for t in range(test_range[0], test_range[1] + 1):
            g = graphs[t].to(device)
            z = encoder(g)  # [N, D] l2-normalized for both graph and sequential
            stack.append(z.cpu().numpy())
    arr = np.stack(stack, axis=0)  # [T, N, D]
    return arr.mean(axis=0)        # [N, D]


def cluster_and_score(Z: np.ndarray, true_labels: list[str], k: int, seed: int = 0) -> dict:
    """k-means + ari/nmi/purity vs true labels."""
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    cl = km.fit_predict(Z)
    return {
        "ari": float(adjusted_rand_score(true_labels, cl)),
        "nmi": float(normalized_mutual_info_score(true_labels, cl)),
        "purity": float(_purity(cl, true_labels)),
        "cluster_labels": cl.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to encoder checkpoint .pt")
    ap.add_argument("--config", required=True, help="path to dataset yaml (must have iso_sorted in meta)")
    ap.add_argument("--condition", required=True, choices=["graph", "sequential"])
    ap.add_argument("--out-dir", default="results/bloc_discovery", help="output directory")
    ap.add_argument("--k", type=int, default=10, help="number of k-means clusters")
    ap.add_argument("--seed", type=int, default=0, help="seed for k-means")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load graphs + meta
    graphs = torch.load(cfg.data.graphs_path, map_location="cpu", weights_only=False)
    meta_path = (
        cfg.data.meta_path if hasattr(cfg.data, "meta_path") and cfg.data.meta_path
        else str(cfg.data.graphs_path).replace("_graphs.pt", "_meta.json")
    )
    meta = json.load(open(meta_path))
    iso_sorted = meta.get("iso_sorted")
    if iso_sorted is None:
        raise ValueError(
            f"meta {meta_path} has no 'iso_sorted' field. bloc discovery needs ISO3 codes "
            f"per country. only datasets with named countries (e.g. baci_gravity, icio) work here."
        )
    test_range = tuple(meta["test_range"])

    # load encoder
    encoder = _build_encoder(args.condition, cfg, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "online" in state:
        encoder.load_state_dict(state["online"])
    elif "model" in state:
        encoder.load_state_dict(state["model"])
    else:
        encoder.load_state_dict(state)

    # embed countries (mean across test snapshots)
    Z = embed_test_mean(encoder, graphs, test_range, device)
    assert Z.shape[0] == len(iso_sorted), f"shape mismatch: Z={Z.shape}, iso={len(iso_sorted)}"
    print(f"[{args.condition}] embeddings: {Z.shape}")

    # bloc labels
    true_labels = labels_for_isos(iso_sorted)
    bloc_dist = bloc_distribution(iso_sorted)
    print(f"bloc distribution in this dataset:")
    for b, c in sorted(bloc_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {b}: {c}")

    # learned-encoder clustering
    enc_result = cluster_and_score(Z, true_labels, args.k, args.seed)
    print(f"\n[{args.condition}] k-means(k={args.k}) on encoder features:")
    print(f"  ARI={enc_result['ari']:.4f}  NMI={enc_result['nmi']:.4f}  purity={enc_result['purity']:.4f}")

    # raw-feature baseline (mean across test snapshots of raw x)
    raw_stack = np.stack([graphs[t].x.numpy() for t in range(test_range[0], test_range[1] + 1)], axis=0)
    raw_mean = raw_stack.mean(axis=0)  # [N, F]
    raw_result = cluster_and_score(raw_mean, true_labels, args.k, args.seed)
    print(f"\nraw-feature baseline k-means(k={args.k}):")
    print(f"  ARI={raw_result['ari']:.4f}  NMI={raw_result['nmi']:.4f}  purity={raw_result['purity']:.4f}")

    # save artifacts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "condition": args.condition,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "n_countries": int(Z.shape[0]),
        "embedding_dim": int(Z.shape[1]),
        "n_clusters": args.k,
        "kmeans_seed": args.seed,
        "encoder_metrics": {
            "ari": enc_result["ari"],
            "nmi": enc_result["nmi"],
            "purity": enc_result["purity"],
        },
        "raw_feature_metrics": {
            "ari": raw_result["ari"],
            "nmi": raw_result["nmi"],
            "purity": raw_result["purity"],
        },
        "bloc_distribution": bloc_dist,
    }
    summary_path = out_dir / f"{args.condition}_bloc_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {summary_path}")

    # save embeddings + cluster labels for downstream viz
    np.savez(
        out_dir / f"{args.condition}_embeddings.npz",
        Z=Z,
        raw_mean=raw_mean,
        iso_sorted=np.array(iso_sorted),
        true_labels=np.array(true_labels),
        encoder_clusters=np.array(enc_result["cluster_labels"]),
        raw_clusters=np.array(raw_result["cluster_labels"]),
    )
    print(f"saved {out_dir / f'{args.condition}_embeddings.npz'}")


if __name__ == "__main__":
    main()
