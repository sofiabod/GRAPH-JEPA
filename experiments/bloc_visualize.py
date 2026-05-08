"""produce a 2D UMAP figure of country embeddings colored by economic bloc.

reads the .npz output of bloc_discovery.py and makes a side-by-side comparison
plot of graph-jepa vs sequential-ablation embeddings projected to 2D, with
points colored by ground-truth bloc membership.

usage:
    python experiments/bloc_visualize.py \\
        --graph-npz results/bloc_discovery/graph_embeddings.npz \\
        --seq-npz results/bloc_discovery/sequential_embeddings.npz \\
        --out figures/bloc_recovery.pdf

if graph-jepa's projection visibly separates EU/USMCA/ASEAN/etc. and sequential's
doesn't, that's the figure for the "what did the model learn" section of the paper.

requires: umap-learn (pip install umap-learn), matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# fixed palette so figures are reproducible
BLOC_COLORS = {
    "EU":          "#1f77b4",
    "USMCA":       "#ff7f0e",
    "ASEAN":       "#2ca02c",
    "GCC":         "#d62728",
    "EAEU":        "#9467bd",
    "MERCOSUR":    "#8c564b",
    "ECOWAS":      "#e377c2",
    "SADC":        "#7f7f7f",
    "SAARC":       "#bcbd22",
    "EFTA_OTHER":  "#17becf",
    "NONE":        "#cccccc",
}


def _load(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "Z": d["Z"],
        "iso": d["iso_sorted"].tolist(),
        "labels": d["true_labels"].tolist(),
    }


def _project_2d(Z: np.ndarray, seed: int = 0) -> np.ndarray:
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(Z)
    except ImportError:
        # fallback to t-SNE if umap not installed
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, perplexity=15).fit_transform(Z)


def _plot_panel(ax, X2: np.ndarray, labels: list[str], title: str):
    for bloc, color in BLOC_COLORS.items():
        mask = [i for i, lbl in enumerate(labels) if lbl == bloc]
        if not mask:
            continue
        ax.scatter(
            X2[mask, 0], X2[mask, 1],
            c=color,
            label=bloc if bloc != "NONE" else None,
            s=20 if bloc != "NONE" else 6,
            alpha=0.85 if bloc != "NONE" else 0.3,
            edgecolors="black" if bloc != "NONE" else "none",
            linewidths=0.3,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-npz", required=True)
    ap.add_argument("--seq-npz", required=True)
    ap.add_argument("--out", default="figures/bloc_recovery.pdf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    g = _load(args.graph_npz)
    s = _load(args.seq_npz)
    assert g["iso"] == s["iso"], "country sets differ between graph and sequential"
    labels = g["labels"]

    print(f"projecting {g['Z'].shape} → 2D for graph...")
    g2 = _project_2d(g["Z"], seed=args.seed)
    print(f"projecting {s['Z'].shape} → 2D for sequential...")
    s2 = _project_2d(s["Z"], seed=args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    _plot_panel(axes[0], g2, labels, "Graph-JEPA encoder embeddings")
    _plot_panel(axes[1], s2, labels, "Sequential-JEPA encoder embeddings")

    # legend on the right
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="center right", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
