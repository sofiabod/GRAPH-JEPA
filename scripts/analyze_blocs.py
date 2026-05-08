"""rich post-hoc analysis of bloc discovery results.

reads results/<dataset>/seed{N}/bloc_discovery.json (saved by the modal
`blocs` entrypoint) and produces:
  - per-bloc purity table (graph vs sequential vs raw_features)
  - cluster × bloc confusion matrix (best seed)
  - UMAP/t-SNE visualization saved to PNG (if matplotlib + umap-learn or sklearn available)

usage:
    python scripts/analyze_blocs.py --dataset baci_gravity
    python scripts/analyze_blocs.py --dataset baci_gravity --seed 0  # confusion matrix uses this seed

note: aggregate ARI is misleading because bloc taxonomy is messy
(countries in multiple blocs, blocs that change over time). per-bloc purity
captures "did the model find EU/USMCA/ASEAN?" cleanly.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# major trade blocs — overlapping ground truth complementing the UN M49 partition.
# a country can be in multiple blocs (e.g., MEX is in USMCA AND OECD).
# per-bloc purity asks "what fraction of bloc X members are in the same cluster?"
TRADE_BLOCS: dict[str, list[str]] = {
    "EU27": ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
             "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
             "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"],
    "USMCA": ["USA", "CAN", "MEX"],
    "ASEAN": ["BRN", "KHM", "IDN", "LAO", "MYS", "MMR", "PHL", "SGP", "THA", "VNM"],
    "MERCOSUR": ["ARG", "BRA", "PRY", "URY", "VEN"],
    "GCC": ["SAU", "ARE", "KWT", "QAT", "BHR", "OMN"],
    "EAEU": ["RUS", "BLR", "KAZ", "ARM", "KGZ"],
    "BRICS": ["BRA", "RUS", "IND", "CHN", "ZAF"],
    "G7": ["USA", "CAN", "GBR", "FRA", "DEU", "ITA", "JPN"],
}


def per_bloc_purity(iso3: list[str], cluster_assignments: list[int]) -> dict:
    """for each known trade bloc, return:
       - n_members_present: how many bloc members are in the dataset
       - dominant_cluster: which cluster has the most bloc members
       - purity: fraction of bloc members in the dominant cluster
    a perfect bloc recovery has purity 1.0 (all bloc members in one cluster).
    """
    iso_to_cluster = dict(zip(iso3, cluster_assignments))
    out: dict = {}
    for bloc, members in TRADE_BLOCS.items():
        present = [m for m in members if m in iso_to_cluster]
        if not present:
            continue
        clusters = [iso_to_cluster[m] for m in present]
        c_count = Counter(clusters)
        dom_cluster, dom_count = c_count.most_common(1)[0]
        out[bloc] = {
            "n_members_in_data": len(present),
            "dominant_cluster": int(dom_cluster),
            "n_in_dominant": int(dom_count),
            "purity": dom_count / len(present),
            "members_present": present,
        }
    return out


def confusion_matrix_text(iso3, cluster_assignments, region_labels):
    """produce a (cluster × region) count matrix, sorted by region size."""
    clusters = sorted(set(cluster_assignments))
    regions = sorted(set(region_labels))
    cm = {c: Counter() for c in clusters}
    for c, r in zip(cluster_assignments, region_labels):
        cm[c][r] += 1
    region_totals = Counter(region_labels)
    region_order = [r for r, _ in region_totals.most_common()]

    header = "| cluster |" + "".join(f" {r[:18]} |" for r in region_order) + " | total |"
    sep = "|---|" + "---|" * len(region_order) + "---|"
    lines = [header, sep]
    for c in clusters:
        row_total = sum(cm[c].values())
        cells = " | ".join(str(cm[c].get(r, 0)) for r in region_order)
        lines.append(f"| {c} | {cells} | {row_total} |")
    totals = " | ".join(str(region_totals[r]) for r in region_order)
    lines.append(f"| total | {totals} | {sum(region_totals.values())} |")
    return "\n".join(lines)


def try_umap_2d(embeddings: np.ndarray, seed: int = 0) -> np.ndarray | None:
    """return 2D projection via UMAP if available, else PCA, else None."""
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15)
        return reducer.fit_transform(embeddings)
    except ImportError:
        pass
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    except ImportError:
        return None


def maybe_save_umap_plot(out_path: Path, condition: str,
                         embeddings: np.ndarray, region_labels: list[str],
                         iso3: list[str], seed: int = 0):
    """save 2D UMAP/PCA plot colored by region. requires matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  skipping {condition} plot — matplotlib not installed")
        return
    proj = try_umap_2d(embeddings, seed=seed)
    if proj is None:
        print(f"  skipping {condition} plot — no umap or sklearn for projection")
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    regions = sorted(set(region_labels))
    cmap = plt.get_cmap("tab20", len(regions))
    for i, r in enumerate(regions):
        mask = np.array([rl == r for rl in region_labels])
        ax.scatter(proj[mask, 0], proj[mask, 1], s=20, color=cmap(i), label=r, alpha=0.75)
    ax.set_title(f"bloc discovery: {condition} (seed {seed})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path}")


def analyze(dataset: str, results_root: Path, seed_for_cm: int):
    ddir = results_root / dataset
    seed_files = sorted(ddir.glob("seed*/bloc_discovery.json"))
    if not seed_files:
        print(f"no bloc_discovery.json files under {ddir}")
        return

    per_seed = []
    for sf in seed_files:
        seed = int(sf.parent.name.replace("seed", ""))
        with open(sf) as f:
            data = json.load(f)
        per_seed.append((seed, data))

    # aggregate ARI/NMI/purity
    cond_metrics: dict = {"graph": [], "sequential": [], "raw_features": []}
    for seed, data in per_seed:
        for cond in cond_metrics:
            cond_metrics[cond].append({
                "seed": seed,
                "ari": data[cond]["ari"],
                "nmi": data[cond]["nmi"],
                "purity": data[cond]["purity"],
            })

    # per-bloc purity per seed per condition
    per_bloc_per_cond: dict = {"graph": defaultdict(list),
                                "sequential": defaultdict(list),
                                "raw_features": defaultdict(list)}
    for _, data in per_seed:
        for cond in per_bloc_per_cond:
            iso3 = data[cond]["iso3"]
            ca = data[cond]["cluster_assignments"]
            pb = per_bloc_purity(iso3, ca)
            for bloc, info in pb.items():
                per_bloc_per_cond[cond][bloc].append(info["purity"])

    # write report
    lines = [f"# bloc discovery analysis: {dataset}", ""]

    lines.append("## aggregate metrics (median across seeds)")
    lines.append("")
    lines.append("| condition | ARI | NMI | purity |")
    lines.append("|---|---:|---:|---:|")
    for cond, values in cond_metrics.items():
        ari = statistics.median(v["ari"] for v in values)
        nmi = statistics.median(v["nmi"] for v in values)
        pur = statistics.median(v["purity"] for v in values)
        lines.append(f"| {cond} | {ari:.3f} | {nmi:.3f} | {pur:.3f} |")
    lines.append("")
    g_ari = statistics.median(v["ari"] for v in cond_metrics["graph"])
    s_ari = statistics.median(v["ari"] for v in cond_metrics["sequential"])
    r_ari = statistics.median(v["ari"] for v in cond_metrics["raw_features"])
    lines.append(f"**ARI gap (graph − sequential): {g_ari - s_ari:+.3f}** "
                 f"(headline metric: > +0.2 = clear bloc-discovery win)")
    lines.append(f"**ARI gap (graph − raw features): {g_ari - r_ari:+.3f}**")
    lines.append("")

    lines.append("## per-bloc purity (median across seeds)")
    lines.append("")
    lines.append("how many of each bloc's members end up in the same cluster?")
    lines.append("a value of 1.0 means the model recovered that bloc perfectly.")
    lines.append("")
    all_blocs = sorted({b for cond in per_bloc_per_cond.values() for b in cond.keys()})
    lines.append("| bloc | graph | sequential | raw features |")
    lines.append("|---|---:|---:|---:|")
    for bloc in all_blocs:
        g = per_bloc_per_cond["graph"].get(bloc, [])
        s = per_bloc_per_cond["sequential"].get(bloc, [])
        r = per_bloc_per_cond["raw_features"].get(bloc, [])
        gm = f"{statistics.median(g):.2f}" if g else "—"
        sm = f"{statistics.median(s):.2f}" if s else "—"
        rm = f"{statistics.median(r):.2f}" if r else "—"
        lines.append(f"| {bloc} | {gm} | {sm} | {rm} |")
    lines.append("")

    # confusion matrix for the requested seed
    seed_data = next((d for s, d in per_seed if s == seed_for_cm), None)
    if seed_data is None:
        seed_data = per_seed[0][1]
        seed_for_cm = per_seed[0][0]
    lines.append(f"## cluster × region confusion matrix (seed {seed_for_cm}, graph)")
    lines.append("")
    g = seed_data["graph"]
    lines.append(confusion_matrix_text(g["iso3"], g["cluster_assignments"], g["region_labels"]))
    lines.append("")

    # save UMAP plots for each condition (best seed)
    for cond in ("graph", "sequential", "raw_features"):
        emb = np.array(seed_data[cond]["embeddings"])
        regions = seed_data[cond]["region_labels"]
        iso3 = seed_data[cond]["iso3"]
        out_png = ddir / f"blocs_umap_{cond}_seed{seed_for_cm}.png"
        maybe_save_umap_plot(out_png, cond, emb, regions, iso3, seed=seed_for_cm)
    lines.append("## visualizations")
    lines.append("")
    for cond in ("graph", "sequential", "raw_features"):
        png_name = f"blocs_umap_{cond}_seed{seed_for_cm}.png"
        if (ddir / png_name).exists():
            lines.append(f"- ![{cond}]({png_name}) — UMAP/PCA projection colored by UN M49 subregion")
    lines.append("")

    md_path = ddir / "BLOCS_ANALYSIS.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name under results/")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed to use for the confusion matrix and visualization")
    args = ap.parse_args()
    analyze(args.dataset, Path(args.results_dir), args.seed)


if __name__ == "__main__":
    main()
