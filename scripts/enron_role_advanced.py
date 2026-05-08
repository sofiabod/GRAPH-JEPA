"""advanced enron role analysis: sub-cluster discovery + role-shuffle null.

reads paper_results/enron/seed{N}/role_recovery.json (which now contains
per-person embeddings) and runs:

E2: sub-cluster within each role
    for each role with >= 4 members, k-means k=2 to test whether the encoder
    found finer-grained structure than the hand-curated labels suggested.
    if govaffairs splits cleanly into California vs federal sub-clusters, the
    "graph anti-clusters govaffairs" finding flips into "graph discovered
    substructure my labels missed."

E3: role-shuffle null distribution
    permute role labels randomly N times, compute within-vs-between cosine gap
    each time, build a null distribution. compare to the actual observed gap
    for graph / sequential / raw_features. tells us whether the +0.87 admin
    signal could plausibly arise from random labeling (it should not).

usage:
    python scripts/enron_role_advanced.py
    python scripts/enron_role_advanced.py --seed 0
    python scripts/enron_role_advanced.py --n-shuffles 1000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_seed(results_root: Path, seed: int) -> dict:
    fp = results_root / "enron" / f"seed{seed}" / "role_recovery.json"
    if not fp.exists():
        raise FileNotFoundError(f"missing {fp}")
    with open(fp) as f:
        return json.load(f)


def _kmeans_k2(X: np.ndarray, seed: int) -> np.ndarray:
    """k=2 k-means via sklearn (or numpy fallback)."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, n_init=20, random_state=seed)
        return km.fit_predict(X)
    except ImportError:
        # simple 2-cluster fallback
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        c = X[rng.choice(n, size=2, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(50):
            d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=-1)
            new = d.argmin(axis=1)
            if np.array_equal(new, labels):
                break
            labels = new
            for k in (0, 1):
                if (labels == k).any():
                    c[k] = X[labels == k].mean(axis=0)
        return labels


def e2_subcluster_within_roles(data: dict, *, conditions=("graph", "sequential"),
                                min_role_size: int = 4) -> dict:
    """For each role with >= min_role_size members, run k=2 k-means within
    the role and report (a) sub-cluster sizes, (b) within-sub-cluster cosine,
    (c) which emails landed in which sub-cluster."""
    out: dict = {}
    for cond in conditions:
        d = data[cond]
        emb = np.asarray(d["embeddings"])
        emails = d["emails_in_order"]
        idx_to_role = {int(k): v for k, v in d["idx_to_role"].items()}

        # group node indices by role
        role_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, role in idx_to_role.items():
            role_to_indices[role].append(idx)

        cond_out: dict = {}
        for role, indices in role_to_indices.items():
            if len(indices) < min_role_size:
                continue
            X = emb[indices]
            sub_labels = _kmeans_k2(X, seed=int(d.get("seed", 0)))
            split_a = [emails[i] for i, sl in zip(indices, sub_labels) if sl == 0]
            split_b = [emails[i] for i, sl in zip(indices, sub_labels) if sl == 1]
            # within-sub-cluster cosine for each sub-cluster
            def _within(idx_set):
                if len(idx_set) < 2:
                    return float("nan")
                sub_X = emb[idx_set]
                sims = sub_X @ sub_X.T
                tri = sims[np.triu_indices(len(sub_X), k=1)]
                return float(np.median(tri))
            sub0_idx = [i for i, sl in zip(indices, sub_labels) if sl == 0]
            sub1_idx = [i for i, sl in zip(indices, sub_labels) if sl == 1]
            cond_out[role] = {
                "n_total": int(len(indices)),
                "sub_cluster_a": split_a,
                "sub_cluster_b": split_b,
                "within_a_median_cos": _within(sub0_idx),
                "within_b_median_cos": _within(sub1_idx),
            }
        out[cond] = cond_out
    return out


def e3_role_shuffle_null(data: dict, *, conditions=("graph", "sequential", "raw_features"),
                          n_shuffles: int = 1000, seed: int = 0) -> dict:
    """Permute role labels N times, compute within-vs-between median cosine gap each time.
    Compare actual observed gap to the null distribution."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    for cond in conditions:
        d = data[cond]
        emb = np.asarray(d["embeddings"])
        idx_to_role = {int(k): v for k, v in d["idx_to_role"].items()}
        labeled_idx = sorted(idx_to_role.keys())
        labels = np.array([idx_to_role[i] for i in labeled_idx])
        n_labeled = len(labeled_idx)

        # compute cosine matrix once
        cos_M = emb @ emb.T

        # actual gap
        within_actual, between_actual = [], []
        for ii in range(n_labeled):
            for jj in range(ii + 1, n_labeled):
                i, j = labeled_idx[ii], labeled_idx[jj]
                v = float(cos_M[i, j])
                if labels[ii] == labels[jj]:
                    within_actual.append(v)
                else:
                    between_actual.append(v)
        actual_gap = float(np.median(within_actual) - np.median(between_actual))

        # null distribution from shuffles
        shuffle_gaps = np.empty(n_shuffles)
        for s in range(n_shuffles):
            shuffled = labels.copy()
            rng.shuffle(shuffled)
            within_s, between_s = [], []
            for ii in range(n_labeled):
                for jj in range(ii + 1, n_labeled):
                    i, j = labeled_idx[ii], labeled_idx[jj]
                    v = float(cos_M[i, j])
                    if shuffled[ii] == shuffled[jj]:
                        within_s.append(v)
                    else:
                        between_s.append(v)
            shuffle_gaps[s] = float(np.median(within_s) - np.median(between_s))

        # one-sided p: how often does the shuffled gap exceed the actual gap?
        # if signal is real, almost never → p near 0
        # if signal is noise, ~50% → p near 0.5
        p_one_sided = float((shuffle_gaps >= actual_gap).mean())
        out[cond] = {
            "actual_gap": actual_gap,
            "null_mean": float(shuffle_gaps.mean()),
            "null_std": float(shuffle_gaps.std()),
            "null_p5": float(np.quantile(shuffle_gaps, 0.05)),
            "null_p95": float(np.quantile(shuffle_gaps, 0.95)),
            "shuffle_p_one_sided": p_one_sided,
            "n_shuffles": int(n_shuffles),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="paper_results")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed to use for sub-cluster + null analyses")
    ap.add_argument("--n-shuffles", type=int, default=1000,
                    help="number of label permutations for the null distribution")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    data = load_seed(results_root, args.seed)
    if "embeddings" not in data["graph"]:
        print("ERROR: role_recovery.json missing 'embeddings' field.")
        print("re-run `modal run experiments/eval_tgjepa.py::enron_roles ...` first")
        return

    print(f"loaded paper_results/enron/seed{args.seed}/role_recovery.json")
    print()
    print("=" * 70)
    print(f"E2: sub-cluster within roles (k-means k=2, min_role_size=4)")
    print("=" * 70)
    e2 = e2_subcluster_within_roles(data, conditions=("graph", "sequential"))
    for cond, role_results in e2.items():
        print(f"\n  ── {cond} ──")
        for role, r in role_results.items():
            print(f"  role={role}  n_total={r['n_total']}")
            print(f"    sub-cluster A (n={len(r['sub_cluster_a'])}): {r['sub_cluster_a']}")
            print(f"      within-cos: {r['within_a_median_cos']:.3f}")
            print(f"    sub-cluster B (n={len(r['sub_cluster_b'])}): {r['sub_cluster_b']}")
            print(f"      within-cos: {r['within_b_median_cos']:.3f}")

    print()
    print("=" * 70)
    print(f"E3: role-shuffle null distribution ({args.n_shuffles} shuffles)")
    print("=" * 70)
    e3 = e3_role_shuffle_null(
        data,
        conditions=("graph", "sequential", "raw_features"),
        n_shuffles=args.n_shuffles,
        seed=args.seed,
    )
    print(f"\n{'condition':<16} {'actual':<10} {'null mean':<12} {'null p95':<12} {'shuffle p':<12}")
    print("-" * 70)
    for cond, r in e3.items():
        print(f"{cond:<16} {r['actual_gap']:<10.3f} {r['null_mean']:<12.3f} "
              f"{r['null_p95']:<12.3f} {r['shuffle_p_one_sided']:<12.4f}")

    # save markdown report
    out_dir = results_root / "enron"
    md_lines = [
        "# Enron role-recovery — advanced analyses",
        "",
        f"Source: paper_results/enron/seed{args.seed}/role_recovery.json",
        f"Date: 2026-05-08",
        "",
        "## E2: sub-cluster discovery within each role (k-means k=2)",
        "",
        "If a role has internal substructure the encoder picked up on, k-means k=2",
        "should split it cleanly. The interesting case is govaffairs: it anti-clustered",
        "in the aggregate (gap −0.62). If sub-clusters split California-regulatory vs",
        "federal/general govaffairs people, the negative gap was actually the encoder",
        "discovering finer-grained structure than the hand-curated labels.",
        "",
    ]
    for cond, role_results in e2.items():
        md_lines.append(f"### {cond} encoder")
        md_lines.append("")
        for role, r in role_results.items():
            md_lines.append(f"**{role}** (n={r['n_total']})")
            md_lines.append("")
            md_lines.append(f"- Sub-cluster A (n={len(r['sub_cluster_a'])}, "
                            f"within-cos {r['within_a_median_cos']:.3f}): {r['sub_cluster_a']}")
            md_lines.append(f"- Sub-cluster B (n={len(r['sub_cluster_b'])}, "
                            f"within-cos {r['within_b_median_cos']:.3f}): {r['sub_cluster_b']}")
            md_lines.append("")

    md_lines.append("## E3: role-shuffle null control")
    md_lines.append("")
    md_lines.append(f"For each condition, role labels are randomly permuted "
                    f"{args.n_shuffles} times and the within-vs-between cosine gap recomputed. "
                    "The null distribution is the gap that would arise from random labeling. "
                    "If the actual observed gap is in the right tail (shuffle_p ≈ 0), the "
                    "real role labels carry signal beyond chance. If shuffle_p ≈ 0.5, the "
                    "signal could plausibly arise from random labels — i.e. an artifact.")
    md_lines.append("")
    md_lines.append("| condition | actual gap | null mean | null p95 | shuffle p (one-sided) |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for cond, r in e3.items():
        md_lines.append(
            f"| {cond} | {r['actual_gap']:+.3f} | {r['null_mean']:+.3f} | "
            f"{r['null_p95']:+.3f} | {r['shuffle_p_one_sided']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("**Reading**: `shuffle_p < 0.05` means the actual gap is in the top 5% "
                    "of the random-label null — i.e. the role labels are doing real work. "
                    "`shuffle_p ≈ 0.5` means the gap could be reproduced by random label "
                    "assignment — the signal is artifact.")

    out_md = out_dir / f"ROLE_ADVANCED_seed{args.seed}.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nsaved {out_md}")

    out_json = out_dir / f"role_advanced_seed{args.seed}.json"
    with open(out_json, "w") as f:
        json.dump({"e2_subclusters": e2, "e3_shuffle_null": e3,
                    "n_shuffles": args.n_shuffles, "seed": args.seed}, f, indent=2)
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
