"""aggregate per-seed JSON results into FINDINGS.md and per-experiment summary.json.

reads results/<dataset>/seed{0..N}/*.json and produces:
  - results/<dataset>/summary.json: mean ± 95% CI per metric, paired wilcoxon, win rate
  - results/FINDINGS.md: human-readable cross-dataset summary table

usage:
    python scripts/aggregate.py --datasets tgbn_trade,metrla --out results/FINDINGS.md
    python scripts/aggregate.py --all  # aggregate every dataset under results/

this is the canonical "every number is reproducible from one command" entry point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_seed_results(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    """load per-seed json files. returns {seed: {group_name: result_dict}}.

    eval1_<dataset>.json contains top-level keys like eval1_node_prediction,
    eval3_multistep_rollout, eval6_representation_quality — those get merged
    into the seed dict (keyed by their inner names). only the graph-jepa
    eval1 file is merged; sequential-ablation results are kept under a
    separate 'sequential' key so callers can compare conditions.
    eval2.json (paired wilcoxon) is kept as a group under 'eval2'.
    eval2_<mode>.json (shared-target variants) kept under their stem.
    """
    seeds = {}
    dataset_name = dataset_dir.name
    for seed_dir in sorted(dataset_dir.glob("seed*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.replace("seed", ""))
        except ValueError:
            continue
        seed_data: dict[str, Any] = {}
        for eval_file in seed_dir.glob("*.json"):
            with open(eval_file) as f:
                data = json.load(f)
            stem = eval_file.stem
            if stem == f"eval1_{dataset_name}":
                # graph-jepa eval1: merge nested groups (eval1_node_prediction etc) at top
                for k, v in data.items():
                    if isinstance(v, dict):
                        seed_data[k] = v
            elif stem == "eval1_sequential-ablation":
                seed_data["sequential"] = data
            elif stem == "eval2":
                seed_data["eval2"] = data
            elif stem.startswith("eval2_"):
                seed_data[stem] = data
        if seed_data:
            seeds[seed] = seed_data
    return seeds


def _ci95(values: list[float]) -> tuple[float, float]:
    """mean and 95% CI half-width from a sample using normal approx."""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr.mean()) if len(arr) else float("nan"), 0.0
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean, 1.96 * sem


def aggregate_dataset(dataset_dir: Path) -> dict[str, Any]:
    """produce summary stats for a single dataset."""
    seed_results = _load_seed_results(dataset_dir)
    if not seed_results:
        return {"dataset": dataset_dir.name,
                "error": f"no seed results found in {dataset_dir}"}

    seeds = sorted(seed_results.keys())
    summary: dict[str, Any] = {
        "dataset": dataset_dir.name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "metrics": {},
    }

    metric_paths = [
        ("eval1_node_prediction", "mean_pred_cos"),
        ("eval1_node_prediction", "mean_copy_cos"),
        ("eval1_node_prediction", "mean_graph_avg_cos"),
        ("eval6_representation_quality", "effective_rank"),
        ("eval6_representation_quality", "mean_pairwise_cosine"),
    ]

    for eval_name, metric in metric_paths:
        values = []
        for seed in seeds:
            val = seed_results[seed].get(eval_name, {}).get(metric)
            if val is not None:
                values.append(val)
        if values:
            mean, ci = _ci95(values)
            summary["metrics"][f"{eval_name}.{metric}"] = {
                "mean": mean,
                "ci95_halfwidth": ci,
                "per_seed": values,
            }

    # eval2 paired wilcoxon: collect per-seed, take min p, mean win rate
    e2_pvals, e2_winrates = [], []
    for seed in seeds:
        e2 = seed_results[seed].get("eval2", {})
        if "wilcoxon_p" in e2:
            e2_pvals.append(e2["wilcoxon_p"])
        if "win_rate" in e2:
            e2_winrates.append(e2["win_rate"])
    if e2_pvals:
        summary["eval2_paired"] = {
            "min_p": min(e2_pvals),
            "max_p": max(e2_pvals),
            "per_seed_p": e2_pvals,
        }
    if e2_winrates:
        mean, ci = _ci95(e2_winrates)
        summary["eval2_paired"] = summary.get("eval2_paired", {})
        summary["eval2_paired"]["win_rate_mean"] = mean
        summary["eval2_paired"]["win_rate_ci95"] = ci

    return summary


def write_findings_md(summaries: list[dict[str, Any]], out_path: Path):
    """write a human-readable markdown summary across all datasets."""
    lines = ["# FINDINGS", "", "Auto-generated by `scripts/aggregate.py`. Do not hand-edit.", ""]
    for s in summaries:
        if "error" in s:
            lines.append(f"## {s.get('dataset', 'unknown')}: {s['error']}")
            continue
        lines.append(f"## {s['dataset']} ({s['n_seeds']} seeds)")
        lines.append("")
        lines.append("| Metric | Mean ± 95% CI |")
        lines.append("|---|---|")
        for name, m in s.get("metrics", {}).items():
            lines.append(f"| {name} | {m['mean']:.4f} ± {m['ci95_halfwidth']:.4f} |")
        if "eval2_paired" in s:
            e2 = s["eval2_paired"]
            lines.append("")
            lines.append("**Paired Graph-JEPA vs Sequential-JEPA:**")
            if "min_p" in e2:
                lines.append(f"- min Wilcoxon p across seeds: {e2['min_p']:.3e}")
                lines.append(f"- max Wilcoxon p across seeds: {e2['max_p']:.3e}")
            if "win_rate_mean" in e2:
                lines.append(f"- win rate: {e2['win_rate_mean']:.3f} ± {e2['win_rate_ci95']:.3f}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results", help="root results directory")
    ap.add_argument("--datasets", help="comma-separated dataset names; overrides --all")
    ap.add_argument("--all", action="store_true", help="aggregate every dataset under --results-dir")
    ap.add_argument("--out", default="results/FINDINGS.md", help="output markdown path")
    args = ap.parse_args()

    results_root = Path(args.results_dir)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    elif args.all:
        # only consider dirs that look like datasets: must contain at least one seed* subdir,
        # and must not be a stray seed dir at the root or a private mirror like _modal_volume
        datasets = []
        for p in sorted(results_root.iterdir()):
            if not p.is_dir():
                continue
            if p.name.startswith("_") or p.name.startswith("seed"):
                continue
            if not any(child.is_dir() and child.name.startswith("seed")
                       for child in p.iterdir()):
                continue
            datasets.append(p.name)
    else:
        ap.error("specify --datasets or --all")

    summaries = []
    for d in datasets:
        ddir = results_root / d
        if not ddir.is_dir():
            print(f"warning: {ddir} not found, skipping")
            continue
        s = aggregate_dataset(ddir)
        summaries.append(s)
        # also write per-dataset summary.json
        with open(ddir / "summary.json", "w") as f:
            json.dump(s, f, indent=2)
        print(f"wrote {ddir / 'summary.json'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_findings_md(summaries, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
