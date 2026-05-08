"""local-only aggregation of anomaly trajectory results.

reads results/<dataset>/seed{N}/anomaly_trajectory.json (already saved by the
modal anomaly entrypoint) and produces:
  - results/<dataset>/ANOMALY_TRAJECTORY.md
  - results/<dataset>/anomaly_summary.json

usage:
    python scripts/aggregate_anomaly.py --dataset baci_gravity
    python scripts/aggregate_anomaly.py --dataset baci_gravity tgbn_trade icio
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

KNOWN_SHOCKS_BY_DATASET = {
    "baci_gravity": {
        1997: "Asian Financial Crisis",
        1998: "Asian Financial Crisis (continued)",
        2001: "Dot-com / 9-11",
        2008: "Global Financial Crisis",
        2009: "GFC trough",
        2014: "Oil price collapse",
        2015: "Oil collapse / Russia sanctions",
        2018: "US-China trade war",
        2020: "COVID-19",
    },
    "tgbn_trade": {
        2008: "Global Financial Crisis",
        2009: "GFC trough",
        2014: "Oil price collapse",
        2018: "US-China trade war",
        2020: "COVID-19",
    },
    "icio": {
        2008: "Global Financial Crisis",
        2009: "GFC trough",
        2020: "COVID-19",
    },
    "enron": {
        "2001-W08": "Q4 2000 results inflate revenues",
        "2001-W32": "Skilling resigns as CEO (Aug 14)",
        "2001-W41": "Q3 earnings restatement window",
        "2001-W42": "SEC inquiry announced (Oct 22), Fastow fired (Oct 24)",
        "2001-W45": "$586M write-down (Nov 8)",
        "2001-W48": "bankruptcy filing (Dec 2)",
    },
}


def aggregate(dataset: str, results_root: Path = Path("results")) -> None:
    ddir = results_root / dataset
    if not ddir.is_dir():
        print(f"warning: {ddir} not found, skipping")
        return
    shocks = KNOWN_SHOCKS_BY_DATASET.get(dataset, {})
    seed_dirs = sorted(p for p in ddir.glob("seed*") if (p / "anomaly_trajectory.json").exists())
    if not seed_dirs:
        print(f"warning: no anomaly_trajectory.json in {ddir}/seed*, skipping")
        return

    cond_to_year_to_devs: dict = {"graph": {}, "sequential": {}}
    year_to_idx: dict = {}
    seeds_used: list = []
    for sd in seed_dirs:
        seed = int(sd.name.replace("seed", ""))
        with open(sd / "anomaly_trajectory.json") as f:
            data = json.load(f)
        seeds_used.append(seed)
        for cond in ("graph", "sequential"):
            for snap in data[cond]["per_snapshot"]:
                y = snap["year"]
                cond_to_year_to_devs[cond].setdefault(y, []).append(snap["deviation"])
                year_to_idx[y] = snap["snapshot_idx"]

    years_sorted = sorted(year_to_idx.keys())
    rows: list = []
    lines = [
        f"# anomaly trajectory: {dataset}",
        "",
        f"per-snapshot prediction deviation (1 - mean_pred_cos), median across seeds {seeds_used}.",
        "shock years are flagged with ★. graph-vs-sequential gap on shock years tells us "
        "whether graph-JEPA picks up real-world dynamics more cleanly than the non-graph baseline.",
        "",
        "| year | snap idx | graph dev (median) | seq dev (median) | shock? | label |",
        "|---|---:|---:|---:|:-:|---|",
    ]
    for y in years_sorted:
        g_devs = cond_to_year_to_devs["graph"].get(y, [])
        s_devs = cond_to_year_to_devs["sequential"].get(y, [])
        if not g_devs or not s_devs:
            continue
        g_med = statistics.median(g_devs)
        s_med = statistics.median(s_devs)
        is_shock = y in shocks
        label = shocks.get(y, "")
        marker = "★" if is_shock else ""
        lines.append(
            f"| {y} | {year_to_idx[y]} | {g_med:.4f} | {s_med:.4f} | {marker} | {label} |"
        )
        rows.append({"year": y, "graph_dev": g_med, "sequential_dev": s_med, "shock": is_shock})

    lines.append("")
    lines.append("## shock detection score")
    lines.append("")
    for cond in ("graph", "sequential"):
        key = f"{cond}_dev"
        non_shock = [r[key] for r in rows if not r["shock"]]
        shock = [r[key] for r in rows if r["shock"]]
        if non_shock and shock:
            base = statistics.median(non_shock)
            shock_med = statistics.median(shock)
            lift = shock_med - base
            ratio = (shock_med / base) if base > 0 else float("inf")
            lines.append(
                f"- **{cond}**: non-shock median dev = {base:.4f}, "
                f"shock median dev = {shock_med:.4f}, "
                f"lift = {lift:+.4f} ({ratio:.2f}× baseline)"
            )

    md_path = ddir / "ANOMALY_TRAJECTORY.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {md_path}")

    summary = {
        "dataset": dataset,
        "seeds": seeds_used,
        "per_year": rows,
        "shocks_known": shocks,
    }
    sj_path = ddir / "anomaly_summary.json"
    with open(sj_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {sj_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True,
                    help="dataset name(s) under results/")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    root = Path(args.results_dir)
    for d in args.dataset:
        aggregate(d, root)


if __name__ == "__main__":
    main()
