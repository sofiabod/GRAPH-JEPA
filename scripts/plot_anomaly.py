"""produce the headline anomaly-trajectory figure(s).

reads results/<dataset>/seed{N}/anomaly_trajectory.json (already saved by
modal `anomaly` runs) and produces a clean PDF + PNG showing:
  - per-year prediction deviation (1 - mean_pred_cos), graph and sequential
  - median across seeds + shaded ±IQR band
  - shock years marked with vertical shaded bands
  - test-split background tinted

usage:
    python scripts/plot_anomaly.py --dataset baci_gravity
    python scripts/plot_anomaly.py --dataset baci_gravity dblp     # multi-panel
    python scripts/plot_anomaly.py --dataset baci_gravity --out figures/anomaly_baci
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


KNOWN_SHOCKS = {
    "baci_gravity": {
        1997: "Asian crisis",
        1998: "Asian crisis",
        2001: "Dot-com / 9-11",
        2008: "GFC",
        2009: "GFC",
        2014: "Oil collapse",
        2015: "Oil / Russia",
        2018: "Trade war",
        2020: "COVID-19",
    },
    "tgbn_trade": {
        2008: "GFC", 2009: "GFC",
        2014: "Oil collapse",
        2018: "Trade war",
    },
    "dblp": {
        2008: "GFC",
        2020: "COVID-19",
    },
    "icio": {
        2008: "GFC", 2009: "GFC",
    },
}


def load_dataset(dataset: str, results_root: Path):
    """returns dict: {year: {graph: [devs across seeds], sequential: [devs]}}"""
    ddir = results_root / dataset
    seed_files = sorted(ddir.glob("seed*/anomaly_trajectory.json"))
    if not seed_files:
        raise FileNotFoundError(f"no anomaly_trajectory.json under {ddir}")
    by_year: dict = defaultdict(lambda: {"graph": [], "sequential": []})
    test_range = None
    for sf in seed_files:
        with open(sf) as f:
            data = json.load(f)
        for cond in ("graph", "sequential"):
            for snap in data[cond]["per_snapshot"]:
                y = snap["year"]
                by_year[y][cond].append(snap["deviation"])
    # try to extract test range from meta
    meta_candidates = [
        Path("data") / f"{dataset}_meta.json",
    ]
    for mp in meta_candidates:
        if mp.exists():
            with open(mp) as f:
                meta = json.load(f)
            tr = meta.get("test_range")
            yk = meta.get("years_kept")
            if tr and yk:
                test_range = (yk[tr[0]], yk[tr[1]])
            break
    return by_year, test_range


def plot_one(ax, dataset: str, by_year: dict, test_range, shocks: dict):
    years = sorted(by_year.keys())
    g_med = np.array([statistics.median(by_year[y]["graph"]) for y in years])
    s_med = np.array([statistics.median(by_year[y]["sequential"]) for y in years])
    # IQR shading (25-75 percentile across seeds)
    g_lo = np.array([np.quantile(by_year[y]["graph"], 0.25) for y in years])
    g_hi = np.array([np.quantile(by_year[y]["graph"], 0.75) for y in years])
    s_lo = np.array([np.quantile(by_year[y]["sequential"], 0.25) for y in years])
    s_hi = np.array([np.quantile(by_year[y]["sequential"], 0.75) for y in years])

    # shock bands first (background)
    for y, label in shocks.items():
        if y in years:
            ax.axvspan(y - 0.5, y + 0.5, alpha=0.18, color="#d62728", linewidth=0)
    # test-split tint
    if test_range is not None:
        ax.axvspan(test_range[0] - 0.5, test_range[1] + 0.5,
                   alpha=0.06, color="#1f77b4", linewidth=0,
                   label="test split (held out)")

    # lines
    ax.plot(years, g_med, color="#1f77b4", linewidth=2.0, label="Graph-JEPA")
    ax.fill_between(years, g_lo, g_hi, color="#1f77b4", alpha=0.18)
    ax.plot(years, s_med, color="#ff7f0e", linewidth=2.0, linestyle="--",
            label="Sequential ablation")
    ax.fill_between(years, s_lo, s_hi, color="#ff7f0e", alpha=0.15)

    # shock labels at top
    ax.set_ylim(0, max(g_hi.max(), s_hi.max()) * 1.15)
    y_top = ax.get_ylim()[1]
    seen_x: list = []
    for y, label in shocks.items():
        if y not in years:
            continue
        # collision-avoid: stagger labels vertically
        rotation = 0
        offset = 0.02 * y_top
        for px in seen_x:
            if abs(y - px) < 1.5:
                offset += 0.04 * y_top
        seen_x.append(y)
        ax.text(y, y_top - offset, label, ha="center", va="top",
                fontsize=8, color="#a02020", rotation=rotation)

    ax.set_xlabel("year")
    ax.set_ylabel("prediction deviation  (1 − mean cosine)")
    ax.set_title(f"{dataset}", loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True,
                    help="one or more dataset names (panels stacked vertically)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None,
                    help="output prefix (default: figures/anomaly_<datasets>)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib")
        return

    results_root = Path(args.results_dir)
    n = len(args.dataset)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.4 * n + 0.3), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, args.dataset):
        try:
            by_year, test_range = load_dataset(ds, results_root)
        except FileNotFoundError as e:
            print(f"  skip {ds}: {e}")
            ax.text(0.5, 0.5, f"no data for {ds}", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        shocks = KNOWN_SHOCKS.get(ds, {})
        # only annual datasets — skip if year is non-int (weekly with date_str)
        sample_year = next(iter(by_year.keys()))
        if not isinstance(sample_year, int):
            print(f"  skip {ds}: weekly (non-int year), use --weekly variant")
            continue
        plot_one(ax, ds, by_year, test_range, shocks)

    fig.suptitle("Graph-JEPA prediction-error trajectory across known economic shocks",
                 fontsize=12, y=0.995)
    fig.tight_layout()

    if args.out:
        out_prefix = args.out
    else:
        Path("figures").mkdir(exist_ok=True)
        out_prefix = "figures/anomaly_" + "_".join(args.dataset)
    pdf_path = f"{out_prefix}.pdf"
    png_path = f"{out_prefix}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
