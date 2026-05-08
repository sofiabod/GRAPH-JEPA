"""local-only analysis: per-person fraud-week anomaly on Enron, grouped by role.

reads paper_results/enron/seed{N}/per_person_cosines.json (saved by Modal entrypoint
`enron_per_person`) and tests:

    H1: people doing role-relevant crisis response (legal, govaffairs) show higher
        prediction-error anomaly during fraud weeks (W41/W42/W45/W48 = snap 100/101/104/107)
        than people doing routine work (admin) during the same weeks.

method:
    1. for each (seed, condition, person), compute the average prediction error
       (1 - mean cosine) across non-fraud snapshots in test split → personal baseline
    2. compute average error during fraud weeks → fraud average
    3. fraud anomaly per person = fraud_avg - baseline_avg
    4. group by role using src/eval/enron_roles.py
    5. compare role groups via Mann-Whitney U test

usage:
    python scripts/enron_fraud_anomaly.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, ".")
from src.eval.enron_roles import EMAIL_TO_ROLE


# fraud-event snapshot indices (per the new enron splits, which dropped the
# bogus 1980-W01 outlier — all snapshots shifted by one)
FRAUD_SNAPS = {
    100: "2001-W41 (Q3 restatement window)",
    101: "2001-W42 (SEC inquiry / Fastow fired)",
    104: "2001-W45 ($586M write-down)",
    107: "2001-W48 (bankruptcy filing)",
}

# config splits (from configs/enron.yaml)
TRAIN_RANGE = (0, 82)
VAL_RANGE = (83, 100)
TEST_RANGE = (101, 119)
CONTEXT_K = 4  # snap 0-3 not used as targets


def load_seed(results_root: Path, seed: int) -> dict:
    fp = results_root / "enron" / f"seed{seed}" / "per_person_cosines.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    with open(fp) as f:
        return json.load(f)


def mann_whitney_u_one_sided(a: np.ndarray, b: np.ndarray) -> float:
    """one-sided MWU p-value: H1 mean(a) > mean(b). normal-approx, no scipy."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="stable")
    ranks = np.empty_like(combined, dtype=np.float64)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[order[j + 1]] == combined[order[i]]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    R_a = ranks[:n_a].sum()
    U = R_a - n_a * (n_a + 1) / 2.0
    mean_U = n_a * n_b / 2.0
    var_U = n_a * n_b * (n_a + n_b + 1) / 12.0
    if var_U <= 0:
        return float("nan")
    z = (U - mean_U) / np.sqrt(var_U)
    from math import erf, sqrt
    return float(0.5 * (1.0 - erf(z / sqrt(2))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="paper_results")
    args = ap.parse_args()

    root = Path(args.results_dir)
    seeds_data = []
    for seed in (0, 1, 2, 3, 4):
        try:
            seeds_data.append((seed, load_seed(root, seed)))
        except FileNotFoundError:
            print(f"warn: missing seed {seed} per_person_cosines.json")
    if not seeds_data:
        print("no data — run the modal entrypoint first")
        return

    # build idx -> email mapping (same across seeds)
    person_index = seeds_data[0][1]["person_index"]
    idx_to_email = {idx: email for email, idx in person_index.items()}
    n_persons = len(person_index)

    # build idx -> role mapping
    idx_to_role = {}
    for idx, email in idx_to_email.items():
        role = EMAIL_TO_ROLE.get(email.strip().lower())
        idx_to_role[idx] = role  # may be None for unlabeled

    # for each condition, build per-person fraud_anomaly = fraud_avg - non_fraud_baseline
    print(f"\n=== Enron per-person fraud-week anomaly (E-P) ===")
    print(f"fraud snaps: {sorted(FRAUD_SNAPS.keys())}")
    print(f"test split (non-fraud test snaps): {TEST_RANGE[0]}–{TEST_RANGE[1]} excluding fraud snaps")
    print()

    role_results = {}
    for condition in ("graph", "sequential"):
        cosine_key = f"{condition}_per_person_cos"
        # collect fraud anomaly per person per seed
        # shape: [n_seeds, n_persons]
        fraud_anomaly = np.zeros((len(seeds_data), n_persons))
        for s_i, (seed, d) in enumerate(seeds_data):
            valid_indices = d["valid_snapshot_indices"]
            cos_per_snap = np.array(d[cosine_key])  # [n_valid_snaps, n_persons]

            # map snap_idx → row in cos_per_snap
            snap_to_row = {idx: i for i, idx in enumerate(valid_indices)}

            # personal non-fraud baseline: average dev across non-fraud test snaps
            non_fraud_test_snaps = [s for s in valid_indices
                                     if TEST_RANGE[0] <= s <= TEST_RANGE[1]
                                     and s not in FRAUD_SNAPS]
            fraud_test_snaps = [s for s in valid_indices if s in FRAUD_SNAPS]

            if not non_fraud_test_snaps or not fraud_test_snaps:
                continue

            baseline_dev = 1.0 - cos_per_snap[[snap_to_row[s] for s in non_fraud_test_snaps]].mean(axis=0)
            fraud_dev = 1.0 - cos_per_snap[[snap_to_row[s] for s in fraud_test_snaps]].mean(axis=0)
            fraud_anomaly[s_i] = fraud_dev - baseline_dev

        # average across seeds: per-person anomaly
        per_person_anomaly = fraud_anomaly.mean(axis=0)

        # group by role
        role_groups = defaultdict(list)
        for idx in range(n_persons):
            role = idx_to_role[idx]
            if role is not None:
                role_groups[role].append(float(per_person_anomaly[idx]))

        print(f"--- {condition} ---")
        print(f"{'role':<14} {'n':<4} {'mean':<10} {'median':<10} {'min':<10} {'max':<10}")
        for role, vals in sorted(role_groups.items(), key=lambda kv: -np.mean(kv[1])):
            arr = np.array(vals)
            print(f"{role:<14} {len(arr):<4} "
                  f"{arr.mean():+.4f}    {np.median(arr):+.4f}    "
                  f"{arr.min():+.4f}    {arr.max():+.4f}")

        # Mann-Whitney: legal+govaffairs vs admin (the H1 test)
        crisis_response = (np.array(role_groups.get("legal", []) + role_groups.get("govaffairs", [])))
        admins = np.array(role_groups.get("admin", []))
        if len(crisis_response) > 0 and len(admins) > 0:
            p = mann_whitney_u_one_sided(crisis_response, admins)
            print(f"\n  MWU one-sided test (legal+govaffairs > admin): p = {p:.4f}")
            print(f"  median anomaly: legal+govaffairs = {np.median(crisis_response):+.4f}, "
                  f"admin = {np.median(admins):+.4f}")
        print()

        role_results[condition] = {
            "per_role": {r: {"n": len(v), "mean": float(np.mean(v)),
                              "median": float(np.median(v)),
                              "values": v}
                          for r, v in role_groups.items()},
            "mwu_legal_govaffairs_vs_admin": (
                mann_whitney_u_one_sided(
                    np.array(role_groups.get("legal", []) + role_groups.get("govaffairs", [])),
                    np.array(role_groups.get("admin", [])),
                )
                if role_groups.get("legal") and role_groups.get("admin") else float("nan")
            ),
        }

    # write report
    md_lines = [
        "# E-P: Enron per-person fraud-week anomaly grouped by role",
        "",
        f"hypothesis: people doing role-relevant crisis response (legal, govaffairs)",
        f"show higher prediction-error anomaly during fraud weeks than admin staff.",
        "",
        f"fraud snaps: {sorted(FRAUD_SNAPS.keys())}",
        f"baseline: non-fraud test snaps in [{TEST_RANGE[0]}, {TEST_RANGE[1]}]",
        f"seeds: {[s for s, _ in seeds_data]}",
        "",
    ]
    for condition, results in role_results.items():
        md_lines.append(f"## {condition}")
        md_lines.append("")
        md_lines.append("| role | n | mean fraud-week anomaly | median |")
        md_lines.append("|---|---:|---:|---:|")
        sorted_roles = sorted(results["per_role"].items(), key=lambda kv: -kv[1]["mean"])
        for role, info in sorted_roles:
            md_lines.append(f"| {role} | {info['n']} | {info['mean']:+.4f} | {info['median']:+.4f} |")
        md_lines.append("")
        md_lines.append(f"MWU one-sided p (legal+govaffairs > admin): "
                        f"{results['mwu_legal_govaffairs_vs_admin']:.4f}")
        md_lines.append("")

    md_path = root / "enron" / "ENRON_FRAUD_ANOMALY.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
