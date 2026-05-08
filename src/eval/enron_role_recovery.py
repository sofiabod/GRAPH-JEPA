"""eval 11: hidden-role recovery on enron.

claim under test: graph-JEPA's frozen encoder, trained only with mask-and-predict
on the email graph, organizes executives by FUNCTIONAL ROLE — lawyers cluster
with lawyers, traders with traders, government-affairs with government-affairs —
even though role labels never appeared in training.

method:
  1. load frozen encoder
  2. embed every person, averaged across test snapshots
  3. for each pair (i, j) with both having known roles, compute cos_sim(emb[i], emb[j])
  4. split pairs into within-role (i, j same role) and between-role (i, j different role)
  5. statistics:
     - median, mean of each distribution
     - mann-whitney U test (one-sided: within > between)
     - bootstrap 95% CI on median(within) - median(between)
     - per-role within-cosine for the major roles (legal, trading, govaffairs, admin)

what makes this non-tautological:
  - role isn't a function of email volume alone (lawyers email everyone for
    contract reviews; traders email each other AND external counterparties;
    admins email a few executives heavily)
  - both graph and sequential encoders see normalized volume features at the
    node level, so the trivial "high-volume people cluster together" signal is
    available to BOTH; if graph clusters by role and sequential doesn't, the
    role information is in the second-order structure (who emails whom),
    which only message-passing can extract
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _embed_all_people(
    online, graphs, context_k: int, device, eval_indices: list[int] | None = None
) -> np.ndarray:
    """produce one L2-normalized embedding per person, averaged across
    eval_indices snapshots. shape [N, D]."""
    online.eval()
    graphs[0].x.shape[0]
    if eval_indices is None:
        eval_indices = list(range(context_k, len(graphs)))

    accum = None
    count = 0
    with torch.no_grad():
        for t_idx in eval_indices:
            g = graphs[t_idx].to(device)
            z = online(g)
            if accum is None:
                accum = torch.zeros_like(z)
            accum = accum + z
            count += 1
    assert accum is not None, "no eval snapshots produced — check eval_indices"
    accum = accum / max(count, 1)
    accum = F.normalize(accum, dim=-1)
    return accum.cpu().numpy()


def _pairwise_cosine_matrix(emb: np.ndarray) -> np.ndarray:
    """upper-triangular pairwise cosine. emb is L2-normalized so cos = dot."""
    return emb @ emb.T


def _mann_whitney_u_one_sided(within: np.ndarray, between: np.ndarray) -> dict:
    """one-sided mann-whitney U: H1 within > between.
    returns dict with U, p, normal-approx z. self-contained (no scipy)."""
    n_w, n_b = len(within), len(between)
    if n_w == 0 or n_b == 0:
        return {
            "U": float("nan"),
            "p": float("nan"),
            "z": float("nan"),
            "n_within": int(n_w),
            "n_between": int(n_b),
        }
    combined = np.concatenate([within, between])
    # ranks (average for ties)
    order = np.argsort(combined, kind="stable")
    ranks = np.empty_like(combined, dtype=np.float64)
    # average rank for ties
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[order[j + 1]] == combined[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed average
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    R_w = ranks[:n_w].sum()
    U_w = R_w - n_w * (n_w + 1) / 2.0
    mean_U = n_w * n_b / 2.0
    var_U = n_w * n_b * (n_w + n_b + 1) / 12.0
    if var_U <= 0:
        return {
            "U": float(U_w),
            "p": float("nan"),
            "z": float("nan"),
            "n_within": int(n_w),
            "n_between": int(n_b),
        }
    z = (U_w - mean_U) / np.sqrt(var_U)
    # one-sided p (within > between → larger U → larger z)
    # use normal approx; for exact p with small n, scipy would be better
    from math import erf, sqrt

    p = 0.5 * (1.0 - erf(z / sqrt(2)))
    return {
        "U": float(U_w),
        "p": float(p),
        "z": float(z),
        "n_within": int(n_w),
        "n_between": int(n_b),
    }


def _bootstrap_median_gap_ci(
    within: np.ndarray,
    between: np.ndarray,
    n_resamples: int = 5000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n_w, n_b = len(within), len(between)
    if n_w == 0 or n_b == 0:
        return {
            "median_gap": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_resamples": 0,
        }
    gaps = np.empty(n_resamples)
    for r in range(n_resamples):
        w = within[rng.integers(0, n_w, size=n_w)]
        b = between[rng.integers(0, n_b, size=n_b)]
        gaps[r] = np.median(w) - np.median(b)
    alpha = (1.0 - ci) / 2.0
    return {
        "median_gap": float(np.median(within) - np.median(between)),
        "ci_low": float(np.quantile(gaps, alpha)),
        "ci_high": float(np.quantile(gaps, 1.0 - alpha)),
        "n_resamples": int(n_resamples),
    }


def run_role_recovery(
    online,
    graphs,
    cfg,
    person_index: dict,
    email_to_role: dict,
    *,
    kind: str = "graph",
    seed: int = 0,
    device=None,
) -> dict:
    """run role-recovery test for one model.

    args:
      online: encoder (None for raw_features baseline)
      person_index: dict email -> node_index from enron_meta.json
      email_to_role: dict email -> role string from enron_roles.py
      kind: "graph" / "sequential" / "raw_features"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_nodes = graphs[0].x.shape[0]
    K = cfg.training.context_k
    # average embeddings across test snapshots only (most relevant for role-period 2001-2002)
    import json

    if hasattr(cfg.data, "test_weeks") and cfg.data.test_weeks is not None:
        test_lo, test_hi = cfg.data.test_weeks
    else:
        with open(f"/app/{cfg.data.meta_path}") as f:
            meta = json.load(f)
        test_lo, test_hi = meta["test_range"]
    eval_indices = list(range(test_lo, test_hi + 1))

    if kind == "raw_features":
        accum = torch.zeros(n_nodes, cfg.encoder.in_dim)
        for t in eval_indices:
            accum = accum + graphs[t].x.cpu()
        emb = (accum / len(eval_indices)).numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
        emb = emb / norms
    else:
        emb = _embed_all_people(online, graphs, K, device, eval_indices=eval_indices)

    cos_M = _pairwise_cosine_matrix(emb)

    # build idx -> role mapping for labeled people
    idx_to_role: dict[int, str] = {}
    for email, idx in person_index.items():
        role = email_to_role.get(email.strip().lower())
        if role is not None:
            idx_to_role[idx] = role

    labeled_idx = sorted(idx_to_role.keys())
    n_labeled = len(labeled_idx)

    # collect pairwise cosines, split into within/between
    within: list[float] = []
    between: list[float] = []
    per_role_within: dict[str, list[float]] = {}
    for ii in range(n_labeled):
        i = labeled_idx[ii]
        for jj in range(ii + 1, n_labeled):
            j = labeled_idx[jj]
            cos_ij = float(cos_M[i, j])
            if idx_to_role[i] == idx_to_role[j]:
                within.append(cos_ij)
                per_role_within.setdefault(idx_to_role[i], []).append(cos_ij)
            else:
                between.append(cos_ij)

    within_arr = np.array(within)
    between_arr = np.array(between)

    mwu = _mann_whitney_u_one_sided(within_arr, between_arr)
    boot = _bootstrap_median_gap_ci(within_arr, between_arr, n_resamples=5000, seed=seed)

    per_role_summary = {}
    for role, vals in per_role_within.items():
        a = np.array(vals)
        per_role_summary[role] = {
            "n_pairs": int(len(a)),
            "median_cos": float(np.median(a)) if len(a) > 0 else float("nan"),
            "mean_cos": float(a.mean()) if len(a) > 0 else float("nan"),
        }

    # build idx -> email mapping (so downstream analyses can identify who's who)
    idx_to_email = {idx: email for email, idx in person_index.items()}
    emails_in_order = [idx_to_email.get(i, f"node_{i}") for i in range(n_nodes)]

    return {
        "kind": kind,
        "n_labeled_people": int(n_labeled),
        "n_within_pairs": int(len(within_arr)),
        "n_between_pairs": int(len(between_arr)),
        "median_within_cos": float(np.median(within_arr)) if len(within_arr) else float("nan"),
        "median_between_cos": float(np.median(between_arr)) if len(between_arr) else float("nan"),
        "mean_within_cos": float(within_arr.mean()) if len(within_arr) else float("nan"),
        "mean_between_cos": float(between_arr.mean()) if len(between_arr) else float("nan"),
        "median_gap": boot["median_gap"],
        "median_gap_ci95_low": boot["ci_low"],
        "median_gap_ci95_high": boot["ci_high"],
        "mwu_U": mwu["U"],
        "mwu_z": mwu["z"],
        "mwu_p_one_sided": mwu["p"],
        "per_role_within_cos": per_role_summary,
        "test_snapshots_used": eval_indices,
        # save embeddings + identity mapping so downstream analyses (sub-cluster
        # discovery, role-shuffle nulls, t-SNE) can run locally without re-running modal
        "embeddings": emb.tolist(),
        "emails_in_order": emails_in_order,
        "idx_to_role": {str(i): r for i, r in idx_to_role.items()},
    }
