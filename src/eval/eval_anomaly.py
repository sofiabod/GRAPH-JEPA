"""eval 7: per-snapshot prediction-error trajectory for anomaly/shock detection.

unlike eval 1/2/3 (which aggregate prediction quality across the test split),
this eval reports per-snapshot prediction cosine across the *entire timeseries*
(train + val + test snapshots that have a valid context window). real-world
shocks should show up as spikes in (1 - mean_pred_cos) because the model has
been fit on "normal" dynamics and a shock breaks them.

the demo claim: if graph-jepa's prediction error spikes at known shocks
(2008 GFC, 2020 COVID, etc.) more cleanly than sequential's, the model is
modelling real-world temporal-graph dynamics, not just self-consistency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.eval.metrics import cosine_sim


def _per_snapshot_cos(online, target, predictor, sample, device):
    """forward graph-jepa once on a single (context, target) sample,
    return mean cosine sim for the masked nodes."""
    from src.train import _build_tokens_for_sample, _encode_context

    tgt_graph = sample["target_graph"].to(device)
    masked_ids = sample["masked_node_ids"].to(device)
    visible_ids = sample["visible_node_ids"].to(device)

    ctx_embs = _encode_context(online, sample["context_graphs"])
    tgt_emb = target(tgt_graph)

    tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
        ctx_embs, tgt_emb, masked_ids, visible_ids, predictor
    )
    out = predictor(
        tokens.unsqueeze(0),
        time_indices.unsqueeze(0),
        node_ids_seq.unsqueeze(0),
    ).squeeze(0)

    z_pred = F.normalize(out[mask_positions], dim=-1)
    z_true = tgt_emb[masked_ids]
    cosines = cosine_sim(z_pred, z_true)
    return float(cosines.mean().item()), int(masked_ids.numel())


def run_anomaly_trajectory(
    online, target, predictor, graphs, cfg, year_labels: list[int], mask_seed: int = 0, device=None
) -> dict:
    """compute mean prediction cosine for every snapshot that has a valid
    context window (idx >= context_k). returns one cosine per year_label,
    plus a (1 - cos) "deviation" trajectory.

    args:
        year_labels: list of length len(graphs); e.g. [1996, 1997, ..., 2020].
            used to label the output for plotting.
        mask_seed: seed for deterministic per-sample masking, same as eval 2.

    note: this iterates one sample per snapshot. masking is deterministic
    via (seed, target_idx) so the result is reproducible.
    """
    from src.data.dataset import TemporalGraphDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    online.eval()
    target.encoder.eval() if hasattr(target, "encoder") else None
    predictor.eval()

    K = cfg.training.context_k
    n_snapshots = len(graphs)
    if n_snapshots != len(year_labels):
        raise ValueError(
            f"year_labels length {len(year_labels)} must match graphs length {n_snapshots}"
        )

    # build a dataset spanning ALL valid target indices (idx >= K), regardless of split
    full_range = (K, n_snapshots - 1)
    dataset = TemporalGraphDataset(
        graphs,
        context_k=K,
        mask_ratio=cfg.training.mask_ratio,
        split="train",
        train_range=full_range,
        val_range=full_range,
        test_range=full_range,
        seed=mask_seed,
    )

    per_snapshot = []
    with torch.no_grad():
        for sample in dataset:
            t_idx = sample["week_idx"]
            year = year_labels[t_idx]
            cos_mean, n_masked = _per_snapshot_cos(online, target, predictor, sample, device)
            # year may be int (annual) or str like "2001-W42" (weekly); preserve as-is
            per_snapshot.append(
                {
                    "snapshot_idx": int(t_idx),
                    "year": year if isinstance(year, str) else int(year),
                    "mean_pred_cos": cos_mean,
                    "deviation": 1.0 - cos_mean,
                    "n_masked": n_masked,
                }
            )

    return {
        "per_snapshot": per_snapshot,
        "context_k": int(K),
        "mask_ratio": float(cfg.training.mask_ratio),
        "mask_seed": int(mask_seed),
        "n_snapshots_evaluated": len(per_snapshot),
    }
