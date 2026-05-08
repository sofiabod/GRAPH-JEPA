"""eval 12: per-country prediction error at a target year.

unlike the aggregate anomaly trajectory (which averages cosine across
randomly-masked nodes), this eval masks ALL nodes at a target snapshot
and records the per-country prediction cosine. that gives us a 1-D
deviation vector over all 227 BACI countries at year 2020 (or any year),
which we can then correlate with real-world ground-truth measures of
country-level COVID trade impact.

claim under test:
  if graph-JEPA's per-country deviation at 2020 correlates with the actual
  per-country 2020 trade decline (ground truth computed from BACI raw),
  the model has recovered the COUNTRY-LEVEL pattern of the shock — not
  just registered "2020 is anomalous" in aggregate.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.metrics import cosine_sim


def per_country_cosines_at_year(
    online, target, predictor, graphs, cfg, year_idx: int, mask_seed: int = 0, device=None
) -> np.ndarray:
    """mask ALL nodes at graphs[year_idx], predict from context window,
    return per-country cosine similarity [N]. nodes whose context spans
    require K previous snapshots, so year_idx must be >= context_k."""
    from src.train import _build_tokens_for_sample, _encode_context

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online.eval()
    if hasattr(target, "encoder"):
        target.encoder.eval()
    predictor.eval()

    K = cfg.training.context_k
    if year_idx < K:
        raise ValueError(f"year_idx={year_idx} too small; need >= context_k={K}")

    target_graph = graphs[year_idx].to(device)
    n_nodes = target_graph.x.shape[0]

    # mask ALL nodes
    masked_ids = torch.arange(n_nodes, device=device)
    visible_ids = torch.tensor([], dtype=torch.long, device=device)

    context_graphs = graphs[year_idx - K : year_idx]
    sample = {
        "context_graphs": context_graphs,
        "target_graph": target_graph,
        "masked_node_ids": masked_ids,
        "visible_node_ids": visible_ids,
    }

    with torch.no_grad():
        ctx_embs = _encode_context(online, sample["context_graphs"])
        tgt_emb = target(target_graph)
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
    return cosines.cpu().numpy()


def compute_actual_trade_decline_per_country(
    graphs, ref_year_idx: int, compare_year_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """compute (volume[compare_year] - volume[ref_year]) / volume[ref_year]
    per country from raw BACI graphs. negative values = trade decline.

    use as ground-truth for COVID-2020 country-level impact ranking.
    """
    n_nodes = graphs[0].x.shape[0]

    def _country_volume(g):
        # sum of incident edge weights per node = total trade volume
        if not hasattr(g, "edge_attr") or g.edge_attr is None:
            return np.zeros(n_nodes)
        ei = g.edge_index.cpu().numpy()
        ew = g.edge_attr.cpu().numpy()
        if ew.ndim > 1:
            ew = ew[:, 0]
        v = np.zeros(n_nodes)
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            v[i] += float(ew[k])
            v[j] += float(ew[k])
        return v

    v_ref = _country_volume(graphs[ref_year_idx])
    v_cmp = _country_volume(graphs[compare_year_idx])
    # avoid division by zero; tiny floor for inactive countries
    safe_ref = np.maximum(v_ref, 1e-8)
    decline = (v_cmp - v_ref) / safe_ref
    # mark countries that were inactive in ref year (no signal)
    active = v_ref > 1e-8
    return decline, active
