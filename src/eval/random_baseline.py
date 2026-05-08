"""eval 13: random untrained encoder baseline.

claim under test: JEPA TRAINING induces the d≈8 attractor.
control: a randomly initialized (untrained) encoder. if its embeddings
also exhibit eff_rank ≈ 8, our claim is wrong — the geometry comes from
architecture, not from training.

method:
  1. instantiate the encoder fresh (random init, no checkpoint load)
  2. run eval6-style effective rank computation on its outputs
  3. report eff_rank for both graph and sequential architectures
  4. compare to trained eff_rank
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.eval.metrics import effective_rank, mean_pairwise_cosine


def random_encoder_eff_rank(
    online, graphs, cfg, *, kind: str = "graph", eval_indices=None, device=None
) -> dict:
    """compute effective rank of a randomly-initialized encoder's outputs.
    do NOT load checkpoint — use random initial weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online.eval()  # eval mode (no dropout) but still random weights
    K = cfg.training.context_k
    if eval_indices is None:
        eval_indices = list(range(K, len(graphs)))

    all_emb = []
    with torch.no_grad():
        for t in eval_indices:
            g = graphs[t].to(device)
            z = online(g)
            z = F.normalize(z, dim=-1)
            all_emb.append(z.cpu())
    full = torch.cat(all_emb, dim=0)  # [T*N, D]
    er = effective_rank(full)
    pc = mean_pairwise_cosine(full)
    return {
        "kind": kind,
        "effective_rank": float(er),
        "mean_pairwise_cosine": float(pc),
        "n_embeddings": int(full.shape[0]),
        "embedding_dim": int(full.shape[1]),
        "snapshots_used": eval_indices,
    }
