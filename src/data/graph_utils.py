import torch


def compute_structural_features(
    edge_index: torch.Tensor,
    n_nodes: int,
    edge_weights: torch.Tensor = None,
) -> torch.Tensor:
    """compute 5d structural node features from edge_index.

    returns [n_nodes, 5]:
      col 0: normalized out-degree
      col 1: normalized in-degree
      col 2: normalized weighted out-degree
      col 3: normalized weighted in-degree
      col 4: active flag (1 if node appears in any edge, else 0)
    """
    out_deg = torch.zeros(n_nodes)
    in_deg = torch.zeros(n_nodes)
    out_w = torch.zeros(n_nodes)
    in_w = torch.zeros(n_nodes)

    if edge_index.shape[1] > 0:
        src = edge_index[0]
        dst = edge_index[1]
        ones = torch.ones(src.shape[0])
        out_deg.scatter_add_(0, src, ones)
        in_deg.scatter_add_(0, dst, ones)
        w = edge_weights if edge_weights is not None else ones
        out_w.scatter_add_(0, src, w)
        in_w.scatter_add_(0, dst, w)

    active = ((out_deg + in_deg) > 0).float()

    feats = torch.stack([out_deg, in_deg, out_w, in_w, active], dim=1)
    col_max = feats[:, :4].max(dim=0).values.clamp(min=1e-8)
    feats[:, :4] = feats[:, :4] / col_max

    return feats
