"""ema-smoothed node features for genre re-attempt (T1.2).

raw weekly tgbn-genre fails to learn because node features lack temporal
autocorrelation (bursty listening events). smoothing with an exponential moving
average over a temporal window injects the autocorrelation jepa requires.

usage:
    from src.data.feature_smoothing import ema_smooth_graph_sequence
    smoothed_graphs = ema_smooth_graph_sequence(graphs, tau_weeks=8)
"""
import torch
from torch_geometric.data import Data


def ema_smooth_graph_sequence(graphs, tau_weeks: int = 8, smooth_dims=None):
    """apply ema smoothing to node features across snapshots.

    args:
        graphs: list of pyg Data, ordered by time
        tau_weeks: smoothing time constant (effective window length)
        smooth_dims: which feature dims to smooth (None = all). e.g. [0] to smooth
                     only the volume feature and leave structural features raw.

    returns: new list of pyg Data with smoothed x; edges unchanged.
    """
    if len(graphs) == 0:
        return graphs

    alpha = 2.0 / (tau_weeks + 1)  # standard ema decay

    n_nodes = graphs[0].x.shape[0]
    feat_dim = graphs[0].x.shape[1]
    ema = graphs[0].x.clone()

    if smooth_dims is None:
        smooth_mask = torch.ones(feat_dim, dtype=torch.bool)
    else:
        smooth_mask = torch.zeros(feat_dim, dtype=torch.bool)
        smooth_mask[smooth_dims] = True

    smoothed = []
    for i, g in enumerate(graphs):
        if i == 0:
            x_smoothed = g.x.clone()
        else:
            x_smoothed = g.x.clone()
            ema[:, smooth_mask] = (
                alpha * g.x[:, smooth_mask] + (1 - alpha) * ema[:, smooth_mask]
            )
            x_smoothed[:, smooth_mask] = ema[:, smooth_mask]

        new_data = Data(
            x=x_smoothed,
            edge_index=g.edge_index,
            node_ids=g.node_ids if hasattr(g, "node_ids") else None,
        )
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            new_data.edge_attr = g.edge_attr
        smoothed.append(new_data)

    return smoothed
