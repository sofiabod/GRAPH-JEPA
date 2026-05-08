"""builder for hungary chickenpox temporal-graph dataset.

source: pytorch_geometric_temporal ChickenpoxDatasetLoader.
20 hungarian counties x 521 weekly chickenpox case counts (z-scored).
edges are fixed county adjacencies (102 edges, includes self-loops, unweighted).

graph topology is fixed across all snapshots. node feature is the per-county
case count at the snapshot week. structural features are computed once on the
fixed graph and copied to every snapshot.

one graph per week, normalized by training-region max-abs of the z-scored signal
to land in roughly [-1, 1] without leaking val/test info.
"""
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


def build_chickenpox_graphs():
    """build hungary chickenpox temporal graph snapshots.

    no args: pulls data from torch_geometric_temporal's ChickenpoxDatasetLoader.

    returns (graphs: list[PyG Data], meta: dict)
    """
    from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader

    loader = ChickenpoxDatasetLoader()
    raw = loader._dataset
    # FX: list of [n_nodes] case counts per week, already z-scored
    fx = np.array(raw["FX"], dtype=np.float32)  # [T, N]
    n_snapshots, n_nodes = fx.shape

    # county-adjacency edges (undirected, listed as ordered pairs).
    # the loader file includes self-loops; drop them so structural degrees
    # match the metrla convention (no self-loops in edge_index).
    raw_edges = np.array(raw["edges"], dtype=np.int64)  # [E, 2]
    keep = raw_edges[:, 0] != raw_edges[:, 1]
    edges = raw_edges[keep].T  # [2, E']
    edge_index = torch.tensor(edges, dtype=torch.long)
    n_edges = edge_index.shape[1]

    # county adjacency is unweighted: edge_weights = ones
    edge_weights = torch.ones(n_edges, dtype=torch.float)
    edge_attr = edge_weights.unsqueeze(1)  # [E, 1]

    # 5d structural features computed once on the fixed graph
    structural = compute_structural_features(
        edge_index, n_nodes=n_nodes, edge_weights=edge_weights
    )  # [N, 5]

    # train-region scaler: max-abs of z-scored cases over training weeks.
    # signal can be negative, so scale into [-1, 1] via /max_abs without
    # leaking val/test extremes.
    train_end = int(n_snapshots * 0.70)
    train_fx = fx[: max(1, train_end)]
    abs_max = float(np.abs(train_fx).max()) if train_fx.size > 0 else 1.0
    if abs_max <= 0:
        abs_max = 1.0

    node_ids_t = torch.arange(n_nodes)

    graphs = []
    for t in range(n_snapshots):
        signal_t = torch.from_numpy(fx[t]).float() / abs_max
        signal_t = signal_t.clamp(-1.0, 1.0).unsqueeze(1)  # [N, 1]

        x = torch.cat([signal_t, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=node_ids_t,
        ))

    train_range, val_range, test_range = compute_split_ranges(n_snapshots)
    meta = {
        "dataset": "chickenpox",
        "n_nodes": n_nodes,
        "n_snapshots": n_snapshots,
        "node_feature_dim": 6,
        "edge_count": int(edge_index.shape[1]),
        "abs_max_train": abs_max,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta
