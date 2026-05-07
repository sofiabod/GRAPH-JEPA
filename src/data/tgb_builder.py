from collections import defaultdict

import torch
from torch_geometric.data import Data

from tgb.nodeproppred.dataset import NodePropPredDataset

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


def download_tgbn_trade(data_dir: str):
    """download tgbn-trade via the pip-installed py-tgb package."""
    dataset = NodePropPredDataset(name="tgbn-trade", root=data_dir)
    return dataset


def build_tgbn_trade_graphs_from_raw(records, country_ids):
    """build annual graph snapshots from trade records.

    args:
        records: list of (year, src_idx, dst_idx, trade_volume) tuples
        country_ids: list of country identifiers (used to fix n_nodes)

    node features: [1d normalized total trade volume, 5d structural] = 6d
    returns (graphs: list[PyG Data], meta: dict)
    """
    n_nodes = len(country_ids)

    by_year = defaultdict(list)
    for year, src, dst, vol in records:
        by_year[year].append((src, dst, float(vol)))

    graphs = []
    for year in sorted(by_year.keys()):
        year_edges = by_year[year]

        src_list = [s for s, d, v in year_edges]
        dst_list = [d for s, d, v in year_edges]
        volumes = [v for s, d, v in year_edges]

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        vol_tensor = torch.tensor(volumes, dtype=torch.float)

        # per-node total trade volume (sum of all edge volumes incident to node)
        node_vol = torch.zeros(n_nodes)
        node_vol.scatter_add_(0, edge_index[0], vol_tensor)
        node_vol.scatter_add_(0, edge_index[1], vol_tensor)
        vol_max = node_vol.max().clamp(min=1e-8)
        node_vol_norm = (node_vol / vol_max).unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=vol_tensor)
        x = torch.cat([node_vol_norm, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            node_ids=torch.arange(n_nodes),
        ))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "tgbn_trade",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta


def build_tgbn_trade_graphs(data_dir: str):
    """full pipeline: download tgbn-trade via TGB, convert to annual snapshots."""
    dataset = NodePropPredDataset(name="tgbn-trade", root=data_dir)
    data = dataset.full_data

    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]
    edge_feats = data.get("edge_feat", None)  # may be None or [E, 1]

    all_nodes = sorted(set(sources.tolist()) | set(destinations.tolist()))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)

    # tgbn-trade timestamps are raw years stored as floats (e.g. 1986.0..2016.0),
    # not unix seconds. cast directly to int.
    records = []
    for i, (s, d, ts) in enumerate(zip(sources, destinations, timestamps)):
        year = int(float(ts))
        vol = float(edge_feats[i, 0]) if edge_feats is not None else 1.0
        records.append((year, node2id[int(s)], node2id[int(d)], vol))

    return build_tgbn_trade_graphs_from_raw(records, list(range(n_nodes)))


def build_tgbn_genre_graphs_from_raw(records, node_ids, min_active_nodes: int = 10):
    """build weekly graph snapshots from genre interaction records.

    args:
        records: list of (week, src_idx, dst_idx, edge_feat_value) tuples
        node_ids: list of node identifiers (used to fix n_nodes)
        min_active_nodes: drop weeks with fewer active nodes than this

    node features: [1d normalized total edge volume, 5d structural] = 6d
    returns (graphs: list[PyG Data], meta: dict)
    """
    n_nodes = len(node_ids)

    by_week = defaultdict(list)
    for week, src, dst, vol in records:
        by_week[week].append((src, dst, float(vol)))

    graphs = []
    for week in sorted(by_week.keys()):
        week_edges = by_week[week]

        active = set()
        for s, d, _ in week_edges:
            active.add(s)
            active.add(d)
        if len(active) < min_active_nodes:
            continue

        src_list = [s for s, d, v in week_edges]
        dst_list = [d for s, d, v in week_edges]
        volumes = [v for s, d, v in week_edges]

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        vol_tensor = torch.tensor(volumes, dtype=torch.float)

        # per-node total edge volume (sum of edge_feat values incident to each node)
        node_vol = torch.zeros(n_nodes)
        node_vol.scatter_add_(0, edge_index[0], vol_tensor)
        node_vol.scatter_add_(0, edge_index[1], vol_tensor)
        vol_max = node_vol.max().clamp(min=1e-8)
        node_vol_norm = (node_vol / vol_max).unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=vol_tensor)
        x = torch.cat([node_vol_norm, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            node_ids=torch.arange(n_nodes),
        ))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "tgbn_genre",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta


def build_tgbn_genre_graphs(data_dir: str):
    """full pipeline: download tgbn-genre via TGB, convert to weekly snapshots.

    timestamps are unix seconds; bucket into weeks of 604800 seconds starting
    from the first observed timestamp. matches the weekly aggregation pattern
    used in jodie_builder.py.
    """
    dataset = NodePropPredDataset(name="tgbn-genre", root=data_dir)
    data = dataset.full_data

    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]
    edge_feats = data.get("edge_feat", None)  # may be None or [E, 1]

    all_nodes = sorted(set(sources.tolist()) | set(destinations.tolist()))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)

    week_sec = 60 * 60 * 24 * 7
    t0 = float(timestamps.min()) if hasattr(timestamps, "min") else float(min(timestamps))

    records = []
    for i in range(len(sources)):
        ts = float(timestamps[i])
        week = int((ts - t0) / week_sec)
        if edge_feats is not None:
            ef = edge_feats[i]
            # edge_feat may be 1d scalar or [1] vector; handle both
            try:
                vol = float(ef[0])
            except (TypeError, IndexError):
                vol = float(ef)
        else:
            vol = 1.0
        records.append((week, node2id[int(sources[i])], node2id[int(destinations[i])], vol))

    return build_tgbn_genre_graphs_from_raw(records, list(range(n_nodes)))
