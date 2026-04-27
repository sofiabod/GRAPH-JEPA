import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


def download_tgbn_trade(data_dir: str):
    """download tgbn-trade using the TGB library (vendored at TGB/ in repo root)."""
    tgb_root = Path(__file__).parent.parent.parent / "TGB"
    if str(tgb_root) not in sys.path:
        sys.path.insert(0, str(tgb_root))
    from tgb.nodeproppred.dataset import NodePropPredDataset
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
    tgb_root = Path(__file__).parent.parent.parent / "TGB"
    if str(tgb_root) not in sys.path:
        sys.path.insert(0, str(tgb_root))
    from tgb.nodeproppred.dataset import NodePropPredDataset

    dataset = NodePropPredDataset(name="tgbn-trade", root=data_dir)
    data = dataset.full_data

    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]
    edge_feats = data.get("edge_feat", None)  # may be None or [E, 1]

    all_nodes = sorted(set(sources.tolist()) | set(destinations.tolist()))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)

    import datetime
    records = []
    for s, d, ts in zip(sources, destinations, timestamps):
        year = datetime.datetime.fromtimestamp(float(ts)).year
        idx = len(records)
        vol = float(edge_feats[idx, 0]) if edge_feats is not None else 1.0
        records.append((year, node2id[int(s)], node2id[int(d)], vol))

    return build_tgbn_trade_graphs_from_raw(records, list(range(n_nodes)))
