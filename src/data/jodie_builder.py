import urllib.request
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges

JODIE_URLS = {
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
}
INTERACTION_FEAT_DIM = 172


def download_jodie(dataset_name: str, data_dir: str) -> str:
    """download jodie csv for 'reddit' or 'wikipedia', return local path."""
    assert dataset_name in JODIE_URLS, f"unknown jodie dataset: {dataset_name}"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / f"{dataset_name}.csv"
    if not csv_path.exists():
        print(f"downloading JODIE {dataset_name}...")
        urllib.request.urlretrieve(JODIE_URLS[dataset_name], csv_path)
    return str(csv_path)


def build_jodie_graphs_from_csv(csv_path: str, min_active_nodes: int = 10):
    """parse jodie csv and return weekly snapshot graphs.

    jodie csv format: user,item,timestamp,state_label,feat_0,...,feat_171

    node features: 172d mean-pooled interaction features + 5 structural = 177d
    inactive nodes this week: zeros for 172d part, structural computed from topology.

    returns (graphs: list[PyG Data], meta: dict)
    """
    user_seq, item_seq, ts_seq, feat_seq = [], [], [], []
    with open(csv_path) as f:
        header = f.readline()
        n_feat_cols = len(header.strip().split(",")) - 4  # subtract user,item,ts,label
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4 + n_feat_cols:
                continue
            user_seq.append(parts[0])
            item_seq.append(parts[1])
            ts_seq.append(float(parts[2]))
            feat_seq.append([float(x) for x in parts[4:4 + n_feat_cols]])

    all_nodes = sorted(set(user_seq) | set(item_seq))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)
    feat_dim = n_feat_cols

    if not ts_seq:
        return [], {"n_snapshots": 0}
    t0 = min(ts_seq)
    week_sec = 7 * 24 * 3600
    edges_by_week = defaultdict(list)  # week -> [(src_id, dst_id, feat_vec)]
    for u, it, ts, feat in zip(user_seq, item_seq, ts_seq, feat_seq):
        week = int((ts - t0) / week_sec)
        edges_by_week[week].append((node2id[u], node2id[it], feat))

    graphs = []
    for week in sorted(edges_by_week.keys()):
        week_edges = edges_by_week[week]
        active_nodes = set()
        for s, d, _ in week_edges:
            active_nodes.add(s)
            active_nodes.add(d)
        if len(active_nodes) < min_active_nodes:
            continue

        src_list = [s for s, d, _ in week_edges]
        dst_list = [d for s, d, _ in week_edges]
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # mean-pool interaction features onto nodes
        node_feat_sum = torch.zeros(n_nodes, feat_dim)
        node_feat_count = torch.zeros(n_nodes)
        for s, d, feat in week_edges:
            fv = torch.tensor(feat, dtype=torch.float)
            node_feat_sum[s] += fv
            node_feat_sum[d] += fv
            node_feat_count[s] += 1
            node_feat_count[d] += 1
        count_safe = node_feat_count.clamp(min=1).unsqueeze(1)
        interaction_feats = node_feat_sum / count_safe
        # zero out nodes with no interactions this week
        inactive = node_feat_count == 0
        interaction_feats[inactive] = 0.0

        structural = compute_structural_features(edge_index, n_nodes=n_nodes)
        x = torch.cat([interaction_feats, structural], dim=1)  # [N, feat_dim + 5]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            node_ids=torch.arange(n_nodes),
        ))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": feat_dim + 5,
        "interaction_feat_dim": feat_dim,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
        "node2id": node2id,
    }
    return graphs, meta
