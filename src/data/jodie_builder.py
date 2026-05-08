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


def build_jodie_user_user_graphs(csv_path: str, top_k_users: int = 500,
                                  bucket_seconds: int = 7 * 24 * 3600,
                                  min_active_nodes: int = 30,
                                  jaccard: bool = False):
    """projection-based jodie builder: user-user co-interaction graph.

    converts the bipartite user-item interaction graph into a unipartite user-user
    graph compatible with the working 6d-feature, ~200-1000-node architecture.

    for each time bucket (default weekly), two users share an edge if they both
    interacted with at least one common item in that bucket. edge weight = number
    of shared items (or jaccard if jaccard=True). filters to top_k_users by total
    activity to keep N tractable for full-graph attention in the predictor.

    matches the cross-dataset 6d convention: [1d normalized incident weight,
    5d structural].

    args:
        csv_path: path to jodie csv
        top_k_users: keep top-k most active users by total interaction count
        bucket_seconds: time bucket in seconds (default 1 week)
        min_active_nodes: drop buckets with fewer active users than this
        jaccard: if True, edge weight = jaccard(item_set_i, item_set_j); else =
            number of co-interacted items

    returns (graphs: list[PyG Data], meta: dict).
    """
    user_seq, item_seq, ts_seq = [], [], []
    with open(csv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            user_seq.append(parts[0])
            item_seq.append(parts[1])
            ts_seq.append(float(parts[2]))

    if not ts_seq:
        return [], {"dataset": "jodie_user_user", "n_snapshots": 0}

    # filter to top-k users by total activity
    user_counts = defaultdict(int)
    for u in user_seq:
        user_counts[u] += 1
    top_users = sorted(user_counts, key=lambda u: -user_counts[u])[:top_k_users]
    user2id = {u: i for i, u in enumerate(top_users)}
    n_nodes = len(user2id)

    # bucket by time
    t0 = min(ts_seq)
    bucket_to_user_items = defaultdict(lambda: defaultdict(set))  # bucket -> user_id -> set of items
    bucket_to_user_count = defaultdict(lambda: defaultdict(int))  # bucket -> user_id -> interaction count
    for u, it, ts in zip(user_seq, item_seq, ts_seq):
        if u not in user2id:
            continue
        bucket = int((ts - t0) / bucket_seconds)
        uid = user2id[u]
        bucket_to_user_items[bucket][uid].add(it)
        bucket_to_user_count[bucket][uid] += 1

    graphs = []
    edge_count_per_snapshot = []
    active_count_per_snapshot = []
    for bucket in sorted(bucket_to_user_items.keys()):
        user_items = bucket_to_user_items[bucket]
        active_users = sorted(user_items.keys())
        if len(active_users) < min_active_nodes:
            continue

        # build inverted index: item -> set of users that touched it this bucket
        item_to_users = defaultdict(set)
        for uid, items in user_items.items():
            for it in items:
                item_to_users[it].add(uid)

        # accumulate co-interaction counts between user pairs
        pair_count = defaultdict(float)
        for it, users in item_to_users.items():
            users_l = sorted(users)
            for i in range(len(users_l)):
                for j in range(i + 1, len(users_l)):
                    a, b = users_l[i], users_l[j]
                    pair_count[(a, b)] += 1.0

        if not pair_count:
            continue

        if jaccard:
            for (a, b), cnt in list(pair_count.items()):
                ia = user_items[a]
                ib = user_items[b]
                union = len(ia | ib)
                pair_count[(a, b)] = cnt / max(union, 1)

        # build undirected (symmetrized) edge_index
        src_list, dst_list, weight_list = [], [], []
        for (a, b), w in pair_count.items():
            src_list.extend([a, b])
            dst_list.extend([b, a])
            weight_list.extend([w, w])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        weights = torch.tensor(weight_list, dtype=torch.float)

        # 1d normalized per-user activity count this bucket
        node_activity = torch.zeros(n_nodes)
        for uid, cnt in bucket_to_user_count[bucket].items():
            node_activity[uid] = float(cnt)
        act_max = node_activity.max().clamp(min=1e-8)
        activity_norm = (node_activity / act_max).unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=weights)
        x = torch.cat([activity_norm, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=weights.unsqueeze(1),
            node_ids=torch.arange(n_nodes),
        ))
        edge_count_per_snapshot.append(edge_index.shape[1])
        active_count_per_snapshot.append(len(active_users))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "jodie_user_user",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "top_k_users": top_k_users,
        "bucket_seconds": bucket_seconds,
        "jaccard": jaccard,
        "edge_counts": edge_count_per_snapshot,
        "active_counts": active_count_per_snapshot,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta
