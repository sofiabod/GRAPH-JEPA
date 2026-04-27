import urllib.request
import gzip
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges

EU_EMAIL_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"


def download_eu_email(data_dir: str) -> str:
    """download eu email snap dataset, return path to unpacked txt file."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    gz_path = data_dir / "email-Eu-core-temporal.txt.gz"
    txt_path = data_dir / "email-Eu-core-temporal.txt"
    if not txt_path.exists():
        print("downloading EU Email SNAP dataset...")
        urllib.request.urlretrieve(EU_EMAIL_URL, gz_path)
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_path.unlink()
    return str(txt_path)


def build_eu_email_graphs_from_edges(txt_path: str, min_active_nodes: int = 10):
    """parse (src dst timestamp) edge list and return weekly snapshot graphs.

    returns (graphs: list[PyG Data], meta: dict)
    node features: 5d structural (out_deg, in_deg, out_w, in_w, active_flag)
    """
    edges_by_week = defaultdict(list)
    all_nodes = set()

    week_sec = 7 * 24 * 3600
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            src, dst, ts = int(parts[0]), int(parts[1]), int(parts[2])
            week = ts // week_sec
            edges_by_week[week].append((src, dst))
            all_nodes.add(src)
            all_nodes.add(dst)

    node2id = {n: i for i, n in enumerate(sorted(all_nodes))}
    n_nodes = len(node2id)

    graphs = []
    for week in sorted(edges_by_week.keys()):
        raw_edges = edges_by_week[week]
        active = {n for pair in raw_edges for n in pair}
        if len(active) < min_active_nodes:
            continue

        src_list = [node2id[s] for s, d in raw_edges]
        dst_list = [node2id[d] for s, d in raw_edges]
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # edge weights = count of repeated (src, dst) pairs this week
        edge_count = defaultdict(int)
        for s, d in raw_edges:
            edge_count[(node2id[s], node2id[d])] += 1
        weights = torch.tensor(
            [edge_count[(s, d)] for s, d in zip(src_list, dst_list)], dtype=torch.float
        )

        x = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=weights)

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            node_ids=torch.arange(n_nodes),
        ))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "eu_email",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 5,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
        "node2id": node2id,
    }
    return graphs, meta
