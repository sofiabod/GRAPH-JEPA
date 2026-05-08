"""builder for metr-la traffic forecasting dataset.

source: dcrnn data release
files:
  metr-la.h5      207 sensors x 34272 timestamps (5min) speed matrix
  adj_mx.pkl      [sensor_ids_list, id_to_idx_dict, adj_mx (207,207)]

graph topology is fixed across all snapshots (sensors at fixed road locations).
node feature is the per-sensor speed at the snapshot time bucket. structural
features are computed once on the fixed graph and copied to every snapshot.

aggregation: 5min raw -> hourly mean (12 step window) -> downsampled by stride
to land in the 120-300 snapshot range to match other datasets in the repo.
"""
from pathlib import Path
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


def _load_raw(raw_dir: str):
    """read metr-la.h5 (speeds) and adj_mx.pkl (adjacency)."""
    import h5py

    raw_path = Path(raw_dir)
    h5_path = raw_path / "metr-la.h5"
    adj_path = raw_path / "adj_mx.pkl"
    if not h5_path.exists():
        raise FileNotFoundError(f"missing {h5_path}")
    if not adj_path.exists():
        raise FileNotFoundError(f"missing {adj_path}")

    with h5py.File(str(h5_path), "r") as f:
        speeds = f["df/block0_values"][:]  # [T, N]
        sensor_ids_h5 = [s.decode() for s in f["df/axis0"][:]]

    with open(adj_path, "rb") as f:
        sensor_ids_adj, _id_to_idx, adj = pickle.load(f, encoding="latin1")

    if sensor_ids_h5 != list(sensor_ids_adj):
        # sanity: dcrnn release ships these with matching order, but if a future
        # mirror differs we permute the speed columns to match adj order
        h5_idx = {sid: i for i, sid in enumerate(sensor_ids_h5)}
        perm = np.array([h5_idx[sid] for sid in sensor_ids_adj], dtype=np.int64)
        speeds = speeds[:, perm]

    return speeds.astype(np.float32), np.asarray(adj, dtype=np.float32), list(sensor_ids_adj)


def _aggregate(speeds: np.ndarray, hour_window: int, hour_stride: int):
    """mean-pool 5min speeds to hourly, then keep every hour_stride-th hour.

    speeds: [T_raw, N] raw 5min observations (zeros mark missing readings)
    hour_window: number of 5min steps per hour (12 for 5min granularity)
    hour_stride: keep every Nth hourly snapshot

    returns [T_snap, N] aggregated speeds
    """
    T_raw, N = speeds.shape
    n_hours = T_raw // hour_window
    trimmed = speeds[: n_hours * hour_window].reshape(n_hours, hour_window, N)

    # mask zeros (missing readings) and average only over valid steps
    valid = (trimmed > 0).astype(np.float32)
    summed = (trimmed * valid).sum(axis=1)
    counts = valid.sum(axis=1).clip(min=1.0)
    hourly = summed / counts  # [n_hours, N]

    # downsample by stride
    hourly = hourly[::hour_stride]
    return hourly


def build_metrla_graphs(raw_dir: str, hour_window: int = 12, hour_stride: int = 12,
                        adj_threshold: float = 0.0):
    """build metr-la temporal graph snapshots.

    args:
        raw_dir: directory containing metr-la.h5 and adj_mx.pkl
        hour_window: number of 5min steps to mean-pool into one hourly value (12)
        hour_stride: keep every Nth hourly snapshot. 12 -> ~12hr blocks ->
            ~238 snapshots over the 4 month span
        adj_threshold: keep edges where adj_mx[i,j] > threshold. exclude self-loops

    returns (graphs: list[PyG Data], meta: dict)
    """
    speeds_raw, adj, sensor_ids = _load_raw(raw_dir)
    n_nodes = len(sensor_ids)

    # aggregate temporal axis
    hourly_speeds = _aggregate(speeds_raw, hour_window=hour_window, hour_stride=hour_stride)
    n_snapshots = hourly_speeds.shape[0]

    # build fixed edge_index from adj_mx (exclude self-loops)
    src_list = []
    dst_list = []
    weight_list = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            w = float(adj[i, j])
            if w > adj_threshold:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(w)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weights = torch.tensor(weight_list, dtype=torch.float)

    # structural features computed once on the fixed graph and shared across snapshots
    structural = compute_structural_features(
        edge_index, n_nodes=n_nodes, edge_weights=edge_weights
    )  # [N, 5]

    # global speed normalizer: max of training-region speeds keeps features in [0, 1]
    # without leaking val/test info. sensors top out around 70 mph in this dataset.
    train_end = int(n_snapshots * 0.70)
    train_speeds = hourly_speeds[: max(1, train_end)]
    speed_max = float(np.max(train_speeds)) if train_speeds.size > 0 else 1.0
    if speed_max <= 0:
        speed_max = 1.0

    edge_attr = edge_weights.unsqueeze(1)  # [E, 1]
    node_ids_t = torch.arange(n_nodes)

    graphs = []
    for t in range(n_snapshots):
        speed_t = torch.from_numpy(hourly_speeds[t]).float() / speed_max
        speed_t = speed_t.clamp(0.0, 1.0).unsqueeze(1)  # [N, 1]

        x = torch.cat([speed_t, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=node_ids_t,
        ))

    train_range, val_range, test_range = compute_split_ranges(n_snapshots)
    meta = {
        "dataset": "metrla",
        "n_nodes": n_nodes,
        "n_snapshots": n_snapshots,
        "node_feature_dim": 6,
        "edge_count": int(edge_index.shape[1]),
        "hour_window": hour_window,
        "hour_stride": hour_stride,
        "speed_max_train": speed_max,
        "adj_threshold": adj_threshold,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta
