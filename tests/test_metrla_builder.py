import pickle

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.data.metrla_builder import build_metrla_graphs


def _write_fake_raw(raw_dir, n_nodes=12, n_timestamps=24 * 12 * 30, seed=0):
    """write fake metr-la.h5 and adj_mx.pkl into raw_dir.

    h5 stores a dataframe-style speed matrix of shape [n_timestamps, n_nodes].
    adj is a list [sensor_ids, id_to_idx, adj_matrix] mirroring the dcrnn format.
    """
    import h5py

    rng = np.random.default_rng(seed)
    sensor_ids = [f"S{i:04d}" for i in range(n_nodes)]
    speeds = rng.uniform(20.0, 70.0, size=(n_timestamps, n_nodes)).astype(np.float64)
    # sprinkle some zeros to simulate missing readings
    miss = rng.random(speeds.shape) < 0.05
    speeds[miss] = 0.0

    h5_path = raw_dir / "metr-la.h5"
    with h5py.File(str(h5_path), "w") as f:
        grp = f.create_group("df")
        # mimic the dcrnn h5 layout: axis0=columns (sensors), block0_values=data
        grp.create_dataset("axis0", data=np.array([s.encode() for s in sensor_ids]))
        grp.create_dataset("axis1", data=np.arange(n_timestamps, dtype=np.int64))
        grp.create_dataset("block0_items", data=np.array([s.encode() for s in sensor_ids]))
        grp.create_dataset("block0_values", data=speeds)

    # build a sparse-ish weighted adj. self-loops on diagonal, mostly zeros elsewhere
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)
    for _ in range(n_nodes * 3):
        i = rng.integers(0, n_nodes)
        j = rng.integers(0, n_nodes)
        if i != j:
            adj[i, j] = float(rng.uniform(0.1, 1.0))
    id_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
    adj_path = raw_dir / "adj_mx.pkl"
    with open(adj_path, "wb") as f:
        pickle.dump([sensor_ids, id_to_idx, adj], f)


def test_output_is_list_of_data(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_node_feature_dim_is_6(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    for g in graphs:
        assert g.x.shape[1] == 6, f"expected in_dim=6, got {g.x.shape[1]}"
        assert g.x.shape[0] == meta["n_nodes"]


def test_edge_index_valid_and_no_self_loops(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        if g.edge_index.shape[1] > 0:
            assert g.edge_index.max() < g.x.shape[0]
            assert (g.edge_index[0] != g.edge_index[1]).all()


def test_edge_index_shared_across_snapshots(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    base = graphs[0].edge_index
    for g in graphs[1:]:
        assert torch.equal(g.edge_index, base)


def test_meta_has_split_ranges(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    assert "train_range" in meta
    assert "val_range" in meta
    assert "test_range" in meta
    assert meta["dataset"] == "metrla"
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_speed_feature_in_unit_range(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    for g in graphs:
        assert g.x[:, 0].min() >= 0.0
        assert g.x[:, 0].max() <= 1.0 + 1e-6


def test_structural_features_constant_across_time(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    # cols 1..5 are structural and should be identical across all snapshots
    base_struct = graphs[0].x[:, 1:]
    for g in graphs[1:]:
        assert torch.equal(g.x[:, 1:], base_struct)


def test_node_ids_present(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    for g in graphs:
        assert hasattr(g, "node_ids")
        assert g.node_ids.shape[0] == g.x.shape[0]


def test_edge_attr_matches_edge_count(tmp_path):
    _write_fake_raw(tmp_path)
    graphs, meta = build_metrla_graphs(str(tmp_path))
    for g in graphs:
        assert g.edge_attr.shape == (g.edge_index.shape[1], 1)


def test_missing_raw_files_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_metrla_graphs(str(tmp_path))
