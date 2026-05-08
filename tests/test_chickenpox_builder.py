import pytest
import torch
from torch_geometric.data import Data

# torch_geometric_temporal pulls data over the network during ChickenpoxDatasetLoader().
# guard the whole module behind a skip if the package or network are unavailable so
# the rest of the suite still runs in offline ci environments.
pytest.importorskip("torch_geometric_temporal")

from src.data.chickenpox_builder import build_chickenpox_graphs


@pytest.fixture(scope="module")
def built():
    try:
        graphs, meta = build_chickenpox_graphs()
    except Exception as e:
        pytest.skip(f"could not fetch chickenpox dataset: {e}")
    return graphs, meta


def test_output_is_list_of_data(built):
    graphs, meta = built
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_n_nodes_is_20(built):
    graphs, meta = built
    assert meta["n_nodes"] == 20
    for g in graphs:
        assert g.x.shape[0] == 20


def test_n_snapshots_is_about_520(built):
    graphs, meta = built
    # hungary chickenpox dataset has 521 weekly snapshots
    assert 510 <= meta["n_snapshots"] <= 530
    assert len(graphs) == meta["n_snapshots"]


def test_node_feature_dim_is_6(built):
    graphs, meta = built
    for g in graphs:
        assert g.x.shape[1] == 6


def test_edge_index_valid_and_no_self_loops(built):
    graphs, meta = built
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        if g.edge_index.shape[1] > 0:
            assert g.edge_index.max() < g.x.shape[0]
            assert (g.edge_index[0] != g.edge_index[1]).all()


def test_edge_index_shared_across_snapshots(built):
    graphs, meta = built
    base = graphs[0].edge_index
    for g in graphs[1:]:
        assert torch.equal(g.edge_index, base)


def test_meta_has_split_ranges(built):
    graphs, meta = built
    assert "train_range" in meta
    assert "val_range" in meta
    assert "test_range" in meta
    assert meta["dataset"] == "chickenpox"
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_signal_in_unit_range(built):
    graphs, meta = built
    # signal is z-scored then scaled by training-region max-abs into [-1, 1]
    for g in graphs:
        assert g.x[:, 0].min() >= -1.0 - 1e-6
        assert g.x[:, 0].max() <= 1.0 + 1e-6


def test_structural_features_constant_across_time(built):
    graphs, meta = built
    base_struct = graphs[0].x[:, 1:]
    for g in graphs[1:]:
        assert torch.equal(g.x[:, 1:], base_struct)


def test_node_ids_present(built):
    graphs, meta = built
    for g in graphs:
        assert hasattr(g, "node_ids")
        assert g.node_ids.shape[0] == g.x.shape[0]
        assert torch.equal(g.node_ids, torch.arange(g.x.shape[0]))


def test_edge_attr_matches_edge_count(built):
    graphs, meta = built
    for g in graphs:
        assert g.edge_attr.shape == (g.edge_index.shape[1], 1)


def test_edge_count_matches_meta(built):
    graphs, meta = built
    for g in graphs:
        assert g.edge_index.shape[1] == meta["edge_count"]
