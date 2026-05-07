import torch
from torch_geometric.data import Data
from src.data.tgb_builder import (
    build_tgbn_trade_graphs_from_raw,
    build_tgbn_genre_graphs_from_raw,
)


def test_tgb_pip_package_importable():
    # smoke test: confirm py-tgb pip install resolves the expected import path
    from tgb.nodeproppred.dataset import NodePropPredDataset  # noqa: F401


def make_fake_trade_data(n_countries=10, n_years=15, edges_per_year=20):
    """generate fake trade data: list of (year, src, dst, volume) tuples."""
    import random
    records = []
    for year in range(2000, 2000 + n_years):
        for _ in range(edges_per_year):
            src = random.randint(0, n_countries - 1)
            dst = random.randint(0, n_countries - 1)
            while dst == src:
                dst = random.randint(0, n_countries - 1)
            vol = random.random() * 1e9
            records.append((year, src, dst, vol))
    return records, list(range(n_countries))


def test_output_is_list_of_data():
    records, countries = make_fake_trade_data()
    graphs, meta = build_tgbn_trade_graphs_from_raw(records, countries)
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_node_feature_dim_is_6():
    records, countries = make_fake_trade_data()
    graphs, meta = build_tgbn_trade_graphs_from_raw(records, countries)
    for g in graphs:
        assert g.x.shape[1] == 6, f"expected in_dim=6, got {g.x.shape[1]}"


def test_edge_index_valid():
    records, countries = make_fake_trade_data()
    graphs, meta = build_tgbn_trade_graphs_from_raw(records, countries)
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        if g.edge_index.shape[1] > 0:
            assert g.edge_index.max() < g.x.shape[0]


def test_meta_has_split_ranges():
    records, countries = make_fake_trade_data(n_years=20)
    graphs, meta = build_tgbn_trade_graphs_from_raw(records, countries)
    assert "train_range" in meta and "val_range" in meta and "test_range" in meta
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_volume_feature_normalized():
    records, countries = make_fake_trade_data()
    graphs, meta = build_tgbn_trade_graphs_from_raw(records, countries)
    for g in graphs:
        # col 0 = normalized trade volume, must be in [0, 1]
        assert g.x[:, 0].min() >= 0.0
        assert g.x[:, 0].max() <= 1.0 + 1e-6


def make_fake_genre_data(n_nodes=30, n_weeks=20, edges_per_week=80):
    """generate fake (week, src, dst, edge_feat_value) tuples."""
    import random
    records = []
    for week in range(n_weeks):
        for _ in range(edges_per_week):
            src = random.randint(0, n_nodes - 1)
            dst = random.randint(0, n_nodes - 1)
            while dst == src:
                dst = random.randint(0, n_nodes - 1)
            vol = random.random()
            records.append((week, src, dst, vol))
    return records, list(range(n_nodes))


def test_genre_output_is_list_of_data():
    records, nodes = make_fake_genre_data()
    graphs, meta = build_tgbn_genre_graphs_from_raw(records, nodes)
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_genre_node_feature_dim_is_6():
    records, nodes = make_fake_genre_data()
    graphs, meta = build_tgbn_genre_graphs_from_raw(records, nodes)
    for g in graphs:
        assert g.x.shape[1] == 6, f"expected in_dim=6, got {g.x.shape[1]}"


def test_genre_edge_index_valid():
    records, nodes = make_fake_genre_data()
    graphs, meta = build_tgbn_genre_graphs_from_raw(records, nodes)
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        if g.edge_index.shape[1] > 0:
            assert g.edge_index.max() < g.x.shape[0]


def test_genre_meta_has_split_ranges():
    records, nodes = make_fake_genre_data(n_weeks=30)
    graphs, meta = build_tgbn_genre_graphs_from_raw(records, nodes)
    assert "train_range" in meta and "val_range" in meta and "test_range" in meta
    assert meta["dataset"] == "tgbn_genre"
    assert meta["n_nodes"] == len(nodes)
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_genre_volume_feature_normalized():
    records, nodes = make_fake_genre_data()
    graphs, meta = build_tgbn_genre_graphs_from_raw(records, nodes)
    for g in graphs:
        # col 0 = normalized total edge volume, must be in [0, 1]
        assert g.x[:, 0].min() >= 0.0
        assert g.x[:, 0].max() <= 1.0 + 1e-6


def test_genre_min_active_nodes_filter():
    # only one edge per week → 2 active nodes; min_active_nodes=10 should drop all
    records = [(w, 0, 1, 0.5) for w in range(20)]
    graphs, meta = build_tgbn_genre_graphs_from_raw(records, list(range(30)), min_active_nodes=10)
    assert len(graphs) == 0
    assert meta["n_snapshots"] == 0
