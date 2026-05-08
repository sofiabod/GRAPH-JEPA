import torch
from torch_geometric.data import Data

from src.data.icio_builder import (
    build_icio_graphs_from_records,
    _parse_partner_country,
)


def make_fake_records(n_countries=8, n_years=15, base_year=2000):
    """generate fake (year, country, partner, value) records.

    every directed pair gets a flow each year, with values drifting smoothly
    so structural features are stable across snapshots.
    """
    countries = [f"C{i:02d}" for i in range(n_countries)]
    records = []
    for y_idx in range(n_years):
        year = base_year + y_idx
        for i, c_o in enumerate(countries):
            for j, c_d in enumerate(countries):
                if i == j:
                    continue
                value = 100.0 + 10.0 * i + j + 0.5 * y_idx
                records.append((year, c_o, c_d, value))
    return records


def test_output_is_list_of_data():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_node_feature_dim_is_6():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    for g in graphs:
        assert g.x.shape[1] == 6, f"expected in_dim=6, got {g.x.shape[1]}"


def test_edge_index_valid():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.max() < g.x.shape[0]


def test_edge_attr_present():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    for g in graphs:
        assert g.edge_attr is not None
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]
        assert g.edge_attr.shape[1] == 1


def test_meta_has_split_ranges():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    assert "train_range" in meta
    assert "val_range" in meta
    assert "test_range" in meta
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_node_ids_present():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    for g in graphs:
        assert hasattr(g, "node_ids")
        assert g.node_ids.shape[0] == g.x.shape[0]


def test_fixed_node_set_across_snapshots():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    n_first = graphs[0].x.shape[0]
    for g in graphs:
        assert g.x.shape[0] == n_first


def test_within_country_flows_dropped():
    # construct records with self-loops; they should be filtered out
    records = [
        (2000, "AAA", "AAA", 999.0),
        (2000, "AAA", "BBB", 1.0),
        (2000, "BBB", "AAA", 2.0),
    ]
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    assert len(graphs) == 1
    assert graphs[0].edge_index.shape[1] == 2


def test_partner_country_parser():
    assert _parse_partner_country("USA12") == "USA"
    assert _parse_partner_country("AUS1") == "AUS"
    assert _parse_partner_country("DEU") == "DEU"
    assert _parse_partner_country("TOT") == "TOT"  # 3 letters: still parsed but caller filters
    assert _parse_partner_country("12") == ""
    assert _parse_partner_country("") == ""


def test_meta_dataset_label():
    records = make_fake_records()
    graphs, meta = build_icio_graphs_from_records(records, min_active_nodes=2)
    assert meta["dataset"] == "icio"
    assert meta["node_feature_dim"] == 6
    assert meta["n_snapshots"] == len(graphs)
