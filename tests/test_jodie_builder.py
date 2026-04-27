import torch
from torch_geometric.data import Data
from src.data.jodie_builder import build_jodie_graphs_from_csv


def make_fake_jodie_csv(n_users=5, n_items=3, n_rows=60, n_feat=172):
    """generate fake jodie CSV content."""
    import random
    lines = ["user,item,timestamp,state_label," + ",".join(f"f{i}" for i in range(n_feat))]
    week_sec = 7 * 24 * 3600
    for i in range(n_rows):
        user = f"u{random.randint(0, n_users-1)}"
        item = f"i{random.randint(0, n_items-1)}"
        ts = (i // (n_rows // 10)) * week_sec + i * 100
        label = 1 if i % 20 == 0 else 0
        feats = ",".join(str(round(random.random(), 4)) for _ in range(n_feat))
        lines.append(f"{user},{item},{ts},{label},{feats}")
    return "\n".join(lines)


def test_output_is_list_of_data(tmp_path):
    csv = tmp_path / "reddit.csv"
    csv.write_text(make_fake_jodie_csv())
    graphs, meta = build_jodie_graphs_from_csv(str(csv), min_active_nodes=2)
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_node_feature_dim_is_177(tmp_path):
    csv = tmp_path / "reddit.csv"
    csv.write_text(make_fake_jodie_csv(n_feat=172))
    graphs, meta = build_jodie_graphs_from_csv(str(csv), min_active_nodes=2)
    for g in graphs:
        assert g.x.shape[1] == 177, f"expected 177, got {g.x.shape[1]}"


def test_edge_index_valid(tmp_path):
    csv = tmp_path / "reddit.csv"
    csv.write_text(make_fake_jodie_csv())
    graphs, meta = build_jodie_graphs_from_csv(str(csv), min_active_nodes=2)
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        if g.edge_index.shape[1] > 0:
            assert g.edge_index.max() < g.x.shape[0]


def test_meta_has_split_ranges(tmp_path):
    csv = tmp_path / "reddit.csv"
    csv.write_text(make_fake_jodie_csv(n_rows=100))
    graphs, meta = build_jodie_graphs_from_csv(str(csv), min_active_nodes=2)
    assert "train_range" in meta and "val_range" in meta and "test_range" in meta
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_inactive_node_interaction_features_are_zero(tmp_path):
    csv = tmp_path / "reddit.csv"
    csv.write_text(make_fake_jodie_csv(n_users=10, n_items=2, n_rows=30))
    graphs, meta = build_jodie_graphs_from_csv(str(csv), min_active_nodes=1)
    for g in graphs:
        # col 176 = active flag (last of 5 structural cols, offset by 172 interaction cols)
        active_mask = g.x[:, 176] == 1.0
        inactive = ~active_mask
        if inactive.any():
            assert g.x[inactive, :172].abs().sum() == 0.0, \
                "inactive nodes should have zero interaction features"
