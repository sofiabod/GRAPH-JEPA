import torch
from torch_geometric.data import Data
from src.data.eu_email_builder import build_eu_email_graphs_from_edges


def make_fake_edge_lines(n_nodes=8, n_edges=20, n_weeks=15):
    """generate fake (src dst timestamp) lines spanning n_weeks."""
    import random
    lines = []
    week_sec = 7 * 24 * 3600
    for i in range(n_edges * n_weeks):
        src = random.randint(0, n_nodes - 1)
        dst = random.randint(0, n_nodes - 1)
        ts = (i // n_edges) * week_sec + (i % n_edges) * 100
        lines.append(f"{src} {dst} {ts}")
    return "\n".join(lines)


def test_output_is_list_of_data(tmp_path):
    txt = tmp_path / "edges.txt"
    txt.write_text(make_fake_edge_lines())
    graphs, meta = build_eu_email_graphs_from_edges(str(txt), min_active_nodes=2)
    assert isinstance(graphs, list)
    assert len(graphs) > 0
    assert isinstance(graphs[0], Data)


def test_node_feature_dim_is_5(tmp_path):
    txt = tmp_path / "edges.txt"
    txt.write_text(make_fake_edge_lines(n_nodes=8))
    graphs, meta = build_eu_email_graphs_from_edges(str(txt), min_active_nodes=2)
    for g in graphs:
        assert g.x.shape[1] == 5, f"expected in_dim=5, got {g.x.shape[1]}"


def test_edge_index_valid(tmp_path):
    txt = tmp_path / "edges.txt"
    txt.write_text(make_fake_edge_lines(n_nodes=8))
    graphs, meta = build_eu_email_graphs_from_edges(str(txt), min_active_nodes=2)
    for g in graphs:
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.max() < g.x.shape[0]


def test_meta_has_split_ranges(tmp_path):
    txt = tmp_path / "edges.txt"
    txt.write_text(make_fake_edge_lines(n_nodes=8, n_weeks=20))
    graphs, meta = build_eu_email_graphs_from_edges(str(txt), min_active_nodes=2)
    assert "train_range" in meta
    assert "val_range" in meta
    assert "test_range" in meta
    assert meta["train_range"][0] == 0
    assert meta["test_range"][1] == meta["n_snapshots"] - 1


def test_node_ids_present(tmp_path):
    txt = tmp_path / "edges.txt"
    txt.write_text(make_fake_edge_lines(n_nodes=8))
    graphs, meta = build_eu_email_graphs_from_edges(str(txt), min_active_nodes=2)
    for g in graphs:
        assert hasattr(g, "node_ids")
        assert g.node_ids.shape[0] == g.x.shape[0]
