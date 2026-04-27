import torch
from src.data.graph_utils import compute_structural_features


def test_output_shape():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    feats = compute_structural_features(edge_index, n_nodes=3)
    assert feats.shape == (3, 5), f"expected [3, 5], got {feats.shape}"


def test_active_flag_set_for_connected_nodes():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    feats = compute_structural_features(edge_index, n_nodes=3)
    assert feats[0, 4] == 1.0, "node 0 is source, should be active"
    assert feats[1, 4] == 1.0, "node 1 is dest, should be active"
    assert feats[2, 4] == 0.0, "node 2 has no edges, should be inactive"


def test_inactive_node_has_zero_degree_features():
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    feats = compute_structural_features(edge_index, n_nodes=3)
    assert feats[2, 0] == 0.0
    assert feats[2, 1] == 0.0
    assert feats[2, 2] == 0.0
    assert feats[2, 3] == 0.0


def test_normalized_columns_max_one():
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    weights = torch.tensor([3.0, 1.0, 2.0])
    feats = compute_structural_features(edge_index, n_nodes=3, edge_weights=weights)
    # cols 0-3 should be in [0, 1]
    assert feats[:, :4].max() <= 1.0 + 1e-6
    assert feats[:, :4].min() >= 0.0


def test_empty_graph_all_zeros():
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    feats = compute_structural_features(edge_index, n_nodes=4)
    assert feats.shape == (4, 5)
    assert feats.sum() == 0.0


def test_weighted_out_degree_correct():
    # node 0 sends 2 edges with weights 3 and 1 -> weighted out = 4
    # node 1 sends 1 edge with weight 2 -> weighted out = 2
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    weights = torch.tensor([3.0, 1.0, 2.0])
    feats = compute_structural_features(edge_index, n_nodes=3, edge_weights=weights)
    # col 2 = weighted out-degree, normalized by max(4)=1.0 for node 0
    assert abs(feats[0, 2].item() - 1.0) < 1e-5
    assert abs(feats[1, 2].item() - 0.5) < 1e-5
