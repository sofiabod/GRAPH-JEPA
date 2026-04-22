import pytest
import torch
from torch_geometric.data import Data

from src.data.dataset import TemporalGraphDataset


def make_fake_graphs(n=200, n_nodes=10):
    """build n fake pyg graphs with n_nodes nodes each"""
    graphs = []
    for _ in range(n):
        graphs.append(Data(
            x=torch.randn(n_nodes, 384),
            edge_index=torch.zeros(2, 5, dtype=torch.long),
            node_ids=torch.arange(n_nodes),
        ))
    return graphs


@pytest.fixture
def fake_dataset():
    graphs = make_fake_graphs(200)
    return TemporalGraphDataset(graphs, context_k=4, mask_ratio=0.20, split='train')


def test_masked_node_ids_not_in_visible(fake_dataset):
    sample = fake_dataset[0]
    masked = set(sample['masked_node_ids'].tolist())
    visible = set(sample['visible_node_ids'].tolist())
    overlap = masked & visible
    assert len(overlap) == 0, f"masked and visible should not overlap, found {overlap}"


def test_masked_and_visible_cover_all_nodes(fake_dataset):
    sample = fake_dataset[0]
    masked = sample['masked_node_ids'].tolist()
    visible = sample['visible_node_ids'].tolist()
    all_ids = sorted(masked + visible)
    assert all_ids == list(range(10)), \
        f"masked + visible should cover all 10 nodes, got {all_ids}"


def test_both_encoders_see_full_graph(fake_dataset):
    sample = fake_dataset[0]
    # context graphs and target graph each contain all nodes (no graph-level masking)
    for g in sample['context_graphs']:
        assert g.x.shape[0] == 10, \
            f"context graph should have 10 nodes, got {g.x.shape[0]}"
    assert sample['target_graph'].x.shape[0] == 10, \
        f"target graph should have 10 nodes, got {sample['target_graph'].x.shape[0]}"


def test_masking_ratio_approximate(fake_dataset):
    # check across multiple samples that ~20% are masked, with tolerance
    ratios = []
    for i in range(20):
        sample = fake_dataset[i]
        ratio = len(sample['masked_node_ids']) / 10.0
        ratios.append(ratio)
    avg_ratio = sum(ratios) / len(ratios)
    assert 0.10 <= avg_ratio <= 0.40, \
        f"average masking ratio {avg_ratio:.2f} outside expected [0.10, 0.40]"


def test_context_window_length(fake_dataset):
    sample = fake_dataset[0]
    assert len(sample['context_graphs']) == 4, \
        f"context_graphs should have 4 entries, got {len(sample['context_graphs'])}"
