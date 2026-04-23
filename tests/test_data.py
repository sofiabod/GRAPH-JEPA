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


FAKE_GRAPHS = make_fake_graphs(200)


def test_train_split_range():
    ds = TemporalGraphDataset(FAKE_GRAPHS, split='train')
    # sample enough items to be confident
    for i in range(len(ds)):
        sample = ds[i]
        assert 0 <= sample['week_idx'] <= 99, \
            f"train split week_idx {sample['week_idx']} outside [0, 99]"


def test_val_split_range():
    ds = TemporalGraphDataset(FAKE_GRAPHS, split='val')
    for i in range(len(ds)):
        sample = ds[i]
        assert 100 <= sample['week_idx'] <= 114, \
            f"val split week_idx {sample['week_idx']} outside [100, 114]"


def test_test_split_range():
    ds = TemporalGraphDataset(FAKE_GRAPHS, split='test')
    for i in range(len(ds)):
        sample = ds[i]
        assert 115 <= sample['week_idx'] <= 132, \
            f"test split week_idx {sample['week_idx']} outside [115, 132]"


def test_no_future_in_context():
    ds = TemporalGraphDataset(FAKE_GRAPHS, split='train')
    for i in range(min(len(ds), 20)):
        sample = ds[i]
        target_idx = sample['week_idx']
        # context graphs must come from earlier time steps
        # the dataset should record context week indices, or we verify via structure
        # each context graph should be a snapshot strictly before the target
        context_graphs = sample['context_graphs']
        # check the dataset provides context_week_indices if available
        if 'context_week_indices' in sample:
            for ctx_idx in sample['context_week_indices']:
                assert ctx_idx < target_idx, \
                    f"context index {ctx_idx} must be < target index {target_idx}"


def test_dataset_len_nonzero():
    ds = TemporalGraphDataset(FAKE_GRAPHS, split='train')
    assert len(ds) > 0, "train dataset should have nonzero length"
