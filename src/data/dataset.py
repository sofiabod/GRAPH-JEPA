import torch
from torch.utils.data import Dataset

TRAIN_END = 119
VAL_START = 120
VAL_END = 139
TEST_START = 140
TEST_END = 179


class TemporalGraphDataset(Dataset):
    def __init__(self, graphs, context_k=4, mask_ratio=0.20, split='train'):
        self.context_k = context_k
        self.mask_ratio = mask_ratio
        # filter graph indices by split
        all_indices = list(range(len(graphs)))
        if split == 'train':
            self.valid_target_indices = [i for i in all_indices if context_k <= i <= TRAIN_END]
        elif split == 'val':
            self.valid_target_indices = [i for i in all_indices if VAL_START <= i <= VAL_END]
        elif split == 'test':
            self.valid_target_indices = [i for i in all_indices if TEST_START <= i <= TEST_END]
        self.graphs = graphs

    def __len__(self):
        return len(self.valid_target_indices)

    def __getitem__(self, idx):
        target_idx = self.valid_target_indices[idx]
        context_start = target_idx - self.context_k
        context_graphs = self.graphs[context_start:target_idx]
        target_graph = self.graphs[target_idx]

        n_nodes = target_graph.x.shape[0]
        n_mask = max(1, round(n_nodes * self.mask_ratio))
        perm = torch.randperm(n_nodes)
        masked_node_ids = perm[:n_mask]
        visible_node_ids = perm[n_mask:]

        return {
            'context_graphs': context_graphs,
            'target_graph': target_graph,
            'masked_node_ids': masked_node_ids,
            'visible_node_ids': visible_node_ids,
            'week_idx': target_idx,
        }
