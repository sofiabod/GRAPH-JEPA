import torch
from torch.utils.data import Dataset

# spec defaults: train weeks 0-119, val 120-139, test 140-179
_DEFAULT_TRAIN = (0, 119)
_DEFAULT_VAL = (120, 139)
_DEFAULT_TEST = (140, 179)


class TemporalGraphDataset(Dataset):
    def __init__(self, graphs, context_k=4, mask_ratio=0.20, split='train',
                 train_range=_DEFAULT_TRAIN, val_range=_DEFAULT_VAL, test_range=_DEFAULT_TEST):
        self.context_k = context_k
        self.mask_ratio = mask_ratio
        all_indices = list(range(len(graphs)))
        if split == 'train':
            lo, hi = train_range
            self.valid_target_indices = [i for i in all_indices if context_k <= i <= hi and i >= lo]
        elif split == 'val':
            lo, hi = val_range
            self.valid_target_indices = [i for i in all_indices if lo <= i <= hi]
        elif split == 'test':
            lo, hi = test_range
            self.valid_target_indices = [i for i in all_indices if lo <= i <= hi]
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
