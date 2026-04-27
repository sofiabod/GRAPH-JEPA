from torch_geometric.nn import GATv2Conv
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=256, n_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout, concat=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # project input features then apply gat layers with residual connections
        h = self.input_proj(data.x)
        for layer, norm in zip(self.layers, self.norms):
            h_new = norm(layer(h, data.edge_index))
            if h_new.shape == h.shape:
                h = h + self.dropout(h_new)
            else:
                h = h_new
        return F.normalize(h, dim=-1)
