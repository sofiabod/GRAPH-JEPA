import torch.nn as nn


class SequentialMLP(nn.Module):
    """param-matched mlp for sequential ablation (no message passing).

    uses data.x only; ignores data.edge_index.
    architecture: input_proj + n_layers of linear+norm+relu, matching
    approximate parameter count of GraphEncoder with the same hidden_dim.
    """

    def __init__(self, in_dim=384, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        # ignores edge_index: no message passing
        h = self.input_proj(data.x)
        h = self.layers(h)
        return h
