import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.head(x)
