import torch.nn as nn
import torch.nn.functional as F


class _FFNBlock(nn.Module):
    """two-linear residual ffn block. mirrors gatv2's per-layer lin_l + lin_r structure."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h_pre = h
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = self.dropout(h)
        return self.norm(h + h_pre)


class SequentialMLP(nn.Module):
    """capacity-matched mlp for sequential ablation (no message passing).

    each block has two linear layers in residual ffn form to mirror gatv2's
    per-layer parameter count (lin_l + lin_r). this fixes the capacity mismatch
    flagged in the 2026-05-07 audit (single-linear sequentialmlp was ~200k while
    gatv2 graphencoder is ~400k, biasing the ablation by 49.8%).

    param count for in_dim=384, hidden_dim=256:
      - n_layers=2: ~363k (within -9% of graphencoder ~400k)
      - n_layers=3: ~495k (~+24%; sequential at least as large as graph)

    uses data.x only; ignores data.edge_index by design.
    output is l2-normalized to match graphencoder's sphere geometry.
    """

    def __init__(self, in_dim=384, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([_FFNBlock(hidden_dim, dropout) for _ in range(n_layers)])

    def forward(self, data):
        # ignores edge_index: no message passing
        h = self.input_proj(data.x)
        for block in self.blocks:
            h = block(h)
        return F.normalize(h, dim=-1)
