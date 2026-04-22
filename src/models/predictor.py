import torch
import torch.nn as nn


class TemporalGraphPredictor(nn.Module):
    def __init__(self, embed_dim=256, n_heads=4, n_layers=2, mlp_ratio=2, dropout=0.1,
                 n_nodes=50, max_time_steps=200, temporal_stride=1):
        super().__init__()
        self.node_id_emb = nn.Embedding(n_nodes, embed_dim)
        self.temporal_pos_emb = nn.Embedding(max_time_steps, embed_dim)
        self.temporal_stride = temporal_stride
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])

    def forward(self, tokens, time_indices, node_ids):
        # tokens: [B, T, D], time_indices: [B, T], node_ids: [B, T]
        x = tokens + self.temporal_pos_emb(time_indices) + self.node_id_emb(node_ids)
        for layer in self.layers:
            x = layer(x)
        return x
