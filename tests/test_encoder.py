import torch
import torch.nn as nn
from types import SimpleNamespace
from torch_geometric.data import Data

from src.models.graph_encoder import GraphEncoder
from src.builders import build_graph_encoder


def test_output_shape(tiny_graph):
    encoder = GraphEncoder()
    out = encoder(tiny_graph)
    assert out.shape == (10, 256), f"expected [10, 256], got {out.shape}"


def test_trainable_params():
    encoder = GraphEncoder()
    assert all(p.requires_grad for p in encoder.parameters()), \
        "all online encoder params should have requires_grad=True"


def test_layernorm_present():
    encoder = GraphEncoder()
    assert isinstance(encoder.norms, nn.ModuleList), "encoder.norms must be nn.ModuleList"
    assert all(isinstance(n, nn.LayerNorm) for n in encoder.norms), \
        "all entries in encoder.norms must be nn.LayerNorm"


def test_input_proj_present():
    encoder = GraphEncoder(in_dim=384, hidden_dim=256)
    assert hasattr(encoder, 'input_proj'), "encoder must have input_proj attribute"
    assert isinstance(encoder.input_proj, nn.Linear), "input_proj must be nn.Linear"
    assert encoder.input_proj.in_features == 384
    assert encoder.input_proj.out_features == 256


def test_different_inputs_different_outputs(tiny_graph):
    encoder = GraphEncoder()
    # build a second graph with very different features
    other_graph = Data(
        x=tiny_graph.x + 100.0,
        edge_index=tiny_graph.edge_index.clone(),
        node_ids=tiny_graph.node_ids.clone(),
    )
    out1 = encoder(tiny_graph)
    out2 = encoder(other_graph)
    assert not torch.allclose(out1, out2), "different inputs should yield different embeddings"


def test_builder_returns_graph_encoder():
    cfg = SimpleNamespace(in_dim=384, hidden_dim=256, n_layers=3, n_heads=4, dropout=0.1)
    encoder = build_graph_encoder(cfg)
    assert isinstance(encoder, GraphEncoder)
