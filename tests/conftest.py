import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def tiny_graph():
    # 10 nodes, 15 edges, node features shape [10, 384]
    n = 10
    x = torch.randn(n, 384)
    edge_index = torch.randint(0, n, (2, 15))
    node_ids = torch.arange(n)
    return Data(x=x, edge_index=edge_index, node_ids=node_ids)


@pytest.fixture
def temporal_sequence(tiny_graph):
    # 8 snapshots with slight perturbation
    graphs = []
    for _ in range(8):
        g = Data(
            x=tiny_graph.x + 0.01 * torch.randn_like(tiny_graph.x),
            edge_index=tiny_graph.edge_index.clone(),
            node_ids=tiny_graph.node_ids.clone(),
        )
        graphs.append(g)
    return graphs
