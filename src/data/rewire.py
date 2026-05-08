"""degree-preserving edge rewiring for the random-edge control (T2.2).

each snapshot's edge_index is randomly rewired while preserving the degree
sequence. used to test whether real graph topology is load-bearing for graph-jepa
or whether any-graph-with-similar-degree-distribution suffices.
"""
import torch
from torch_geometric.data import Data


def _double_edge_swap(edge_index, n_swaps, generator):
    """networkx-style double edge swap; preserves degree sequence.

    edge_index: [2, E] tensor (long)
    returns rewired edge_index of the same shape.
    """
    src = edge_index[0].clone()
    dst = edge_index[1].clone()
    E = src.shape[0]
    if E < 2:
        return edge_index

    attempts = 0
    swaps = 0
    max_attempts = 10 * n_swaps

    while swaps < n_swaps and attempts < max_attempts:
        attempts += 1
        i = int(torch.randint(0, E, (1,), generator=generator).item())
        j = int(torch.randint(0, E, (1,), generator=generator).item())
        if i == j:
            continue
        u, v = src[i].item(), dst[i].item()
        x, y = src[j].item(), dst[j].item()
        # avoid self-loops post-swap
        if u == y or x == v or u == x or v == y:
            continue
        # swap: (u, v), (x, y) -> (u, y), (x, v)
        src[i], dst[i] = u, y
        src[j], dst[j] = x, v
        swaps += 1

    return torch.stack([src, dst], dim=0)


def rewire_snapshot(data: Data, seed: int, swap_factor: float = 1.0) -> Data:
    """produce a new pyg Data with rewired edges. node features unchanged.

    args:
        data: pyg Data with edge_index, x, optional edge_attr
        seed: rng seed for this snapshot
        swap_factor: number of swap attempts as a fraction of edge count
    """
    g = torch.Generator()
    g.manual_seed(seed)
    n_swaps = max(1, int(swap_factor * data.edge_index.shape[1]))
    new_edge_index = _double_edge_swap(data.edge_index, n_swaps, generator=g)

    new_data = Data(
        x=data.x,
        edge_index=new_edge_index,
        node_ids=data.node_ids if hasattr(data, "node_ids") else None,
    )
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        # edge attrs are correlated with original edges; for the random-edge control
        # we drop them rather than carry stale values to the new edges.
        pass
    return new_data


def rewire_graph_sequence(graphs, seed: int, swap_factor: float = 1.0):
    """apply degree-preserving rewire to every snapshot.

    each snapshot gets a deterministic rewire seed derived from (seed, snapshot_idx)
    so the same seed produces the same rewired sequence across runs.
    """
    rewired = []
    for i, g in enumerate(graphs):
        snap_seed = seed * 100003 + i
        rewired.append(rewire_snapshot(g, seed=snap_seed, swap_factor=swap_factor))
    return rewired
