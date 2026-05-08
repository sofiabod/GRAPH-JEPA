from collections import defaultdict

import torch
from torch_geometric.data import Data

from tgb.nodeproppred.dataset import NodePropPredDataset

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


def download_tgbn_trade(data_dir: str):
    """download tgbn-trade via the pip-installed py-tgb package."""
    dataset = NodePropPredDataset(name="tgbn-trade", root=data_dir)
    return dataset


def build_tgbn_trade_graphs_from_raw(records, country_ids):
    """build annual graph snapshots from trade records.

    args:
        records: list of (year, src_idx, dst_idx, trade_volume) tuples
        country_ids: list of country identifiers (used to fix n_nodes)

    node features: [1d normalized total trade volume, 5d structural] = 6d
    returns (graphs: list[PyG Data], meta: dict)
    """
    n_nodes = len(country_ids)

    by_year = defaultdict(list)
    for year, src, dst, vol in records:
        by_year[year].append((src, dst, float(vol)))

    graphs = []
    for year in sorted(by_year.keys()):
        year_edges = by_year[year]

        src_list = [s for s, d, v in year_edges]
        dst_list = [d for s, d, v in year_edges]
        volumes = [v for s, d, v in year_edges]

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        vol_tensor = torch.tensor(volumes, dtype=torch.float)

        # per-node total trade volume (sum of all edge volumes incident to node)
        node_vol = torch.zeros(n_nodes)
        node_vol.scatter_add_(0, edge_index[0], vol_tensor)
        node_vol.scatter_add_(0, edge_index[1], vol_tensor)
        vol_max = node_vol.max().clamp(min=1e-8)
        node_vol_norm = (node_vol / vol_max).unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=vol_tensor)
        x = torch.cat([node_vol_norm, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=vol_tensor.unsqueeze(1),
            node_ids=torch.arange(n_nodes),
        ))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "tgbn_trade",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "directed": True,
        "edge_attr": "trade_volume_usd",
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta


def build_tgbn_trade_graphs(data_dir: str):
    """full pipeline: download tgbn-trade via TGB, convert to annual snapshots."""
    dataset = NodePropPredDataset(name="tgbn-trade", root=data_dir)
    data = dataset.full_data

    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]
    edge_feats = data.get("edge_feat", None)  # may be None or [E, 1]

    all_nodes = sorted(set(sources.tolist()) | set(destinations.tolist()))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)

    # tgbn-trade timestamps are raw years stored as floats (e.g. 1986.0..2016.0),
    # not unix seconds. cast directly to int.
    records = []
    for i, (s, d, ts) in enumerate(zip(sources, destinations, timestamps)):
        year = int(float(ts))
        vol = float(edge_feats[i, 0]) if edge_feats is not None else 1.0
        records.append((year, node2id[int(s)], node2id[int(d)], vol))

    return build_tgbn_trade_graphs_from_raw(records, list(range(n_nodes)))


def build_tgbn_genre_graphs_from_raw(records, node_ids, min_active_nodes: int = 10):
    """build weekly graph snapshots from genre interaction records.

    args:
        records: list of (week, src_idx, dst_idx, edge_feat_value) tuples
        node_ids: list of node identifiers (used to fix n_nodes)
        min_active_nodes: drop weeks with fewer active nodes than this

    node features: [1d normalized total edge volume, 5d structural] = 6d
    returns (graphs: list[PyG Data], meta: dict)
    """
    n_nodes = len(node_ids)

    by_week = defaultdict(list)
    for week, src, dst, vol in records:
        by_week[week].append((src, dst, float(vol)))

    graphs = []
    for week in sorted(by_week.keys()):
        week_edges = by_week[week]

        active = set()
        for s, d, _ in week_edges:
            active.add(s)
            active.add(d)
        if len(active) < min_active_nodes:
            continue

        src_list = [s for s, d, v in week_edges]
        dst_list = [d for s, d, v in week_edges]
        volumes = [v for s, d, v in week_edges]

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        vol_tensor = torch.tensor(volumes, dtype=torch.float)

        # per-node total edge volume (sum of edge_feat values incident to each node)
        node_vol = torch.zeros(n_nodes)
        node_vol.scatter_add_(0, edge_index[0], vol_tensor)
        node_vol.scatter_add_(0, edge_index[1], vol_tensor)
        vol_max = node_vol.max().clamp(min=1e-8)
        node_vol_norm = (node_vol / vol_max).unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=vol_tensor)
        x = torch.cat([node_vol_norm, structural], dim=1)  # [N, 6]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=vol_tensor.unsqueeze(1),
            node_ids=torch.arange(n_nodes),
        ))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "tgbn_genre",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "directed": True,
        "edge_attr": "interaction_count",
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta


def build_tgbn_genre_graphs(data_dir: str):
    """full pipeline: download tgbn-genre via TGB, convert to weekly snapshots.

    timestamps are unix seconds; bucket into weeks of 604800 seconds starting
    from the first observed timestamp. matches the weekly aggregation pattern
    used in jodie_builder.py.
    """
    dataset = NodePropPredDataset(name="tgbn-genre", root=data_dir)
    data = dataset.full_data

    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]
    edge_feats = data.get("edge_feat", None)  # may be None or [E, 1]

    all_nodes = sorted(set(sources.tolist()) | set(destinations.tolist()))
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)

    week_sec = 60 * 60 * 24 * 7
    t0 = float(timestamps.min()) if hasattr(timestamps, "min") else float(min(timestamps))

    records = []
    for i in range(len(sources)):
        ts = float(timestamps[i])
        week = int((ts - t0) / week_sec)
        if edge_feats is not None:
            ef = edge_feats[i]
            # edge_feat may be 1d scalar or [1] vector; handle both
            try:
                vol = float(ef[0])
            except (TypeError, IndexError):
                vol = float(ef)
        else:
            vol = 1.0
        records.append((week, node2id[int(sources[i])], node2id[int(destinations[i])], vol))

    return build_tgbn_genre_graphs_from_raw(records, list(range(n_nodes)))


def build_tgbn_genre_v2_graphs_from_raw(
    records,
    node_ids,
    src_only_ids=None,
    dst_only_ids=None,
    active_week_threshold: float = 0.25,
    min_active_nodes: int = 10,
):
    """build weekly graph snapshots from genre interaction records, bipartite-aware.

    fixes vs v1:
      a) detect bipartite user/item structure, then symmetrize edges (add reverse
         edges) so homogeneous gatv2 sees both directions
      b) deduplicate multi-edges per week and use the count as edge weight
      c) filter to nodes active in at least active_week_threshold fraction of weeks
      d) extra node feature: bipartite type indicator (0=user/src, 1=item/dst)

    args:
        records: list of (week, src_idx, dst_idx, edge_feat_value) tuples
        node_ids: list of node identifiers spanning [0, N) raw id space
        src_only_ids: set of remapped ids that only appear as sources (users)
        dst_only_ids: set of remapped ids that only appear as destinations (items)
                      (passed in so we don't re-detect from filtered records)
        active_week_threshold: keep nodes active in >= this fraction of weeks
        min_active_nodes: drop weeks with fewer active nodes than this

    node features (7d): [1d normalized total volume, 1d node type, 5d structural]
    returns (graphs: list[PyG Data], meta: dict)

    KNOWN CAVEAT (2026-05-07 audit):
      after the active-week filter selects n_filtered globally-kept nodes, every
      snapshot uses x.shape[0] == n_filtered. nodes that are kept globally but
      inactive in a particular week have all-zero features in that week's snapshot
      (no incident edges → node_vol=0, structural features=0). when masking with
      mask_ratio=0.20, the mask may include semantically-inactive nodes whose
      target embedding is the trivial zero vector — degenerate but not statistically
      breaking (paired wilcoxon n_pairs is still n_filtered × test_snapshots × ratio).
      to address before publication: either restrict masking to active nodes per
      snapshot via a `mask_active_only` flag in TemporalGraphDataset, or add an
      eval-time filter discarding zero-feature mask positions. tgbn-trade is
      unaffected (every country has volume every year).
    """
    n_nodes_raw = len(node_ids)

    # detect bipartite structure if not supplied (used by tests with fake records)
    if src_only_ids is None or dst_only_ids is None:
        srcs_seen = set()
        dsts_seen = set()
        for _, s, d, _ in records:
            srcs_seen.add(s)
            dsts_seen.add(d)
        src_only_ids = srcs_seen - dsts_seen
        dst_only_ids = dsts_seen - srcs_seen

    by_week = defaultdict(list)
    for week, src, dst, vol in records:
        by_week[week].append((src, dst, float(vol)))

    weeks_sorted = sorted(by_week.keys())
    n_weeks_total = len(weeks_sorted)

    # count weeks each raw node appears in
    weeks_active = defaultdict(set)
    for week, src, dst, _ in records:
        weeks_active[src].add(week)
        weeks_active[dst].add(week)

    min_weeks = max(1, int(active_week_threshold * n_weeks_total))
    kept_raw = sorted([n for n, ws in weeks_active.items() if len(ws) >= min_weeks])

    # remap kept raw ids to 0..N_filtered-1
    raw2new = {raw: new for new, raw in enumerate(kept_raw)}
    n_filtered = len(kept_raw)

    # node type: 1 if raw id was an item (destination only), else 0 (user/src or mixed)
    node_type = torch.zeros(n_filtered)
    for raw, new in raw2new.items():
        if raw in dst_only_ids:
            node_type[new] = 1.0

    graphs = []
    edge_count_per_snapshot = []
    active_count_per_snapshot = []
    for week in weeks_sorted:
        week_edges = by_week[week]

        # dedupe (src, dst) -> total count weight, after filtering to kept nodes
        pair_to_count = defaultdict(float)
        for s, d, _ in week_edges:
            if s in raw2new and d in raw2new:
                ns = raw2new[s]
                nd = raw2new[d]
                pair_to_count[(ns, nd)] += 1.0

        if len(pair_to_count) == 0:
            continue

        # symmetrize: add reverse edges, accumulate counts on already-existing pairs
        symm = defaultdict(float)
        for (s, d), c in pair_to_count.items():
            symm[(s, d)] += c
            symm[(d, s)] += c

        src_list = []
        dst_list = []
        weight_list = []
        for (s, d), w in symm.items():
            src_list.append(s)
            dst_list.append(d)
            weight_list.append(w)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        weight_tensor = torch.tensor(weight_list, dtype=torch.float)

        active = set(src_list) | set(dst_list)
        if len(active) < min_active_nodes:
            continue

        # per-node total weighted volume
        node_vol = torch.zeros(n_filtered)
        node_vol.scatter_add_(0, edge_index[0], weight_tensor)
        node_vol.scatter_add_(0, edge_index[1], weight_tensor)
        vol_max = node_vol.max().clamp(min=1e-8)
        node_vol_norm = (node_vol / vol_max).unsqueeze(1)  # [N, 1]

        type_feat = node_type.unsqueeze(1)  # [N, 1]

        structural = compute_structural_features(
            edge_index, n_nodes=n_filtered, edge_weights=weight_tensor
        )
        x = torch.cat([node_vol_norm, type_feat, structural], dim=1)  # [N, 7]

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=weight_tensor.unsqueeze(1),
            node_ids=torch.arange(n_filtered),
        ))
        edge_count_per_snapshot.append(edge_index.shape[1])
        active_count_per_snapshot.append(len(active))

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "tgbn_genre_v2",
        "n_nodes": n_filtered,
        "n_nodes_raw": n_nodes_raw,
        "n_snapshots": n,
        "n_weeks_total": n_weeks_total,
        "min_weeks_active": min_weeks,
        "active_week_threshold": active_week_threshold,
        "node_feature_dim": 7,
        "n_users": int((node_type == 0).sum().item()),
        "n_items": int((node_type == 1).sum().item()),
        "edge_counts": edge_count_per_snapshot,
        "active_counts": active_count_per_snapshot,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
    }
    return graphs, meta


def build_tgbn_genre_v2_graphs(data_dir: str):
    """full pipeline: download tgbn-genre via TGB, build bipartite-aware weekly snapshots.

    differences vs build_tgbn_genre_graphs:
      - detects bipartite source/destination structure
      - adds reverse edges so message passing flows both ways
      - deduplicates multi-edges into edge weights (count of listening events)
      - filters out nodes active in < 25% of weeks
      - adds a node type indicator feature
    """
    dataset = NodePropPredDataset(name="tgbn-genre", root=data_dir)
    data = dataset.full_data

    sources = data["sources"]
    destinations = data["destinations"]
    timestamps = data["timestamps"]

    src_set = set(int(s) for s in sources.tolist())
    dst_set = set(int(d) for d in destinations.tolist())
    all_nodes = sorted(src_set | dst_set)
    node2id = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2id)

    # bipartite detection in raw id space, then translate to remapped ids
    raw_src_only = src_set - dst_set
    raw_dst_only = dst_set - src_set
    src_only_ids = {node2id[n] for n in raw_src_only}
    dst_only_ids = {node2id[n] for n in raw_dst_only}

    week_sec = 60 * 60 * 24 * 7
    t0 = float(timestamps.min()) if hasattr(timestamps, "min") else float(min(timestamps))

    records = []
    for i in range(len(sources)):
        ts = float(timestamps[i])
        week = int((ts - t0) / week_sec)
        # edge_feat ignored: v2 uses raw event count via dedup
        records.append((week, node2id[int(sources[i])], node2id[int(destinations[i])], 1.0))

    return build_tgbn_genre_v2_graphs_from_raw(
        records,
        list(range(n_nodes)),
        src_only_ids=src_only_ids,
        dst_only_ids=dst_only_ids,
    )
