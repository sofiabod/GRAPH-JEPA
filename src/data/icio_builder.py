"""builder for wiod 2016 release inter-country input-output tables.

source: world input-output database (wiod) 2016 release, hosted at the
groningen growth and development centre. long-format rds files mirrored at
the wiener institut fuer internationale wirtschaftsvergleiche (wiiw):
https://wiiw.ac.at/files/staff-content/reiter/WIOT{year}_October16_ROW_long.rds

we use wiod 2016 as a stand-in for the oecd icio database since both have the
same country x industry x year inter-country input-output structure. wiod is
released openly while oecd icio is gated behind a cloudflare-protected portal.

shape: same as TGBN-Trade and BACI-gravity. one snapshot per year, fixed
country node set. node features = [1d normalized total trade volume, 5d
structural] = 6d.

each row in the long-format file is:
    IndustryCode (origin sector), IndustryDescription, Country (origin iso3),
    RNr, Year, Partner (destination iso3 + sector index), value

we aggregate over origin industry and destination sector to produce a single
directed bilateral flow per (origin_country, destination_country, year).
"""
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


# industry codes that are not real production sectors in wiod 2016 long format.
# they encode taxes, value added, gross output, etc., not bilateral trade flows.
_NON_INDUSTRY_CODES = {
    "II_fob", "TXSP", "EXP_adj", "PURR", "PURNR", "VA", "IntTTM", "GO",
}


def _parse_partner_country(partner: str) -> str:
    """extract iso3 country code from a wiod partner string like 'USA12' or 'TOT'."""
    if not isinstance(partner, str) or len(partner) < 3:
        return ""
    head = partner[:3]
    if head.isalpha() and head.isupper():
        return head
    return ""


def build_icio_graphs(rds_dir: str, min_active_nodes: int = 30,
                      year_min: int = 2000, year_max: int = 2014):
    """parse wiod 2016 long-format rds files, build annual graph snapshots.

    args:
        rds_dir: directory containing WIOT{year}_long.rds files
        min_active_nodes: drop years with fewer active countries than this
        year_min, year_max: year range to keep (wiod 2016 covers 2000-2014)

    each snapshot has:
        x: [N, 6] (1d normalized incident trade volume, 5d structural)
        edge_index: [2, E]
        edge_attr: [E, 1] (bilateral trade volume)
        node_ids: arange(N)
    """
    import pyreadr

    rds_dir = Path(rds_dir)
    by_year_pair = defaultdict(lambda: defaultdict(float))  # year -> (src_iso, dst_iso) -> total_volume
    countries_seen = set()

    for year in range(year_min, year_max + 1):
        rds_path = rds_dir / f"WIOT{year}_long.rds"
        if not rds_path.exists():
            continue
        result = pyreadr.read_r(str(rds_path))
        df = result[None]

        # drop totals rows and non-industry rows (taxes, value added, etc.)
        df = df[df["Country"] != "TOT"]
        df = df[~df["IndustryCode"].isin(_NON_INDUSTRY_CODES)]
        df = df[df["Partner"] != "TOT"]

        # parse partner into country code
        df = df.copy()
        df["partner_country"] = df["Partner"].map(_parse_partner_country)
        df = df[df["partner_country"] != ""]

        # drop within-country flows (we want cross-border only)
        df = df[df["Country"] != df["partner_country"]]

        # clip negative adjustments to zero
        df = df[df["value"].notna()]
        vals = df["value"].clip(lower=0).to_numpy()
        origins = df["Country"].to_numpy()
        dests = df["partner_country"].to_numpy()

        # aggregate over industries and partner sectors
        for o, d, v in zip(origins, dests, vals):
            if v <= 0:
                continue
            by_year_pair[year][(o, d)] += float(v)
            countries_seen.add(o)
            countries_seen.add(d)

    # fixed node set across all years
    iso_sorted = sorted(countries_seen)
    iso2id = {iso: i for i, iso in enumerate(iso_sorted)}
    n_nodes = len(iso2id)

    graphs = []
    edge_count_per_snapshot = []
    active_count_per_snapshot = []
    years_kept = []
    for year in sorted(by_year_pair.keys()):
        pairs = by_year_pair[year]

        active = set()
        src_list = []
        dst_list = []
        vol_list = []
        for (iso_o, iso_d), vol in pairs.items():
            s = iso2id[iso_o]
            d = iso2id[iso_d]
            src_list.append(s)
            dst_list.append(d)
            vol_list.append(vol)
            active.add(s)
            active.add(d)

        if len(active) < min_active_nodes:
            continue

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        vol_tensor = torch.tensor(vol_list, dtype=torch.float)

        # per-node total trade volume (sum of incident edge values, both directions)
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
        edge_count_per_snapshot.append(edge_index.shape[1])
        active_count_per_snapshot.append(len(active))
        years_kept.append(year)

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "icio",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "years_kept": years_kept,
        "edge_counts": edge_count_per_snapshot,
        "active_counts": active_count_per_snapshot,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
        "iso_sorted": iso_sorted,
    }
    return graphs, meta


def build_icio_graphs_from_records(records, min_active_nodes: int = 2):
    """build graphs from an in-memory list of records.

    each record is a tuple (year, country, partner_country, value).
    used by the test suite to avoid needing rds fixtures.
    """
    by_year_pair = defaultdict(lambda: defaultdict(float))
    countries_seen = set()
    for year, country, partner, value in records:
        if country == partner:
            continue
        if value <= 0:
            continue
        by_year_pair[int(year)][(country, partner)] += float(value)
        countries_seen.add(country)
        countries_seen.add(partner)

    iso_sorted = sorted(countries_seen)
    iso2id = {iso: i for i, iso in enumerate(iso_sorted)}
    n_nodes = len(iso2id)

    graphs = []
    edge_count_per_snapshot = []
    active_count_per_snapshot = []
    years_kept = []
    for year in sorted(by_year_pair.keys()):
        pairs = by_year_pair[year]

        active = set()
        src_list = []
        dst_list = []
        vol_list = []
        for (iso_o, iso_d), vol in pairs.items():
            s = iso2id[iso_o]
            d = iso2id[iso_d]
            src_list.append(s)
            dst_list.append(d)
            vol_list.append(vol)
            active.add(s)
            active.add(d)

        if len(active) < min_active_nodes:
            continue

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        vol_tensor = torch.tensor(vol_list, dtype=torch.float)

        node_vol = torch.zeros(n_nodes)
        node_vol.scatter_add_(0, edge_index[0], vol_tensor)
        node_vol.scatter_add_(0, edge_index[1], vol_tensor)
        vol_max = node_vol.max().clamp(min=1e-8)
        node_vol_norm = (node_vol / vol_max).unsqueeze(1)

        structural = compute_structural_features(edge_index, n_nodes=n_nodes, edge_weights=vol_tensor)
        x = torch.cat([node_vol_norm, structural], dim=1)

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=vol_tensor.unsqueeze(1),
            node_ids=torch.arange(n_nodes),
        ))
        edge_count_per_snapshot.append(edge_index.shape[1])
        active_count_per_snapshot.append(len(active))
        years_kept.append(year)

    n = len(graphs)
    train_range, val_range, test_range = compute_split_ranges(n)
    meta = {
        "dataset": "icio",
        "n_nodes": n_nodes,
        "n_snapshots": n,
        "node_feature_dim": 6,
        "years_kept": years_kept,
        "edge_counts": edge_count_per_snapshot,
        "active_counts": active_count_per_snapshot,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "test_range": list(test_range),
        "iso_sorted": iso_sorted,
    }
    return graphs, meta
