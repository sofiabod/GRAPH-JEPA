"""builder for cepii gravity bilateral trade flows.

source: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=8
file: Gravity_V202211.csv (inside Gravity_csv_V202211.zip)

shape: same as TGBN-Trade. one snapshot per year, fixed country node set,
node features = [1d normalized total trade volume, 5d structural] = 6d.

key column: tradeflow_baci (bilateral trade value from BACI, in current usd thousands)
"""
from collections import defaultdict
from pathlib import Path
import csv
import zipfile

import torch
from torch_geometric.data import Data

from src.data.graph_utils import compute_structural_features
from src.data.factory import compute_split_ranges


def build_baci_gravity_graphs(zip_path: str, min_active_nodes: int = 30,
                              year_min: int = 1995, year_max: int = 2020):
    """parse cepii gravity csv inside the zip, build annual graph snapshots.

    args:
        zip_path: path to Gravity_csv_V202211.zip
        min_active_nodes: drop years with fewer active countries than this
        year_min, year_max: year range to keep (default 1995-2020 to match BACI coverage
            with reasonable sample sizes per year)

    each snapshot has:
        x: [N, 6] (1d normalized incident trade volume, 5d structural)
        edge_index: [2, E]
        edge_attr: [E, 1] (bilateral trade volume)
        node_ids: arange(N)
    """
    # streaming parse: read only the columns we need
    by_year_pair = defaultdict(lambda: defaultdict(float))  # year -> (src_iso, dst_iso) -> total_volume
    countries_seen = set()

    with zipfile.ZipFile(zip_path) as z:
        with z.open("Gravity_V202211.csv") as f:
            text = (line.decode("utf-8") for line in f)
            reader = csv.DictReader(text)
            for i, row in enumerate(reader):
                try:
                    year = int(row["year"])
                except (ValueError, TypeError):
                    continue
                if year < year_min or year > year_max:
                    continue
                iso_o = row["iso3_o"].strip().strip('"')
                iso_d = row["iso3_d"].strip().strip('"')
                if not iso_o or not iso_d or iso_o == iso_d:
                    continue
                tf = row.get("tradeflow_baci", "")
                if tf == "" or tf == "NA":
                    continue
                try:
                    vol = float(tf)
                except ValueError:
                    continue
                if vol <= 0:
                    continue
                by_year_pair[year][(iso_o, iso_d)] += vol
                countries_seen.add(iso_o)
                countries_seen.add(iso_d)

    # fixed node set: all countries that appeared as origin or destination at least once in range
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

        # per-node total trade volume (sum of all incident edge values)
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
        "dataset": "baci_gravity",
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
