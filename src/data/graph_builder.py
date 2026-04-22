import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric.data import Data


def build_weekly_graphs(emails, top_n=50, min_active=10):
    # groups emails by iso week, builds pyg data objects
    # returns (graphs, meta) where graphs is list of Data ordered by week

    # parse timestamps and group
    weekly_edges = defaultdict(lambda: defaultdict(int))
    weekly_texts = defaultdict(lambda: defaultdict(list))

    for e in emails:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(e["date_str"])
            week_key = dt.isocalendar()[:2]  # (year, week)
        except Exception:
            continue
        sender = e["sender"]
        for r in e["recipients"]:
            weekly_edges[week_key][(sender, r)] += 1
        if e["body"]:
            weekly_texts[week_key][sender].append(e["body"][:512])

    # find top n people by total email volume
    person_counts = defaultdict(int)
    for week_data in weekly_edges.values():
        for (s, r), cnt in week_data.items():
            person_counts[s] += cnt
            person_counts[r] += cnt
    top_people = sorted(person_counts, key=lambda p: -person_counts[p])[:top_n]
    person_to_idx = {p: i for i, p in enumerate(top_people)}
    n_people = len(top_people)

    # sort weeks
    all_weeks = sorted(weekly_edges.keys())
    graphs = []
    meta_weeks = []

    for week_key in all_weeks:
        edges = weekly_edges[week_key]
        # count active nodes
        active = set()
        for (s, r) in edges:
            if s in person_to_idx:
                active.add(s)
            if r in person_to_idx:
                active.add(r)
        if len(active) < min_active:
            continue

        # node features: zeros for now (bge embedding happens in separate step)
        x = torch.zeros(n_people, 384)

        # build edge_index and edge_attr
        src_list, dst_list, weight_list = [], [], []
        for (s, r), cnt in edges.items():
            if s in person_to_idx and r in person_to_idx:
                src_list.append(person_to_idx[s])
                dst_list.append(person_to_idx[r])
                weight_list.append(float(cnt))

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(weight_list, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 1)

        node_ids = torch.arange(n_people)
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, node_ids=node_ids)
        graphs.append(g)
        meta_weeks.append({"year": week_key[0], "week": week_key[1], "n_active": len(active)})

    meta = {
        "person_index": {p: i for p, i in person_to_idx.items()},
        "n_people": n_people,
        "n_weeks": len(graphs),
        "top_n": top_n,
        "min_active": min_active,
        "weeks": meta_weeks,
    }
    return graphs, meta
