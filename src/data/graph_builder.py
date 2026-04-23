import pickle
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data


def build_weekly_graphs(emails, top_n=50, min_active=10, embed_model=None, cache_path=None):
    # node features x: [N, 389]
    #   bge mean-pool of outgoing emails [384]
    #   normalized out-degree [1]
    #   normalized in-degree [1]
    #   normalized weighted out-degree [1]
    #   normalized weighted in-degree [1]
    #   active flag: 1 if person sent >= 1 email this week [1]
    #
    # edge features edge_attr: [E, 2]
    #   normalized email count [1]
    #   recurring flag: 1 if edge also present in previous week [1]

    weekly_edges = defaultdict(lambda: defaultdict(int))
    weekly_texts = defaultdict(lambda: defaultdict(list))

    for e in emails:
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(e["date_str"])
            week_key = dt.isocalendar()[:2]
        except Exception:
            continue
        sender = e["sender"]
        for r in e["recipients"]:
            weekly_edges[week_key][(sender, r)] += 1
        if e["body"]:
            weekly_texts[week_key][sender].append(e["body"][:512])

    # top n people by total email volume
    person_counts = defaultdict(int)
    for week_data in weekly_edges.values():
        for (s, r), cnt in week_data.items():
            person_counts[s] += cnt
            person_counts[r] += cnt
    top_people = sorted(person_counts, key=lambda p: -person_counts[p])[:top_n]
    person_to_idx = {p: i for i, p in enumerate(top_people)}
    n_people = len(top_people)

    bge_embs = _compute_bge_embeddings(weekly_texts, top_people, embed_model, cache_path)

    all_weeks = sorted(weekly_edges.keys())
    graphs = []
    meta_weeks = []
    prev_edge_keys = set()
    week_seq = 0

    for week_key in all_weeks:
        edges = weekly_edges[week_key]

        active_senders = {s for (s, r) in edges if s in person_to_idx}
        active = {p for pair in edges for p in pair if p in person_to_idx}

        if len(active) < min_active:
            prev_edge_keys = set(edges.keys())
            continue

        # structural features per node
        out_deg = defaultdict(float)
        in_deg = defaultdict(float)
        out_w = defaultdict(float)
        in_w = defaultdict(float)

        for (s, r), cnt in edges.items():
            if s in person_to_idx:
                out_deg[s] += 1
                out_w[s] += cnt
            if r in person_to_idx:
                in_deg[r] += 1
                in_w[r] += cnt

        max_out_deg = max(out_deg.values(), default=1.0)
        max_in_deg = max(in_deg.values(), default=1.0)
        max_out_w = max(out_w.values(), default=1.0)
        max_in_w = max(in_w.values(), default=1.0)

        # build x [N, 389]
        x = torch.zeros(n_people, 389)
        week_bge = bge_embs.get(week_key, {})

        for person, idx in person_to_idx.items():
            if person in week_bge:
                x[idx, :384] = torch.tensor(week_bge[person], dtype=torch.float)
            x[idx, 384] = out_deg[person] / max_out_deg
            x[idx, 385] = in_deg[person] / max_in_deg
            x[idx, 386] = out_w[person] / max_out_w
            x[idx, 387] = in_w[person] / max_in_w
            x[idx, 388] = 1.0 if person in active_senders else 0.0

        # build edges [E, 2]
        src_list, dst_list, weight_list, recur_list = [], [], [], []
        week_max_cnt = max(edges.values(), default=1.0)

        for (s, r), cnt in edges.items():
            if s in person_to_idx and r in person_to_idx:
                src_list.append(person_to_idx[s])
                dst_list.append(person_to_idx[r])
                weight_list.append(cnt / week_max_cnt)
                recur_list.append(1.0 if (s, r) in prev_edge_keys else 0.0)

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(
                list(zip(weight_list, recur_list)), dtype=torch.float
            )
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 2)

        g = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=torch.arange(n_people),
            week_idx=week_seq,
        )
        graphs.append(g)
        meta_weeks.append({
            "week_idx": week_seq,
            "year": week_key[0],
            "week": week_key[1],
            "date_str": f"{week_key[0]}-W{week_key[1]:02d}",
            "n_active": len(active),
            "n_edges": len(src_list),
            "email_volume": int(sum(edges.values())),
        })

        prev_edge_keys = set(edges.keys())
        week_seq += 1

    meta = {
        "person_index": {p: i for p, i in person_to_idx.items()},
        "n_people": n_people,
        "n_weeks": len(graphs),
        "top_n": top_n,
        "min_active": min_active,
        "node_feature_dim": 389,
        "edge_feature_dim": 2,
        "weeks": meta_weeks,
    }
    return graphs, meta


def _compute_bge_embeddings(weekly_texts, top_people, embed_model=None, cache_path=None):
    # returns dict[(year, week)][person] -> list[float] length 384
    # loads from cache if available, otherwise embeds and caches
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    top_set = set(top_people)
    all_texts = []
    text_to_idx = {}

    for person_texts in weekly_texts.values():
        for person, texts in person_texts.items():
            if person not in top_set:
                continue
            for t in texts:
                if t not in text_to_idx:
                    text_to_idx[t] = len(all_texts)
                    all_texts.append(t)

    if not all_texts:
        return {}

    print(f"embedding {len(all_texts)} unique email bodies...")
    embeddings = embed_model.encode(all_texts, batch_size=256, show_progress_bar=True)

    result = {}
    for week_key, person_texts in weekly_texts.items():
        result[week_key] = {}
        for person, texts in person_texts.items():
            if person not in top_set:
                continue
            idxs = [text_to_idx[t] for t in texts if t in text_to_idx]
            if not idxs:
                continue
            result[week_key][person] = embeddings[idxs].mean(axis=0).tolist()

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result
