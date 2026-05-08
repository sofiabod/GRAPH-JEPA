"""download and preprocess all non-enron benchmark datasets.

usage:
    python experiments/download_benchmarks.py --dataset all
    python experiments/download_benchmarks.py --dataset eu_email
    python experiments/download_benchmarks.py --dataset jodie_reddit
    python experiments/download_benchmarks.py --dataset jodie_wikipedia
    python experiments/download_benchmarks.py --dataset tgbn_trade
    python experiments/download_benchmarks.py --dataset tgbn_genre
    python experiments/download_benchmarks.py --dataset tgbn_genre_v2
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.factory import save_meta


def build_eu_email(data_dir="data"):
    from src.data.eu_email_builder import download_eu_email, build_eu_email_graphs_from_edges
    txt_path = download_eu_email(data_dir)
    print("building EU Email weekly graphs...")
    graphs, meta = build_eu_email_graphs_from_edges(txt_path, min_active_nodes=10)
    torch.save(graphs, f"{data_dir}/eu_email_graphs.pt")
    save_meta(meta, f"{data_dir}/eu_email_meta.json")
    print(f"saved {len(graphs)} snapshots. splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def build_jodie(dataset_name, data_dir="data"):
    assert dataset_name in ("reddit", "wikipedia")
    from src.data.jodie_builder import download_jodie, build_jodie_graphs_from_csv
    csv_path = download_jodie(dataset_name, data_dir)
    print(f"building JODIE {dataset_name} weekly graphs...")
    graphs, meta = build_jodie_graphs_from_csv(csv_path, min_active_nodes=10)
    out_key = f"jodie_{dataset_name}"
    torch.save(graphs, f"{data_dir}/{out_key}_graphs.pt")
    save_meta(meta, f"{data_dir}/{out_key}_meta.json")
    print(f"saved {len(graphs)} snapshots. splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def build_jodie_user_user(dataset_name, data_dir="data", top_k_users=500):
    """user-user co-interaction projection variant — unipartite, ~top_k_users nodes,
    6d node features. compatible with the cross-dataset architecture.

    uses daily buckets (86400s) because jodie covers only ~30 days; weekly would
    give only 4-5 snapshots which is below context_k+1=5 needed for valid samples.
    """
    assert dataset_name in ("reddit", "wikipedia")
    from src.data.jodie_builder import download_jodie, build_jodie_user_user_graphs
    csv_path = download_jodie(dataset_name, data_dir)
    print(f"building JODIE {dataset_name} user-user projection graphs (top_k={top_k_users}, daily buckets)...")
    graphs, meta = build_jodie_user_user_graphs(
        csv_path, top_k_users=top_k_users, bucket_seconds=86400, min_active_nodes=20
    )
    out_key = f"jodie_{dataset_name}_uu"
    torch.save(graphs, f"{data_dir}/{out_key}_graphs.pt")
    save_meta(meta, f"{data_dir}/{out_key}_meta.json")
    print(f"saved {len(graphs)} snapshots, n_nodes={meta['n_nodes']}")
    if meta.get('edge_counts'):
        ec = sorted(meta['edge_counts'])
        ac = sorted(meta['active_counts'])
        print(f"  edges per snapshot: min={ec[0]} median={ec[len(ec)//2]} max={ec[-1]}")
        print(f"  active nodes per snapshot: min={ac[0]} median={ac[len(ac)//2]} mean={sum(ac)/len(ac):.1f}")
    print(f"  splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def build_tgbn_trade(data_dir="data"):
    from src.data.tgb_builder import build_tgbn_trade_graphs
    print("building TGBN-Trade annual graphs...")
    graphs, meta = build_tgbn_trade_graphs(data_dir)
    torch.save(graphs, f"{data_dir}/tgbn_trade_graphs.pt")
    save_meta(meta, f"{data_dir}/tgbn_trade_meta.json")
    print(f"saved {len(graphs)} snapshots. splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def build_tgbn_genre(data_dir="data"):
    from src.data.tgb_builder import build_tgbn_genre_graphs
    print("building TGBN-Genre weekly graphs...")
    graphs, meta = build_tgbn_genre_graphs(data_dir)
    torch.save(graphs, f"{data_dir}/tgbn_genre_graphs.pt")
    save_meta(meta, f"{data_dir}/tgbn_genre_meta.json")
    print(f"saved {len(graphs)} snapshots. splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def build_dblp(data_dir="data", year_min=1995, year_max=2020,
               min_papers=5, min_active_nodes=30, max_authors=1000):
    from src.data.dblp_builder import download_dblp, build_dblp_graphs
    xml_gz = download_dblp(data_dir)
    print("building DBLP annual coauthorship graphs...")
    graphs, meta = build_dblp_graphs(
        xml_gz,
        year_min=year_min,
        year_max=year_max,
        min_papers=min_papers,
        min_active_nodes=min_active_nodes,
        max_authors=max_authors,
    )
    torch.save(graphs, f"{data_dir}/dblp_graphs.pt")
    save_meta(meta, f"{data_dir}/dblp_meta.json")
    print(f"saved {len(graphs)} snapshots, n_nodes={meta['n_nodes']}")
    if meta['edge_counts']:
        ec = sorted(meta['edge_counts'])
        ac = sorted(meta['active_counts'])
        med = ec[len(ec) // 2]
        med_a = ac[len(ac) // 2]
        print(f"  edges per snapshot: min={ec[0]} median={med} max={ec[-1]}")
        print(f"  active nodes per snapshot: min={ac[0]} median={med_a} mean={sum(ac)/len(ac):.1f}")
    print(f"  splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def build_tgbn_genre_v2(data_dir="data"):
    from src.data.tgb_builder import build_tgbn_genre_v2_graphs
    print("building TGBN-Genre v2 (bipartite-aware) weekly graphs...")
    graphs, meta = build_tgbn_genre_v2_graphs(data_dir)
    torch.save(graphs, f"{data_dir}/tgbn_genre_v2_graphs.pt")
    save_meta(meta, f"{data_dir}/tgbn_genre_v2_meta.json")
    print(f"saved {len(graphs)} snapshots. n_filtered={meta['n_nodes']} (raw={meta['n_nodes_raw']})")
    print(f"  users={meta['n_users']}, items={meta['n_items']}, min_weeks_active={meta['min_weeks_active']}")
    if meta['edge_counts']:
        ec = sorted(meta['edge_counts'])
        ac = sorted(meta['active_counts'])
        med = ec[len(ec) // 2]
        med_a = ac[len(ac) // 2]
        print(f"  edges per snapshot: min={ec[0]} median={med} max={ec[-1]}")
        print(f"  active nodes per snapshot: min={ac[0]} median={med_a} mean={sum(ac)/len(ac):.1f}")
    print(f"  splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "eu_email", "jodie_reddit", "jodie_wikipedia",
                                 "jodie_reddit_uu", "jodie_wikipedia_uu",
                                 "tgbn_trade", "tgbn_genre", "tgbn_genre_v2", "dblp"])
    parser.add_argument("--data_dir", default="data")
    # accept --data-dir alias to match prompt-style invocations
    parser.add_argument("--data-dir", dest="data_dir_alias", default=None)
    args = parser.parse_args()
    if args.data_dir_alias is not None:
        args.data_dir = args.data_dir_alias

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "eu_email"):
        build_eu_email(args.data_dir)

    if args.dataset in ("all", "jodie_reddit"):
        build_jodie("reddit", args.data_dir)

    if args.dataset in ("all", "jodie_wikipedia"):
        build_jodie("wikipedia", args.data_dir)

    if args.dataset in ("all", "tgbn_trade"):
        build_tgbn_trade(args.data_dir)

    if args.dataset in ("all", "tgbn_genre"):
        build_tgbn_genre(args.data_dir)

    if args.dataset == "tgbn_genre_v2":
        build_tgbn_genre_v2(args.data_dir)

    if args.dataset == "jodie_reddit_uu":
        build_jodie_user_user("reddit", args.data_dir)

    if args.dataset == "jodie_wikipedia_uu":
        build_jodie_user_user("wikipedia", args.data_dir)

    # dblp lowest priority: only run if explicitly requested
    if args.dataset == "dblp":
        build_dblp(args.data_dir)

    print("done.")


if __name__ == "__main__":
    main()
