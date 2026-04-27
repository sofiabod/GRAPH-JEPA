"""download and preprocess all non-enron benchmark datasets.

usage:
    python experiments/download_benchmarks.py --dataset all
    python experiments/download_benchmarks.py --dataset eu_email
    python experiments/download_benchmarks.py --dataset jodie_reddit
    python experiments/download_benchmarks.py --dataset jodie_wikipedia
    python experiments/download_benchmarks.py --dataset tgbn_trade
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


def build_tgbn_trade(data_dir="data"):
    from src.data.tgb_builder import build_tgbn_trade_graphs
    print("building TGBN-Trade annual graphs...")
    graphs, meta = build_tgbn_trade_graphs(data_dir)
    torch.save(graphs, f"{data_dir}/tgbn_trade_graphs.pt")
    save_meta(meta, f"{data_dir}/tgbn_trade_meta.json")
    print(f"saved {len(graphs)} snapshots. splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "eu_email", "jodie_reddit", "jodie_wikipedia", "tgbn_trade"])
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset in ("all", "eu_email"):
        build_eu_email(args.data_dir)

    if args.dataset in ("all", "jodie_reddit"):
        build_jodie("reddit", args.data_dir)

    if args.dataset in ("all", "jodie_wikipedia"):
        build_jodie("wikipedia", args.data_dir)

    if args.dataset in ("all", "tgbn_trade"):
        build_tgbn_trade(args.data_dir)

    print("done.")


if __name__ == "__main__":
    main()
