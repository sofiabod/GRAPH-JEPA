import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.enron_loader import download_enron, load_emails
from src.data.graph_builder import build_weekly_graphs_6d

DATA_DIR = "data"


def main():
    print("downloading enron corpus...")
    download_enron(DATA_DIR)
    print("parsing emails...")
    emails = load_emails(DATA_DIR)
    print(f"loaded {len(emails)} emails")
    # 6d-feature variant matches cross-dataset architecture (in_dim=6 in configs/enron.yaml).
    # the older 389d BGE-text variant is still in graph_builder.py if richer features are
    # ever needed, but 6d is what the configs and trained models expect.
    print("building weekly graphs (6d cross-dataset features)...")
    graphs, meta = build_weekly_graphs_6d(emails, top_n=50, min_active=10)
    print(f"built {len(graphs)} weekly snapshots, n_nodes={meta['n_nodes']}")
    print(f"  splits: train={meta['train_range']}, val={meta['val_range']}, test={meta['test_range']}")
    Path(DATA_DIR).mkdir(exist_ok=True)
    torch.save(graphs, f"{DATA_DIR}/enron_graphs.pt")
    with open(f"{DATA_DIR}/enron_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved to {DATA_DIR}/enron_graphs.pt and {DATA_DIR}/enron_meta.json")


if __name__ == "__main__":
    main()
