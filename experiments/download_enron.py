import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.enron_loader import download_enron, load_emails
from src.data.graph_builder import build_weekly_graphs

DATA_DIR = "data"


def main():
    print("downloading enron corpus...")
    download_enron(DATA_DIR)
    print("parsing emails...")
    emails = load_emails(DATA_DIR)
    print(f"loaded {len(emails)} emails")
    print("building weekly graphs...")
    graphs, meta = build_weekly_graphs(emails, top_n=50, min_active=10, cache_path=f"{DATA_DIR}/bge_cache.pkl")
    print(f"built {len(graphs)} weekly snapshots")
    Path(DATA_DIR).mkdir(exist_ok=True)
    torch.save(graphs, f"{DATA_DIR}/enron_graphs.pt")
    with open(f"{DATA_DIR}/enron_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved to {DATA_DIR}/enron_graphs.pt and {DATA_DIR}/enron_meta.json")


if __name__ == "__main__":
    main()
