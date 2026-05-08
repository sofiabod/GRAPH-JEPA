"""one-shot modal entrypoint to prep enron data.

bypasses the local torch_geometric dependency: the modal container has PyG
installed, downloads the corpus to its tmp space, builds the weekly graphs,
and writes them to the tgjepa-results volume under /_data/. then pull
locally with `modal volume get`.

usage:
    modal run experiments/prep_enron_modal.py::main
    modal volume get tgjepa-results /_data/enron_graphs.pt data/enron_graphs.pt
    modal volume get tgjepa-results /_data/enron_meta.json data/enron_meta.json
"""
import modal

app = modal.App("tgjepa-prep-enron")

TORCH_VERSION = "2.1.0"
CUDA = "cu121"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "scipy", "scikit-learn")
    .pip_install(f"torch=={TORCH_VERSION}", index_url=f"https://download.pytorch.org/whl/{CUDA}")
    .pip_install(
        "torch-scatter",
        "torch-sparse",
        "torch-geometric",
        find_links=f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html",
    )
    .pip_install("omegaconf", "einops")
    .add_local_dir("src", remote_path="/app/src")
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=True)


@app.function(
    image=image,
    timeout=3600,
    cpu=4.0,
    memory=8192,
    volumes={"/results": vol},
)
def prep_enron():
    """download + parse + 6d-build enron, write to /results/_data/."""
    import os
    import sys
    import json
    import torch
    sys.path.insert(0, "/app")
    os.makedirs("/tmp/enron_data", exist_ok=True)
    os.chdir("/tmp/enron_data")

    from src.data.enron_loader import download_enron, load_emails
    from src.data.graph_builder import build_weekly_graphs_6d

    print("downloading enron corpus (~430MB)...")
    download_enron(".")
    print("parsing emails (this is slow — ~500k emails)...")
    emails = load_emails(".")
    print(f"loaded {len(emails)} emails")
    print("building weekly graphs (6d cross-dataset features)...")
    graphs, meta = build_weekly_graphs_6d(emails, top_n=50, min_active=10)
    print(f"built {len(graphs)} weekly snapshots, n_nodes={meta['n_nodes']}")
    print(f"  default splits: train={meta['train_range']}, "
          f"val={meta['val_range']}, test={meta['test_range']}")
    if meta["weeks"]:
        first = meta["weeks"][0]["date_str"]
        last = meta["weeks"][-1]["date_str"]
        print(f"  date range: {first} to {last}")

    out_dir = "/results/_data"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(graphs, f"{out_dir}/enron_graphs.pt")
    with open(f"{out_dir}/enron_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()

    return {
        "n_snapshots": meta["n_snapshots"],
        "n_nodes": meta["n_nodes"],
        "first_week": meta["weeks"][0]["date_str"] if meta["weeks"] else None,
        "last_week": meta["weeks"][-1]["date_str"] if meta["weeks"] else None,
        "default_splits": {
            "train": meta["train_range"],
            "val": meta["val_range"],
            "test": meta["test_range"],
        },
        # snapshot-index lookup table for the fraud-event weeks (helps decide split adjustments)
        "fraud_event_weeks": {
            w["date_str"]: w["week_idx"]
            for w in meta["weeks"]
            if w["date_str"] in {
                "2001-W08", "2001-W32", "2001-W41", "2001-W42",
                "2001-W45", "2001-W48",
            }
        },
    }


@app.local_entrypoint()
def main():
    print("running enron data prep on modal (will take ~30 min for download + parse)")
    result = prep_enron.remote()
    import json
    print("\nresult:")
    print(json.dumps(result, indent=2))
    print("\nnow pull locally:")
    print("  modal volume get tgjepa-results /_data/enron_graphs.pt data/enron_graphs.pt")
    print("  modal volume get tgjepa-results /_data/enron_meta.json data/enron_meta.json")
