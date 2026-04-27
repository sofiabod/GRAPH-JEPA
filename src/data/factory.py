import json
import torch
from pathlib import Path


def create_dataset(cfg):
    """load or build a dataset based on cfg.dataset.

    for pre-built datasets (graphs already saved to disk), loads from cfg.data.graphs_path.
    for datasets that need building, call the appropriate builder.

    returns list[PyG Data] — weekly graph snapshots.
    """
    dataset = cfg.dataset

    if dataset in ("enron", "eu_email", "jodie_reddit", "jodie_wikipedia", "tgbn_trade"):
        return _load_prebuilt(cfg)

    raise ValueError(
        f"unknown dataset '{dataset}'. "
        f"valid options: enron, eu_email, jodie_reddit, jodie_wikipedia, tgbn_trade"
    )


def _load_prebuilt(cfg):
    """load graphs from cfg.data.graphs_path (pre-built by download script)."""
    graphs = torch.load(cfg.data.graphs_path, weights_only=False)
    return graphs


def compute_split_ranges(n_snapshots: int, train_frac=0.70, val_frac=0.15):
    """compute absolute [lo, hi] split ranges for a dataset with n_snapshots total.

    returns (train_range, val_range, test_range) as (int, int) tuples (inclusive).
    """
    train_end = int(n_snapshots * train_frac) - 1
    val_end = int(n_snapshots * (train_frac + val_frac)) - 1
    test_end = n_snapshots - 1
    return (0, train_end), (train_end + 1, val_end), (val_end + 1, test_end)


def save_meta(meta: dict, meta_path: str):
    """save dataset meta dict to json."""
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
