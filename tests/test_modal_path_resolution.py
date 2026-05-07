"""regression tests for the path-resolution bug that crashed the first modal run.

cfg.data.graphs_path is a repo-relative path. modal containers don't have
the repo as cwd by default, so torch.load(cfg.data.graphs_path) fails
unless the entrypoint chdir's to the mount root or absolute paths are used.

these tests catch the class of bug locally.
"""
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [
    REPO_ROOT / "configs" / "tgbn_trade.yaml",
    REPO_ROOT / "configs" / "enron.yaml",
    REPO_ROOT / "configs" / "eu_email.yaml",
    REPO_ROOT / "configs" / "jodie_reddit.yaml",
    REPO_ROOT / "configs" / "jodie_wikipedia.yaml",
]


@pytest.mark.parametrize("config_path", CONFIGS, ids=[c.name for c in CONFIGS])
def test_config_paths_are_relative_to_repo_root(config_path):
    """config graph_path should resolve when cwd is the repo root."""
    cfg = OmegaConf.load(config_path)
    assert hasattr(cfg, "data"), f"{config_path.name}: missing 'data' section"
    assert hasattr(cfg.data, "graphs_path"), f"{config_path.name}: missing 'data.graphs_path'"
    # path is repo-relative; resolve from repo root and confirm it's a string
    p = Path(cfg.data.graphs_path)
    assert not p.is_absolute(), \
        f"{config_path.name}: graphs_path should be repo-relative, got absolute {p}"


def test_modal_entrypoint_files_chdir_or_abs_path():
    """modal training entrypoints must either chdir to /app or use absolute paths.

    if neither, torch.load(cfg.data.graphs_path) will fail because modal cwd
    is not /app by default.
    """
    targets = [
        REPO_ROOT / "experiments" / "train_tgjepa.py",
        REPO_ROOT / "experiments" / "train_sequential_ablation.py",
    ]
    for path in targets:
        text = path.read_text()
        # the function must either chdir to /app or prefix data path with /app/
        ok = ('os.chdir("/app")' in text) or ('f"/app/{cfg.data.graphs_path}"' in text)
        assert ok, (
            f"{path.name}: training function must either chdir('/app') or "
            f"resolve cfg.data.graphs_path with '/app/' prefix. Otherwise "
            f"torch.load(cfg.data.graphs_path) raises FileNotFoundError on Modal."
        )


def test_train_handles_relative_path_from_repo_root(tmp_path):
    """confirms train() can load graphs via cfg.data.graphs_path when cwd is repo root.

    this is the path that succeeded locally and failed on modal. the test
    locks in the local-path contract: relative paths in cfg.data.graphs_path
    resolve when cwd is the repo root.
    """
    tgbn_path = REPO_ROOT / "data" / "tgbn_trade_graphs.pt"
    if not tgbn_path.exists():
        pytest.skip("tgbn_trade_graphs.pt not built locally; run download_benchmarks.py")
    import torch
    cwd_before = os.getcwd()
    try:
        os.chdir(REPO_ROOT)
        cfg = OmegaConf.load(REPO_ROOT / "configs" / "tgbn_trade.yaml")
        # this is what train() does internally
        graphs = torch.load(cfg.data.graphs_path, weights_only=False)
        assert len(graphs) > 0
    finally:
        os.chdir(cwd_before)
