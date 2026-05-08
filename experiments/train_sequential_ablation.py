import modal

app = modal.App("tgjepa-sequential-ablation")

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
    .pip_install("sentence-transformers", "omegaconf", "einops")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("data", remote_path="/app/data",
                   ignore=["*_raw", "*_raw/**", "*.zip", "*.gz", "*.tab", "*.csv"])
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=3600 * 8,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def train_seed_ablation(seed: int, config_path: str = "configs/tgbn_trade.yaml",
                         git_sha: str = ""):
    import os
    import sys
    sys.path.insert(0, "/app")
    # cfg.data.graphs_path is a repo-relative path; chdir so torch.load can find it
    os.chdir("/app")
    from omegaconf import OmegaConf
    from src.train import train
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    cfg = OmegaConf.load(f"/app/{config_path}")
    set_seed(seed)
    # namespace by dataset so per-dataset runs don't clobber each other
    dataset = cfg.get("dataset", "tgjepa")
    # ablation=True swaps GraphEncoder for param-matched SequentialMLP (no message passing)
    log = train(
        cfg,
        seed=seed,
        ablation=True,
        out_dir=f"/results/{dataset}/sequential-ablation/seed{seed}",
    )
    vol.commit()
    return log


def _format_log(log: dict, seed: int) -> str:
    train_l = log.get("train_losses") or []
    val_l = log.get("val_losses") or []
    best = log.get("best_val_loss")
    n_epochs = len(train_l)
    if not val_l:
        return f"\nseed {seed} (sequential-ablation) done | epochs={n_epochs}  no val data"
    best_idx = val_l.index(min(val_l))
    last_train = train_l[-1] if train_l else float("nan")
    best_str = f"{best:.4f}" if best is not None else "n/a"
    return (
        f"\nseed {seed} (sequential-ablation) done | epochs={n_epochs}  "
        f"best_val={best_str} (epoch {best_idx + 1})  "
        f"val first->last: {val_l[0]:.4f} -> {val_l[-1]:.4f}  "
        f"final_train: {last_train:.4f}"
    )


def _read_dataset_name_local(config_path: str) -> str:
    # parse 'dataset: <name>' from yaml without depending on omegaconf in the local env
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("dataset:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    return "tgjepa"


def _local_git_sha() -> str:
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else ""
    except Exception:
        return ""


@app.local_entrypoint()
def main(config: str = "configs/tgbn_trade.yaml", seeds: str = "0,1,2,3,4",
         out_root: str = "results"):
    import json
    from pathlib import Path
    seed_list = [int(s) for s in seeds.split(",")]
    dataset = _read_dataset_name_local(config)
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")
    print(f"running {len(seed_list)} seeds; saves to {out_root}/{dataset}/")
    logs = list(train_seed_ablation.map(seed_list,
                                          kwargs={"config_path": config, "git_sha": git_sha}))
    print("all seeds complete; saving locally")
    for seed, future in zip(seed_list, logs):
        print(_format_log(future, seed))
        out_dir = Path(out_root) / dataset / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_log_seq.json", "w") as f:
            json.dump(future, f, indent=2)
        print(f"saved {out_dir / 'train_log_seq.json'}")
