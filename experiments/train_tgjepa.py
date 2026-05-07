import modal

app = modal.App("tgjepa-training")

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
    .pip_install("sentence-transformers", "omegaconf", "einops", "pytest")
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("tests", remote_path="/app/tests")
)

# training image also mounts the preprocessed graph files; ship the whole data dir
# so any dataset (tgbn-trade, eu_email, jodie, enron) is available based on cfg.data.graphs_path
train_image = image.add_local_dir("data", remote_path="/app/data")

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=3600 * 8,
    image=train_image,
    volumes={"/results": vol},
    max_containers=1,
)
def train_seed(seed: int, config_path: str = "configs/tgbn_trade.yaml"):
    import os
    import sys
    sys.path.insert(0, "/app")
    # cfg.data.graphs_path is a repo-relative path; chdir so torch.load can find it
    os.chdir("/app")
    from omegaconf import OmegaConf
    from src.train import train
    from src.utils.seed import set_seed

    cfg = OmegaConf.load(f"/app/{config_path}")
    set_seed(seed)
    condition = cfg.get("dataset", "tgjepa")
    log = train(cfg, seed=seed, out_dir=f"/results/{condition}/seed{seed}")
    vol.commit()
    return log


@app.function(image=image, timeout=600)
def run_tests():
    import sys
    import subprocess
    sys.path.insert(0, "/app")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd="/app",
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def _format_log(log: dict, seed: int) -> str:
    train_l = log.get("train_losses") or []
    val_l = log.get("val_losses") or []
    best = log.get("best_val_loss")
    n_epochs = len(train_l)
    if not val_l:
        return f"\nseed {seed} done | epochs={n_epochs}  no val data"
    best_idx = val_l.index(min(val_l))
    last_train = train_l[-1] if train_l else float("nan")
    best_str = f"{best:.4f}" if best is not None else "n/a"
    return (
        f"\nseed {seed} done | epochs={n_epochs}  "
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


@app.local_entrypoint()
def main(config: str = "configs/tgbn_trade.yaml", seeds: str = "0,1,2,3,4"):
    import json
    from pathlib import Path
    seed_list = [int(s) for s in seeds.split(",")]
    dataset = _read_dataset_name_local(config)
    # collect all futures into a list first so modal's iterator cleanup is fully done
    # before we run local save logic (streaming iteration races with cleanup, drops side effects)
    print(f"running {len(seed_list)} seeds; saves will land after all complete")
    logs = list(train_seed.map(seed_list, kwargs={"config_path": config}))
    print("all seeds complete; saving locally")
    for seed, future in zip(seed_list, logs):
        print(_format_log(future, seed))
        out_dir = Path("results") / dataset / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train_log_graph.json", "w") as f:
            json.dump(future, f, indent=2)
        print(f"saved {out_dir / 'train_log_graph.json'}")


@app.local_entrypoint()
def test():
    rc = run_tests.remote()
    print(f"\nexit code: {rc}")
