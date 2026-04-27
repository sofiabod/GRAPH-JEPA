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
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("data", remote_path="/app/data")
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=3600 * 8,
    image=image,
    volumes={"/results": vol},
)
def train_seed_ablation(seed: int, config_path: str = "configs/enron.yaml"):
    import sys
    sys.path.insert(0, "/app")
    from omegaconf import OmegaConf
    from src.train import train
    from src.utils.seed import set_seed

    cfg = OmegaConf.load(f"/app/{config_path}")
    set_seed(seed)
    # ablation=True swaps GraphEncoder for param-matched SequentialMLP (no message passing)
    log = train(
        cfg,
        seed=seed,
        ablation=True,
        out_dir=f"/results/sequential-ablation/seed{seed}",
    )
    vol.commit()
    return log


@app.local_entrypoint()
def main(seeds: str = "0,1,2,3,4"):
    seed_list = [int(s) for s in seeds.split(",")]
    for future in train_seed_ablation.map(seed_list):
        print(future)
