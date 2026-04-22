import modal

app = modal.App("tgjepa-sequential-ablation")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "torch-geometric",
    "torch-scatter",
    "torch-sparse",
    "sentence-transformers",
    "omegaconf",
    "einops",
    "numpy",
    "scipy",
    "scikit-learn",
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=3600 * 8,
    image=image,
    volumes={"/results": vol},
    mounts=[modal.Mount.from_local_dir(".", remote_path="/app")],
)
def train_seed_ablation(seed: int, config_path: str = "configs/tgjepa_base.yaml"):
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
