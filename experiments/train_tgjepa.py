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

# training image also mounts the preprocessed graph files
train_image = image.add_local_file(
    "data/enron_graphs.pt", remote_path="/app/data/enron_graphs.pt"
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=3600 * 8,
    image=train_image,
    volumes={"/results": vol},
)
def train_seed(seed: int, config_path: str = "configs/enron.yaml"):
    import sys
    sys.path.insert(0, "/app")
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


@app.local_entrypoint()
def main(config: str = "configs/enron.yaml", seeds: str = "0,1,2,3,4"):
    seed_list = [int(s) for s in seeds.split(",")]
    for future in train_seed.map(seed_list, kwargs=[{"config_path": config}] * len(seed_list)):
        print(future)


@app.local_entrypoint()
def test():
    rc = run_tests.remote()
    print(f"\nexit code: {rc}")
