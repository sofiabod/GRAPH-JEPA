"""modal orchestration for graph-jepa-2 experiments.

provides run_one (single experiment, single seed) and run_all (full sweep) entry
points. each experiment is a function decorated with @modal.function.

usage:
    modal run --detach modal_app/app.py::run_one --exp e1_tgbn_trade --seed 0
    modal run --detach modal_app/app.py::run_all --exp e1_tgbn_trade --seeds 0,1,2,3,4
    modal run --detach modal_app/app.py::run_metrla --seeds 0,1,2,3,4

this is a SKELETON. fill in:
  - the modal Image with your repo + dependencies
  - the modal Volume for cached datasets and checkpoints
  - secrets (HF_TOKEN, etc.) if needed
  - per-experiment dispatch in run_one
"""
from __future__ import annotations

import modal

app = modal.App("graph-jepa-2")

# image: clone repo, install editable, install experiment deps
# adjust to point at your fork once published
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.3.1", "torch-geometric==2.5.3")
    .run_commands(
        "git clone https://github.com/sofiabodnar/GRAPH-JEPA-2.git /workspace",
        "pip install -e /workspace[experiments]",
    )
)

# persistent volume for datasets, checkpoints, results
volume = modal.Volume.from_name("graph-jepa-2-vol", create_if_missing=True)

GPU_CONFIG = modal.gpu.H100(count=1)
TIMEOUT_HOURS = 6


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/data": volume},
    timeout=TIMEOUT_HOURS * 3600,
)
def run_one(exp: str, seed: int = 0, config_override: str | None = None) -> dict:
    """run a single (exp, seed) cell.

    args:
        exp: experiment id matching configs/<exp>.yaml — e.g. "tgbn_trade", "metrla"
        seed: rng seed
        config_override: optional path to a yaml override (for ablations)
    """
    import subprocess

    cmd = [
        "python", "/workspace/experiments/train_tgjepa.py",
        "--config", f"/workspace/configs/{exp}.yaml",
        "--seed", str(seed),
        "--out", f"/data/results/{exp}/seed{seed}",
    ]
    if config_override:
        cmd.extend(["--override", config_override])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "exp": exp,
        "seed": seed,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=TIMEOUT_HOURS * 3600,
)
def run_eval(exp: str, seed: int) -> dict:
    """re-run eval on an existing checkpoint. populates eval3 rollout for stale jsons."""
    import subprocess
    cmd = [
        "python", "/workspace/experiments/eval_tgjepa.py",
        "--checkpoint", f"/data/results/{exp}/seed{seed}/best_model.pt",
        "--out", f"/data/results/{exp}/seed{seed}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"exp": exp, "seed": seed, "returncode": result.returncode}


@app.local_entrypoint()
def run_all(exp: str = "tgbn_trade", seeds: str = "0,1,2,3,4"):
    """fan out (exp, seed) cells in parallel. usage:
        modal run modal_app/app.py::run_all --exp tgbn_trade --seeds 0,1,2,3,4
    """
    seed_list = [int(s) for s in seeds.split(",")]
    results = list(run_one.map(
        [exp] * len(seed_list),
        seed_list,
    ))
    print(f"completed {len(results)} cells")
    for r in results:
        if r["returncode"] != 0:
            print(f"FAILED: {r['exp']} seed={r['seed']}")
            print(r["stderr_tail"])
    return results


@app.local_entrypoint()
def run_metrla(seeds: str = "0,1,2,3,4"):
    """metr-la-specific entrypoint (different config + horizon eval)."""
    return run_all(exp="metrla", seeds=seeds)
