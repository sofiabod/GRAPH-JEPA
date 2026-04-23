import modal

app = modal.App("tgjepa-eval")

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
    .add_local_file("data/enron_graphs.pt", remote_path="/app/data/enron_graphs.pt")
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=False)


@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    volumes={"/results": vol},
)
def eval_seed(seed: int, condition: str):
    import sys
    sys.path.insert(0, "/app")
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_runner import EvalRunner

    cfg = OmegaConf.load("/app/configs/tgjepa_base.yaml")
    device = torch.device("cuda")

    graphs = torch.load("/app/data/enron_graphs.pt", map_location="cpu")

    ckpt_path = f"/results/{condition}/seed{seed}/checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    if condition == "sequential-ablation":
        online = SequentialMLP(
            in_dim=cfg.encoder.in_dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        ).to(device)
    else:
        online = build_graph_encoder(cfg.encoder).to(device)

    predictor = build_predictor(cfg.predictor).to(device)
    online.load_state_dict(ckpt["online"])
    predictor.load_state_dict(ckpt["predictor"])

    target = build_target_encoder(online)
    target.encoder = target.encoder.to(device)

    runner = EvalRunner(online, target, predictor, graphs, cfg)
    results = runner.run_all(f"/results/eval/{condition}/seed{seed}")
    vol.commit()
    return {"seed": seed, "condition": condition, **results}


@app.local_entrypoint()
def main():
    import json

    conditions = ["tgjepa", "sequential-ablation"]
    seeds = [0, 1, 2, 3, 4]
    args = [(seed, cond) for cond in conditions for seed in seeds]

    all_results = []
    for result in eval_seed.starmap(args):
        print(json.dumps(result, indent=2))
        all_results.append(result)

    print("\n--- summary ---")
    for cond in conditions:
        cond_results = [r for r in all_results if r["condition"] == cond]
        pred_cos = [r["eval1_node_prediction"]["mean_pred_cos"] for r in cond_results
                    if "mean_pred_cos" in r.get("eval1_node_prediction", {})]
        copy_cos = [r["eval1_node_prediction"]["mean_copy_cos"] for r in cond_results
                    if "mean_copy_cos" in r.get("eval1_node_prediction", {})]
        if pred_cos:
            print(f"{cond}: pred_cos={sum(pred_cos)/len(pred_cos):.4f}  "
                  f"copy_cos={sum(copy_cos)/len(copy_cos):.4f}")
