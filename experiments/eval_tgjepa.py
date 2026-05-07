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
    .add_local_dir("data", remote_path="/app/data")
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=False)


@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def eval_seed(seed: int, condition: str, config_path: str = "configs/tgbn_trade.yaml"):
    import sys
    sys.path.insert(0, "/app")
    import json
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_runner import EvalRunner

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    # condition selects checkpoint dir; do not clobber with cfg.dataset (training writes per-condition)
    ckpt_path = f"/results/{condition}/seed{seed}/checkpoint.pt"

    if hasattr(cfg.data, 'train_weeks') and cfg.data.train_weeks is not None:
        split_kwargs = dict(
            train_range=tuple(cfg.data.train_weeks),
            val_range=tuple(cfg.data.val_weeks),
            test_range=tuple(cfg.data.test_weeks),
        )
    else:
        with open(f"/app/{cfg.data.meta_path}") as f:
            meta = json.load(f)
        split_kwargs = dict(
            train_range=tuple(meta['train_range']),
            val_range=tuple(meta['val_range']),
            test_range=tuple(meta['test_range']),
        )
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
    if "target_encoder" in ckpt:
        target.encoder.load_state_dict(ckpt["target_encoder"])

    runner = EvalRunner(online, target, predictor, graphs, cfg, mask_seed=seed, **split_kwargs)
    results = runner.run_all(f"/results/eval/{condition}/seed{seed}")
    vol.commit()
    return {"seed": seed, "condition": condition, **results}


@app.function(
    gpu="A10G",
    timeout=3600,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def eval_paired_seed(seed: int, config_path: str = "configs/tgbn_trade.yaml"):
    """run eval 2 paired comparison for one seed.

    loads both graph-jepa and sequential-ablation checkpoints from the
    conventional /results paths, runs eval2_compare on the same masked
    test set, returns paired wilcoxon p-value and win rate.
    """
    import sys
    sys.path.insert(0, "/app")
    import json
    import torch
    from omegaconf import OmegaConf
    from src.eval.eval2 import eval2_compare

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")

    if hasattr(cfg.data, 'train_weeks') and cfg.data.train_weeks is not None:
        splits = (
            tuple(cfg.data.train_weeks),
            tuple(cfg.data.val_weeks),
            tuple(cfg.data.test_weeks),
        )
    else:
        with open(f"/app/{cfg.data.meta_path}") as f:
            meta = json.load(f)
        splits = (
            tuple(meta['train_range']),
            tuple(meta['val_range']),
            tuple(meta['test_range']),
        )

    main_condition = cfg.get("dataset", "tgjepa")
    graph_ckpt_path = f"/results/{main_condition}/seed{seed}/checkpoint.pt"
    seq_ckpt_path = f"/results/sequential-ablation/seed{seed}/checkpoint.pt"

    result = eval2_compare(
        graph_ckpt_path=graph_ckpt_path,
        sequential_ckpt_path=seq_ckpt_path,
        graphs=graphs,
        cfg=cfg,
        splits=splits,
        mask_seed=seed,
        device=device,
    )

    out_path = f"/results/eval2/seed{seed}"
    import os
    os.makedirs(out_path, exist_ok=True)
    with open(f"{out_path}/eval2.json", 'w') as f:
        json.dump(result, f, indent=2)
    vol.commit()
    return {"seed": seed, **result}


def _read_dataset_name(config_path: str) -> str:
    # parse 'dataset: <name>' from a yaml config without depending on omegaconf locally
    # (omegaconf is only installed in the modal container image, not in the local env)
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

    # use cfg.dataset as the graph-jepa condition so eval matches training output dir
    main_condition = _read_dataset_name(config)
    conditions = [main_condition, "sequential-ablation"]
    seed_list = [int(s) for s in seeds.split(",")]
    args = [(seed, cond, config) for cond in conditions for seed in seed_list]

    results_root = Path("results") / main_condition

    # collect all futures into lists first so modal's iterator cleanup is fully done
    # before we run local save logic (streaming iteration races with cleanup, drops side effects)
    print(f"running eval 1 on {len(args)} (seed, condition) pairs")
    all_results = list(eval_seed.starmap(args))
    print("eval 1 complete; saving locally")
    for result in all_results:
        print(json.dumps(result, indent=2))
        cond = result.get("condition", "unknown")
        seed = result.get("seed", -1)
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval1_{cond}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved {out_path}")

    # eval 2: paired comparison per seed
    print("\n--- eval 2 (paired graph vs sequential) ---")
    eval2_args = [(seed, config) for seed in seed_list]
    print(f"running eval 2 on {len(eval2_args)} seeds")
    eval2_results = list(eval_paired_seed.starmap(eval2_args))
    print("eval 2 complete; saving locally")
    for result in eval2_results:
        print(json.dumps(result, indent=2))
        seed = result.get("seed", -1)
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval2.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"saved {out_path}")

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

    if eval2_results:
        graph_means = [r["mean_graph_cos"] for r in eval2_results if "mean_graph_cos" in r]
        seq_means = [r["mean_sequential_cos"] for r in eval2_results if "mean_sequential_cos" in r]
        p_vals = [r["wilcoxon_p"] for r in eval2_results if "wilcoxon_p" in r]
        win_rates = [r["win_rate"] for r in eval2_results if "win_rate" in r]
        if graph_means:
            print(f"eval 2: graph={sum(graph_means)/len(graph_means):.4f}  "
                  f"seq={sum(seq_means)/len(seq_means):.4f}  "
                  f"mean_win_rate={sum(win_rates)/len(win_rates):.3f}  "
                  f"per-seed p-values: {[f'{p:.4g}' for p in p_vals]}")

    # write a markdown summary across seeds
    summary_path = results_root / "SUMMARY.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# eval summary: {main_condition}", "", f"seeds: {seed_list}", ""]
    for cond in conditions:
        cond_results = [r for r in all_results if r.get("condition") == cond]
        if not cond_results:
            continue
        lines.append(f"## {cond}")
        for r in cond_results:
            e1 = r.get("eval1_node_prediction", {})
            e6 = r.get("eval6_representation_quality", {})
            seed = r.get("seed", "?")
            lines.append(
                f"- seed {seed}: "
                f"pred_cos={e1.get('mean_pred_cos', 'n/a'):.4f}  "
                f"copy_cos={e1.get('mean_copy_cos', 'n/a'):.4f}  "
                f"graph_avg_cos={e1.get('mean_graph_avg_cos', 0.0):.4f}  "
                f"eff_rank={e6.get('effective_rank', 'n/a'):.2f}  "
                f"mean_pair_cos={e6.get('mean_pairwise_cosine', 'n/a'):.3f}"
            )
        lines.append("")
    if eval2_results:
        lines.append("## eval 2 (paired graph vs sequential)")
        for r in eval2_results:
            seed = r.get("seed", "?")
            lines.append(
                f"- seed {seed}: "
                f"graph_cos={r.get('mean_graph_cos', 'n/a'):.4f}  "
                f"seq_cos={r.get('mean_sequential_cos', 'n/a'):.4f}  "
                f"wilcoxon_p={r.get('wilcoxon_p', 'n/a'):.3e}  "
                f"win_rate={r.get('win_rate', 'n/a'):.3f}  "
                f"n={r.get('n_pairs', 'n/a')}"
            )
        lines.append("")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {summary_path}")
