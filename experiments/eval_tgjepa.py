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
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
    .add_local_dir("data", remote_path="/app/data",
                   ignore=["*_raw", "*_raw/**", "*.zip", "*.gz", "*.tab", "*.csv"])
)

vol = modal.Volume.from_name("tgjepa-results", create_if_missing=False)


@app.function(
    gpu="H100",
    timeout=3600,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def eval_seed(seed: int, condition: str, config_path: str = "configs/tgbn_trade.yaml",
               git_sha: str = ""):
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_runner import EvalRunner
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    # call set_seed inside eval to flip torch.use_deterministic_algorithms(True)
    # so the captured _repro metadata reads "deterministic: true"
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    # condition selects checkpoint dir. for sequential-ablation we use the
    # dataset-namespaced path (added 2026-05; the old shared path
    # /results/sequential-ablation/ was clobbered by per-dataset reruns).
    if condition == "sequential-ablation":
        ckpt_path = f"/results/{cfg.dataset}/sequential-ablation/seed{seed}/checkpoint.pt"
    else:
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
    from src.utils.repro import capture_metadata
    meta_paths = []
    if hasattr(cfg.data, "graphs_path"):
        meta_paths.append(f"/app/{cfg.data.graphs_path}")
    if hasattr(cfg.data, "meta_path") and cfg.data.meta_path:
        meta_paths.append(f"/app/{cfg.data.meta_path}")
    repro = capture_metadata(
        dataset_paths=meta_paths,
        config_path=f"/app/{config_path}",
        extra={"checkpoint_path": ckpt_path},
    )
    vol.commit()
    return {"seed": seed, "condition": condition, **results, "_repro": repro}


@app.function(
    gpu="H100",
    timeout=3600,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def eval_paired_seed(seed: int, config_path: str = "configs/tgbn_trade.yaml",
                      shared_target_mode: str = "self",
                      eval_split: str = "test",
                      git_sha: str = ""):
    """run eval 2 paired comparison for one seed.

    loads both graph-jepa and sequential-ablation checkpoints from the
    conventional /results paths, runs eval2_compare on the same masked
    test set, returns paired wilcoxon p-value and win rate.

    shared_target_mode controls the cosine reference:
        "self": each model vs its own target encoder (within-condition)
        "graph": both vs graph-jepa's target (biases away from graph)
        "sequential": both vs sequential's target (biases away from sequential)
    """
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    import torch
    from omegaconf import OmegaConf
    from src.eval.eval2 import eval2_compare
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

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
    # sequential-ablation is now namespaced by dataset to prevent cross-dataset clobbering
    seq_ckpt_path = f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt"

    result = eval2_compare(
        graph_ckpt_path=graph_ckpt_path,
        sequential_ckpt_path=seq_ckpt_path,
        graphs=graphs,
        cfg=cfg,
        splits=splits,
        mask_seed=seed,
        device=device,
        shared_target_mode=shared_target_mode,
        eval_split=eval_split,
    )

    out_path = f"/results/eval2/seed{seed}"
    import os
    os.makedirs(out_path, exist_ok=True)
    if eval_split != "test":
        fname = f"eval2_split-{eval_split}.json"
    elif shared_target_mode != "self":
        fname = f"eval2_{shared_target_mode}.json"
    else:
        fname = "eval2.json"
    from src.utils.repro import capture_metadata
    meta_paths = []
    if hasattr(cfg.data, "graphs_path"):
        meta_paths.append(f"/app/{cfg.data.graphs_path}")
    if hasattr(cfg.data, "meta_path") and cfg.data.meta_path:
        meta_paths.append(f"/app/{cfg.data.meta_path}")
    repro = capture_metadata(
        dataset_paths=meta_paths,
        config_path=f"/app/{config_path}",
        extra={"graph_ckpt_path": graph_ckpt_path,
               "sequential_ckpt_path": seq_ckpt_path,
               "shared_target_mode": shared_target_mode,
               "eval_split": eval_split},
    )
    result_with_repro = {**result, "_repro": repro}
    with open(f"{out_path}/{fname}", 'w') as f:
        json.dump(result_with_repro, f, indent=2)
    vol.commit()
    return {"seed": seed, **result_with_repro}


def _read_dataset_name(config_path: str) -> str:
    # parse 'dataset: <name>' from a yaml config without depending on omegaconf locally
    # (omegaconf is only installed in the modal container image, not in the local env)
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("dataset:"):
                return line.split(":", 1)[1].strip().strip('"').strip("'")
    return "tgjepa"


def _local_git_sha() -> str:
    """capture local git sha + dirty flag for stamping result files. modal
    containers don't inherit local env, so the caller passes this string
    as a function argument and the modal function sets os.environ['GIT_SHA']
    inside the container before calling capture_metadata."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        if sha:
            return sha + ("-dirty" if dirty else "")
    except Exception:
        pass
    return ""


@app.local_entrypoint()
def main(config: str = "configs/tgbn_trade.yaml", seeds: str = "0,1,2,3,4",
         out_root: str = "results"):
    import json
    from pathlib import Path
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git sha: {git_sha}")

    # use cfg.dataset as the graph-jepa condition so eval matches training output dir
    main_condition = _read_dataset_name(config)
    conditions = [main_condition, "sequential-ablation"]
    seed_list = [int(s) for s in seeds.split(",")]
    # eval_seed signature: (seed, condition, config_path, git_sha)
    args = [(seed, cond, config, git_sha) for cond in conditions for seed in seed_list]

    results_root = Path(out_root) / main_condition
    print(f"saving locally to {results_root}/")

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
    print(f"running eval 2 on {len(seed_list)} seeds (sequential .remote calls)")
    eval2_results = []
    for seed in seed_list:
        try:
            result = eval_paired_seed.remote(seed, config, "self", "test", git_sha)
        except Exception as e:
            print(f"eval 2 seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        print(json.dumps(result, indent=2))
        eval2_results.append(result)
        out_dir = results_root / f"seed{result.get('seed', seed)}"
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


@app.function(
    gpu="H100",
    timeout=3600,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def anomaly_trajectory(seed: int, config_path: str = "configs/baci_gravity.yaml",
                        git_sha: str = ""):
    """run per-snapshot prediction trajectory for both graph-jepa and
    sequential-ablation, on every valid snapshot of the dataset (train+val+test)."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_anomaly import run_anomaly_trajectory
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    with open(f"/app/{cfg.data.meta_path}") as f:
        meta = json.load(f)
    # snapshot label resolution priority: years_kept (annual datasets like baci/trade/icio)
    # → weeks date_str list (weekly datasets like enron/eu_email) → snapshot index fallback
    if meta.get("years_kept"):
        year_labels = meta["years_kept"]
    elif meta.get("weeks"):
        year_labels = [w.get("date_str", w.get("week_idx", i))
                        for i, w in enumerate(meta["weeks"])]
    else:
        year_labels = list(range(len(graphs)))

    out: dict = {"seed": seed, "dataset": main_condition, "year_labels": year_labels}

    for cond_name, ckpt_subdir, builder in [
        ("graph", main_condition, "graph"),
        ("sequential", f"{main_condition}/sequential-ablation", "sequential"),
    ]:
        ckpt_path = f"/results/{ckpt_subdir}/seed{seed}/checkpoint.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        if builder == "graph":
            online = build_graph_encoder(cfg.encoder).to(device)
        else:
            online = SequentialMLP(
                in_dim=cfg.encoder.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                n_layers=cfg.encoder.n_layers,
                dropout=cfg.encoder.dropout,
            ).to(device)
        predictor = build_predictor(cfg.predictor).to(device)
        online.load_state_dict(ckpt["online"])
        predictor.load_state_dict(ckpt["predictor"])
        target = build_target_encoder(online)
        target.encoder = target.encoder.to(device)
        if "target_encoder" in ckpt:
            target.encoder.load_state_dict(ckpt["target_encoder"])

        traj = run_anomaly_trajectory(
            online, target, predictor, graphs, cfg,
            year_labels=year_labels,
            mask_seed=seed,
            device=device,
        )
        out[cond_name] = traj

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "anomaly_trajectory"},
    )
    out_dir = f"/results/eval_anomaly/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/anomaly_trajectory.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


# documented economic shocks per dataset (year -> short label).
# add or correct as needed; the demo highlights these on the trajectory plot.
KNOWN_SHOCKS_BY_DATASET = {
    "baci_gravity": {
        1997: "Asian Financial Crisis",
        1998: "Asian Financial Crisis (continued)",
        2001: "Dot-com / 9-11",
        2008: "Global Financial Crisis",
        2009: "GFC trough",
        2014: "Oil price collapse",
        2015: "Oil collapse / Russia sanctions",
        2018: "US-China trade war",
        2020: "COVID-19",
    },
    "tgbn_trade": {
        2008: "Global Financial Crisis",
        2009: "GFC trough",
        2014: "Oil price collapse",
        2018: "US-China trade war",
        2020: "COVID-19",
    },
    "icio": {
        2008: "Global Financial Crisis",
        2009: "GFC trough",
        2020: "COVID-19",
    },
    # Enron uses weekly cadence with date_str labels like "2001-W42"
    "enron": {
        "2001-W08": "Q4 2000 results inflate revenues",
        "2001-W32": "Skilling resigns as CEO (Aug 14)",
        "2001-W41": "Q3 earnings restatement window",
        "2001-W42": "SEC inquiry announced (Oct 22), Fastow fired (Oct 24)",
        "2001-W45": "$586M write-down (Nov 8)",
        "2001-W48": "bankruptcy filing (Dec 2)",
    },
}


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def bloc_discovery_seed(seed: int, config_path: str = "configs/baci_gravity.yaml",
                         partition: str = "subregion", k_override: int = 0):
    """run bloc-discovery for one seed:
       - graph-JEPA frozen embeddings
       - sequential-ablation frozen embeddings
       - raw-features baseline (mean of node feature vectors over test snapshots)
    returns ARI/NMI/purity for each + embeddings + cluster assignments."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_blocs import run_bloc_discovery, _kmeans_clustering, _adjusted_rand_index, _normalized_mutual_info, _cluster_purity
    from src.eval.iso3_regions import label_iso3_list

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    with open(f"/app/{cfg.data.meta_path}") as f:
        meta = json.load(f)
    iso3 = meta["iso_sorted"] if "iso_sorted" in meta else None
    if iso3 is None:
        return {"error": "iso_sorted not in meta — bloc discovery only works on country-keyed datasets"}

    out: dict = {"seed": seed, "dataset": main_condition, "n_countries": len(iso3),
                 "partition": partition, "k_override": k_override}

    k_arg = k_override if k_override > 0 else None
    # graph-jepa
    online_g = build_graph_encoder(cfg.encoder).to(device)
    ckpt_g = torch.load(f"/results/{main_condition}/seed{seed}/checkpoint.pt", map_location=device)
    online_g.load_state_dict(ckpt_g["online"])
    out["graph"] = run_bloc_discovery(online_g, graphs, cfg, iso3, seed=seed, device=device,
                                       partition=partition, k_override=k_arg)

    # sequential-ablation
    online_s = SequentialMLP(
        in_dim=cfg.encoder.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    ckpt_s = torch.load(f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt",
                         map_location=device)
    online_s.load_state_dict(ckpt_s["online"])
    out["sequential"] = run_bloc_discovery(online_s, graphs, cfg, iso3, seed=seed, device=device,
                                            partition=partition, k_override=k_arg)

    # raw-features baseline: mean of node feature vectors across test snapshots.
    # PyG Data.to(device) mutates in place, so after the run_bloc_discovery calls
    # above, graphs[t].x lives on cuda. accumulate on cpu for portability.
    test_lo, test_hi = meta["test_range"]
    feats_accum = torch.zeros(len(iso3), cfg.encoder.in_dim)
    for t in range(test_lo, test_hi + 1):
        feats_accum = feats_accum + graphs[t].x.cpu()
    feats_mean = (feats_accum / max(test_hi - test_lo + 1, 1)).numpy()
    if partition == "continent":
        from src.eval.iso3_regions import label_iso3_list_continent
        region_labels, _ = label_iso3_list_continent(iso3)
    else:
        region_labels, _ = label_iso3_list(iso3)
    region_to_int = {r: i for i, r in enumerate(sorted(set(region_labels)))}
    labels_true = np.array([region_to_int[r] for r in region_labels])
    k = k_arg if k_arg is not None else len(region_to_int)
    raw_clusters = _kmeans_clustering(feats_mean, k=k, seed=seed)
    out["raw_features"] = {
        "n_clusters": int(k),
        "n_countries": int(len(iso3)),
        "ari": _adjusted_rand_index(labels_true, raw_clusters),
        "nmi": _normalized_mutual_info(labels_true, raw_clusters),
        "purity": _cluster_purity(labels_true, raw_clusters),
        "embeddings": feats_mean.tolist(),
        "iso3": list(iso3),
        "region_labels": region_labels,
        "cluster_assignments": raw_clusters.tolist(),
    }

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "bloc_discovery"},
    )
    out_dir = f"/results/eval_blocs/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/bloc_discovery.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def probe_seed(seed: int, config_path: str = "configs/baci_gravity.yaml"):
    """linear probe for transferable representations.
    target: next-period log trade growth.
    returns R²/MAE per condition (graph / sequential / raw_features) via 5-fold CV."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_probe import run_probe

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    if hasattr(cfg.data, 'train_weeks') and cfg.data.train_weeks is not None:
        splits = (tuple(cfg.data.train_weeks), tuple(cfg.data.val_weeks), tuple(cfg.data.test_weeks))
    else:
        with open(f"/app/{cfg.data.meta_path}") as f:
            meta = json.load(f)
        splits = (tuple(meta['train_range']), tuple(meta['val_range']), tuple(meta['test_range']))

    out: dict = {"seed": seed, "dataset": main_condition}

    # graph
    online_g = build_graph_encoder(cfg.encoder).to(device)
    ckpt_g = torch.load(f"/results/{main_condition}/seed{seed}/checkpoint.pt", map_location=device)
    online_g.load_state_dict(ckpt_g["online"])
    out["graph"] = run_probe(online_g, graphs, cfg, splits, kind="graph",
                              folds=5, seed=seed, device=device)

    # sequential
    online_s = SequentialMLP(
        in_dim=cfg.encoder.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    ckpt_s = torch.load(f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt",
                         map_location=device)
    online_s.load_state_dict(ckpt_s["online"])
    out["sequential"] = run_probe(online_s, graphs, cfg, splits, kind="sequential",
                                    folds=5, seed=seed, device=device)

    # raw features
    out["raw_features"] = run_probe(None, graphs, cfg, splits, kind="raw_features",
                                      folds=5, seed=seed, device=device)

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "linear_probe_growth"},
    )
    out_dir = f"/results/eval_probe/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/probe.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def topology_seed(seed: int, config_path: str = "configs/baci_gravity.yaml"):
    """test whether frozen embeddings preserve bilateral trade topology.
    spearman correlation of pairwise cosine similarity vs aggregated trade weight."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.eval_topology import run_topology_test

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    out: dict = {"seed": seed, "dataset": main_condition}

    online_g = build_graph_encoder(cfg.encoder).to(device)
    ckpt_g = torch.load(f"/results/{main_condition}/seed{seed}/checkpoint.pt", map_location=device)
    online_g.load_state_dict(ckpt_g["online"])
    out["graph"] = run_topology_test(online_g, graphs, cfg, kind="graph", seed=seed, device=device)

    online_s = SequentialMLP(
        in_dim=cfg.encoder.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    ckpt_s = torch.load(f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt",
                         map_location=device)
    online_s.load_state_dict(ckpt_s["online"])
    out["sequential"] = run_topology_test(online_s, graphs, cfg, kind="sequential",
                                            seed=seed, device=device)

    out["raw_features"] = run_topology_test(None, graphs, cfg, kind="raw_features",
                                              seed=seed, device=device)

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "topology_preservation"},
    )
    out_dir = f"/results/eval_topology/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/topology.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def enron_role_seed(seed: int, config_path: str = "configs/enron.yaml",
                     git_sha: str = ""):
    """test whether frozen enron embeddings cluster people by functional role.
    runs graph + sequential + raw_features and returns within-vs-between cosine
    statistics for each."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.enron_role_recovery import run_role_recovery
    from src.eval.enron_roles import EMAIL_TO_ROLE
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")
    if main_condition != "enron":
        raise ValueError(
            f"enron_role_seed expects enron dataset, got '{main_condition}'"
        )

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    with open(f"/app/{cfg.data.meta_path}") as f:
        meta = json.load(f)
    person_index = meta["person_index"]

    out: dict = {"seed": seed, "dataset": main_condition}

    online_g = build_graph_encoder(cfg.encoder).to(device)
    ckpt_g = torch.load(f"/results/{main_condition}/seed{seed}/checkpoint.pt", map_location=device)
    online_g.load_state_dict(ckpt_g["online"])
    out["graph"] = run_role_recovery(online_g, graphs, cfg, person_index, EMAIL_TO_ROLE,
                                       kind="graph", seed=seed, device=device)

    online_s = SequentialMLP(
        in_dim=cfg.encoder.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    ckpt_s = torch.load(f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt",
                         map_location=device)
    online_s.load_state_dict(ckpt_s["online"])
    out["sequential"] = run_role_recovery(online_s, graphs, cfg, person_index, EMAIL_TO_ROLE,
                                            kind="sequential", seed=seed, device=device)

    out["raw_features"] = run_role_recovery(None, graphs, cfg, person_index, EMAIL_TO_ROLE,
                                              kind="raw_features", seed=seed, device=device)

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "enron_role_recovery"},
    )
    out_dir = f"/results/eval_enron_roles/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/role_recovery.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.local_entrypoint()
def enron_roles(config: str = "configs/enron.yaml", seeds: str = "0,1,2,3,4",
                 out_root: str = "results"):
    """run hidden-role recovery test on Enron across seeds.
    produces results/<dataset>/seed{N}/role_recovery.json + results/enron/ROLE_RECOVERY.md"""
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    if main_condition != "enron":
        print("WARNING: this entrypoint only makes sense on Enron")
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")

    print(f"running enron role recovery on {len(seed_list)} seeds (saving to {results_root}/)")
    all_results = []
    for seed in seed_list:
        try:
            r = enron_role_seed.remote(seed, config, git_sha)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "role_recovery.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        g = r["graph"]; s = r["sequential"]; raw = r["raw_features"]
        print(f"seed {seed}:")
        print(f"  graph: median within={g['median_within_cos']:.3f} between={g['median_between_cos']:.3f} "
              f"gap={g['median_gap']:+.3f} CI=[{g['median_gap_ci95_low']:+.3f}, {g['median_gap_ci95_high']:+.3f}] "
              f"MWU p={g['mwu_p_one_sided']:.2e}")
        print(f"  seq:   median within={s['median_within_cos']:.3f} between={s['median_between_cos']:.3f} "
              f"gap={s['median_gap']:+.3f} CI=[{s['median_gap_ci95_low']:+.3f}, {s['median_gap_ci95_high']:+.3f}] "
              f"MWU p={s['mwu_p_one_sided']:.2e}")
        print(f"  raw:   median within={raw['median_within_cos']:.3f} between={raw['median_between_cos']:.3f} "
              f"gap={raw['median_gap']:+.3f} CI=[{raw['median_gap_ci95_low']:+.3f}, {raw['median_gap_ci95_high']:+.3f}] "
              f"MWU p={raw['mwu_p_one_sided']:.2e}")

    if not all_results:
        return

    # cross-seed median of (median_gap)
    def med(xs): return statistics.median(xs) if xs else float("nan")
    rows = {}
    for cond in ("graph", "sequential", "raw_features"):
        gaps = [r[cond]["median_gap"] for r in all_results]
        ps = [r[cond]["mwu_p_one_sided"] for r in all_results]
        within = [r[cond]["median_within_cos"] for r in all_results]
        between = [r[cond]["median_between_cos"] for r in all_results]
        rows[cond] = {
            "median_gap": med(gaps), "gap_per_seed": gaps,
            "min_p": min(ps) if ps else float("nan"),
            "max_p": max(ps) if ps else float("nan"),
            "median_within": med(within),
            "median_between": med(between),
        }

    g_gap = rows["graph"]["median_gap"]
    s_gap = rows["sequential"]["median_gap"]
    r_gap = rows["raw_features"]["median_gap"]

    lines = [
        f"# enron hidden-role recovery: {main_condition}",
        "",
        "test: do frozen embeddings cluster Enron executives by FUNCTIONAL ROLE",
        "(legal, govaffairs, trading, exec, admin, research, operations) without",
        "any role labels appearing in training? compare pairwise cosine similarity",
        "for within-role pairs vs between-role pairs.",
        "",
        f"seeds: {seed_list}.  n_labeled_people: {all_results[0]['graph']['n_labeled_people']}",
        f"n_within_pairs: {all_results[0]['graph']['n_within_pairs']}, "
        f"n_between_pairs: {all_results[0]['graph']['n_between_pairs']}",
        "",
        "## median(within) − median(between), median across seeds",
        "",
        "| condition | median within | median between | gap | min MWU p | max MWU p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cond in ("graph", "sequential", "raw_features"):
        r = rows[cond]
        lines.append(
            f"| {cond} | {r['median_within']:.3f} | {r['median_between']:.3f} | "
            f"**{r['median_gap']:+.3f}** | {r['min_p']:.2e} | {r['max_p']:.2e} |"
        )
    lines.append("")
    lines.append(f"**gap delta (graph − sequential): {g_gap - s_gap:+.3f}**")
    lines.append(f"**gap delta (graph − raw features): {g_gap - r_gap:+.3f}**")
    lines.append("")
    lines.append("interpretation: a positive gap means within-role pairs are "
                 "MORE similar than between-role pairs. graph >> sequential "
                 "(and graph >> raw_features) means the graph encoder discovered "
                 "role structure that the per-node baseline could not.")

    md_path = results_root / "ROLE_RECOVERY.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")

    sj = results_root / "role_recovery_summary.json"
    with open(sj, "w") as f:
        json.dump({"dataset": main_condition, "seeds": seed_list, "rows": rows}, f, indent=2)
    print(f"saved {sj}")


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def per_country_anomaly_seed(seed: int, config_path: str = "configs/baci_gravity.yaml",
                              year_idx: int = 24, git_sha: str = ""):
    """compute per-country prediction cosine at a target year by masking
    ALL nodes at graphs[year_idx]. returns per-country cosine vector for
    both graph-jepa and sequential-ablation conditions, plus actual trade
    decline ground truth from BACI raw."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.per_country_anomaly import (
        per_country_cosines_at_year,
        compute_actual_trade_decline_per_country,
    )
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")

    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    with open(f"/app/{cfg.data.meta_path}") as f:
        meta = json.load(f)
    year_labels = meta.get("years_kept") or list(range(len(graphs)))

    out: dict = {
        "seed": seed, "dataset": main_condition,
        "year_idx": year_idx,
        "year_label": year_labels[year_idx] if year_idx < len(year_labels) else None,
        "n_nodes": graphs[0].x.shape[0],
    }

    def _load_and_eval(condition: str, builder_kind: str):
        ckpt_path = (f"/results/{main_condition}/seed{seed}/checkpoint.pt"
                      if builder_kind == "graph"
                      else f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt")
        if builder_kind == "graph":
            online = build_graph_encoder(cfg.encoder).to(device)
        else:
            online = SequentialMLP(
                in_dim=cfg.encoder.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                n_layers=cfg.encoder.n_layers,
                dropout=cfg.encoder.dropout,
            ).to(device)
        predictor = build_predictor(cfg.predictor).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        online.load_state_dict(ckpt["online"])
        predictor.load_state_dict(ckpt["predictor"])
        target = build_target_encoder(online)
        target.encoder = target.encoder.to(device)
        if "target_encoder" in ckpt:
            target.encoder.load_state_dict(ckpt["target_encoder"])
        return per_country_cosines_at_year(online, target, predictor, graphs, cfg,
                                             year_idx=year_idx, mask_seed=seed, device=device)

    out["graph_per_country_cos"] = _load_and_eval("graph", "graph").tolist()
    out["sequential_per_country_cos"] = _load_and_eval("sequential", "sequential").tolist()

    # ground truth: actual trade decline from BACI raw
    # ref = year before target (e.g., 2019 if target=2020)
    ref_year_idx = year_idx - 1
    decline, active = compute_actual_trade_decline_per_country(graphs, ref_year_idx, year_idx)
    out["actual_trade_decline"] = decline.tolist()
    out["country_active_in_ref_year"] = active.tolist()
    out["ref_year_idx"] = ref_year_idx
    out["ref_year_label"] = year_labels[ref_year_idx] if ref_year_idx < len(year_labels) else None
    if hasattr(meta, "get") and meta.get("iso_sorted"):
        out["iso_sorted"] = meta["iso_sorted"]
    elif "iso_sorted" in meta:
        out["iso_sorted"] = meta["iso_sorted"]

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "per_country_anomaly", "year_idx": year_idx},
    )
    out_dir = f"/results/eval_per_country/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/per_country_anomaly_year{year_idx}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.local_entrypoint()
def per_country_anomaly(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4",
                         year_idx: int = 24, out_root: str = "results"):
    """run per-country anomaly across seeds + compute correlation to actual
    trade decline. for BACI default year_idx=24 (year 2020, COVID).

    produces:
      paper_results/<dataset>/seed{N}/per_country_anomaly_year{Y}.json
      paper_results/<dataset>/PER_COUNTRY_ANOMALY_year{Y}.md
    """
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")

    print(f"running per-country anomaly at year_idx={year_idx} on {len(seed_list)} seeds")

    import numpy as np
    all_results = []
    for seed in seed_list:
        try:
            r = per_country_anomaly_seed.remote(seed, config, year_idx, git_sha)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"per_country_anomaly_year{year_idx}.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)

    if not all_results:
        print("no results")
        return

    # average per-country cosine across seeds
    n_nodes = all_results[0]["n_nodes"]
    graph_cos = np.stack([np.array(r["graph_per_country_cos"]) for r in all_results])  # [n_seeds, N]
    seq_cos = np.stack([np.array(r["sequential_per_country_cos"]) for r in all_results])
    decline = np.array(all_results[0]["actual_trade_decline"])
    active = np.array(all_results[0]["country_active_in_ref_year"], dtype=bool)
    iso = all_results[0].get("iso_sorted", [str(i) for i in range(n_nodes)])

    graph_dev = 1.0 - graph_cos.mean(axis=0)  # [N], higher = more anomalous
    seq_dev = 1.0 - seq_cos.mean(axis=0)

    # restrict to active countries (had any trade in reference year)
    mask = active

    # spearman correlation between predicted deviation and actual trade decline magnitude
    # decline is signed (negative = drop). use |decline| for "magnitude of impact"
    # OR use -decline (so larger value = bigger drop)
    impact = -decline  # higher = bigger drop in 2020
    def _spearman(a, b):
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    rho_graph = _spearman(graph_dev[mask], impact[mask])
    rho_seq = _spearman(seq_dev[mask], impact[mask])

    # bootstrap CI on the spearman gap
    rng = np.random.default_rng(0)
    n_active = int(mask.sum())
    n_resamples = 5000
    boot_graph = np.empty(n_resamples)
    boot_seq = np.empty(n_resamples)
    g_active = graph_dev[mask]
    s_active = seq_dev[mask]
    i_active = impact[mask]
    for r in range(n_resamples):
        idx = rng.integers(0, n_active, size=n_active)
        boot_graph[r] = _spearman(g_active[idx], i_active[idx])
        boot_seq[r] = _spearman(s_active[idx], i_active[idx])

    print()
    print(f"=== per-country anomaly at year {all_results[0]['year_label']} ===")
    print(f"n_active countries: {n_active} / {n_nodes}")
    print()
    print(f"  graph: Spearman ρ(predicted deviation, actual trade decline) = {rho_graph:.3f}")
    print(f"         bootstrap CI95 = [{np.quantile(boot_graph, 0.025):.3f}, "
          f"{np.quantile(boot_graph, 0.975):.3f}]")
    print(f"  seq:   Spearman ρ = {rho_seq:.3f}")
    print(f"         bootstrap CI95 = [{np.quantile(boot_seq, 0.025):.3f}, "
          f"{np.quantile(boot_seq, 0.975):.3f}]")
    print(f"  Δρ (graph − seq) = {rho_graph - rho_seq:+.3f}")

    # top-K anomalous countries by graph
    top_idx_graph = np.argsort(graph_dev)[::-1][:20]
    top_decline_idx = np.argsort(impact)[::-1][:20]

    print()
    print("top-20 countries by GRAPH-JEPA 2020 deviation:")
    for i, ix in enumerate(top_idx_graph):
        if mask[ix]:
            iso_code = iso[ix] if ix < len(iso) else f"node_{ix}"
            print(f"  {i+1:2d}. {iso_code} dev={graph_dev[ix]:.3f} actual_decline={decline[ix]*100:+.1f}%")

    print()
    print("top-20 countries by ACTUAL 2020 trade decline:")
    for i, ix in enumerate(top_decline_idx):
        if mask[ix]:
            iso_code = iso[ix] if ix < len(iso) else f"node_{ix}"
            print(f"  {i+1:2d}. {iso_code} actual={decline[ix]*100:+.1f}% graph_dev={graph_dev[ix]:.3f}")

    # write markdown
    lines = [
        f"# per-country anomaly: {main_condition} year {all_results[0]['year_label']}",
        "",
        f"target year: {all_results[0]['year_label']} (snap_idx {year_idx})",
        f"reference year: {all_results[0]['ref_year_label']} (snap_idx {all_results[0]['ref_year_idx']})",
        f"n_active countries: {n_active} of {n_nodes}",
        f"n_seeds: {len(all_results)}",
        "",
        "## Spearman correlation: predicted deviation vs actual trade decline",
        "",
        "| condition | Spearman ρ | bootstrap CI95 |",
        "|---|---:|---|",
        f"| graph-JEPA | {rho_graph:.3f} | "
        f"[{np.quantile(boot_graph, 0.025):.3f}, {np.quantile(boot_graph, 0.975):.3f}] |",
        f"| sequential | {rho_seq:.3f} | "
        f"[{np.quantile(boot_seq, 0.025):.3f}, {np.quantile(boot_seq, 0.975):.3f}] |",
        "",
        f"**Δρ (graph − sequential): {rho_graph - rho_seq:+.3f}**",
        "",
        "interpretation: a positive Spearman ρ means countries flagged as more anomalous "
        "by the model also experienced larger actual trade declines. ρ > 0.3 with CI excluding 0 "
        "supports the claim that the model recovered the country-level pattern of the shock.",
    ]
    md_path = results_root / f"PER_COUNTRY_ANOMALY_year{year_idx}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")


@app.function(
    gpu="H100",
    timeout=900,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def random_baseline_seed(seed: int, config_path: str = "configs/baci_gravity.yaml",
                          git_sha: str = ""):
    """RG-1 + RG-10: untrained random encoder eff_rank, plus random-projection
    baseline. tests whether the d≈8 compression is a property of JEPA training
    (vs of the architecture or of random projection)."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    import torch.nn.functional as F
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.random_baseline import random_encoder_eff_rank
    from src.eval.metrics import effective_rank, mean_pairwise_cosine
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")
    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")

    # RG-1a: untrained random GraphEncoder
    online_g = build_graph_encoder(cfg.encoder).to(device)
    out_g = random_encoder_eff_rank(online_g, graphs, cfg, kind="random_graph_encoder", device=device)

    # RG-1b: untrained random SequentialMLP
    online_s = SequentialMLP(
        in_dim=cfg.encoder.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    out_s = random_encoder_eff_rank(online_s, graphs, cfg, kind="random_sequential_encoder", device=device)

    # RG-10: random-projection baseline (project raw 6d features to hidden_dim via random matrix)
    rng_torch = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(cfg.encoder.in_dim, cfg.encoder.hidden_dim, generator=rng_torch).to(device)
    R = F.normalize(R, dim=0)  # unit-norm columns for fair comparison
    K = cfg.training.context_k
    all_emb_proj = []
    for t in range(K, len(graphs)):
        x = graphs[t].x.to(device)  # [N, in_dim]
        z = x @ R  # [N, hidden_dim]
        z = F.normalize(z, dim=-1)
        all_emb_proj.append(z.cpu())
    full = torch.cat(all_emb_proj, dim=0)
    out_rp = {
        "kind": "random_projection_baseline",
        "effective_rank": float(effective_rank(full)),
        "mean_pairwise_cosine": float(mean_pairwise_cosine(full)),
        "n_embeddings": int(full.shape[0]),
        "embedding_dim": int(full.shape[1]),
    }

    # also report PARAMETER COUNTS (RG-4 — the capacity-matched audit)
    def _count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    param_counts = {
        "graph_encoder": _count_params(online_g),
        "sequential_encoder": _count_params(online_s),
    }

    out = {
        "seed": seed,
        "dataset": main_condition,
        "RG_1a_random_graph": out_g,
        "RG_1b_random_sequential": out_s,
        "RG_10_random_projection": out_rp,
        "RG_4_param_counts": param_counts,
    }

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "random_baseline + random_projection"},
    )
    out_dir = f"/results/eval_random_baseline/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/random_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def enron_per_person_anomaly_seed(seed: int, config_path: str = "configs/enron.yaml",
                                    git_sha: str = ""):
    """E-P: per-person prediction cosines across all valid snapshots in Enron.
    used to compute fraud-week anomaly (per-person dev minus per-person baseline)
    grouped by role. tests whether the model registers ROLE-RELEVANT anomalous
    behavior during fraud events: hypothesis is that legal + govaffairs spike
    more than admin at W41/W42/W45/W48."""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    import numpy as np
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.models.sequential_encoder import SequentialMLP
    from src.eval.per_country_anomaly import per_country_cosines_at_year
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")
    if main_condition != "enron":
        raise ValueError(f"this is enron-specific, got {main_condition}")
    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")
    with open(f"/app/{cfg.data.meta_path}") as f:
        meta = json.load(f)

    K = cfg.training.context_k
    valid_indices = list(range(K, len(graphs)))

    def _load_and_score(condition: str):
        ckpt_path = (f"/results/{main_condition}/seed{seed}/checkpoint.pt"
                      if condition == "graph"
                      else f"/results/{main_condition}/sequential-ablation/seed{seed}/checkpoint.pt")
        if condition == "graph":
            online = build_graph_encoder(cfg.encoder).to(device)
        else:
            online = SequentialMLP(
                in_dim=cfg.encoder.in_dim,
                hidden_dim=cfg.encoder.hidden_dim,
                n_layers=cfg.encoder.n_layers,
                dropout=cfg.encoder.dropout,
            ).to(device)
        predictor = build_predictor(cfg.predictor).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        online.load_state_dict(ckpt["online"])
        predictor.load_state_dict(ckpt["predictor"])
        target = build_target_encoder(online)
        target.encoder = target.encoder.to(device)
        if "target_encoder" in ckpt:
            target.encoder.load_state_dict(ckpt["target_encoder"])
        # cosines for all valid snapshots: shape [n_snaps, n_persons]
        cosines_per_snap = []
        for idx in valid_indices:
            cosines = per_country_cosines_at_year(online, target, predictor, graphs, cfg,
                                                    year_idx=idx, mask_seed=seed, device=device)
            cosines_per_snap.append(cosines.tolist())
        return cosines_per_snap

    out: dict = {
        "seed": seed,
        "dataset": main_condition,
        "valid_snapshot_indices": valid_indices,
        "person_index": meta["person_index"],
        "weeks_meta": [{"week_idx": w["week_idx"], "date_str": w["date_str"]}
                        for w in meta["weeks"]],
        "graph_per_person_cos": _load_and_score("graph"),
        "sequential_per_person_cos": _load_and_score("sequential"),
    }

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "enron_per_person_anomaly"},
    )
    out_dir = f"/results/eval_enron_per_person/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/per_person_cosines.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.local_entrypoint()
def enron_per_person(config: str = "configs/enron.yaml", seeds: str = "0,1,2,3,4",
                      out_root: str = "results"):
    """E-P: per-person fraud-week anomaly grouped by role.
    saves per-seed JSONs; downstream analysis runs locally via
    scripts/enron_fraud_anomaly.py."""
    import json
    from pathlib import Path

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")

    print(f"running enron per-person anomaly on {len(seed_list)} seeds")
    for seed in seed_list:
        try:
            r = enron_per_person_anomaly_seed.remote(seed, config, git_sha)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "per_person_cosines.json", "w") as f:
            json.dump(r, f, indent=2)
        n_snaps = len(r["valid_snapshot_indices"])
        print(f"seed {seed}: collected {n_snaps} snapshots × 50 persons saved")

    print("\nrun local analysis: python scripts/enron_fraud_anomaly.py")


@app.function(
    gpu="H100",
    timeout=1800,
    image=image,
    volumes={"/results": vol},
    max_containers=1,
)
def synthetic_shock_seed(seed: int, config_path: str = "configs/baci_gravity.yaml",
                          year_idx: int = 14, n_top_countries: int = 30,
                          git_sha: str = ""):
    """RG-A: synthetic shock injection. perturb the graph at year_idx by
    zeroing out incident edges for top-N countries (by trade volume), measure
    prediction-error increase. tests whether the model has internalized graph
    structure — does dropping high-volume countries cause more error than
    dropping low-volume ones?"""
    import sys
    sys.path.insert(0, "/app")
    import json
    import os
    os.chdir("/app")
    import torch
    from omegaconf import OmegaConf
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    from src.eval.synthetic_shock import run_synthetic_shock
    from src.utils.seed import set_seed

    if git_sha:
        os.environ["GIT_SHA"] = git_sha
    set_seed(seed)

    cfg = OmegaConf.load(f"/app/{config_path}")
    device = torch.device("cuda")
    main_condition = cfg.get("dataset", "tgjepa")
    graphs = torch.load(f"/app/{cfg.data.graphs_path}", map_location="cpu")

    online = build_graph_encoder(cfg.encoder).to(device)
    ckpt = torch.load(f"/results/{main_condition}/seed{seed}/checkpoint.pt", map_location=device)
    online.load_state_dict(ckpt["online"])
    predictor = build_predictor(cfg.predictor).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    target = build_target_encoder(online)
    target.encoder = target.encoder.to(device)
    if "target_encoder" in ckpt:
        target.encoder.load_state_dict(ckpt["target_encoder"])

    out = run_synthetic_shock(online, target, predictor, graphs, cfg,
                                year_idx=year_idx, n_top_countries=n_top_countries,
                                seed=seed, device=device)
    out["seed"] = seed
    out["dataset"] = main_condition

    from src.utils.repro import capture_metadata
    out["_repro"] = capture_metadata(
        dataset_paths=[f"/app/{cfg.data.graphs_path}", f"/app/{cfg.data.meta_path}"],
        config_path=f"/app/{config_path}",
        extra={"eval": "synthetic_shock", "year_idx": year_idx, "n_top": n_top_countries},
    )
    out_dir = f"/results/eval_synthetic_shock/{main_condition}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/synthetic_shock_year{year_idx}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.local_entrypoint()
def synthetic_shock(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4",
                     year_idx: int = 14, n_top_countries: int = 30,
                     out_root: str = "results"):
    """run synthetic shock injection across seeds + correlate prediction-error
    increase with knocked-out country's trade volume + degree centrality."""
    import json
    from pathlib import Path
    import statistics
    import numpy as np

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    git_sha = _local_git_sha()

    print(f"running synthetic shock at year_idx={year_idx}, top {n_top_countries} countries")
    all_results = []
    for seed in seed_list:
        try:
            r = synthetic_shock_seed.remote(seed, config, year_idx, n_top_countries, git_sha)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"synthetic_shock_year{year_idx}.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        print(f"seed {seed}: ρ(Δdev_excl_self vs volume) = {r['spearman_dev_vs_volume']:+.3f}, "
              f"ρ(vs degree) = {r['spearman_dev_vs_degree']:+.3f}, "
              f"max Δdev = {r['max_delta_dev_excl_self']:+.4f}")

    if not all_results:
        return

    rho_vol = [r["spearman_dev_vs_volume"] for r in all_results]
    rho_deg = [r["spearman_dev_vs_degree"] for r in all_results]
    max_delta = [r["max_delta_dev_excl_self"] for r in all_results]

    print(f"\n=== synthetic shock summary across {len(all_results)} seeds ===")
    print(f"  Spearman ρ(Δdev vs volume):  median {statistics.median(rho_vol):+.3f}, "
          f"range [{min(rho_vol):+.3f}, {max(rho_vol):+.3f}]")
    print(f"  Spearman ρ(Δdev vs degree):  median {statistics.median(rho_deg):+.3f}, "
          f"range [{min(rho_deg):+.3f}, {max(rho_deg):+.3f}]")
    print(f"  max Δdev (excluding self):   median {statistics.median(max_delta):+.4f}")

    # write report
    lines = [
        f"# RG-A: synthetic shock injection on {main_condition}",
        "",
        f"perturbation year: snap_idx {year_idx} (model has seen this in training)",
        f"prediction target: snap_idx {year_idx + 1}",
        f"top {n_top_countries} countries tested (by trade volume)",
        f"seeds: {seed_list}",
        "",
        "for each candidate country, we zero out all incident edges in the perturbation",
        "snapshot, then predict the next snapshot. we measure prediction-error increase",
        "for the OTHER N-1 countries (not the knocked-out one). if the model has internalized",
        "graph structure, dropping high-volume countries should produce larger error.",
        "",
        "## Spearman ρ between (Δprediction-error excluding self) and (knocked-out country's centrality)",
        "",
        "| measure | median ρ across seeds | min | max |",
        "|---|---:|---:|---:|",
        f"| volume centrality | {statistics.median(rho_vol):+.3f} | {min(rho_vol):+.3f} | {max(rho_vol):+.3f} |",
        f"| degree centrality | {statistics.median(rho_deg):+.3f} | {min(rho_deg):+.3f} | {max(rho_deg):+.3f} |",
        "",
        f"max delta-deviation across knockouts: median {statistics.median(max_delta):+.4f}",
        "",
        "**reading**: ρ > 0 means dropping more-central countries causes more disruption to",
        "the model's prediction of OTHER countries' next-snapshot embeddings. positive ρ is",
        "evidence the model has internalized which countries are graph-structurally important.",
    ]
    md_path = results_root / f"SYNTHETIC_SHOCK_year{year_idx}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")


@app.local_entrypoint()
def random_baseline(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4",
                     out_root: str = "results"):
    """RG-1 + RG-10 + RG-4: random untrained encoder eff_rank + random projection
    eff_rank + parameter count audit. tests whether d≈8 attractor is a property
    of JEPA training (not of architecture or of random projection)."""
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")

    print(f"running random-encoder + random-projection baselines on {len(seed_list)} seeds")
    all_results = []
    for seed in seed_list:
        try:
            r = random_baseline_seed.remote(seed, config, git_sha)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "random_baseline.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        a, b, c = r["RG_1a_random_graph"], r["RG_1b_random_sequential"], r["RG_10_random_projection"]
        print(f"seed {seed}: random_graph eff_rank={a['effective_rank']:.2f} | "
              f"random_seq eff_rank={b['effective_rank']:.2f} | "
              f"random_proj eff_rank={c['effective_rank']:.2f}")
        pc = r["RG_4_param_counts"]
        print(f"  param counts: graph={pc['graph_encoder']:,}, sequential={pc['sequential_encoder']:,}")

    if not all_results:
        return

    def med(xs): return statistics.median(xs)
    rg = [r["RG_1a_random_graph"]["effective_rank"] for r in all_results]
    rs = [r["RG_1b_random_sequential"]["effective_rank"] for r in all_results]
    rp = [r["RG_10_random_projection"]["effective_rank"] for r in all_results]

    lines = [
        f"# RG-1 + RG-10 + RG-4: random baselines for {main_condition}",
        "",
        f"5-seed median effective rank for **untrained** encoders compared to JEPA-trained.",
        "",
        "| condition | median eff_rank | min | max |",
        "|---|---:|---:|---:|",
        f"| **random graph encoder** (RG-1a) | {med(rg):.2f} | {min(rg):.2f} | {max(rg):.2f} |",
        f"| **random sequential encoder** (RG-1b) | {med(rs):.2f} | {min(rs):.2f} | {max(rs):.2f} |",
        f"| **random 6d→256 projection** (RG-10) | {med(rp):.2f} | {min(rp):.2f} | {max(rp):.2f} |",
        "",
        "Compare to JEPA-trained eff_rank (from main eval). If random encoders produce eff_rank ≈ 8,",
        "the 'JEPA training induces d≈8 attractor' claim is wrong — geometry is from architecture",
        "or from the input dim, not from training.",
        "",
        "## RG-4: parameter counts",
        "",
        f"- graph encoder: **{all_results[0]['RG_4_param_counts']['graph_encoder']:,}** parameters",
        f"- sequential encoder: **{all_results[0]['RG_4_param_counts']['sequential_encoder']:,}** parameters",
        "",
        "predictor + target encoder shared across both conditions; the difference between",
        "encoder param counts is the 'capacity-matched' margin.",
    ]
    md_path = results_root / "RANDOM_BASELINE.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")


@app.local_entrypoint()
def topology(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4"):
    """run topology preservation test across seeds. produces:
       results/<dataset>/seed{N}/topology.json + results/<dataset>/TOPOLOGY.md"""
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path("results") / main_condition

    print(f"running topology preservation on {len(seed_list)} seeds for '{main_condition}'")
    all_results = []
    for seed in seed_list:
        try:
            r = topology_seed.remote(seed, config)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "topology.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        g = r["graph"]; s = r["sequential"]; raw = r["raw_features"]
        print(f"seed {seed}: "
              f"graph ρ={g['spearman_rho']:.3f} CI95=[{g['spearman_ci95_low']:.3f}, {g['spearman_ci95_high']:.3f}] | "
              f"seq ρ={s['spearman_rho']:.3f} | "
              f"raw ρ={raw['spearman_rho']:.3f}")

    if not all_results:
        return

    def med(values): return statistics.median(values) if values else float("nan")
    rows = {}
    for cond in ("graph", "sequential", "raw_features"):
        rows[cond] = {
            "rho_median": med([r[cond]["spearman_rho"] for r in all_results]),
            "rho_per_seed": [r[cond]["spearman_rho"] for r in all_results],
            "ci_low_median": med([r[cond]["spearman_ci95_low"] for r in all_results]),
            "ci_high_median": med([r[cond]["spearman_ci95_high"] for r in all_results]),
        }

    g_med = rows["graph"]["rho_median"]
    s_med = rows["sequential"]["rho_median"]
    r_med = rows["raw_features"]["rho_median"]

    n_pairs = all_results[0]["graph"]["n_pairs"]
    lines = [
        f"# topology preservation: {main_condition}",
        "",
        f"Spearman correlation of pairwise embedding cosine similarity vs "
        f"log(aggregate bilateral trade weight) across the test split.",
        f"n_pairs = {n_pairs} (upper triangle of {all_results[0]['graph']['n_countries']}×country matrix).",
        f"seeds: {seed_list}.",
        "",
        "## Spearman ρ (median across seeds)",
        "",
        "| condition | ρ | bootstrap CI95 (median) | per-seed ρ |",
        "|---|---:|---:|---|",
    ]
    for cond in ("graph", "sequential", "raw_features"):
        r = rows[cond]
        per_seed = ", ".join(f"{x:.3f}" for x in r["rho_per_seed"])
        lines.append(
            f"| {cond} | {r['rho_median']:.3f} | "
            f"[{r['ci_low_median']:.3f}, {r['ci_high_median']:.3f}] | {per_seed} |"
        )
    lines.append("")
    lines.append(f"**ρ gap (graph − sequential): {g_med - s_med:+.3f}**")
    lines.append(f"**ρ gap (graph − raw features): {g_med - r_med:+.3f}**")
    lines.append("")
    lines.append("interpretation: a positive ρ means countries that trade heavily "
                 "have similar embeddings. graph ρ >> sequential ρ would mean "
                 "the graph encoder learned bilateral trade topology that the "
                 "non-graph baseline could not.")

    md_path = results_root / "TOPOLOGY.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")

    sj = results_root / "topology_summary.json"
    with open(sj, "w") as f:
        json.dump({"dataset": main_condition, "seeds": seed_list, "rows": rows}, f, indent=2)
    print(f"saved {sj}")


@app.local_entrypoint()
def probe(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4"):
    """run linear probe across seeds, aggregate R² and bootstrap CIs.
    produces results/<dataset>/seed{N}/probe.json + results/<dataset>/PROBE.md."""
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path("results") / main_condition

    print(f"running linear probe on {len(seed_list)} seeds for '{main_condition}'")
    all_results = []
    for seed in seed_list:
        try:
            r = probe_seed.remote(seed, config)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "probe.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        g = r["graph"]; s = r["sequential"]; raw = r["raw_features"]
        print(f"seed {seed}: "
              f"graph R²={g['r2_mean']:.3f}±{g['r2_std']:.3f} | "
              f"seq R²={s['r2_mean']:.3f}±{s['r2_std']:.3f} | "
              f"raw R²={raw['r2_mean']:.3f}±{raw['r2_std']:.3f}")

    if not all_results:
        return

    # aggregate
    def med(values): return statistics.median(values) if values else float("nan")
    rows = {}
    for cond in ("graph", "sequential", "raw_features"):
        rows[cond] = {
            "r2_median": med([r[cond]["r2_mean"] for r in all_results]),
            "r2_per_seed": [r[cond]["r2_mean"] for r in all_results],
            "mae_median": med([r[cond]["mae_mean"] for r in all_results]),
            "mae_per_seed": [r[cond]["mae_mean"] for r in all_results],
        }

    g_med = rows["graph"]["r2_median"]
    s_med = rows["sequential"]["r2_median"]
    r_med = rows["raw_features"]["r2_median"]

    lines = [
        f"# linear probe: {main_condition}",
        "",
        "frozen-encoder 5-fold ridge regression. target: log(trade_vol[t+1]) − log(trade_vol[t]).",
        f"seeds: {seed_list}.",
        "",
        "## R² (5-fold CV, median across seeds)",
        "",
        "| condition | R² | MAE | per-seed R² |",
        "|---|---:|---:|---|",
    ]
    for cond in ("graph", "sequential", "raw_features"):
        r = rows[cond]
        per_seed = ", ".join(f"{x:.3f}" for x in r["r2_per_seed"])
        lines.append(f"| {cond} | {r['r2_median']:.3f} | {r['mae_median']:.3f} | {per_seed} |")
    lines.append("")
    lines.append(f"**R² gap (graph − sequential): {g_med - s_med:+.3f}**")
    lines.append(f"**R² gap (graph − raw features): {g_med - r_med:+.3f}**")
    lines.append("")
    lines.append("a positive gap (especially > +0.05) means graph-JEPA's frozen "
                 "embeddings linearly carry information about next-period growth "
                 "that the corresponding baseline does not.")

    md_path = results_root / "PROBE.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")

    sj = results_root / "probe_summary.json"
    with open(sj, "w") as f:
        json.dump({"dataset": main_condition, "seeds": seed_list, "rows": rows}, f, indent=2)
    print(f"saved {sj}")


@app.local_entrypoint()
def blocs(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4",
          partition: str = "subregion", k: int = 0):
    """run bloc discovery across seeds, aggregate ARI/NMI/purity for graph
    vs sequential vs raw-features baseline. produces:
       results/<dataset>/seed{N}/bloc_discovery_<partition>_k<k>.json (saved to subdir)
       results/<dataset>/BLOCS_<partition>_k<k>.md
       results/<dataset>/blocs_summary_<partition>_k<k>.json

    args:
        partition: "subregion" (UN M49, 17 buckets, default) or "continent" (5 buckets)
        k: cluster count (default 0 → use the partition's natural count)
    """
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path("results") / main_condition
    suffix = f"_{partition}" + (f"_k{k}" if k > 0 else "")

    print(f"running bloc discovery on {len(seed_list)} seeds for '{main_condition}' "
          f"(partition={partition}, k={k or 'auto'})")
    all_results = []
    for seed in seed_list:
        try:
            r = bloc_discovery_seed.remote(seed, config, partition, k)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        if "error" in r:
            print(f"seed {seed}: {r['error']}")
            return
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"bloc_discovery{suffix}.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        g = r["graph"]; s = r["sequential"]; raw = r["raw_features"]
        print(f"seed {seed}: "
              f"graph ARI={g['ari']:.3f} NMI={g['nmi']:.3f} purity={g['purity']:.3f} | "
              f"seq ARI={s['ari']:.3f} NMI={s['nmi']:.3f} purity={s['purity']:.3f} | "
              f"raw ARI={raw['ari']:.3f} NMI={raw['nmi']:.3f} purity={raw['purity']:.3f}")

    if not all_results:
        return

    # cross-seed medians
    def med(values): return statistics.median(values) if values else float("nan")
    rows = {}
    for cond in ("graph", "sequential", "raw_features"):
        rows[cond] = {
            "ari": med([r[cond]["ari"] for r in all_results]),
            "nmi": med([r[cond]["nmi"] for r in all_results]),
            "purity": med([r[cond]["purity"] for r in all_results]),
            "ari_per_seed": [r[cond]["ari"] for r in all_results],
            "nmi_per_seed": [r[cond]["nmi"] for r in all_results],
            "purity_per_seed": [r[cond]["purity"] for r in all_results],
        }

    lines = [
        f"# bloc discovery: {main_condition}",
        "",
        f"unsupervised k-means clustering of frozen country embeddings, scored against "
        f"UN M49 subregion ground truth ({all_results[0]['graph']['n_clusters']} clusters, "
        f"{all_results[0]['n_countries']} countries). seeds: {seed_list}.",
        "",
        "## results (median across seeds)",
        "",
        "| condition | ARI | NMI | purity |",
        "|---|---:|---:|---:|",
    ]
    for cond in ("graph", "sequential", "raw_features"):
        r = rows[cond]
        lines.append(f"| {cond} | {r['ari']:.3f} | {r['nmi']:.3f} | {r['purity']:.3f} |")
    lines.append("")
    lines.append("## per-seed breakdown")
    lines.append("")
    for cond in ("graph", "sequential", "raw_features"):
        lines.append(f"### {cond}")
        lines.append("")
        lines.append("| seed | ARI | NMI | purity |")
        lines.append("|---|---:|---:|---:|")
        for i, seed in enumerate(seed_list[:len(all_results)]):
            r = rows[cond]
            lines.append(f"| {seed} | {r['ari_per_seed'][i]:.3f} | "
                         f"{r['nmi_per_seed'][i]:.3f} | "
                         f"{r['purity_per_seed'][i]:.3f} |")
        lines.append("")

    md_path = results_root / f"BLOCS{suffix}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {md_path}")

    summary = {"dataset": main_condition, "seeds": seed_list,
               "partition": partition, "k_override": k, "rows": rows}
    sj = results_root / f"blocs_summary{suffix}.json"
    with open(sj, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {sj}")


@app.local_entrypoint()
def anomaly(config: str = "configs/baci_gravity.yaml", seeds: str = "0,1,2,3,4",
            out_root: str = "results"):
    """run per-snapshot anomaly trajectory across seeds and emit a markdown
    report flagging known economic shocks."""
    import json
    from pathlib import Path
    import statistics

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    shocks = KNOWN_SHOCKS_BY_DATASET.get(main_condition, {})
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")

    print(f"running anomaly trajectory on {len(seed_list)} seeds for '{main_condition}' "
          f"(saving to {results_root}/)")
    all_results = []
    for seed in seed_list:
        try:
            r = anomaly_trajectory.remote(seed, config, git_sha)
        except Exception as e:
            print(f"seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "anomaly_trajectory.json", "w") as f:
            json.dump(r, f, indent=2)
        all_results.append(r)
        print(f"saved {out_dir / 'anomaly_trajectory.json'}")

    if not all_results:
        print("no results — exiting")
        return

    # build per-year median deviation across seeds, per condition
    year_to_idx: dict = {}
    cond_to_year_to_devs: dict = {"graph": {}, "sequential": {}}
    for r in all_results:
        for cond in ("graph", "sequential"):
            for snap in r[cond]["per_snapshot"]:
                y = snap["year"]
                cond_to_year_to_devs[cond].setdefault(y, []).append(snap["deviation"])
                year_to_idx[y] = snap["snapshot_idx"]

    years_sorted = sorted(year_to_idx.keys())

    # markdown report
    lines = [
        f"# anomaly trajectory: {main_condition}",
        "",
        f"per-snapshot prediction deviation (1 - mean_pred_cos), median across seeds {seed_list}.",
        "shock years are flagged with ★. graph-vs-sequential gap on shock years tells us "
        "whether graph-JEPA picks up real-world dynamics more cleanly than the non-graph baseline.",
        "",
        "| year | snap idx | graph dev (median) | seq dev (median) | shock? | label |",
        "|---|---:|---:|---:|:-:|---|",
    ]
    rows: list[dict] = []
    for y in years_sorted:
        g_devs = cond_to_year_to_devs["graph"].get(y, [])
        s_devs = cond_to_year_to_devs["sequential"].get(y, [])
        if not g_devs or not s_devs:
            continue
        g_med = statistics.median(g_devs)
        s_med = statistics.median(s_devs)
        is_shock = y in shocks
        label = shocks.get(y, "")
        marker = "★" if is_shock else ""
        lines.append(
            f"| {y} | {year_to_idx[y]} | {g_med:.4f} | {s_med:.4f} | {marker} | {label} |"
        )
        rows.append({"year": y, "graph_dev": g_med, "sequential_dev": s_med, "shock": is_shock})

    # shock-spike score: did each model show elevated deviation on shock years
    # compared to non-shock years? a clean shock-detector should show shock
    # deviations significantly above the non-shock median.
    lines.append("")
    lines.append("## shock detection score")
    lines.append("")
    for cond in ("graph", "sequential"):
        non_shock = [r[f"{cond}_dev"] for r in rows if not r["shock"]]
        shock = [r[f"{cond}_dev"] for r in rows if r["shock"]]
        if non_shock and shock:
            base = statistics.median(non_shock)
            lift = statistics.median(shock) - base
            ratio = (statistics.median(shock) / base) if base > 0 else float("inf")
            lines.append(
                f"- **{cond}**: non-shock median dev = {base:.4f}, "
                f"shock median dev = {statistics.median(shock):.4f}, "
                f"lift = {lift:+.4f} ({ratio:.2f}× baseline)"
            )

    summary_path = results_root / "ANOMALY_TRAJECTORY.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {summary_path}")

    # also emit a small json with per-year medians so the result can be plotted
    summary_json = {
        "dataset": main_condition,
        "seeds": seed_list,
        "per_year": rows,
        "shocks_known": shocks,
    }
    with open(results_root / "anomaly_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"saved {results_root / 'anomaly_summary.json'}")


@app.local_entrypoint()
def rigor(config: str = "configs/tgbn_trade.yaml", seeds: str = "0,1,2,3,4",
          out_root: str = "results"):
    """rigor adds for the headline paired comparison:
    - bootstrap 95% CI on per-seed mean Δ (test split)
    - train-split paired diagnostic (memorization sanity check)

    runs against existing checkpoints; no retraining. produces a
    RIGOR_SUMMARY.md with both bootstrap and train-vs-test comparisons.
    """
    import json
    from pathlib import Path

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    results_root = Path(out_root) / main_condition
    git_sha = _local_git_sha()
    if git_sha:
        print(f"[repro] git_sha={git_sha}")

    print(f"running rigor checks on {len(seed_list)} seeds for '{main_condition}' "
          f"(saving to {results_root}/)")

    # 1. test-split with bootstrap CI
    print("\n--- bootstrap CI on Δ (test split) ---")
    test_results = []
    for seed in seed_list:
        try:
            result = eval_paired_seed.remote(seed, config, "self", "test", git_sha)
        except Exception as e:
            print(f"test seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        print(f"seed {seed}: Δ={result.get('mean_delta', 'n/a'):.4f}  "
              f"CI95=[{result.get('delta_ci95_low', 'n/a'):.4f}, "
              f"{result.get('delta_ci95_high', 'n/a'):.4f}]  "
              f"win_rate={result.get('win_rate', 'n/a'):.3f}  "
              f"n={result.get('n_pairs', 'n/a')}")
        test_results.append(result)
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval2_bootstrap.json", "w") as f:
            json.dump(result, f, indent=2)

    # 2. train-split diagnostic
    print("\n--- train-split diagnostic (memorization check) ---")
    train_results = []
    for seed in seed_list:
        try:
            result = eval_paired_seed.remote(seed, config, "self", "train", git_sha)
        except Exception as e:
            print(f"train seed {seed} FAILED: {type(e).__name__}: {e}")
            continue
        print(f"seed {seed}: train_graph={result.get('mean_graph_cos', 'n/a'):.4f}  "
              f"train_seq={result.get('mean_sequential_cos', 'n/a'):.4f}  "
              f"train_Δ={result.get('mean_delta', 'n/a'):.4f}")
        train_results.append(result)
        out_dir = results_root / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "eval2_train.json", "w") as f:
            json.dump(result, f, indent=2)

    # write summary
    print("\n--- rigor summary ---")
    lines = [f"# rigor checks: {main_condition}", "", f"seeds: {seed_list}", ""]
    if test_results:
        lines.append("## bootstrap 95% CI on Δ (test split)")
        lines.append("")
        lines.append("| seed | mean Δ | CI95 low | CI95 high | win rate | n_pairs |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in test_results:
            lines.append(
                f"| {r.get('seed', '?')} | {r.get('mean_delta', 0):.4f} "
                f"| {r.get('delta_ci95_low', 0):.4f} "
                f"| {r.get('delta_ci95_high', 0):.4f} "
                f"| {r.get('win_rate', 0):.3f} | {r.get('n_pairs', 0)} |"
            )
        lines.append("")
    if train_results:
        lines.append("## train-split diagnostic (does not generalize → train >> test)")
        lines.append("")
        lines.append("| seed | train graph | train seq | train Δ | test Δ | train-test gap |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        # match train and test by seed
        test_by_seed = {r.get('seed'): r for r in test_results}
        for r in train_results:
            seed = r.get('seed', '?')
            t_test = test_by_seed.get(seed, {})
            train_d = r.get('mean_delta', 0)
            test_d = t_test.get('mean_delta', 0)
            lines.append(
                f"| {seed} | {r.get('mean_graph_cos', 0):.4f} "
                f"| {r.get('mean_sequential_cos', 0):.4f} "
                f"| {train_d:.4f} | {test_d:.4f} | {train_d - test_d:+.4f} |"
            )
        lines.append("")
        lines.append("a small (or negative) train-test gap means the model is generalizing, "
                     "not memorizing. a large positive gap (train Δ >> test Δ) suggests "
                     "graph-jepa's advantage relies on memorizing training graphs.")
        lines.append("")
    summary_path = results_root / "RIGOR_SUMMARY.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {summary_path}")


@app.local_entrypoint()
def eval2b(config: str = "configs/tgbn_trade.yaml",
           seeds: str = "0,1,2,3,4",
           modes: str = "graph,sequential"):
    """bulletproof paired comparison: run eval 2 with shared target encoders.

    "self" mode is already covered by main(); this entrypoint runs the two
    cross-target modes ("graph", "sequential") that hold the comparison
    target fixed across both models. if graph wins under self AND graph AND
    sequential, the result is rock-solid against the per-condition target
    asymmetry concern.
    """
    import json
    from pathlib import Path

    main_condition = _read_dataset_name(config)
    seed_list = [int(s) for s in seeds.split(",")]
    mode_list = [m.strip() for m in modes.split(",")]

    results_root = Path("results") / main_condition

    print(f"running eval2b on {len(seed_list)} seeds x {len(mode_list)} modes "
          f"({mode_list}) for condition '{main_condition}'")

    all_by_mode: dict[str, list] = {m: [] for m in mode_list}
    for mode in mode_list:
        print(f"\n--- shared_target_mode = '{mode}' ---")
        for seed in seed_list:
            try:
                result = eval_paired_seed.remote(seed, config, mode)
            except Exception as e:
                print(f"eval2b seed {seed} mode {mode} FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            print(json.dumps(result, indent=2))
            all_by_mode[mode].append(result)
            out_dir = results_root / f"seed{result.get('seed', seed)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"eval2_{mode}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"saved {out_path}")

    print("\n--- eval2b summary ---")
    for mode, results in all_by_mode.items():
        if not results:
            print(f"{mode}: no results")
            continue
        graph_means = [r["mean_graph_cos"] for r in results if "mean_graph_cos" in r]
        seq_means = [r["mean_sequential_cos"] for r in results if "mean_sequential_cos" in r]
        p_vals = [r["wilcoxon_p"] for r in results if "wilcoxon_p" in r]
        win_rates = [r["win_rate"] for r in results if "win_rate" in r]
        if graph_means:
            print(f"{mode}: graph={sum(graph_means)/len(graph_means):.4f}  "
                  f"seq={sum(seq_means)/len(seq_means):.4f}  "
                  f"mean_win_rate={sum(win_rates)/len(win_rates):.3f}  "
                  f"per-seed p-values: {[f'{p:.4g}' for p in p_vals]}")

    summary_path = results_root / "EVAL2B_SUMMARY.md"
    lines = [f"# eval2b (shared-target paired comparison): {main_condition}",
             "", f"seeds: {seed_list}", f"modes: {mode_list}", ""]
    for mode, results in all_by_mode.items():
        if not results:
            continue
        lines.append(f"## shared_target_mode = `{mode}`")
        for r in results:
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
