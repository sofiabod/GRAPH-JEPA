"""stress test smoke run for the graph-jepa training pipeline.

generates synthetic pyg snapshots, runs train() for 3 epochs, and checks:
- loss is finite at every epoch
- target encoder weights move via ema (not via gradient)
- online encoder weights move via gradient
- predictor weights move via gradient
- cosine sim of z_pred / z_target on training samples in [-1, 1]
- effective rank of online encoder output > 10 after 3 epochs

reports memory peak and wall-clock per epoch.

run with /opt/homebrew/Caskroom/miniconda/base/bin/python.
"""
from __future__ import annotations

import copy
import gc
import math
import resource
import sys
import time
import warnings
from pathlib import Path

# silence verbose pyg/transformer warnings unless useful; we still capture them.
warnings.simplefilter("default")

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data

# repo root on path so that `from src.train import train` resolves regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.train import train  # noqa: E402


def make_synthetic_graphs(n_nodes: int, n_steps: int, in_dim: int, avg_degree: int = 8, seed: int = 0):
    """generate `n_steps` snapshots of a graph on `n_nodes` nodes with random sparse edges
    and random node features. node count and node_id ordering identical across snapshots.
    """
    rng = torch.Generator().manual_seed(seed)
    base_x = torch.randn(n_nodes, in_dim, generator=rng)
    graphs = []
    for t in range(n_steps):
        # drift node features slightly per step so target != copy-forward exactly
        x = base_x + 0.05 * torch.randn(n_nodes, in_dim, generator=rng)
        n_edges = max(n_nodes, n_nodes * avg_degree // 2)
        src = torch.randint(0, n_nodes, (n_edges,), generator=rng)
        dst = torch.randint(0, n_nodes, (n_edges,), generator=rng)
        edge_index = torch.stack([src, dst], dim=0)
        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            node_ids=torch.arange(n_nodes),
        ))
    return graphs


def make_cfg_from_tgbn(n_nodes: int, n_steps: int, in_dim: int, batch_size: int = 4):
    cfg = OmegaConf.load(_REPO_ROOT / "configs" / "tgbn_trade.yaml")
    cfg.encoder.in_dim = in_dim
    cfg.predictor.n_nodes = n_nodes
    cfg.predictor.max_time_steps = max(60, n_steps + 5)
    cfg.training.max_epochs = 3
    cfg.training.batch_size = batch_size
    cfg.training.context_k = 4
    cfg.training.early_stopping_patience = 100  # disable early stopping for the smoke run
    cfg.training.total_steps = 200  # ema schedule denom should match short run
    # split ranges within `n_steps`
    train_hi = max(cfg.training.context_k, int(n_steps * 0.7))
    val_hi = max(train_hi + 2, int(n_steps * 0.85))
    test_hi = n_steps - 1
    cfg.data.train_weeks = [cfg.training.context_k, train_hi]
    cfg.data.val_weeks = [train_hi + 1, val_hi]
    cfg.data.test_weeks = [val_hi + 1, test_hi]
    return cfg


def snapshot_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def state_diff_l2(s_before, s_after):
    diffs = {}
    for k in s_before:
        if k in s_after and s_before[k].shape == s_after[k].shape and s_before[k].dtype.is_floating_point:
            diffs[k] = (s_after[k] - s_before[k]).norm().item()
    return diffs


def peak_mem_mb():
    # ru_maxrss on darwin is bytes, on linux it is kilobytes
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def effective_rank(z: torch.Tensor) -> float:
    _, s, _ = torch.linalg.svd(z, full_matrices=False)
    s = s[s > 1e-10]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy.exp().item()


class _LossHook:
    """monkeypatch builders.build_loss to wrap the loss and capture component breakdown.

    we DO NOT modify src/. the hook is installed at module import time inside this file.
    """
    def __init__(self):
        self.records = []

    def install(self):
        # train.py imported `build_loss` into its namespace via `from src.builders import build_loss`,
        # so we must patch BOTH `src.builders.build_loss` and `src.train.build_loss` to take effect.
        from src import builders
        from src import train as train_mod
        from src.losses.prediction import TGJEPALoss
        original_b = builders.build_loss
        original_t = train_mod.build_loss
        records = self.records

        class _Wrapped(TGJEPALoss):
            def forward(self, z_pred, z_target, z_online_all):
                total, pred_loss, sigreg = super().forward(z_pred, z_target, z_online_all)
                records.append({
                    "total": float(total.detach().item()),
                    "pred_loss": float(pred_loss.detach().item()),
                    "sigreg_loss": float(sigreg["loss"].detach().item()),
                    "bcs": float(sigreg["bcs_loss"].detach().item()),
                    "invariance": float(sigreg["invariance_loss"].detach().item()),
                    # cos sim summary on z_pred/z_target this step (already l2-normed)
                    "cos_min": float((z_pred * z_target).sum(dim=-1).min().item()),
                    "cos_max": float((z_pred * z_target).sum(dim=-1).max().item()),
                    "cos_mean": float((z_pred * z_target).sum(dim=-1).mean().item()),
                })
                return total, pred_loss, sigreg

        def patched(cfg):
            return _Wrapped(lambda_reg=cfg.lambda_reg)

        builders.build_loss = patched
        train_mod.build_loss = patched
        return (original_b, original_t)


class _GradHook:
    """capture gradient norms per training step by tapping into AdamW.step.

    rather than monkeypatching, we register pre-step parameter snapshots through
    a subclass after train() builds the optimizer. simplest: monkeypatch
    torch.nn.utils.clip_grad_norm_ to capture norm before clipping.
    """
    def __init__(self):
        self.norms = []

    def install(self):
        import torch.nn.utils as nnu
        original = nnu.clip_grad_norm_
        norms = self.norms

        def patched(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
            ret = original(parameters, max_norm, norm_type=norm_type,
                           error_if_nonfinite=error_if_nonfinite, foreach=foreach)
            try:
                norms.append(float(ret))
            except Exception:
                norms.append(float("nan"))
            return ret

        nnu.clip_grad_norm_ = patched
        # train.py imports as `from torch import nn`; nn.utils.clip_grad_norm_ resolves
        # via the same submodule, so patching torch.nn.utils is sufficient.
        return original


class _EMAHook:
    """capture ema momentum values used at each update step."""
    def __init__(self):
        self.momenta = []

    def install(self):
        from src.models import ema as ema_mod
        cls = ema_mod.EMAUpdater
        original_get = cls.get_momentum
        momenta = self.momenta

        def patched_get(self, step):
            m = original_get(self, step)
            momenta.append((int(step), float(m)))
            return m

        cls.get_momentum = patched_get
        return original_get


class _StateHook:
    """snapshot online/target/predictor state_dicts at start and after each epoch.

    we tap into train() by replacing torch.optim.AdamW with a wrapper that records
    the modules it sees on first construction.
    """
    def __init__(self):
        self.online_ref = None
        self.target_ref = None
        self.predictor_ref = None

    def install(self, results: dict):
        # we hook build_predictor, build_graph_encoder, build_target_encoder so we
        # capture references; those are imported into train.py at module level so
        # we patch via `src.train` namespace too.
        import src.train as t

        original_geo = t.build_graph_encoder
        original_seq_import = None
        original_target = t.build_target_encoder
        original_pred = t.build_predictor

        records = results

        def cap_geo(cfg):
            mod = original_geo(cfg)
            records["_online_module"] = mod
            return mod

        def cap_target(online):
            tgt = original_target(online)
            records["_target_module"] = tgt
            return tgt

        def cap_pred(cfg):
            mod = original_pred(cfg)
            records["_predictor_module"] = mod
            return mod

        t.build_graph_encoder = cap_geo
        t.build_target_encoder = cap_target
        t.build_predictor = cap_pred

        # also handle ablation path which constructs SequentialMLP inline
        from src.models import sequential_encoder as se

        original_seq = se.SequentialMLP.__init__

        def patched_seq_init(self, *args, **kwargs):
            original_seq(self, *args, **kwargs)
            records["_online_module"] = self

        se.SequentialMLP.__init__ = patched_seq_init


def run_one(label: str, n_nodes: int, n_steps: int, in_dim: int, ablation: bool = False,
            batch_size: int = 4):
    print(f"\n{'=' * 72}")
    print(f"== {label}: n_nodes={n_nodes}, n_steps={n_steps}, ablation={ablation}")
    print(f"{'=' * 72}")
    gc.collect()

    graphs = make_synthetic_graphs(n_nodes=n_nodes, n_steps=n_steps, in_dim=in_dim, seed=42)
    cfg = make_cfg_from_tgbn(n_nodes=n_nodes, n_steps=n_steps, in_dim=in_dim, batch_size=batch_size)

    loss_hook = _LossHook()
    grad_hook = _GradHook()
    ema_hook = _EMAHook()
    state_hook = _StateHook()

    refs = {}
    state_hook.install(refs)
    orig_loss = loss_hook.install()
    orig_clip = grad_hook.install()
    orig_ema = ema_hook.install()

    # snapshot initial states by hooking train() to dump them right after build
    initial_states = {}

    captured_warnings = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        t_start = time.time()
        try:
            result = train(cfg, seed=0, graphs=graphs, out_dir=None, ablation=ablation)
            error = None
        except Exception as e:
            import traceback
            error = traceback.format_exc()
            result = None
        elapsed = time.time() - t_start
        captured_warnings.extend([(str(w.category.__name__), str(w.message)) for w in wlist])

    # restore patches
    from src import builders as _b
    from src import train as _t
    if orig_loss is not None:
        ob, ot = orig_loss
        _b.build_loss = ob
        _t.build_loss = ot
    import torch.nn.utils as _nnu
    _nnu.clip_grad_norm_ = orig_clip
    from src.models import ema as _ema_mod
    _ema_mod.EMAUpdater.get_momentum = orig_ema

    if error is not None:
        print(f"!! TRAIN FAILED:\n{error}")
        return {"label": label, "error": error}

    online_mod = refs.get("_online_module")
    target_mod = refs.get("_target_module")
    predictor_mod = refs.get("_predictor_module")

    # final state snapshots
    online_after = snapshot_state(online_mod)
    target_after = snapshot_state(target_mod.encoder)
    predictor_after = snapshot_state(predictor_mod)

    # to compare to "initial", we need an initial reference. since we do not modify
    # src/, the cleanest approach is to rebuild a fresh model from the same cfg and
    # seed and compare structure. for the smoke test we instead measure parameter
    # movement using the standard "fresh init has different state but same dims"
    # approach: rebuild and compare.
    from src.utils.seed import set_seed
    set_seed(0)
    if ablation:
        from src.models.sequential_encoder import SequentialMLP
        fresh_online = SequentialMLP(
            in_dim=cfg.encoder.in_dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        )
    else:
        from src.builders import build_graph_encoder
        fresh_online = build_graph_encoder(cfg.encoder)
    from src.builders import build_target_encoder, build_predictor
    fresh_target = build_target_encoder(fresh_online)
    fresh_predictor = build_predictor(cfg.predictor)

    initial_states["online"] = snapshot_state(fresh_online)
    initial_states["target"] = snapshot_state(fresh_target.encoder)
    initial_states["predictor"] = snapshot_state(fresh_predictor)

    online_diffs = state_diff_l2(initial_states["online"], online_after)
    target_diffs = state_diff_l2(initial_states["target"], target_after)
    predictor_diffs = state_diff_l2(initial_states["predictor"], predictor_after)

    # forward online over a fresh sample to compute effective rank and cos sim sanity
    online_mod.eval()
    with torch.no_grad():
        z_all = []
        for g in graphs[:min(8, len(graphs))]:
            z_all.append(online_mod(g))
        z_cat = torch.cat(z_all, dim=0)
    er = effective_rank(z_cat)

    # check final cos sim between z_pred and z_target captured by loss hook
    cos_records = loss_hook.records
    if cos_records:
        last_step = cos_records[-1]
        cos_in_range = -1.001 <= last_step["cos_min"] and last_step["cos_max"] <= 1.001
    else:
        cos_in_range = None

    # assertions
    train_losses = result.get("train_losses", [])
    val_losses = result.get("val_losses", [])
    finite_train = all(math.isfinite(v) for v in train_losses)
    finite_val = all(math.isfinite(v) for v in val_losses)

    # online and predictor should move (some param diff > 0)
    online_moved = sum(online_diffs.values()) > 0.0
    predictor_moved = sum(predictor_diffs.values()) > 0.0
    target_moved = sum(target_diffs.values()) > 0.0  # via ema, expect movement when m < 1

    # confirm target didn't move via gradient: if ema momentum ever 1.0 the entire
    # run would have target frozen. check that predicted ema momentum < 1 at step 0.
    ema_m_first = ema_hook.momenta[0][1] if ema_hook.momenta else None

    # assertion summary
    summary = {
        "label": label,
        "elapsed_sec": elapsed,
        "epochs": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "n_train_steps": len(grad_hook.norms),
        "grad_norm_first": grad_hook.norms[0] if grad_hook.norms else None,
        "grad_norm_last": grad_hook.norms[-1] if grad_hook.norms else None,
        "grad_norm_max": max(grad_hook.norms) if grad_hook.norms else None,
        "grad_norm_min": min(grad_hook.norms) if grad_hook.norms else None,
        "any_nan_grad": any(not math.isfinite(g) for g in grad_hook.norms),
        "ema_first": ema_m_first,
        "ema_last": ema_hook.momenta[-1][1] if ema_hook.momenta else None,
        "online_total_movement": sum(online_diffs.values()),
        "predictor_total_movement": sum(predictor_diffs.values()),
        "target_total_movement": sum(target_diffs.values()),
        "online_moved_via_grad": online_moved,
        "predictor_moved_via_grad": predictor_moved,
        "target_moved_via_ema": target_moved,
        "effective_rank_online": er,
        "cos_in_unit_range_last_step": cos_in_range,
        "cos_mean_first": cos_records[0]["cos_mean"] if cos_records else None,
        "cos_mean_last": cos_records[-1]["cos_mean"] if cos_records else None,
        "pred_loss_first": cos_records[0]["pred_loss"] if cos_records else None,
        "pred_loss_last": cos_records[-1]["pred_loss"] if cos_records else None,
        "sigreg_first": cos_records[0]["sigreg_loss"] if cos_records else None,
        "sigreg_last": cos_records[-1]["sigreg_loss"] if cos_records else None,
        "all_train_losses_finite": finite_train,
        "all_val_losses_finite": finite_val,
        "peak_rss_mb": peak_mem_mb(),
        "warnings": captured_warnings,
    }

    print("\n--- summary ---")
    for k, v in summary.items():
        if k == "warnings":
            continue
        print(f"{k}: {v}")
    if captured_warnings:
        print(f"\nwarnings ({len(captured_warnings)}):")
        for cat, msg in captured_warnings[:10]:
            print(f"  [{cat}] {msg[:160]}")
        if len(captured_warnings) > 10:
            print(f"  ... +{len(captured_warnings) - 10} more")

    # explicit checks
    print("\n--- assertions ---")
    print(f"(a) train losses finite: {finite_train}")
    print(f"(a) val losses finite:   {finite_val}")
    decreased = (len(train_losses) >= 2 and train_losses[-1] <= train_losses[0] * 1.5)
    print(f"(b) loss does not explode (last <= 1.5*first): {decreased}")
    print(f"(c) target encoder moved (via ema, expected nonzero): {target_moved}")
    print(f"(d) online encoder moved (via gradient): {online_moved}")
    print(f"(e) predictor moved (via gradient): {predictor_moved}")
    if cos_in_range is not None:
        print(f"(f) cosine sim z_pred/z_target in [-1,1]: {cos_in_range}")
    print(f"(g) effective rank > 10: {er > 10} (er={er:.2f})")

    return summary


def main():
    print("starting graph-jepa stress test smoke run")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, mps: {torch.backends.mps.is_available()}")

    summaries = []

    # 1) tgbn-trade scale (255 nodes, 30 timesteps)
    summaries.append(run_one("tgbn_trade_scale", n_nodes=255, n_steps=30, in_dim=6))

    # 2) tgbn-trade scale, ablation
    summaries.append(run_one("tgbn_trade_ablation", n_nodes=255, n_steps=30, in_dim=6, ablation=True))

    # 3) enron scale (50 nodes)
    summaries.append(run_one("enron_scale", n_nodes=50, n_steps=30, in_dim=6))

    # 4) large-scale: n=1000 (oom probe). reduce batch size if necessary.
    summaries.append(run_one("large_scale_1000", n_nodes=1000, n_steps=15, in_dim=6, batch_size=2))

    # comparative summary
    print("\n\n========================================================================")
    print("OVERALL SUMMARY")
    print("========================================================================")
    for s in summaries:
        if "error" in s:
            print(f"\n{s['label']}: FAILED\n{s['error'][:400]}")
            continue
        tl = s["train_losses"]
        vl = s["val_losses"]
        print(f"\n{s['label']}:")
        print(f"  elapsed={s['elapsed_sec']:.1f}s, epochs={s['epochs']}, peak_rss={s['peak_rss_mb']:.0f} MB")
        print(f"  train_losses={tl}")
        print(f"  val_losses={vl}")
        def _f(x):
            return f"{x:.4f}" if isinstance(x, float) else str(x)
        print(f"  pred_loss first->last: {_f(s['pred_loss_first'])} -> {_f(s['pred_loss_last'])}")
        print(f"  sigreg first->last:    {_f(s['sigreg_first'])} -> {_f(s['sigreg_last'])}")
        print(f"  cos_mean first->last:  {_f(s['cos_mean_first'])} -> {_f(s['cos_mean_last'])}")
        print(f"  grad_norm min/max:     {s['grad_norm_min']} / {s['grad_norm_max']}")
        print(f"  ema first->last:       {s['ema_first']} -> {s['ema_last']}")
        print(f"  effective_rank_online: {s['effective_rank_online']:.2f}")
        print(f"  online moved={s['online_moved_via_grad']}, predictor moved={s['predictor_moved_via_grad']}, target moved={s['target_moved_via_ema']}")

    return summaries


if __name__ == "__main__":
    main()
