"""eval 2: graph-jepa vs sequential-ablation paired comparator.

this is the thesis experiment. both models are run on identical masked
node sets (deterministic per-sample masking), per-sample cosines are
collected, and a paired wilcoxon tests graph_sim > sequential_sim.
"""
import torch
import torch.nn.functional as F
import numpy as np

from src.eval.metrics import cosine_sim
from src.eval.wilcoxon import paired_wilcoxon, bootstrap_ci_on_delta


def _load_graph_jepa(ckpt_path, cfg, device):
    from src.builders import build_graph_encoder, build_target_encoder, build_predictor
    online = build_graph_encoder(cfg.encoder).to(device)
    target = build_target_encoder(online)
    target.encoder = target.encoder.to(device)
    predictor = build_predictor(cfg.predictor).to(device)
    state = torch.load(ckpt_path, map_location=device)
    online.load_state_dict(state['online'])
    predictor.load_state_dict(state['predictor'])
    target.encoder.load_state_dict(state['target_encoder'])
    online.eval()
    predictor.eval()
    return online, target, predictor


def _load_sequential(ckpt_path, cfg, device):
    from src.models.sequential_encoder import SequentialMLP
    from src.builders import build_target_encoder, build_predictor
    online = SequentialMLP(
        in_dim=cfg.encoder.in_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    target = build_target_encoder(online)
    target.encoder = target.encoder.to(device)
    predictor = build_predictor(cfg.predictor).to(device)
    state = torch.load(ckpt_path, map_location=device)
    online.load_state_dict(state['online'])
    predictor.load_state_dict(state['predictor'])
    target.encoder.load_state_dict(state['target_encoder'])
    online.eval()
    predictor.eval()
    return online, target, predictor


def _per_sample_cos(online, target, predictor, sample, device, shared_target=None):
    # forward both encoder and predictor on a single sample, return cos sims
    # for masked nodes against the target embedding.
    #
    # if shared_target is None (default), each model uses its own target encoder
    # — measures self-consistency: how well does this model predict its own
    # target distribution.
    #
    # if shared_target is provided (a target encoder), both models predict against
    # the same reference. used for the bulletproof comparison: "graph and sequential
    # both try to match the SAME target signal" rather than each matching its own.
    from src.train import _encode_context, _build_tokens_for_sample

    tgt_graph = sample['target_graph'].to(device)
    masked_ids = sample['masked_node_ids'].to(device)
    visible_ids = sample['visible_node_ids'].to(device)

    ctx_embs = _encode_context(online, sample['context_graphs'])
    # the predictor's input mask token positions are constructed from whatever
    # target encoder we pass to _build_tokens_for_sample; for the predictor's
    # own input we use the model's own target so the prediction has the right
    # context geometry. the comparison target (z_true) is the shared one.
    tgt_emb_for_predictor = target(tgt_graph)
    if shared_target is not None:
        tgt_emb_for_compare = shared_target(tgt_graph)
    else:
        tgt_emb_for_compare = tgt_emb_for_predictor

    tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
        ctx_embs, tgt_emb_for_predictor, masked_ids, visible_ids, predictor
    )
    out = predictor(
        tokens.unsqueeze(0),
        time_indices.unsqueeze(0),
        node_ids_seq.unsqueeze(0),
    ).squeeze(0)

    z_pred = F.normalize(out[mask_positions], dim=-1)
    z_true = tgt_emb_for_compare[masked_ids]
    return cosine_sim(z_pred, z_true).cpu().numpy().tolist()


def eval2_compare(graph_models=None, graph_ckpt_path=None,
                  sequential_ckpt_path=None, graphs=None, cfg=None,
                  splits=None, mask_seed=0, device=None,
                  shared_target_mode="self",
                  eval_split="test") -> dict:
    """run the paired graph-vs-sequential comparison.

    args:
        graph_models: optional tuple (online, target, predictor) already loaded.
            if None, graph_ckpt_path is used to load fresh.
        graph_ckpt_path: path to graph-jepa checkpoint (used if graph_models None).
        sequential_ckpt_path: path to sequential-ablation checkpoint (required).
        graphs: list of pyg Data objects.
        cfg: omegaconf cfg.
        splits: tuple (train_range, val_range, test_range).
        mask_seed: seed for deterministic per-sample masking.
        device: torch device; inferred if None.
        shared_target_mode: how to handle target encoder for the cosine comparison.
            "self" (default): each model uses its own target encoder. measures
                self-consistency of prediction; the reading is "graph predicts
                graph's target better than seq predicts seq's target."
            "graph": both models compare against graph-jepa's target encoder.
                both try to match the same reference; biases AWAY from graph
                (sequential is forced to predict graph's manifold).
            "sequential": both models compare against sequential-ablation's
                target encoder. biases AWAY from sequential.
            run all three for the bulletproof headline: if graph wins under
            "self", "graph", AND "sequential", the result is rock-solid.

    returns dict with mean_graph_cos, mean_sequential_cos, wilcoxon_p,
    wilcoxon_stat, n_pairs, win_rate, shared_target_mode.
    """
    from src.data.dataset import TemporalGraphDataset

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if graph_models is None:
        if graph_ckpt_path is None:
            raise ValueError('either graph_models or graph_ckpt_path must be provided')
        g_online, g_target, g_predictor = _load_graph_jepa(graph_ckpt_path, cfg, device)
    else:
        g_online, g_target, g_predictor = graph_models
        g_online.eval()
        g_predictor.eval()

    if sequential_ckpt_path is None:
        raise ValueError('sequential_ckpt_path must be provided')
    if splits is None:
        raise ValueError('splits must be provided as (train_range, val_range, test_range)')
    if cfg is None:
        raise ValueError('cfg must be provided')
    if graphs is None:
        raise ValueError('graphs must be provided')
    s_online, s_target, s_predictor = _load_sequential(sequential_ckpt_path, cfg, device)

    # decide which target encoder is used for the cosine comparison.
    # "self": each model compared to its own target (default).
    # "graph": both compared to graph-jepa's target encoder.
    # "sequential": both compared to sequential ablation's target encoder.
    if shared_target_mode == "self":
        shared_for_graph = None
        shared_for_seq = None
    elif shared_target_mode == "graph":
        shared_for_graph = None  # graph already uses its own
        shared_for_seq = g_target
    elif shared_target_mode == "sequential":
        shared_for_graph = s_target
        shared_for_seq = None  # seq already uses its own
    else:
        raise ValueError(f"unknown shared_target_mode {shared_target_mode!r}")

    train_range, val_range, test_range = splits
    dataset = TemporalGraphDataset(
        graphs,
        context_k=cfg.training.context_k,
        mask_ratio=cfg.training.mask_ratio,
        split=eval_split,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
        seed=mask_seed,
    )
    if len(dataset) == 0:
        return {'error': 'no test data'}

    graph_sims = []
    seq_sims = []

    with torch.no_grad():
        for sample in dataset:
            g_cos = _per_sample_cos(g_online, g_target, g_predictor, sample, device,
                                     shared_target=shared_for_graph)
            s_cos = _per_sample_cos(s_online, s_target, s_predictor, sample, device,
                                     shared_target=shared_for_seq)
            # both should produce identical lengths since masking is deterministic
            assert len(g_cos) == len(s_cos), \
                f"mask-set length mismatch ({len(g_cos)} vs {len(s_cos)}); " \
                "deterministic masking is broken"
            graph_sims.extend(g_cos)
            seq_sims.extend(s_cos)

    g_arr = np.array(graph_sims)
    s_arr = np.array(seq_sims)
    p, stat = paired_wilcoxon(g_arr, s_arr)
    win_rate = float((g_arr > s_arr).mean()) if g_arr.size > 0 else 0.0
    boot = bootstrap_ci_on_delta(g_arr, s_arr, n_resamples=10000, seed=mask_seed)

    return {
        'mean_graph_cos': float(g_arr.mean()),
        'mean_sequential_cos': float(s_arr.mean()),
        'wilcoxon_p': p,
        'wilcoxon_stat': stat,
        'n_pairs': int(g_arr.size),
        'win_rate': win_rate,
        'shared_target_mode': shared_target_mode,
        'eval_split': eval_split,
        'mean_delta': boot['mean_delta'],
        'delta_ci95_low': boot['ci_low'],
        'delta_ci95_high': boot['ci_high'],
        'bootstrap_n_resamples': boot['n_resamples'],
    }
