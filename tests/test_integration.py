import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.models.graph_encoder import GraphEncoder
from src.models.target_encoder import TargetEncoder
from src.models.ema import EMAUpdater
from src.models.predictor import TemporalGraphPredictor
from src.losses.prediction import TGJEPALoss


# use small hidden_dim=32 to keep tests fast
HIDDEN_DIM = 32
N_NODES = 10
IN_DIM = 384


def make_tiny_graph():
    return Data(
        x=torch.randn(N_NODES, IN_DIM),
        edge_index=torch.randint(0, N_NODES, (2, 15)),
        node_ids=torch.arange(N_NODES),
    )


def build_components():
    """assemble all components with small hidden dim"""
    online = GraphEncoder(
        in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, n_layers=1, n_heads=2, dropout=0.0
    )
    target = TargetEncoder(online)
    predictor = TemporalGraphPredictor(
        embed_dim=HIDDEN_DIM, n_heads=2, n_layers=1, mlp_ratio=2,
        dropout=0.0, n_nodes=N_NODES, max_time_steps=20, temporal_stride=1
    )
    loss_fn = TGJEPALoss(lambda_reg=0.01)
    ema = EMAUpdater(momentum_start=0.996, momentum_end=1.0, total_steps=100)
    return online, target, predictor, loss_fn, ema


def run_forward(online, target, predictor, loss_fn):
    """run one forward pass, return (loss, total, pred_loss, sigreg_dict)"""
    graph = make_tiny_graph()

    # encode with online and target
    z_online = online(graph)  # [N, D]
    with torch.no_grad():
        z_target = target(graph)  # [N, D], no grad

    # build a token sequence: [1, N*k + N + 1, D]
    # k=2 context steps + 1 target step + 1 mask token
    k = 2
    context_tokens = z_online.unsqueeze(0).expand(k, -1, -1)  # [k, N, D]
    context_flat = context_tokens.reshape(1, k * N_NODES, HIDDEN_DIM)
    target_tokens = z_target.unsqueeze(0)  # [1, N, D]
    mask_token = torch.zeros(1, 1, HIDDEN_DIM)

    tokens = torch.cat([context_flat, target_tokens, mask_token], dim=1)  # [1, k*N+N+1, D]
    T = tokens.shape[1]

    time_indices = torch.zeros(1, T, dtype=torch.long)
    node_ids = torch.zeros(1, T, dtype=torch.long)

    pred_out = predictor(tokens, time_indices, node_ids)  # [1, T, D]

    # use last token output as prediction, first node of z_target as target
    z_pred = pred_out[0, -1:].expand(N_NODES, -1)  # [N, D]
    z_online_all = z_online  # [N, D]
    z_target_all = z_target  # [N, D]

    total, pred_loss, sigreg_dict = loss_fn(z_pred, z_target, z_online_all, z_target_all)
    return total, pred_loss, sigreg_dict


def test_full_forward_no_error():
    online, target, predictor, loss_fn, ema = build_components()
    # should not raise
    total, pred_loss, sigreg_dict = run_forward(online, target, predictor, loss_fn)
    assert total is not None


def test_online_encoder_params_update():
    online, target, predictor, loss_fn, ema = build_components()
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    # snapshot online params
    online_before = {n: p.data.clone() for n, p in online.named_parameters()}

    optimizer.zero_grad()
    total, pred_loss, sigreg_dict = run_forward(online, target, predictor, loss_fn)
    total.backward()
    optimizer.step()

    # at least one param should have changed
    any_changed = any(
        not torch.allclose(p.data, online_before[n])
        for n, p in online.named_parameters()
    )
    assert any_changed, "at least one online encoder param should change after optimizer step"


def test_predictor_params_update():
    online, target, predictor, loss_fn, ema = build_components()
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    predictor_before = {n: p.data.clone() for n, p in predictor.named_parameters()}

    optimizer.zero_grad()
    total, pred_loss, sigreg_dict = run_forward(online, target, predictor, loss_fn)
    total.backward()
    optimizer.step()

    any_changed = any(
        not torch.allclose(p.data, predictor_before[n])
        for n, p in predictor.named_parameters()
    )
    assert any_changed, "at least one predictor param should change after optimizer step"


def test_target_encoder_unchanged_by_optimizer():
    online, target, predictor, loss_fn, ema = build_components()
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    # snapshot target params before training step
    target_before = {n: p.data.clone() for n, p in target.named_parameters()}

    optimizer.zero_grad()
    total, pred_loss, sigreg_dict = run_forward(online, target, predictor, loss_fn)
    total.backward()
    optimizer.step()
    # do NOT call ema.update here

    for n, p in target.named_parameters():
        assert torch.allclose(p.data, target_before[n]), \
            f"target param {n} should not change after optimizer.step (no ema.update)"


def test_target_updates_via_ema():
    online, target, predictor, loss_fn, ema = build_components()

    # use low momentum so the update is visible
    ema_fast = EMAUpdater(momentum_start=0.5, momentum_end=0.5, total_steps=100)
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=0.1
    )

    target_before = {n: p.data.clone() for n, p in target.named_parameters()}

    optimizer.zero_grad()
    total, pred_loss, sigreg_dict = run_forward(online, target, predictor, loss_fn)
    total.backward()
    optimizer.step()

    # now call ema.update
    ema_fast.update(online, target, step=0)

    any_changed = any(
        not torch.allclose(p.data, target_before[n])
        for n, p in target.named_parameters()
    )
    assert any_changed, "target encoder params should change after ema.update"


def test_target_not_in_optimizer():
    online, target, predictor, loss_fn, ema = build_components()
    optimizer = torch.optim.Adam(
        list(online.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    target_param_ids = {id(p) for p in target.parameters()}
    optimizer_param_ids = {
        id(p) for group in optimizer.param_groups for p in group['params']
    }

    overlap = target_param_ids & optimizer_param_ids
    assert len(overlap) == 0, \
        "target encoder parameters must not appear in any optimizer param group"


def test_loss_finite():
    online, target, predictor, loss_fn, ema = build_components()
    total, pred_loss, sigreg_dict = run_forward(online, target, predictor, loss_fn)
    assert torch.isfinite(total), f"loss should be finite, got {total.item()}"


def test_stop_gradient_verified():
    online, target, predictor, loss_fn, ema = build_components()
    graph = make_tiny_graph()

    # target encoder output should not allow gradient flow to target params
    try:
        out = target(graph)
        out.sum().backward()
        # if backward succeeds, target params should still have None or zero grad
        for p in target.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                raise AssertionError(
                    "target encoder params received nonzero gradients via backward"
                )
    except RuntimeError:
        # raising RuntimeError is also acceptable (e.g., from stop-grad)
        pass
