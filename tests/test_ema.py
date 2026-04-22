import torch
from torch_geometric.data import Data

from src.models.graph_encoder import GraphEncoder
from src.models.target_encoder import TargetEncoder
from src.models.ema import EMAUpdater


def test_target_no_grad():
    online = GraphEncoder()
    target = TargetEncoder(online)
    assert all(not p.requires_grad for p in target.parameters()), \
        "all target encoder params must have requires_grad=False"


def test_ema_formula(tiny_graph):
    online = GraphEncoder(hidden_dim=32, n_layers=1, n_heads=2)
    target = TargetEncoder(online)
    ema = EMAUpdater(momentum_start=0.9, momentum_end=0.9, total_steps=100)

    # snapshot target params before update
    target_params_before = {
        name: p.data.clone() for name, p in target.named_parameters()
    }

    # take one optimizer step on online encoder
    optimizer = torch.optim.SGD(online.parameters(), lr=0.1)
    out = online(tiny_graph)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    # snapshot online params after step
    online_params_after = {
        name: p.data.clone() for name, p in online.named_parameters()
    }

    # run ema update at step 0
    m = ema.get_momentum(0)
    ema.update(online, target, step=0)

    # check formula: p_target_new = m * p_target_old + (1-m) * p_online_new
    for name, p_target_new in target.named_parameters():
        expected = m * target_params_before[name] + (1 - m) * online_params_after[name]
        assert torch.allclose(p_target_new.data, expected, atol=1e-5), \
            f"EMA formula violated for param {name}"


def test_ema_diverges_from_online():
    online = GraphEncoder(hidden_dim=32, n_layers=1, n_heads=2)
    target = TargetEncoder(online)

    # snapshot initial target params
    target_params_initial = {
        name: p.data.clone() for name, p in target.named_parameters()
    }

    # take 10 gradient steps on online without calling ema.update
    optimizer = torch.optim.SGD(online.parameters(), lr=0.5)
    dummy_input = Data(
        x=torch.randn(5, 384),
        edge_index=torch.zeros(2, 4, dtype=torch.long),
        node_ids=torch.arange(5),
    )
    for _ in range(10):
        optimizer.zero_grad()
        out = online(dummy_input)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    # target should still match initial (no ema.update was called)
    for name, p_target in target.named_parameters():
        assert torch.allclose(p_target.data, target_params_initial[name]), \
            "target params should not change without ema.update"

    # online params should differ from target params now
    any_diff = False
    for (_, p_o), (_, p_t) in zip(
        online.named_parameters(), target.named_parameters()
    ):
        if not torch.allclose(p_o.data, p_t.data):
            any_diff = True
            break
    assert any_diff, "online and target params should diverge after optimizer steps without EMA"


def test_momentum_schedule():
    ema = EMAUpdater(momentum_start=0.996, momentum_end=1.0, total_steps=100)
    m_start = ema.get_momentum(0)
    m_end = ema.get_momentum(100)
    assert abs(m_start - 0.996) < 1e-4, f"momentum at step 0 should be ~0.996, got {m_start}"
    assert abs(m_end - 1.0) < 1e-4, f"momentum at step 100 should be ~1.0, got {m_end}"


def test_target_outputs_no_grad(tiny_graph):
    online = GraphEncoder()
    target = TargetEncoder(online)
    out = target(tiny_graph)
    assert not out.requires_grad, "target encoder output must have requires_grad=False"
