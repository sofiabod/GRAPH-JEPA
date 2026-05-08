"""tests for the eval runner: eval 1 (with graph-average baseline) and eval 2."""
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.builders import build_graph_encoder, build_target_encoder, build_predictor
from src.eval.eval_runner import EvalRunner, _graph_average_baseline
from src.eval.eval2 import eval2_compare
from src.train import train


def _make_graphs(n=30, n_nodes=10, n_edges=20):
    graphs = []
    for _ in range(n):
        graphs.append(Data(
            x=torch.randn(n_nodes, 384),
            edge_index=torch.randint(0, n_nodes, (2, n_edges)),
            node_ids=torch.arange(n_nodes),
        ))
    return graphs


def _tiny_cfg(n_nodes=10):
    return OmegaConf.create({
        'encoder': {
            'in_dim': 384, 'hidden_dim': 32, 'n_layers': 1,
            'n_heads': 2, 'dropout': 0.0,
        },
        'predictor': {
            'embed_dim': 32, 'n_heads': 2, 'n_layers': 1, 'mlp_ratio': 2,
            'dropout': 0.0, 'n_nodes': n_nodes, 'max_time_steps': 50,
        },
        'training': {
            'lr': 1e-3, 'weight_decay': 0.01, 'lr_min': 1e-5,
            'batch_size': 2, 'max_epochs': 1, 'early_stopping_patience': 30,
            'mask_ratio': 0.20, 'context_k': 4,
            'ema_momentum_start': 0.9, 'ema_momentum_end': 1.0,
            'grad_clip_max_norm': 1.0, 'total_steps': 100,
        },
        'loss': {'lambda_reg': 0.01, 'bcs_num_slices': 16, 'bcs_lmbd': 0.1},
        'data': {
            'graphs_path': 'data/none.pt', 'meta_path': 'data/none.json',
            'train_weeks': [0, 9], 'val_weeks': [10, 14], 'test_weeks': [15, 24],
        },
    })


def test_graph_average_baseline_isolated_falls_back_to_global_mean():
    n_nodes = 6
    last_ctx = torch.randn(n_nodes, 16)
    # node 3 is isolated (no edges); expect global mean (normalized)
    edge_index = torch.tensor([[0, 1, 2, 4], [1, 2, 0, 5]], dtype=torch.long)
    g = Data(x=torch.zeros(n_nodes, 1), edge_index=edge_index)
    masked = torch.tensor([3])
    out = _graph_average_baseline(masked, g, last_ctx)
    # value should equal global mean direction
    target = torch.nn.functional.normalize(last_ctx.mean(dim=0, keepdim=True), dim=-1)
    assert torch.allclose(out, target, atol=1e-5)


def test_graph_average_baseline_uses_neighbors():
    n_nodes = 4
    last_ctx = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    # node 0 connected to node 1 and node 2 (undirected via union)
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    g = Data(x=torch.zeros(n_nodes, 1), edge_index=edge_index)
    masked = torch.tensor([0])
    out = _graph_average_baseline(masked, g, last_ctx)
    # mean of nodes 1 and 2 is (-0.5, 0.5); normalized
    expected_raw = torch.tensor([[-0.5, 0.5]])
    expected = torch.nn.functional.normalize(expected_raw, dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_eval1_returns_graph_avg_keys():
    cfg = _tiny_cfg()
    graphs = _make_graphs(n=30)

    online = build_graph_encoder(cfg.encoder)
    target = build_target_encoder(online)
    predictor = build_predictor(cfg.predictor)

    runner = EvalRunner(
        online, target, predictor, graphs, cfg,
        train_range=(0, 9), val_range=(10, 14), test_range=(15, 24),
        mask_seed=0,
    )
    out = runner._eval1_node_prediction()
    assert 'mean_pred_cos' in out
    assert 'mean_copy_cos' in out
    assert 'mean_graph_avg_cos' in out
    assert 'wilcoxon_p_vs_copy' in out
    assert 'wilcoxon_p_vs_graph_avg' in out
    assert out['n_pairs'] > 0


def test_eval2_end_to_end_runnable(tmp_path):
    cfg = _tiny_cfg()
    graphs = _make_graphs(n=30)

    # train graph-jepa
    graph_dir = tmp_path / 'graph'
    train(cfg, seed=0, graphs=graphs, out_dir=str(graph_dir), ablation=False)
    assert (graph_dir / 'checkpoint.pt').exists()

    # train sequential ablation
    seq_dir = tmp_path / 'seq'
    train(cfg, seed=0, graphs=graphs, out_dir=str(seq_dir), ablation=True)
    assert (seq_dir / 'checkpoint.pt').exists()

    # eval 2 paired comparator: load from disk, run on identical mask sets
    out = eval2_compare(
        graph_ckpt_path=str(graph_dir / 'checkpoint.pt'),
        sequential_ckpt_path=str(seq_dir / 'checkpoint.pt'),
        graphs=graphs,
        cfg=cfg,
        splits=((0, 9), (10, 14), (15, 24)),
        mask_seed=0,
        device=torch.device('cpu'),
    )
    assert 'mean_graph_cos' in out
    assert 'mean_sequential_cos' in out
    assert 'wilcoxon_p' in out
    assert 'n_pairs' in out
    assert 'win_rate' in out
    assert out['n_pairs'] > 0
    assert 0.0 <= out['win_rate'] <= 1.0


def test_run_all_with_eval2_writes_summary(tmp_path):
    cfg = _tiny_cfg()
    graphs = _make_graphs(n=30)

    graph_dir = tmp_path / 'graph'
    train(cfg, seed=0, graphs=graphs, out_dir=str(graph_dir), ablation=False)
    seq_dir = tmp_path / 'seq'
    train(cfg, seed=0, graphs=graphs, out_dir=str(seq_dir), ablation=True)

    # rebuild graph-jepa models from the checkpoint we just trained
    state = torch.load(graph_dir / 'checkpoint.pt', map_location='cpu')
    online = build_graph_encoder(cfg.encoder)
    online.load_state_dict(state['online'])
    target = build_target_encoder(online)
    target.encoder.load_state_dict(state['target_encoder'])
    predictor = build_predictor(cfg.predictor)
    predictor.load_state_dict(state['predictor'])

    runner = EvalRunner(
        online, target, predictor, graphs, cfg,
        train_range=(0, 9), val_range=(10, 14), test_range=(15, 24),
        mask_seed=0,
    )
    out_dir = tmp_path / 'results'
    results = runner.run_all_with_eval2(
        str(out_dir), sequential_ckpt_path=str(seq_dir / 'checkpoint.pt')
    )
    assert (out_dir / 'eval_summary.json').exists()
    assert 'eval2_graph_vs_sequential' in results
    assert 'bonferroni' in results
    assert results['family_size'] >= 2

    with open(out_dir / 'eval_summary.json') as f:
        loaded = json.load(f)
    assert 'eval2_graph_vs_sequential' in loaded


def test_train_raises_on_n_nodes_mismatch():
    cfg = _tiny_cfg(n_nodes=10)
    # mismatched: graphs say 12 nodes, cfg says 10
    bad_graphs = _make_graphs(n=20, n_nodes=12)
    with pytest.raises(ValueError, match='n_nodes'):
        train(cfg, seed=0, graphs=bad_graphs, out_dir=None)


def test_train_raises_on_inconsistent_n_nodes_across_graphs():
    cfg = _tiny_cfg(n_nodes=10)
    a = _make_graphs(n=10, n_nodes=10)
    b = _make_graphs(n=10, n_nodes=8)
    with pytest.raises(ValueError, match='n_nodes'):
        train(cfg, seed=0, graphs=a + b, out_dir=None)


def test_eval3_multistep_rollout_returns_horizons():
    """eval 3 should return entries for h1, h2, h4 with paired wilcoxon vs copy-forward."""
    n_nodes = 10
    graphs = _make_graphs(n=30, n_nodes=n_nodes, n_edges=20)
    cfg = _tiny_cfg(n_nodes=n_nodes)

    online = build_graph_encoder(cfg.encoder)
    predictor = build_predictor(cfg.predictor)
    target = build_target_encoder(online)

    runner = EvalRunner(
        online, target, predictor, graphs, cfg,
        train_range=(0, 19), val_range=(20, 23), test_range=(24, 29),
        mask_seed=0,
    )
    result = runner._eval3_multistep_rollout(horizons=(1, 2, 4))
    assert 'horizons_evaluated' in result, "must report which horizons ran"
    assert 'h1' in result, "h1 must be present"
    for h in result['horizons_evaluated']:
        entry = result[f'h{h}']
        assert 'mean_pred_cos' in entry
        assert 'mean_copy_cos' in entry
        assert 'wilcoxon_p_vs_copy' in entry
        assert 'n_pairs' in entry
        assert -1.0 - 1e-5 <= entry['mean_pred_cos'] <= 1.0 + 1e-5, \
            f"cos must be in [-1,1], got pred_cos={entry['mean_pred_cos']} at h={h}"
        assert -1.0 - 1e-5 <= entry['mean_copy_cos'] <= 1.0 + 1e-5
        assert entry['n_pairs'] > 0


def test_eval3_handles_empty_test_range():
    """eval 3 must return a clean error dict if test set is empty."""
    n_nodes = 10
    graphs = _make_graphs(n=10, n_nodes=n_nodes, n_edges=20)
    cfg = _tiny_cfg(n_nodes=n_nodes)

    online = build_graph_encoder(cfg.encoder)
    predictor = build_predictor(cfg.predictor)
    target = build_target_encoder(online)

    runner = EvalRunner(
        online, target, predictor, graphs, cfg,
        train_range=(0, 5), val_range=(6, 7), test_range=(100, 200),
        mask_seed=0,
    )
    result = runner._eval3_multistep_rollout(horizons=(1, 2, 4))
    assert 'error' in result


def test_eval3_horizon_runs_out_of_data():
    """if horizon h would index past the graph list, it should be skipped, not crash."""
    n_nodes = 10
    graphs = _make_graphs(n=15, n_nodes=n_nodes, n_edges=20)
    cfg = _tiny_cfg(n_nodes=n_nodes)

    online = build_graph_encoder(cfg.encoder)
    predictor = build_predictor(cfg.predictor)
    target = build_target_encoder(online)

    # test range with small space for rollout: only 1 sample fits in test
    runner = EvalRunner(
        online, target, predictor, graphs, cfg,
        train_range=(0, 9), val_range=(10, 11), test_range=(12, 13),
        mask_seed=0,
    )
    # h=4 needs target_idx + 3 to exist; with only 2 test indices and graphs ending at 14,
    # h=4 may have fewer pairs than h=1
    result = runner._eval3_multistep_rollout(horizons=(1, 2, 4))
    if 'error' not in result:
        # h1 should always run if any test sample exists
        if 'h1' in result:
            assert result['h1']['n_pairs'] > 0
        # h4 may run with fewer pairs or be absent — either is fine, not a crash
