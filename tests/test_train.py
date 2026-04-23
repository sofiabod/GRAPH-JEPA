import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.train import train


def make_graphs(n=30, n_nodes=10, n_edges=15):
    graphs = []
    for _ in range(n):
        graphs.append(Data(
            x=torch.randn(n_nodes, 384),
            edge_index=torch.randint(0, n_nodes, (2, n_edges)),
            node_ids=torch.arange(n_nodes),
        ))
    return graphs


def make_test_cfg(n_nodes=10):
    return OmegaConf.create({
        'encoder': {
            'in_dim': 384,
            'hidden_dim': 32,
            'n_layers': 1,
            'n_heads': 2,
            'dropout': 0.0,
            'temporal_stride': 1,
        },
        'predictor': {
            'embed_dim': 32,
            'n_heads': 2,
            'n_layers': 1,
            'mlp_ratio': 2,
            'dropout': 0.0,
            'n_nodes': n_nodes,
            'max_time_steps': 50,
            'temporal_stride': 1,
        },
        'training': {
            'lr': 1e-3,
            'weight_decay': 0.01,
            'lr_min': 1e-5,
            'batch_size': 2,
            'max_epochs': 2,
            'early_stopping_patience': 30,
            'mask_ratio': 0.20,
            'context_k': 4,
            'ema_momentum_start': 0.9,
            'ema_momentum_end': 1.0,
            'grad_clip_max_norm': 1.0,
            'total_steps': 100,
        },
        'loss': {
            'lambda_reg': 0.01,
            'bcs_num_slices': 16,
            'bcs_lmbd': 0.1,
        },
        'data': {
            'enron_data_path': 'data/enron_graphs.pt',
            'enron_meta_path': 'data/enron_meta.json',
        },
    })


def test_train_two_epochs_no_error(tmp_path):
    graphs = make_graphs(30)
    cfg = make_test_cfg()
    result = train(cfg, seed=0, graphs=graphs, out_dir=str(tmp_path))
    assert result is not None


def test_train_returns_loss_history(tmp_path):
    graphs = make_graphs(30)
    cfg = make_test_cfg()
    result = train(cfg, seed=0, graphs=graphs, out_dir=str(tmp_path))
    assert 'train_losses' in result
    assert isinstance(result['train_losses'], list)
    assert len(result['train_losses']) > 0
    for v in result['train_losses']:
        assert isinstance(v, float)


def test_train_loss_finite(tmp_path):
    graphs = make_graphs(30)
    cfg = make_test_cfg()
    result = train(cfg, seed=0, graphs=graphs, out_dir=str(tmp_path))
    for v in result['train_losses']:
        assert math.isfinite(v), f"loss {v} is not finite"


def test_checkpoint_saved(tmp_path):
    graphs = make_graphs(30)
    cfg = make_test_cfg()
    train(cfg, seed=0, graphs=graphs, out_dir=str(tmp_path))
    assert (Path(tmp_path) / 'checkpoint.pt').exists()


def test_target_encoder_not_in_optimizer():
    # target encoder params must stay frozen; verify via builders directly
    from src.builders import build_graph_encoder, build_target_encoder
    from omegaconf import OmegaConf
    enc_cfg = OmegaConf.create({
        'in_dim': 384, 'hidden_dim': 32, 'n_layers': 1,
        'n_heads': 2, 'dropout': 0.0, 'temporal_stride': 1,
    })
    online = build_graph_encoder(enc_cfg)
    target = build_target_encoder(online)
    for p in target.parameters():
        assert not p.requires_grad, "target encoder param has requires_grad=True"


def test_val_losses_tracked(tmp_path):
    # with only 30 graphs, val split is empty, so val_losses should be an empty list
    # but the key must exist
    graphs = make_graphs(30)
    cfg = make_test_cfg()
    result = train(cfg, seed=0, graphs=graphs, out_dir=str(tmp_path))
    assert 'val_losses' in result
    assert isinstance(result['val_losses'], list)
