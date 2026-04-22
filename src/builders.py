import copy
from src.models.graph_encoder import GraphEncoder
from src.models.target_encoder import TargetEncoder
from src.models.predictor import TemporalGraphPredictor
from src.models.ema import EMAUpdater
from src.losses.prediction import TGJEPALoss


def build_graph_encoder(cfg):
    return GraphEncoder(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        temporal_stride=cfg.temporal_stride,
    )


def build_target_encoder(online_encoder):
    return TargetEncoder(online_encoder)


def build_predictor(cfg):
    return TemporalGraphPredictor(
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        n_nodes=cfg.n_nodes,
        max_time_steps=cfg.max_time_steps,
        temporal_stride=cfg.temporal_stride,
    )


def build_loss(cfg):
    return TGJEPALoss(lambda_reg=cfg.lambda_reg)


def build_ema(cfg):
    return EMAUpdater(
        momentum_start=cfg.ema_momentum_start,
        momentum_end=cfg.ema_momentum_end,
        total_steps=cfg.total_steps,
    )
