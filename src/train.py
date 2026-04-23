from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.builders import (
    build_graph_encoder,
    build_target_encoder,
    build_predictor,
    build_loss,
    build_ema,
)
from src.data.dataset import TemporalGraphDataset
from src.utils.seed import set_seed


def _collate_fn(batch):
    # keep samples as a list; no default collation because context_graphs vary
    return batch


def _encode_context(online, context_graphs):
    # returns list of [N, D] tensors, one per context snapshot
    device = next(online.parameters()).device
    return [online(g.to(device)) for g in context_graphs]


def _build_tokens_for_sample(
    ctx_embs, tgt_emb_sg, masked_ids, visible_ids, predictor
):
    """build token sequence for one sample.

    returns (tokens [T, D], time_indices [T], node_ids [T], mask_positions [M])
    where T = k*N + N and mask_positions indexes into the final N slots.
    """
    k = len(ctx_embs)
    n_nodes = ctx_embs[0].shape[0]
    D = ctx_embs[0].shape[1]
    device = ctx_embs[0].device

    # time indices: 0..k-1 for history, k for target
    # node ids: 0..N-1 repeated k+1 times (node ordering matches graph)
    all_node_ids = torch.arange(n_nodes, device=device)

    tokens_list = []
    time_list = []
    node_list = []

    # history tokens: shape [k*N, D]
    for t, emb in enumerate(ctx_embs):
        tokens_list.append(emb)  # [N, D]
        time_list.append(torch.full((n_nodes,), t, dtype=torch.long, device=device))
        node_list.append(all_node_ids)

    # target time tokens: visible nodes use tgt_emb_sg, masked nodes use positional token
    tgt_time_idx = k
    tgt_tokens = torch.zeros(n_nodes, D, device=device)

    # visible nodes: use stop-grad target embedding
    if visible_ids.numel() > 0:
        tgt_tokens[visible_ids] = tgt_emb_sg[visible_ids]

    # masked nodes: temporal_pos_emb + node_id_emb (no content, only position)
    if masked_ids.numel() > 0:
        t_idx = torch.tensor([tgt_time_idx], dtype=torch.long, device=device)
        time_emb = predictor.temporal_pos_emb(t_idx)  # [1, D]
        node_emb = predictor.node_id_emb(masked_ids)  # [M, D]
        tgt_tokens[masked_ids] = time_emb + node_emb

    tokens_list.append(tgt_tokens)
    time_list.append(torch.full((n_nodes,), tgt_time_idx, dtype=torch.long, device=device))
    node_list.append(all_node_ids)

    tokens = torch.cat(tokens_list, dim=0)       # [(k+1)*N, D]
    time_indices = torch.cat(time_list, dim=0)   # [(k+1)*N]
    node_ids_seq = torch.cat(node_list, dim=0)   # [(k+1)*N]

    # mask positions are at the end (last N slots), offset by k*N
    mask_positions = k * n_nodes + masked_ids  # absolute indices in token sequence

    return tokens, time_indices, node_ids_seq, mask_positions


def _step(batch, online, target, predictor, loss_fn):
    """run one forward pass over a list-batch of samples, return total loss."""
    device = next(online.parameters()).device
    z_pred_list = []
    z_target_list = []
    z_online_all_list = []
    z_target_all_list = []

    # process each sample separately (loop over batch)
    for sample in batch:
        tgt_graph = sample['target_graph'].to(device)
        masked_ids = sample['masked_node_ids'].to(device)
        visible_ids = sample['visible_node_ids'].to(device)
        # online encoder on all context snapshots
        ctx_embs = _encode_context(online, sample['context_graphs'])  # list of [N, D]

        # target encoder on target graph (stop-grad already applied in TargetEncoder)
        tgt_emb_sg = target(tgt_graph)  # [N, D], no grad

        # online encoder on target graph (for BCS anti-collapse)
        tgt_emb_online = online(tgt_graph)  # [N, D]

        # build token sequence for this sample
        tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
            ctx_embs, tgt_emb_sg, masked_ids, visible_ids, predictor
        )

        # predictor forward: [1, T, D]
        out = predictor(
            tokens.unsqueeze(0),
            time_indices.unsqueeze(0),
            node_ids_seq.unsqueeze(0),
        )  # [1, T, D]

        out = out.squeeze(0)  # [T, D]

        # extract predictions at mask positions
        z_pred = out[mask_positions]  # [M, D]
        z_tgt = tgt_emb_sg[masked_ids]  # [M, D]

        z_pred_list.append(z_pred)
        z_target_list.append(z_tgt)
        z_online_all_list.append(tgt_emb_online)   # [N, D]
        z_target_all_list.append(tgt_emb_sg)        # [N, D]

    z_pred_all = torch.cat(z_pred_list, dim=0)      # [sum(M), D]
    z_tgt_all = torch.cat(z_target_list, dim=0)     # [sum(M), D]
    z_online_all = torch.cat(z_online_all_list, dim=0)   # [B*N, D]
    z_target_all = torch.cat(z_target_all_list, dim=0)   # [B*N, D]

    total, _, _ = loss_fn(z_pred_all, z_tgt_all, z_online_all, z_target_all)
    return total


def train(cfg, seed=0, graphs=None, out_dir=None, ablation=False):
    """train the temporal graph jepa model.

    args:
        cfg: omegaconf config
        seed: random seed
        graphs: optional list of pyg Data objects (if None, loads from cfg.data.enron_data_path)
        out_dir: optional path string for checkpoint (if None, skips saving)
        ablation: if True, use SequentialMLP instead of GraphEncoder
    returns:
        dict with 'train_losses', 'val_losses', 'target_encoder'
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load graphs if not provided
    if graphs is None:
        graphs = torch.load(cfg.data.enron_data_path)

    # build models
    if ablation:
        from src.models.sequential_encoder import SequentialMLP
        online = SequentialMLP(
            in_dim=cfg.encoder.in_dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        ).to(device)
    else:
        online = build_graph_encoder(cfg.encoder).to(device)

    target = build_target_encoder(online)
    # move target encoder inner module to device
    target.encoder = target.encoder.to(device)

    predictor = build_predictor(cfg.predictor).to(device)
    loss_fn = build_loss(cfg.loss).to(device)
    ema_updater = build_ema(cfg.training)

    # optimizer: online encoder + predictor only
    opt_params = list(online.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # cosine annealing lr scheduler
    max_epochs = cfg.training.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=cfg.training.lr_min,
    )

    # datasets
    context_k = cfg.training.context_k
    mask_ratio = cfg.training.mask_ratio
    batch_size = cfg.training.batch_size

    train_dataset = TemporalGraphDataset(
        graphs, context_k=context_k, mask_ratio=mask_ratio, split='train'
    )
    val_dataset = TemporalGraphDataset(
        graphs, context_k=context_k, mask_ratio=mask_ratio, split='val'
    )

    train_losses = []
    val_losses = []
    global_step = 0

    for _ in range(max_epochs):
        online.train()
        predictor.train()

        if len(train_dataset) == 0:
            # no training data (e.g. synthetic test graphs with fewer than 120 items)
            pass
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=_collate_fn,
            )
            epoch_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                loss = _step(batch, online, target, predictor, loss_fn)
                loss.backward()
                nn.utils.clip_grad_norm_(opt_params, cfg.training.grad_clip_max_norm)
                optimizer.step()
                ema_updater.update(online, target, global_step)
                global_step += 1
                epoch_losses.append(loss.item())

            train_losses.append(sum(epoch_losses) / len(epoch_losses))

        scheduler.step()

        # validation
        if len(val_dataset) > 0:
            online.eval()
            predictor.eval()
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_fn,
            )
            vlosses = []
            with torch.no_grad():
                for batch in val_loader:
                    vloss = _step(batch, online, target, predictor, loss_fn)
                    vlosses.append(vloss.item())
            val_losses.append(sum(vlosses) / len(vlosses))

    # save checkpoint
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'online': online.state_dict(),
                'predictor': predictor.state_dict(),
                'step': global_step,
            },
            out_path / 'checkpoint.pt',
        )

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
