from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

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


def _encode_batched(encoder, graphs, n_nodes, device):
    # batch a list of pyg data objects, run encoder once, split back to [B, N, D]
    batched = Batch.from_data_list([g.to(device) for g in graphs])
    out = encoder(batched)  # [B*N, D]
    return out.view(len(graphs), n_nodes, -1)


def _step(batch, online, target, predictor, loss_fn):
    """run one forward pass over a list-batch of samples, return total loss."""
    device = next(online.parameters()).device
    B = len(batch)
    k = len(batch[0]['context_graphs'])
    n_nodes = batch[0]['target_graph'].x.shape[0]

    # vectorize encoder calls across the batch
    # for each timestep t, encode the t-th context graph for all samples in one call
    ctx_embs_per_sample = [[None] * k for _ in range(B)]  # ctx_embs_per_sample[b][t] = [N, D]
    for t in range(k):
        graphs_t = [batch[b]['context_graphs'][t] for b in range(B)]
        out_t = _encode_batched(online, graphs_t, n_nodes, device)  # [B, N, D]
        for b in range(B):
            ctx_embs_per_sample[b][t] = out_t[b]

    # batched encoders on target graphs
    tgt_graphs = [batch[b]['target_graph'] for b in range(B)]
    tgt_emb_online_batched = _encode_batched(online, tgt_graphs, n_nodes, device)  # [B, N, D]
    # target encoder applies stop-grad inside; pass batched data directly
    batched_tgt = Batch.from_data_list([g.to(device) for g in tgt_graphs])
    tgt_emb_sg_flat = target(batched_tgt)  # [B*N, D], no grad
    tgt_emb_sg_batched = tgt_emb_sg_flat.view(B, n_nodes, -1)  # [B, N, D]

    # build per-sample token sequences then stack for one predictor forward
    tokens_list = []
    time_list = []
    node_list = []
    z_tgt_list = []
    mask_positions_list = []
    for b in range(B):
        sample = batch[b]
        masked_ids = sample['masked_node_ids'].to(device)
        visible_ids = sample['visible_node_ids'].to(device)
        ctx_embs = ctx_embs_per_sample[b]
        tgt_emb_sg = tgt_emb_sg_batched[b]

        tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
            ctx_embs, tgt_emb_sg, masked_ids, visible_ids, predictor
        )
        tokens_list.append(tokens)
        time_list.append(time_indices)
        node_list.append(node_ids_seq)
        z_tgt_list.append(tgt_emb_sg[masked_ids])
        mask_positions_list.append(mask_positions)

    # stack into [B, T, D]; T is identical across samples since N and k are fixed
    tokens_b = torch.stack(tokens_list, dim=0)
    time_b = torch.stack(time_list, dim=0)
    node_b = torch.stack(node_list, dim=0)

    out_b = predictor(tokens_b, time_b, node_b)  # [B, T, D]

    # extract predictions at per-sample mask positions and l2-normalize onto sphere
    z_pred_list = []
    for b in range(B):
        z_pred_b = out_b[b][mask_positions_list[b]]
        z_pred_b = F.normalize(z_pred_b, dim=-1)
        z_pred_list.append(z_pred_b)

    z_pred_all = torch.cat(z_pred_list, dim=0)              # [sum(M), D]
    z_tgt_all = torch.cat(z_tgt_list, dim=0)                # [sum(M), D]
    z_online_all = tgt_emb_online_batched.reshape(B * n_nodes, -1)  # [B*N, D]

    total, _, _ = loss_fn(z_pred_all, z_tgt_all, z_online_all)
    return total


def train(cfg, seed=0, graphs=None, out_dir=None, ablation=False):
    """train the temporal graph jepa model.

    args:
        cfg: omegaconf config
        seed: random seed
        graphs: optional list of pyg Data objects (if None, loads from cfg.data.graphs_path)
        out_dir: optional path string for checkpoint (if None, skips saving)
        ablation: if True, use SequentialMLP instead of GraphEncoder
    returns:
        dict with 'train_losses', 'val_losses', 'target_encoder'
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load graphs if not provided
    if graphs is None:
        import json
        # weights_only=False: pyg data objects rely on pickle
        graphs = torch.load(cfg.data.graphs_path, weights_only=False)

    # all snapshots must share the same node count and match cfg.predictor.n_nodes
    # catches dataset/config drift (e.g. JODIE) before mid-training IndexErrors
    if len(graphs) > 0:
        n_nodes_set = {g.x.shape[0] for g in graphs}
        if len(n_nodes_set) != 1:
            raise ValueError(
                f"all graphs must share n_nodes; found {sorted(n_nodes_set)}"
            )
        n_nodes_data = next(iter(n_nodes_set))
        if n_nodes_data != cfg.predictor.n_nodes:
            raise ValueError(
                f"n_nodes mismatch: data has {n_nodes_data}, "
                f"cfg.predictor.n_nodes is {cfg.predictor.n_nodes}"
            )

    # resolve split ranges from config or meta.json
    if hasattr(cfg.data, 'train_weeks') and cfg.data.train_weeks is not None:
        train_range = tuple(cfg.data.train_weeks)
        val_range = tuple(cfg.data.val_weeks)
        test_range = tuple(cfg.data.test_weeks)
    else:
        import json
        with open(cfg.data.meta_path) as f:
            meta = json.load(f)
        train_range = tuple(meta['train_range'])
        val_range = tuple(meta['val_range'])
        test_range = tuple(meta['test_range'])

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

    # mask_seed is plumbed so paired evals share the same mask sets
    mask_seed = cfg.training.get('mask_seed', seed)

    train_dataset = TemporalGraphDataset(
        graphs, context_k=context_k, mask_ratio=mask_ratio, split='train',
        train_range=train_range, val_range=val_range, test_range=test_range,
        seed=mask_seed,
    )
    val_dataset = TemporalGraphDataset(
        graphs, context_k=context_k, mask_ratio=mask_ratio, split='val',
        train_range=train_range, val_range=val_range, test_range=test_range,
        seed=mask_seed,
    )

    train_losses = []
    val_losses = []
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = cfg.training.early_stopping_patience

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
            epoch_val_loss = sum(vlosses) / len(vlosses)
            val_losses.append(epoch_val_loss)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                # save best checkpoint on every val improvement
                if out_dir is not None:
                    out_path = Path(out_dir)
                    out_path.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            'online': online.state_dict(),
                            'predictor': predictor.state_dict(),
                            'target_encoder': target.encoder.state_dict(),
                            'step': global_step,
                            'val_loss': epoch_val_loss,
                        },
                        out_path / 'checkpoint_best.pt',
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    # save final checkpoint as well
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'online': online.state_dict(),
                'predictor': predictor.state_dict(),
                'target_encoder': target.encoder.state_dict(),
                'step': global_step,
            },
            out_path / 'checkpoint.pt',
        )

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
    }
