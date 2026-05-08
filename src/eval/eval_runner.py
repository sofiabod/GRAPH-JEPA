"""eval runner: loads checkpoint, runs all 6 evals, saves results."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.metrics import cosine_sim, effective_rank, mean_pairwise_cosine
from src.eval.wilcoxon import bonferroni_correct, paired_wilcoxon


class EvalRunner:
    def __init__(
        self,
        online,
        target,
        predictor,
        graphs,
        cfg,
        train_range=(0, 119),
        val_range=(120, 139),
        test_range=(140, 179),
        mask_seed=0,
    ):
        self.online = online
        self.target = target
        self.predictor = predictor
        self.graphs = graphs
        self.cfg = cfg
        self.device = next(online.parameters()).device
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range
        self.mask_seed = mask_seed

    def run_all(self, out_dir: str) -> dict:
        # runs the eval family minus eval 2 (which needs a paired ablation ckpt)
        # applies bonferroni across the family of paired tests for eval 1
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}
        results["eval1_node_prediction"] = self._eval1_node_prediction()
        results["eval3_multistep_rollout"] = self._eval3_multistep_rollout()
        results["eval6_representation_quality"] = self._eval6_representation_quality()

        # bonferroni across eval 1 family (model vs copy-forward, model vs graph-average)
        e1 = results["eval1_node_prediction"]
        if "wilcoxon_p_vs_copy" in e1 and "wilcoxon_p_vs_graph_avg" in e1:
            corrected = bonferroni_correct(
                [e1["wilcoxon_p_vs_copy"], e1["wilcoxon_p_vs_graph_avg"]], n_tests=2
            )
            e1["wilcoxon_p_vs_copy_bonferroni"] = corrected[0]
            e1["wilcoxon_p_vs_graph_avg_bonferroni"] = corrected[1]
            results["family_size"] = 2

        with open(out_path / "eval_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    def run_all_with_eval2(self, out_dir: str, sequential_ckpt_path=None) -> dict:
        """eval family including eval 2 if a sequential-ablation checkpoint is given.

        bonferroni correction is applied across the whole family of paired tests:
        eval1 (model vs copy, model vs graph-avg) + eval2 (graph vs sequential).
        """
        from src.eval.eval2 import eval2_compare

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}
        results["eval1_node_prediction"] = self._eval1_node_prediction()
        results["eval3_multistep_rollout"] = self._eval3_multistep_rollout()
        results["eval6_representation_quality"] = self._eval6_representation_quality()

        e1 = results["eval1_node_prediction"]
        pvals = []
        labels = []
        if "wilcoxon_p_vs_copy" in e1:
            pvals.append(e1["wilcoxon_p_vs_copy"])
            labels.append("eval1_vs_copy")
        if "wilcoxon_p_vs_graph_avg" in e1:
            pvals.append(e1["wilcoxon_p_vs_graph_avg"])
            labels.append("eval1_vs_graph_avg")

        if sequential_ckpt_path is not None:
            # graph_ckpt_path is implicit: the runner already holds the graph-jepa models
            e2 = eval2_compare(
                graph_models=(self.online, self.target, self.predictor),
                sequential_ckpt_path=sequential_ckpt_path,
                graphs=self.graphs,
                cfg=self.cfg,
                splits=(self.train_range, self.val_range, self.test_range),
                mask_seed=self.mask_seed,
                device=self.device,
            )
            results["eval2_graph_vs_sequential"] = e2
            pvals.append(e2["wilcoxon_p"])
            labels.append("eval2_graph_vs_sequential")

        if len(pvals) > 0:
            corrected = bonferroni_correct(pvals, n_tests=len(pvals))
            results["bonferroni"] = {
                label: {"raw": pvals[i], "corrected": corrected[i]}
                for i, label in enumerate(labels)
            }
            results["family_size"] = len(pvals)

        with open(out_path / "eval_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    def _eval1_node_prediction(self) -> dict:
        # cos(z_pred, z_target) vs copy-forward and graph-average baselines
        # uses test split (week indices 140-179 by default)
        from src.data.dataset import TemporalGraphDataset
        from src.train import _build_tokens_for_sample, _encode_context

        dataset = TemporalGraphDataset(
            self.graphs,
            context_k=self.cfg.training.context_k,
            mask_ratio=self.cfg.training.mask_ratio,
            split="test",
            train_range=self.train_range,
            val_range=self.val_range,
            test_range=self.test_range,
            seed=self.mask_seed,
        )
        if len(dataset) == 0:
            return {"error": "no test data"}

        pred_sims = []
        copy_sims = []
        graph_avg_sims = []

        self.online.eval()
        self.predictor.eval()

        with torch.no_grad():
            for sample in dataset:
                tgt_graph = sample["target_graph"].to(self.device)
                masked_ids = sample["masked_node_ids"].to(self.device)
                visible_ids = sample["visible_node_ids"].to(self.device)

                ctx_embs = _encode_context(self.online, sample["context_graphs"])
                tgt_emb = self.target(tgt_graph)

                tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
                    ctx_embs, tgt_emb, masked_ids, visible_ids, self.predictor
                )
                out = self.predictor(
                    tokens.unsqueeze(0),
                    time_indices.unsqueeze(0),
                    node_ids_seq.unsqueeze(0),
                ).squeeze(0)

                # normalize predictor output to match training-time sphere geometry
                z_pred = F.normalize(out[mask_positions], dim=-1)
                z_true = tgt_emb[masked_ids]
                z_last = ctx_embs[-1][masked_ids]  # copy-forward baseline

                # graph-average baseline: mean of neighbor embeddings at last context step
                z_graph_avg = _graph_average_baseline(masked_ids, tgt_graph, ctx_embs[-1])

                pred_sims.extend(cosine_sim(z_pred, z_true).cpu().numpy().tolist())
                copy_sims.extend(cosine_sim(z_last, z_true).cpu().numpy().tolist())
                graph_avg_sims.extend(cosine_sim(z_graph_avg, z_true).cpu().numpy().tolist())

        pred_arr = np.array(pred_sims)
        copy_arr = np.array(copy_sims)
        graph_avg_arr = np.array(graph_avg_sims)

        p_copy, stat_copy = paired_wilcoxon(pred_arr, copy_arr)
        p_ga, stat_ga = paired_wilcoxon(pred_arr, graph_avg_arr)

        return {
            "mean_pred_cos": float(pred_arr.mean()),
            "mean_copy_cos": float(copy_arr.mean()),
            "mean_graph_avg_cos": float(graph_avg_arr.mean()),
            "wilcoxon_p_vs_copy": p_copy,
            "wilcoxon_stat_vs_copy": stat_copy,
            "wilcoxon_p_vs_graph_avg": p_ga,
            "wilcoxon_stat_vs_graph_avg": stat_ga,
            "n_pairs": len(pred_arr),
        }

    def _eval3_multistep_rollout(self, horizons=(1, 2, 4)) -> dict:
        """autoregressive multi-step rollout in latent space.

        for each test sample with target index T, predict masked-node embeddings
        at horizons T+(h-1) for h in {1,2,4} by feeding earlier predictions back
        as context. compare to ground-truth target encoder output and to
        copy-forward (last-known online embedding) at each horizon.

        the masked node set is fixed across horizons (deterministic masking).
        non-masked nodes at each predicted timestep use the target encoder's
        ground-truth embedding (revealed); masked nodes use the model's
        previous-step prediction. this is the standard v-jepa rollout protocol.
        """
        from src.data.dataset import TemporalGraphDataset
        from src.train import _build_tokens_for_sample

        k_ctx = self.cfg.training.context_k
        max_h = max(horizons)

        dataset = TemporalGraphDataset(
            self.graphs,
            context_k=k_ctx,
            mask_ratio=self.cfg.training.mask_ratio,
            split="test",
            train_range=self.train_range,
            val_range=self.val_range,
            test_range=self.test_range,
            seed=self.mask_seed,
        )
        if len(dataset) == 0:
            return {"error": "no test data"}

        pred_cos_per_h = {h: [] for h in horizons}
        copy_cos_per_h = {h: [] for h in horizons}

        self.online.eval()
        self.predictor.eval()

        with torch.no_grad():
            for sample_idx in range(len(dataset)):
                sample = dataset[sample_idx]
                target_idx = dataset.valid_target_indices[sample_idx]
                masked_ids = sample["masked_node_ids"].to(self.device)
                visible_ids = sample["visible_node_ids"].to(self.device)

                # encode k context graphs once with online encoder
                ctx_embs = [self.online(g.to(self.device)) for g in sample["context_graphs"]]
                z_last_ctx = ctx_embs[-1]  # used for copy-forward baseline at every horizon

                current_ctx = list(ctx_embs)
                for h in range(1, max_h + 1):
                    t_h = target_idx + h - 1
                    if t_h >= len(self.graphs):
                        break

                    target_graph = self.graphs[t_h].to(self.device)
                    z_target_full = self.target(target_graph)  # [N, D] target encoder, normalized
                    z_target_masked = z_target_full[masked_ids]

                    tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
                        current_ctx, z_target_full, masked_ids, visible_ids, self.predictor
                    )
                    out = self.predictor(
                        tokens.unsqueeze(0),
                        time_indices.unsqueeze(0),
                        node_ids_seq.unsqueeze(0),
                    ).squeeze(0)
                    z_pred = F.normalize(out[mask_positions], dim=-1)

                    if h in horizons:
                        # both z_pred and z_target_masked are unit vectors; cos = dot product
                        cos = (z_pred * z_target_masked).sum(dim=-1).cpu().numpy()
                        pred_cos_per_h[h].extend(cos.tolist())

                        # copy-forward baseline at this horizon: z_last_ctx[masked] vs z_target at t_h
                        z_last = z_last_ctx[masked_ids]
                        cos_cf = (z_last * z_target_masked).sum(dim=-1).cpu().numpy()
                        copy_cos_per_h[h].extend(cos_cf.tolist())

                    # slide window for next iteration: composed embedding at t_h
                    # non-masked nodes use ground-truth target encoder; masked nodes use prediction
                    composed = z_target_full.clone()
                    composed[masked_ids] = z_pred
                    current_ctx = current_ctx[1:] + [composed]

        result = {"horizons_evaluated": [h for h in horizons if len(pred_cos_per_h[h]) > 0]}
        for h in horizons:
            if len(pred_cos_per_h[h]) == 0:
                continue
            pa = np.array(pred_cos_per_h[h])
            ca = np.array(copy_cos_per_h[h])
            p, stat = paired_wilcoxon(pa, ca)
            result[f"h{h}"] = {
                "mean_pred_cos": float(pa.mean()),
                "mean_copy_cos": float(ca.mean()),
                "wilcoxon_p_vs_copy": p,
                "wilcoxon_stat_vs_copy": stat,
                "n_pairs": len(pa),
            }
        return result

    def _eval6_representation_quality(self) -> dict:
        # effective rank and mean pairwise cosine on test set embeddings
        from src.data.dataset import TemporalGraphDataset

        dataset = TemporalGraphDataset(
            self.graphs,
            context_k=4,
            mask_ratio=0.0,
            split="test",
            train_range=self.train_range,
            val_range=self.val_range,
            test_range=self.test_range,
            seed=self.mask_seed,
        )
        if len(dataset) == 0:
            return {"error": "no test data"}

        all_embs = []
        self.online.eval()
        with torch.no_grad():
            for sample in dataset:
                g = sample["target_graph"].to(self.device)
                z = self.online(g)
                all_embs.append(z)

        z_all = torch.cat(all_embs, dim=0)
        return {
            "effective_rank": effective_rank(z_all),
            "mean_pairwise_cosine": mean_pairwise_cosine(z_all),
        }


def _graph_average_baseline(masked_ids, tgt_graph, last_ctx_emb):
    """mean neighbor embedding at the last context timestep, l2-normalized.

    for each masked node v, find neighbors in the target graph snapshot
    and average their last-context-step embeddings. isolated nodes fall
    back to the global mean of last_ctx_emb.
    """
    device = last_ctx_emb.device
    n_nodes = last_ctx_emb.shape[0]
    edge_index = tgt_graph.edge_index
    # treat edges as undirected for neighborhood: union of source and dest
    src = edge_index[0]
    dst = edge_index[1]

    global_mean = last_ctx_emb.mean(dim=0, keepdim=True)
    out = torch.zeros(masked_ids.shape[0], last_ctx_emb.shape[1], device=device)
    for i, v in enumerate(masked_ids.tolist()):
        # neighbors from both directions
        mask_out = src == v
        mask_in = dst == v
        neighbors = torch.cat([dst[mask_out], src[mask_in]], dim=0)
        if neighbors.numel() == 0:
            out[i] = global_mean.squeeze(0)
        else:
            # restrict to valid node ids
            neighbors = neighbors[(neighbors >= 0) & (neighbors < n_nodes)]
            if neighbors.numel() == 0:
                out[i] = global_mean.squeeze(0)
            else:
                out[i] = last_ctx_emb[neighbors].mean(dim=0)
    return F.normalize(out, dim=-1)
