"""eval runner: loads checkpoint, runs all 6 evals, saves results."""
from pathlib import Path
import json
import torch
import numpy as np

from src.eval.metrics import cosine_sim, effective_rank, mean_pairwise_cosine
from src.eval.wilcoxon import paired_wilcoxon


class EvalRunner:
    def __init__(self, online, target, predictor, graphs, cfg,
                 train_range=(0, 119), val_range=(120, 139), test_range=(140, 179)):
        self.online = online
        self.target = target
        self.predictor = predictor
        self.graphs = graphs
        self.cfg = cfg
        self.device = next(online.parameters()).device
        self.train_range = train_range
        self.val_range = val_range
        self.test_range = test_range

    def run_all(self, out_dir: str) -> dict:
        # runs all 6 evals, saves eval_summary.json to out_dir
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}
        results['eval1_node_prediction'] = self._eval1_node_prediction()
        results['eval3_multistep_rollout'] = self._eval3_multistep_rollout()
        results['eval6_representation_quality'] = self._eval6_representation_quality()

        with open(out_path / 'eval_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def _eval1_node_prediction(self) -> dict:
        # cos(z_pred, z_target) vs copy-forward baseline
        # uses test split (week indices 140-179)
        from src.data.dataset import TemporalGraphDataset
        from src.train import _encode_context, _build_tokens_for_sample

        dataset = TemporalGraphDataset(
            self.graphs,
            context_k=self.cfg.training.context_k,
            mask_ratio=self.cfg.training.mask_ratio,
            split='test',
            train_range=self.train_range,
            val_range=self.val_range,
            test_range=self.test_range,
        )
        if len(dataset) == 0:
            return {'error': 'no test data'}

        pred_sims = []
        copy_sims = []

        self.online.eval()
        self.predictor.eval()

        with torch.no_grad():
            for sample in dataset:
                tgt_graph = sample['target_graph'].to(self.device)
                masked_ids = sample['masked_node_ids'].to(self.device)
                visible_ids = sample['visible_node_ids'].to(self.device)

                ctx_embs = _encode_context(self.online, sample['context_graphs'])
                tgt_emb = self.target(tgt_graph)

                tokens, time_indices, node_ids_seq, mask_positions = _build_tokens_for_sample(
                    ctx_embs, tgt_emb, masked_ids, visible_ids, self.predictor
                )
                out = self.predictor(
                    tokens.unsqueeze(0),
                    time_indices.unsqueeze(0),
                    node_ids_seq.unsqueeze(0),
                ).squeeze(0)

                z_pred = out[mask_positions]
                z_true = tgt_emb[masked_ids]
                z_last = ctx_embs[-1][masked_ids]  # copy-forward baseline

                pred_sims.extend(cosine_sim(z_pred, z_true).cpu().numpy().tolist())
                copy_sims.extend(cosine_sim(z_last, z_true).cpu().numpy().tolist())

        pred_arr = np.array(pred_sims)
        copy_arr = np.array(copy_sims)
        p_val, stat = paired_wilcoxon(pred_arr, copy_arr)

        return {
            'mean_pred_cos': float(pred_arr.mean()),
            'mean_copy_cos': float(copy_arr.mean()),
            'wilcoxon_p': p_val,
            'wilcoxon_stat': stat,
            'n_pairs': len(pred_arr),
        }

    def _eval3_multistep_rollout(self) -> dict:
        # placeholder: autoregressive rollout at horizons 1, 2, 4
        # full implementation requires iterative forward pass
        return {'status': 'not_implemented'}

    def _eval6_representation_quality(self) -> dict:
        # effective rank and mean pairwise cosine on test set embeddings
        from src.data.dataset import TemporalGraphDataset
        dataset = TemporalGraphDataset(
            self.graphs, context_k=4, mask_ratio=0.0, split='test',
            train_range=self.train_range,
            val_range=self.val_range,
            test_range=self.test_range,
        )
        if len(dataset) == 0:
            return {'error': 'no test data'}

        all_embs = []
        self.online.eval()
        with torch.no_grad():
            for sample in dataset:
                g = sample['target_graph'].to(self.device)
                z = self.online(g)
                all_embs.append(z)

        z_all = torch.cat(all_embs, dim=0)
        return {
            'effective_rank': effective_rank(z_all),
            'mean_pairwise_cosine': mean_pairwise_cosine(z_all),
        }
