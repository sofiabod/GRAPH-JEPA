import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.builders import build_graph_encoder, build_target_encoder, build_predictor
from src.eval.eval_runner import EvalRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/enron.yaml')
    parser.add_argument('--data', type=str, default=None,
                        help='path to graphs .pt file (defaults to cfg.data.graphs_path)')
    parser.add_argument('--out_dir', type=str, default='results/eval')
    args = parser.parse_args()

    import json
    cfg = OmegaConf.load(args.config)
    data_path = args.data if args.data is not None else cfg.data.graphs_path
    graphs = torch.load(data_path)

    if hasattr(cfg.data, 'train_weeks') and cfg.data.train_weeks is not None:
        split_kwargs = dict(
            train_range=tuple(cfg.data.train_weeks),
            val_range=tuple(cfg.data.val_weeks),
            test_range=tuple(cfg.data.test_weeks),
        )
    else:
        with open(cfg.data.meta_path) as f:
            meta = json.load(f)
        split_kwargs = dict(
            train_range=tuple(meta['train_range']),
            val_range=tuple(meta['val_range']),
            test_range=tuple(meta['test_range']),
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    online = build_graph_encoder(cfg.encoder).to(device)
    target = build_target_encoder(online)
    target.encoder = target.encoder.to(device)
    predictor = build_predictor(cfg.predictor).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    online.load_state_dict(ckpt['online'])
    predictor.load_state_dict(ckpt['predictor'])
    if 'target_encoder' in ckpt:
        target.encoder.load_state_dict(ckpt['target_encoder'])

    runner = EvalRunner(online, target, predictor, graphs, cfg, **split_kwargs)
    results = runner.run_all(args.out_dir)
    print(results)


if __name__ == '__main__':
    main()
