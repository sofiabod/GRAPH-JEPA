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
    parser.add_argument('--config', type=str, default='configs/tgjepa_base.yaml')
    parser.add_argument('--data', type=str, default='data/enron_graphs.pt')
    parser.add_argument('--out_dir', type=str, default='results/eval')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    graphs = torch.load(args.data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    online = build_graph_encoder(cfg.encoder).to(device)
    target = build_target_encoder(online)
    target.encoder = target.encoder.to(device)
    predictor = build_predictor(cfg.predictor).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    online.load_state_dict(ckpt['online'])
    predictor.load_state_dict(ckpt['predictor'])

    runner = EvalRunner(online, target, predictor, graphs, cfg)
    results = runner.run_all(args.out_dir)
    print(results)


if __name__ == '__main__':
    main()
