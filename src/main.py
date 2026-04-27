import argparse
from omegaconf import OmegaConf
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/enron.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg, seed=args.seed)


if __name__ == "__main__":
    main()
