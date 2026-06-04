"""CLI: GRPO fine-tuning after SFT (action or text+action)."""

from __future__ import annotations

import argparse

from grpo_config import GRPOConfig
from grpo_train import train_grpo


def main() -> None:
    p = argparse.ArgumentParser(description="GRPO fine-tuning for MiniGrid VLM")
    p.add_argument("--sft-checkpoint", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--mode", choices=("action", "text_action"), default="action")
    p.add_argument("--env", choices=("empty", "manifest"), default="empty")
    p.add_argument("--empty-size", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--num-updates", type=int, default=200)
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=25)
    args = p.parse_args()

    cfg = GRPOConfig(output_mode=args.mode, env_mode=args.env)
    if args.sft_checkpoint:
        cfg.sft_checkpoint = args.sft_checkpoint
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.empty_size = args.empty_size
    cfg.group_size = args.group_size
    cfg.num_updates = args.num_updates
    cfg.kl_coef = args.kl_coef
    cfg.eval_every = args.eval_every

    train_grpo(cfg)


if __name__ == "__main__":
    main()
