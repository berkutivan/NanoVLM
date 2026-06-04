"""Hyperparameters for GRPO fine-tuning after SFT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
SFT_CKPT_DIR = ROOT / "checkpoints" / "sft-minigrid"
CURATED_MANIFEST = ROOT / "Datasets" / "babyai_curated" / "manifest.json"

GrpoOutputMode = Literal["action", "text_action"]
GrpoEnvMode = Literal["empty", "manifest"]


@dataclass
class GRPOConfig:
    # Init from SFT checkpoint (required)
    sft_checkpoint: str = str(SFT_CKPT_DIR / "best")
    output_dir: str = str(ROOT / "checkpoints" / "grpo-minigrid")

    output_mode: GrpoOutputMode = "action"
    env_mode: GrpoEnvMode = "empty"

    # MiniGrid-Empty-* (assignment) or curated manifest episodes (BabyAI, like SFT eval)
    empty_size: int = 8
    tile_size: int = 8
    curated_manifest_path: str = str(CURATED_MANIFEST)
    train_episode_ids: list[int] | None = None  # None = use train split from manifest

    # GRPO
    group_size: int = 4
    num_updates: int = 200
    episodes_per_update: int = 1  # how many independent seeds per update (each spawns group_size rollouts)
    temperature: float = 1.0
    kl_coef: float = 0.02
    adv_eps: float = 1e-6
    clip_adv: float = 5.0

    max_agent_steps: int | None = None
    seed: int = 0

    # Optimizer (same recipe as SFT)
    lr_mp: float = 5e-4
    lr_backbones: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    freeze_vision: bool = True
    use_8bit_optimizer: bool = True
    unfreeze_lm_blocks: int = 0

    # Generation (text_action mode)
    max_new_tokens_text: int = 64

    log_every: int = 10
    eval_every: int = 25
    eval_episodes: int = 8
    save_every: int = 50

    use_amp: bool = True
