"""Hyperparameters for MiniGrid SFT fine-tuning of nanoVLM-222M."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_JSON = ROOT / "Datasets" / "dataset.json"
PRETRAINED_CKPT = ROOT / "checkpoints" / "nanoVLM-222M"
SFT_OUTPUT_DIR = ROOT / "checkpoints" / "sft-minigrid"


@dataclass
class SFTConfig:
    # Fine-tuning: load full pretrained VLM (vision + language + MP), not random init
    pretrained_path: str = str(PRETRAINED_CKPT)
    output_dir: str = str(SFT_OUTPUT_DIR)

    dataset_path: str = str(DATASET_JSON)
    val_ratio: float = 0.1
    seed: int = 0

    epochs: int = 3
    batch_size: int = 8
    num_workers: int = 0  # trajectories cached; avoid fork + pygame issues
    grad_accum_steps: int = 1

    # Two LR groups (nanoVLM recipe): MP faster, backbones slower
    lr_mp: float = 1e-3
    lr_backbones: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    log_every: int = 10
    eval_every: int = 50
    save_every_epoch: bool = True

    compile_model: bool = False
    use_amp: bool = True  # only applied when CUDA is available

    max_objects: int | None = None  # debug: cap number of mazes
