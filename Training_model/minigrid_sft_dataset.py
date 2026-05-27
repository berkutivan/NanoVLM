"""SFT dataset: (RGB frame, mission) -> next action from BFS expert."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "Datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from maze_expert import ExpertStep, rollout_expert_trajectory  # noqa: E402

PROMPT_TEMPLATE = (
    "Mission: {mission}\n"
    "What is the next action? Answer with one word: left, right, or forward.\n"
    "Answer:"
)


def load_objects(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("objects", []))


def build_flat_index(
    objects: list[dict[str, Any]],
    trajectory_cache: dict[int, list[ExpertStep]],
) -> list[tuple[int, int]]:
    flat: list[tuple[int, int]] = []
    for obj_idx, steps in trajectory_cache.items():
        for step_idx in range(len(steps)):
            flat.append((obj_idx, step_idx))
    return flat


class MiniGridSFTDataset(Dataset):
    def __init__(
        self,
        objects: list[dict[str, Any]],
        tokenizer,
        image_processor,
        trajectory_cache: dict[int, list[ExpertStep]] | None = None,
    ):
        self.objects = objects
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        if trajectory_cache is None:
            trajectory_cache = {}
            for i, obj in enumerate(objects):
                trajectory_cache[i] = rollout_expert_trajectory(obj)

        self.trajectory_cache = trajectory_cache
        self.flat = build_flat_index(objects, trajectory_cache)

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        obj_idx, step_idx = self.flat[idx]
        step = self.trajectory_cache[obj_idx][step_idx]

        image = Image.fromarray(step.image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        processed_image = self.image_processor(image)

        prompt = PROMPT_TEMPLATE.format(mission=step.mission)
        answer = f" {step.action}{self.tokenizer.eos_token}"

        return {
            "image": processed_image,
            "text_data": prompt,
            "answer": answer,
            "action": step.action,
            "object_id": self.objects[obj_idx].get("id", obj_idx),
            "step_index": step.step_index,
        }


def split_objects(
    objects: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(objects)))
    rng.shuffle(indices)
    val_count = max(1, int(len(objects) * val_ratio)) if len(objects) > 1 else 0
    val_idx = set(indices[:val_count])
    train = [objects[i] for i in range(len(objects)) if i not in val_idx]
    val = [objects[i] for i in range(len(objects)) if i in val_idx]
    return train, val


def precompute_trajectories(objects: list[dict[str, Any]]) -> dict[int, list[ExpertStep]]:
    cache: dict[int, list[ExpertStep]] = {}
    for i, obj in enumerate(objects):
        cache[i] = rollout_expert_trajectory(obj)
    return cache
