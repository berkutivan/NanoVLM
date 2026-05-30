"""SFT dataset: (RGB frame, mission) -> next expert action (BabyAI / Minari)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "Datasets"
TRAINING_DIR = Path(__file__).resolve().parent
for p in (str(DATASETS_DIR), str(TRAINING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from maze_expert import ExpertStep, rollout_expert_trajectory  # noqa: E402

ACTIONS_PROMPT = "left, right, forward, pickup, drop, toggle, or done"
PROMPT_TEMPLATE = (
    "Mission: {mission}\n"
    f"What is the next action? Answer with one word: {ACTIONS_PROMPT}.\n"
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


def build_minari_flat_index(
    trajectory_cache: dict[tuple[str, int], list[ExpertStep]],
) -> list[tuple[str, int, int]]:
    flat: list[tuple[str, int, int]] = []
    for (dataset_id, episode_id), steps in trajectory_cache.items():
        for step_idx in range(len(steps)):
            flat.append((dataset_id, episode_id, step_idx))
    return flat


class MiniGridSFTDataset(Dataset):
    def __init__(
        self,
        objects: list[dict[str, Any]],
        tokenizer,
        image_processor,
        trajectory_cache: dict[int, list[ExpertStep]] | None = None,
    ):
        self.source = "json"
        self.objects = objects
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.minari_cache: dict[tuple[str, int], list[ExpertStep]] | None = None
        self.minari_flat: list[tuple[str, int, int]] = []

        if trajectory_cache is None:
            trajectory_cache = {}
            for i, obj in enumerate(objects):
                trajectory_cache[i] = rollout_expert_trajectory(obj)

        self.trajectory_cache = trajectory_cache
        self.flat = build_flat_index(objects, trajectory_cache)

    @classmethod
    def from_minari(
        cls,
        dataset_ids: list[str],
        tokenizer,
        image_processor,
        *,
        trajectory_cache: dict[tuple[str, int], list[ExpertStep]] | None = None,
        download: bool = True,
        tile_size: int = 8,
        max_episodes_per_dataset: int | None = None,
        log_every: int = 100,
    ) -> MiniGridSFTDataset:
        from minari_adapter import (  # noqa: WPS433
            iterate_episode_ids,
            load_minari_dataset,
            precompute_minari_trajectories,
        )

        if trajectory_cache is None:
            trajectory_cache = {}
            for dataset_id in dataset_ids:
                ds = load_minari_dataset(dataset_id, download=download)
                ep_ids = iterate_episode_ids(ds, max_episodes_per_dataset)
                partial = precompute_minari_trajectories(
                    dataset_id,
                    ep_ids,
                    download=False,
                    tile_size=tile_size,
                    log_every=log_every,
                )
                for ep_id, steps in partial.items():
                    trajectory_cache[(dataset_id, ep_id)] = steps

        self = cls.__new__(cls)
        self.source = "minari"
        self.objects = []
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.trajectory_cache = {}
        self.minari_cache = trajectory_cache
        self.minari_flat = build_minari_flat_index(trajectory_cache)
        self.flat = []
        return self

    def __len__(self) -> int:
        if self.source == "minari":
            return len(self.minari_flat)
        return len(self.flat)

    def _item_from_step(
        self,
        step: ExpertStep,
        *,
        object_id: int | str,
        step_index: int,
    ) -> dict[str, Any]:
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
            "allowed_actions": list(step.allowed_actions),
            "object_id": object_id,
            "step_index": step_index,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.source == "minari":
            assert self.minari_cache is not None
            dataset_id, episode_id, step_idx = self.minari_flat[idx]
            step = self.minari_cache[(dataset_id, episode_id)][step_idx]
            return self._item_from_step(
                step,
                object_id=f"{dataset_id}:{episode_id}",
                step_index=step.step_index,
            )

        obj_idx, step_idx = self.flat[idx]
        step = self.trajectory_cache[obj_idx][step_idx]
        return self._item_from_step(
            step,
            object_id=self.objects[obj_idx].get("id", obj_idx),
            step_index=step.step_index,
        )


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


def split_minari_episodes(
    episode_keys: list[tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Episode-level train/val split (no step leakage)."""
    rng = random.Random(seed)
    keys = list(episode_keys)
    rng.shuffle(keys)
    val_count = max(1, int(len(keys) * val_ratio)) if len(keys) > 1 else 0
    val_set = set(keys[:val_count])
    train = [k for k in keys if k not in val_set]
    val = [k for k in keys if k in val_set]
    return train, val


def filter_minari_cache(
    cache: dict[tuple[str, int], list[ExpertStep]],
    keys: list[tuple[str, int]],
) -> dict[tuple[str, int], list[ExpertStep]]:
    key_set = set(keys)
    return {k: v for k, v in cache.items() if k in key_set}


def precompute_trajectories(objects: list[dict[str, Any]]) -> dict[int, list[ExpertStep]]:
    cache: dict[int, list[ExpertStep]] = {}
    for i, obj in enumerate(objects):
        cache[i] = rollout_expert_trajectory(obj)
    return cache


def precompute_minari_for_keys(
    keys: list[tuple[str, int]],
    *,
    download: bool = True,
    tile_size: int = 8,
    log_every: int = 100,
) -> dict[tuple[str, int], list[ExpertStep]]:
    """Replay only the requested (dataset_id, episode_id) pairs."""
    from collections import defaultdict

    from minari_adapter import precompute_minari_trajectories  # noqa: WPS433

    by_dataset: dict[str, list[int]] = defaultdict(list)
    for dataset_id, episode_id in keys:
        by_dataset[dataset_id].append(episode_id)

    cache: dict[tuple[str, int], list[ExpertStep]] = {}
    for dataset_id, ep_ids in by_dataset.items():
        partial = precompute_minari_trajectories(
            dataset_id,
            sorted(set(ep_ids)),
            download=download,
            tile_size=tile_size,
            log_every=log_every,
        )
        for ep_id, steps in partial.items():
            cache[(dataset_id, ep_id)] = steps
    return cache
