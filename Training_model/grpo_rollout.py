"""Environment factories and episode rollouts for GRPO."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper

TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
for p in (str(TRAINING_DIR), str(ROOT / "Datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from minari_adapter import ACTION_NAME_TO_ID, load_minari_dataset, make_rgb_env  # noqa: E402


@dataclass
class RolloutStep:
    rgb: np.ndarray
    mission: str
    action: str
    action_idx: int
    log_prob: float
    kl: float = 0.0
    completion_ids: list[int] | None = None
    prompt_text: str = ""


@dataclass
class EpisodeTrajectory:
    seed: int
    env_id: str
    steps: list[RolloutStep] = field(default_factory=list)
    return_: float = 0.0
    success: bool = False
    terminated: bool = False
    truncated: bool = False

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_log_prob(self) -> float:
        return sum(s.log_prob for s in self.steps)

    @property
    def total_kl(self) -> float:
        return sum(s.kl for s in self.steps)


def empty_env_id(size: int) -> str:
    if size not in (5, 6, 8, 16):
        raise ValueError(f"unsupported Empty size: {size}")
    return f"MiniGrid-Empty-{size}x{size}-v0"


def make_empty_rgb_env(*, size: int = 8, tile_size: int = 8) -> gym.Env:
    base = gym.make(empty_env_id(size))
    return RGBImgPartialObsWrapper(base, tile_size=tile_size)


def make_manifest_rgb_env(
    ep_meta: dict[str, Any],
    *,
    tile_size: int,
    dataset_cache: dict[str, Any],
) -> tuple[gym.Env, dict[str, Any], str]:
    dataset_id = ep_meta["dataset_id"]
    episode_id = int(ep_meta["episode_id"])
    if dataset_id not in dataset_cache:
        dataset_cache[dataset_id] = load_minari_dataset(dataset_id, download=True)
    dataset = dataset_cache[dataset_id]
    meta = list(dataset.storage.get_episode_metadata([episode_id]))[0]
    env = make_rgb_env(dataset, tile_size=tile_size)
    env_key = f"{dataset_id}#{episode_id}"
    return env, meta, env_key


def rollout_episode(
    env: gym.Env,
    policy,
    *,
    seed: int,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    env_id: str = "env",
) -> EpisodeTrajectory:
    """env -> policy -> step loop; stops on done or step limit."""
    obs, _ = env.reset(seed=seed, options=options)
    mission = str(obs["mission"])
    env_max = int(getattr(env.unwrapped, "max_steps", 256))
    limit = env_max if max_steps is None else min(int(max_steps), env_max)

    traj = EpisodeTrajectory(seed=seed, env_id=env_id)
    total_reward = 0.0
    terminated = truncated = False

    for _ in range(limit):
        rgb = np.asarray(obs["image"], dtype=np.uint8)
        action, log_prob, kl, extra = policy.act(rgb, mission, sample=True)
        action_id = ACTION_NAME_TO_ID.get(action, ACTION_NAME_TO_ID["forward"])

        step = RolloutStep(
            rgb=rgb,
            mission=mission,
            action=action,
            action_idx=ACTION_NAME_TO_ID[action],
            log_prob=float(log_prob),
            kl=float(kl),
            completion_ids=extra.get("completion_ids"),
            prompt_text=extra.get("prompt_text", ""),
        )
        traj.steps.append(step)

        obs, reward, terminated, truncated, _ = env.step(action_id)
        total_reward += float(reward)
        mission = str(obs["mission"])

        if terminated or truncated:
            break

    traj.return_ = total_reward
    traj.terminated = terminated
    traj.truncated = truncated
    traj.success = terminated and not truncated and total_reward > 0.0
    return traj


def collect_grpo_group(
    env_factory,
    policy,
    *,
    seed: int,
    group_size: int,
    max_steps: int | None,
    env_id: str,
    reset_options: dict[str, Any] | None = None,
) -> list[EpisodeTrajectory]:
    """G rollouts on the same reset seed (GRPO group)."""
    group: list[EpisodeTrajectory] = []
    for _ in range(group_size):
        env = env_factory()
        try:
            traj = rollout_episode(
                env,
                policy,
                seed=seed,
                options=reset_options,
                max_steps=max_steps,
                env_id=env_id,
            )
            group.append(traj)
        finally:
            env.close()
    return group


def group_advantages(returns: list[float], eps: float = 1e-6, clip: float = 5.0) -> list[float]:
    arr = np.asarray(returns, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < eps:
        adv = arr - mean
    else:
        adv = (arr - mean) / (std + eps)
    if clip > 0:
        adv = np.clip(adv, -clip, clip)
    return adv.tolist()


def iter_training_seeds(cfg_seed: int, num: int) -> list[int]:
    rng = random.Random(cfg_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(num)]
