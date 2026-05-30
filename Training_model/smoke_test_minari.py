"""Smoke test: RGB replay + MinariSFT sample (no HF download)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401 — register envs
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Training_model"
DATASETS = ROOT / "Datasets"
for p in (str(TRAINING), str(DATASETS)):
    if p not in sys.path:
        sys.path.insert(0, p)


class _FakeMinariDataset:
    """Minimal stand-in for minari.MinariDataset in offline smoke tests."""

    def recover_environment(self, **kwargs):
        return gym.make("BabyAI-Open-v0")


def _collect_episode(seed: int = 42, max_steps: int = 30):
    from minari.dataset.episode_data import EpisodeData

    env = gym.make("BabyAI-Open-v0")
    obs, _ = env.reset(seed=seed)
    observations = [obs]
    actions = []
    rewards = []
    terminations = []
    truncations = []

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, rew, term, trunc, _ = env.step(action)
        actions.append(action)
        rewards.append(rew)
        terminations.append(term)
        truncations.append(trunc)
        observations.append(obs)
        if term or trunc:
            break
    env.close()

    return EpisodeData(
        id=0,
        observations=observations,
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        terminations=np.array(terminations, dtype=bool),
        truncations=np.array(truncations, dtype=bool),
        infos={},
    ), {"seed": seed, "options": None}


def main() -> None:
    from minari_adapter import replay_episode_rgb
    from minigrid_sft_dataset import MiniGridSFTDataset, precompute_minari_for_keys

    episode, meta = _collect_episode()
    ds = _FakeMinariDataset()
    steps = replay_episode_rgb(ds, episode, meta, tile_size=8)
    assert steps, "replay returned no steps"
    assert steps[0].image.ndim == 3 and steps[0].image.shape[2] == 3
    print(f"replay ok: {len(steps)} steps, image shape={steps[0].image.shape}")

    # Patch load path: inject fake cache directly
    cache = {( "minigrid/smoke-local", 0): steps}

    class _Tok:
        eos_token = "</s>"

    class _Proc:
        def __call__(self, img):
            return img

    sft_ds = MiniGridSFTDataset.from_minari(
        ["minigrid/smoke-local"],
        _Tok(),
        _Proc(),
        trajectory_cache=cache,
        download=False,
    )
    item = sft_ds[0]
    assert item["action"] in steps[0].action
    print(f"sft sample action={item['action']!r}")
    print("smoke_test_minari: OK")


if __name__ == "__main__":
    main()
