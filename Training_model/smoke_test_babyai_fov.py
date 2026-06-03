"""Smoke test for BabyAI FOV target detection (no Minari required)."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgPartialObsWrapper

ROOT = Path(__file__).resolve().parents[1]
TRAINING = Path(__file__).resolve().parent
for p in (str(ROOT / "Datasets"), str(TRAINING)):
    if p not in sys.path:
        sys.path.insert(0, p)

from babyai_fov import classify_step_mission, has_visible_target  # noqa: E402


def main() -> None:
    env = RGBImgPartialObsWrapper(gym.make("BabyAI-Open-v0"), tile_size=8)
    obs, _ = env.reset(seed=0)
    base_mission = str(obs["mission"])

    def make_env_fn():
        return RGBImgPartialObsWrapper(gym.make("BabyAI-Open-v0"), tile_size=8)

    mission, tag = classify_step_mission(
        base_mission,
        env,
        make_env_fn=make_env_fn,
        seed=0,
        options=None,
        prefix_action_ids=[],
    )
    print("mission:", mission)
    print("tag:", tag)
    print("has_target:", has_visible_target(env))

    env.step(int(Actions.forward))
    print("after forward has_target:", has_visible_target(env))
    env.close()
    print("smoke_test_babyai_fov OK")


if __name__ == "__main__":
    main()
