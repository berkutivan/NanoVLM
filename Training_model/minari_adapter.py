"""Load Minari BabyAI datasets and replay episodes with RGB partial observations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgPartialObsWrapper

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "Datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from maze_expert import ExpertStep  # noqa: E402
from maze_presample import _patch_minigrid_render_font  # noqa: E402

_patch_minigrid_render_font()

# BabyAI / MiniGrid discrete actions (Minari optimal-v0 datasets).
ACTION_ID_TO_NAME: dict[int, str] = {
    int(Actions.left): "left",
    int(Actions.right): "right",
    int(Actions.forward): "forward",
    int(Actions.pickup): "pickup",
    int(Actions.drop): "drop",
    int(Actions.toggle): "toggle",
    int(Actions.done): "done",
}
ACTION_NAME_TO_ID: dict[str, int] = {v: k for k, v in ACTION_ID_TO_NAME.items()}

MINARI_MAZE_DATASETS: tuple[str, ...] = (
    "minigrid/BabyAI-GoToObjMazeOpen/optimal-v0",
    "minigrid/BabyAI-GoToObjMaze/optimal-v0",
    "minigrid/BabyAI-GoToObjMazeS4/optimal-v0",
    "minigrid/BabyAI-GoToObjMazeS5/optimal-v0",
    "minigrid/BabyAI-GoToObjMazeS6/optimal-v0",
    "minigrid/BabyAI-GoToObjMazeS7/optimal-v0",
    "minigrid/BabyAI-Open/optimal-v0",
    "minigrid/BabyAI-KeyCorridorS3R1/optimal-v0",
    "minigrid/BabyAI-KeyCorridorS4R3/optimal-v0",
    "minigrid/BabyAI-KeyCorridorS5R3/optimal-v0",
)


def action_name(action_id: int | np.integer) -> str:
    aid = int(action_id)
    if aid not in ACTION_ID_TO_NAME:
        raise ValueError(f"unknown MiniGrid action id: {aid}")
    return ACTION_ID_TO_NAME[aid]


def make_rgb_env(dataset: Any, tile_size: int = 8):
    """Recover Minari env and wrap with RGB partial-observation rendering."""
    base = dataset.recover_environment()
    return RGBImgPartialObsWrapper(base, tile_size=tile_size)


def replay_episode_rgb(
    dataset: Any,
    episode: Any,
    episode_metadata: dict[str, Any] | None = None,
    *,
    tile_size: int = 8,
) -> list[ExpertStep]:
    """
    Replay a Minari episode in a fresh RGB-wrapped env.

    Uses episode metadata seed/options (Minari autoseed) so layouts match collection.
    """
    if episode_metadata is None:
        metas = list(dataset.storage.get_episode_metadata([episode.id]))
        episode_metadata = metas[0] if metas else {}

    env = make_rgb_env(dataset, tile_size=tile_size)
    seed = episode_metadata.get("seed")
    options = episode_metadata.get("options")
    obs, _ = env.reset(seed=seed, options=options)
    mission = str(obs["mission"])
    steps: list[ExpertStep] = []

    for step_index, action in enumerate(episode.actions):
        aid = int(action)
        name = action_name(aid)
        steps.append(
            ExpertStep(
                image=np.asarray(obs["image"], dtype=np.uint8),
                mission=mission,
                action=name,
                action_id=aid,
                allowed_actions=(name,),
                step_index=step_index,
            )
        )
        obs, _, terminated, truncated, _ = env.step(aid)
        if terminated or truncated:
            break

    env.close()
    if not steps:
        raise RuntimeError(f"empty replay for episode id={episode.id}")
    return steps


def ensure_h5py() -> None:
    """Minari BabyAI datasets use HDF5; fail fast with install hint."""
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read Minari HDF5 datasets. Install into .venv:\n"
            '  pip install h5py\n'
            '  pip install "minari[hdf5]"\n'
            "If pip times out on Windows, use conda:\n"
            "  conda install -y h5py -c conda-forge\n"
            "Then copy h5py into .venv or recreate the venv with h5py included."
        ) from exc


def load_minari_dataset(dataset_id: str, *, download: bool = True):
    ensure_h5py()
    import minari

    return minari.load_dataset(dataset_id, download=download)


def iterate_episode_ids(dataset: Any, max_episodes: int | None = None) -> list[int]:
    total = int(dataset.total_episodes)
    ids = list(range(total))
    if max_episodes is not None:
        ids = ids[: max(0, max_episodes)]
    return ids


def precompute_minari_trajectories(
    dataset_id: str,
    episode_ids: list[int],
    *,
    download: bool = True,
    tile_size: int = 8,
    log_every: int = 100,
) -> dict[int, list[ExpertStep]]:
    """Replay selected episodes; cache keyed by episode id within one dataset."""
    dataset = load_minari_dataset(dataset_id, download=download)
    metas: dict[int, dict[str, Any]] = {}
    for ep_id, meta in zip(episode_ids, dataset.storage.get_episode_metadata(episode_ids)):
        metas[int(ep_id)] = meta
    cache: dict[int, list[ExpertStep]] = {}
    episodes = list(dataset.iterate_episodes(episode_indices=episode_ids))
    for i, episode in enumerate(episodes):
        meta = metas.get(episode.id, {})
        cache[episode.id] = replay_episode_rgb(
            dataset, episode, meta, tile_size=tile_size
        )
        if log_every and (i + 1) % log_every == 0:
            print(f"  [{dataset_id}] replayed {i + 1}/{len(episodes)} episodes", flush=True)
    return cache


def dataset_step_count(cache: dict[int, list[ExpertStep]]) -> int:
    return sum(len(steps) for steps in cache.values())
