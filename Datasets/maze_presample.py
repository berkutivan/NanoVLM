"""
Procedural maze pre-sample for VLM SFT (one episode -> slice later).

Layout + expert from graph distance scores (official MiniGrid env pattern):
https://minigrid.farama.org/content/create_env_tutorial/

RGB observations via RGBImgPartialObsWrapper (no PNG on disk):
https://minigrid.farama.org/ (wrappers docs / manual_control)
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper


def _patch_minigrid_render_font() -> None:
    """MiniGrid passes get_default_font() ('freesansbold.ttf') to freetype.SysFont."""
    import pygame

    _render = MiniGridEnv.render

    def render(self, *args, **kwargs):
        orig = pygame.font.get_default_font
        default = orig()
        if default.endswith(".ttf"):
            pygame.font.get_default_font = lambda: default[:-4]
        try:
            return _render(self, *args, **kwargs)
        finally:
            pygame.font.get_default_font = orig

    MiniGridEnv.render = render


_patch_minigrid_render_font()

DATASET_DIR = Path(__file__).resolve().parent
DATASET_JSON = DATASET_DIR / "dataset.json"

@dataclass
class MazePreSampleConfig:
    size: int = 9
    n_walls: int = 4
    seed: int = 0
    agent_start_pos: tuple[int, int] = (1, 1)
    agent_start_dir: int = 0
    max_steps: int | None = None
    tile_size: int = 8
    block_score: float = 100.0
    render_mode: str | None = None
    fixed_walls: list[tuple[int, int]] | None = None
    fixed_goal: tuple[int, int] | None = None


class MazePreSampleBuilder:
    """
    One pre-sample = generated maze (layout + mission + predicate_space).

    Pipeline:
      1) field + mission, agent in top-left interior cell
      2) random internal walls (more walls -> harder)
      3) graph on walkable cells; keep component reachable from agent
      4) random goal inside that component
    """

    class ProceduralMazeEnv(MiniGridEnv):
        """Custom env following the official create-env tutorial."""

        def __init__(
            self,
            builder: MazePreSampleBuilder,
            size: int,
            n_walls: int,
            agent_start_pos: tuple[int, int],
            agent_start_dir: int,
            max_steps: int | None,
            render_mode: str | None,
            **kwargs,
        ):
            self._builder = builder
            self.n_walls = n_walls
            self.agent_start_pos = agent_start_pos
            self.agent_start_dir = agent_start_dir
            self.goal_pos: tuple[int, int] | None = None
            self.internal_walls: list[tuple[int, int]] = []
            self.reachable_cells: list[tuple[int, int]] = []

            mission_space = MissionSpace(mission_func=self._gen_mission)

            if max_steps is None:
                max_steps = 4 * size * size

            super().__init__(
                mission_space=mission_space,
                grid_size=size,
                see_through_walls=True,
                max_steps=max_steps,
                render_mode=render_mode,
                **kwargs,
            )

        @staticmethod
        def _gen_mission() -> str:
            return "reach the green goal square in the maze"

        def _gen_grid(self, width: int, height: int) -> None:
            self.grid = Grid(width, height)
            self.grid.wall_rect(0, 0, width, height)

            ax, ay = self.agent_start_pos
            self.agent_pos = (ax, ay)
            self.agent_dir = self.agent_start_dir

            self._builder._place_internal_walls(self)
            component = self._builder._connected_component(self, self.agent_pos)
            if len(component) < 2:
                raise RuntimeError("agent component too small after wall generation")

            self.reachable_cells = sorted(component)
            if self._builder.cfg.fixed_goal is not None:
                self.goal_pos = tuple(self._builder.cfg.fixed_goal)
                if self.goal_pos not in component:
                    raise RuntimeError("fixed goal not reachable from agent")
            else:
                candidates = [c for c in component if c != tuple(self.agent_pos)]
                gx, gy = self.np_random.choice(candidates)
                self.goal_pos = (int(gx), int(gy))
            self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
            self.mission = self._gen_mission()

    def __init__(self, config: MazePreSampleConfig | None = None):
        self.cfg = config or MazePreSampleConfig()
        if self.cfg.max_steps is None:
            self.cfg.max_steps = 4 * self.cfg.size * self.cfg.size

    # --- graph helpers (on walkable cells: empty or goal) ---

    @staticmethod
    def _is_walkable(env: MiniGridEnv, x: int, y: int) -> bool:
        cell = env.grid.get(x, y)
        if cell is None:
            return True
        return cell.can_overlap()

    def _connected_component(self, env: MiniGridEnv, start: tuple[int, int]) -> set[tuple[int, int]]:
        sx, sy = start
        if not self._is_walkable(env, sx, sy):
            return set()

        seen: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            if (x, y) in seen:
                continue
            if not self._is_walkable(env, x, y):
                continue
            seen.add((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                queue.append((x + dx, y + dy))
        return seen

    def _place_internal_walls(self, env: ProceduralMazeEnv) -> None:
        if self.cfg.fixed_walls is not None:
            env.internal_walls = []
            for x, y in self.cfg.fixed_walls:
                env.grid.set(x, y, Wall())
                env.internal_walls.append((x, y))
            return

        w, h = env.width, env.height
        ax, ay = env.agent_start_pos
        interior = [
            (x, y)
            for x in range(1, w - 1)
            for y in range(1, h - 1)
            if (x, y) != (ax, ay)
        ]
        env.np_random.shuffle(interior)
        env.internal_walls = []
        placed = 0
        for x, y in interior:
            if placed >= env.n_walls:
                break
            env.grid.set(x, y, Wall())
            component = self._connected_component(env, env.agent_pos)
            if tuple(env.agent_pos) in component and len(component) >= 2:
                env.internal_walls.append((x, y))
                placed += 1
            else:
                env.grid.set(x, y, None)

    def make_env(self) -> gym.Env:
        base = self.ProceduralMazeEnv(
            builder=self,
            size=self.cfg.size,
            n_walls=self.cfg.n_walls,
            agent_start_pos=self.cfg.agent_start_pos,
            agent_start_dir=self.cfg.agent_start_dir,
            max_steps=self.cfg.max_steps,
            render_mode=self.cfg.render_mode,
        )
        return RGBImgPartialObsWrapper(base, tile_size=self.cfg.tile_size)

    def build(self) -> dict[str, Any]:
        env = self.make_env()
        obs, _ = env.reset(seed=self.cfg.seed)
        mission = str(obs["mission"])
        u = env.unwrapped
        assert u.goal_pos is not None
        env.close()

        return {
            "env_type": "ProceduralMazeEnv",
            "seed": self.cfg.seed,
            "predicate_space": {
                "grid_size": [u.width, u.height],
                "n_walls": self.cfg.n_walls,
                "actions": ["left", "right", "forward"],
                "action_ids": {k: int(getattr(Actions, k)) for k in ("left", "right", "forward")},
                "observation": {
                    "wrapper": "RGBImgPartialObsWrapper",
                    "tile_size": self.cfg.tile_size,
                    "shape": list(obs["image"].shape),
                    "dtype": str(obs["image"].dtype),
                },
                "score_rule": {
                    "blocked_forward": self.cfg.block_score,
                    "allowed": "graph_shortest_path_distance_to_goal",
                },
            },
            "mission": mission,
            "layout": {
                "size": self.cfg.size,
                "agent_start": list(self.cfg.agent_start_pos),
                "agent_start_dir": self.cfg.agent_start_dir,
                "goal_pos": list(u.goal_pos),
                "walls": [list(w) for w in u.internal_walls],
                "reachable_count": len(u.reachable_cells),
            },
            "target": {
                "goal_pos": list(u.goal_pos),
            },
        }


# --- dataset io ---

CURRICULUM = [
    {"size": 7, "n_walls": 0},
    {"size": 7, "n_walls": 2},
    {"size": 9, "n_walls": 4},
    {"size": 9, "n_walls": 8},
    {"size": 11, "n_walls": 12},
    {"size": 13, "n_walls": 18},
]


def load_dataset() -> dict[str, Any]:
    if DATASET_JSON.exists():
        with DATASET_JSON.open(encoding="utf-8") as f:
            return json.load(f)
    return {"version": 2, "format": "pre_sample", "objects": []}


def save_dataset(data: dict[str, Any]) -> None:
    with DATASET_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def curriculum_for_index(index: int) -> dict[str, int]:
    if index < len(CURRICULUM):
        return CURRICULUM[index]
    last = CURRICULUM[-1]
    return {"size": last["size"], "n_walls": min(last["n_walls"] + (index - len(CURRICULUM) + 1), last["size"] ** 2)}


def append_presample(seed: int | None = None, size: int | None = None, n_walls: int | None = None) -> dict[str, Any]:
    data = load_dataset()
    obj_id = len(data["objects"])
    cur = curriculum_for_index(obj_id)
    cfg = MazePreSampleConfig(
        size=size or cur["size"],
        n_walls=n_walls if n_walls is not None else cur["n_walls"],
        seed=seed if seed is not None else obj_id,
    )
    sample = MazePreSampleBuilder(cfg).build()
    sample["id"] = obj_id
    data["objects"].append(sample)
    save_dataset(data)
    return sample
