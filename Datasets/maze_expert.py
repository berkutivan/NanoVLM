"""
BFS expert on (x, y, direction) and rollout for SFT samples.

The planning graph has one vertex per (cell, facing): left/right only change facing;
forward moves to an adjacent cell if there is no wall. This matches MiniGrid physics.

Uses layout from pre_sample objects; RGB frames from MazePreSampleBuilder env.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from minigrid.core.actions import Actions

from maze_presample import MazePreSampleBuilder, MazePreSampleConfig

# MiniGrid agent_dir: 0=right, 1=down, 2=left, 3=up
_DIR_VEC = ((1, 0), (0, 1), (-1, 0), (0, -1))
_ACTION_NAMES = ("left", "right", "forward")
_ACTION_TO_ID = {
    "left": int(Actions.left),
    "right": int(Actions.right),
    "forward": int(Actions.forward),
}


@dataclass(frozen=True)
class ExpertStep:
    image: np.ndarray
    mission: str
    action: str
    action_id: int
    allowed_actions: tuple[str, ...]
    step_index: int


def _walkable_cells(layout: dict[str, Any]) -> set[tuple[int, int]]:
    size = layout["size"]
    walls = {tuple(w) for w in layout["walls"]}
    cells: set[tuple[int, int]] = set()
    for x in range(1, size - 1):
        for y in range(1, size - 1):
            if (x, y) not in walls:
                cells.add((x, y))
    return cells


def _turn_left(d: int) -> int:
    return (d - 1) % 4


def _turn_right(d: int) -> int:
    return (d + 1) % 4


def _forward_cell(x: int, y: int, d: int) -> tuple[int, int]:
    dx, dy = _DIR_VEC[d]
    return x + dx, y + dy


def plan_expert_actions(obj: dict[str, Any]) -> list[str]:
    """Shortest action sequence from agent start to goal on the maze grid."""
    layout = obj["layout"]
    walkable = _walkable_cells(layout)
    goal = tuple(layout["goal_pos"])
    start = (
        layout["agent_start"][0],
        layout["agent_start"][1],
        layout["agent_start_dir"],
    )

    if start[:2] not in walkable or goal not in walkable:
        raise ValueError(f"object id={obj.get('id')}: invalid start or goal")

    queue: deque[tuple[tuple[int, int, int], list[str]]] = deque([(start, [])])
    visited: set[tuple[int, int, int]] = {start}

    while queue:
        (x, y, d), path = queue.popleft()
        if (x, y) == goal:
            return path

        for action in _ACTION_NAMES:
            if action == "left":
                nd = _turn_left(d)
                nx, ny = x, y
            elif action == "right":
                nd = _turn_right(d)
                nx, ny = x, y
            else:
                nd = d
                nx, ny = _forward_cell(x, y, d)
                if (nx, ny) not in walkable:
                    continue

            state = (nx, ny, nd)
            if state in visited:
                continue
            visited.add(state)
            queue.append((state, path + [action]))

    raise RuntimeError(f"object id={obj.get('id')}: no path from start to goal")


def _compute_state_distances(
    layout: dict[str, Any],
) -> dict[tuple[int, int, int], int]:
    """
    Compute shortest distance (#actions) from every reachable (x,y,dir) to the goal.

    We do a reverse BFS from all goal states (goal_x, goal_y, dir in 0..3).
    """
    walkable = _walkable_cells(layout)
    goal = tuple(layout["goal_pos"])
    if goal not in walkable:
        return {}

    dist: dict[tuple[int, int, int], int] = {}
    q: deque[tuple[int, int, int]] = deque()
    for d in range(4):
        s = (goal[0], goal[1], d)
        dist[s] = 0
        q.append(s)

    while q:
        x, y, d = q.popleft()
        cur = dist[(x, y, d)]

        # Predecessors via turns:
        # If we are at (x,y,d) now, we could have come from:
        # - action 'left' from (x,y, turn_right(d))
        # - action 'right' from (x,y, turn_left(d))
        pred_left = (x, y, _turn_right(d))
        pred_right = (x, y, _turn_left(d))
        for ps in (pred_left, pred_right):
            if ps not in dist:
                dist[ps] = cur + 1
                q.append(ps)

        # Predecessor via forward:
        # If we are at (x,y,d) now, we could have come from (x-dx, y-dy, d) with action 'forward'
        dx, dy = _DIR_VEC[d]
        px, py = x - dx, y - dy
        if (px, py) in walkable:
            ps = (px, py, d)
            if ps not in dist:
                dist[ps] = cur + 1
                q.append(ps)

    return dist


def _allowed_optimal_actions(
    state: tuple[int, int, int],
    *,
    walkable: set[tuple[int, int]],
    dist: dict[tuple[int, int, int], int],
) -> tuple[str, ...]:
    """
    Return all actions that keep the path length optimal (i.e. reduce dist by 1).
    """
    if state not in dist:
        return tuple()
    x, y, d = state
    cur = dist[state]
    allowed: list[str] = []

    # left / right always valid
    for action in ("left", "right"):
        if action == "left":
            ns = (x, y, _turn_left(d))
        else:
            ns = (x, y, _turn_right(d))
        if dist.get(ns, 10**9) == cur - 1:
            allowed.append(action)

    # forward only if walkable
    nx, ny = _forward_cell(x, y, d)
    if (nx, ny) in walkable:
        ns = (nx, ny, d)
        if dist.get(ns, 10**9) == cur - 1:
            allowed.append("forward")

    # Fallback: if something is off, at least allow the expert path to remain usable
    return tuple(allowed)


def make_env_from_object(obj: dict[str, Any]):
    layout = obj["layout"]
    pred = obj["predicate_space"]
    cfg = MazePreSampleConfig(
        size=layout["size"],
        n_walls=len(layout["walls"]),
        seed=obj["seed"],
        agent_start_pos=tuple(layout["agent_start"]),
        agent_start_dir=layout["agent_start_dir"],
        tile_size=pred["observation"]["tile_size"],
        fixed_walls=[tuple(w) for w in layout["walls"]],
        fixed_goal=tuple(layout["goal_pos"]),
        render_mode=None,
    )
    return MazePreSampleBuilder(cfg).make_env()


def rollout_expert_trajectory(obj: dict[str, Any]) -> list[ExpertStep]:
    """RGB observation before each expert action until the goal is reached."""
    actions = plan_expert_actions(obj)
    layout = obj["layout"]
    walkable = _walkable_cells(layout)
    dist = _compute_state_distances(layout)
    env = make_env_from_object(obj)
    obs, _ = env.reset(seed=obj["seed"])
    mission = str(obs["mission"])
    steps: list[ExpertStep] = []

    for step_index, action in enumerate(actions):
        # Compute allowed optimal actions for the CURRENT state (before executing expert action)
        # MiniGrid keeps agent state on env.unwrapped
        ax, ay = int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1])
        ad = int(env.unwrapped.agent_dir)
        allowed = _allowed_optimal_actions((ax, ay, ad), walkable=walkable, dist=dist)
        if not allowed:
            allowed = (action,)

        steps.append(
            ExpertStep(
                image=np.asarray(obs["image"], dtype=np.uint8),
                mission=mission,
                action=action,
                action_id=_ACTION_TO_ID[action],
                allowed_actions=allowed,
                step_index=step_index,
            )
        )
        obs, _, terminated, truncated, _ = env.step(_ACTION_TO_ID[action])
        if terminated or truncated:
            break

    env.close()
    if not steps:
        raise RuntimeError(f"object id={obj.get('id')}: empty trajectory")
    return steps
