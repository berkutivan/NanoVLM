"""
Detect mission targets (key / finish) in the agent's partial observation.

Used when curating BabyAI SFT data: keep steps where a target is visible,
or where a target becomes visible after one explore action (left/right/forward).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.world_object import Ball, Goal, Key

EXPLORE_ACTION_IDS: tuple[int, ...] = (
    int(Actions.left),
    int(Actions.right),
    int(Actions.forward),
)
EXPLORE_HINT = "You should explore space"


def visible_target_types(env) -> frozenset[str]:
    """
    Return target kinds currently in the agent FOV.

    - ``key`` — Key on the grid or carried by the agent
    - ``finish`` — Goal tile or Ball (BabyAI goto objective)
    """
    u = env.unwrapped
    found: set[str] = set()

    if u.carrying is not None and isinstance(u.carrying, Key):
        found.add("key")

    grid, vis_mask = u.gen_obs_grid()
    for x in range(grid.width):
        for y in range(grid.height):
            if not vis_mask[x, y]:
                continue
            cell = grid.get(x, y)
            if cell is None:
                continue
            if isinstance(cell, Key):
                found.add("key")
            elif isinstance(cell, (Goal, Ball)):
                found.add("finish")
    return frozenset(found)


def has_visible_target(env) -> bool:
    types = visible_target_types(env)
    return "key" in types or "finish" in types


def targets_visible_after_explore(
    make_env_fn,
    *,
    seed: int | None,
    options: dict[str, Any] | None,
    prefix_action_ids: Iterable[int],
) -> bool:
    """
    True if key or finish appears in FOV after one left/right/forward from the current state.
    """
    prefix = [int(a) for a in prefix_action_ids]
    for probe_id in EXPLORE_ACTION_IDS:
        env = make_env_fn()
        try:
            env.reset(seed=seed, options=options)
            for aid in prefix:
                _, _, terminated, truncated, _ = env.step(aid)
                if terminated or truncated:
                    break
            else:
                env.step(probe_id)
                if has_visible_target(env):
                    return True
        finally:
            env.close()
    return False


def classify_step_mission(
    base_mission: str,
    env,
    *,
    make_env_fn,
    seed: int | None,
    options: dict[str, Any] | None,
    prefix_action_ids: Iterable[int],
) -> tuple[str | None, str]:
    """
    Decide whether to keep a step and which mission text to use.

    Returns ``(mission, visibility_tag)`` or ``(None, "")`` if the step should be dropped.
    ``visibility_tag`` is ``direct`` or ``explore``.
    """
    if has_visible_target(env):
        return base_mission, "direct"
    if targets_visible_after_explore(
        make_env_fn,
        seed=seed,
        options=options,
        prefix_action_ids=prefix_action_ids,
    ):
        mission = base_mission.rstrip()
        if not mission.endswith("."):
            mission = f"{mission}."
        return f"{mission} {EXPLORE_HINT}", "explore"
    return None, ""


def mission_with_explore_hint(base_mission: str) -> str:
    mission = base_mission.rstrip()
    if not mission.endswith("."):
        mission = f"{mission}."
    return f"{mission} {EXPLORE_HINT}"
