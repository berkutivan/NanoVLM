"""Prompt templates for GRPO: direct action vs text + action."""

from __future__ import annotations

import re

from minigrid_sft_dataset import ACTIONS_PROMPT, PROMPT_TEMPLATE

# §2 GRPO: same format as SFT (direct one-word action)
PROMPT_ACTION = PROMPT_TEMPLATE

# §3 GRPO: brief state/plan + action (2–3 sentences, then one action word)
# Rationale: forces explicit spatial reasoning before committing; chain-of-thought
# improves credit assignment vs a single token when the layout is partially observable.
PROMPT_TEXT_ACTION = (
    "Mission: {mission}\n"
    "Describe what you see and your plan in 2-3 short sentences, "
    f"then write one action word: {ACTIONS_PROMPT}.\n"
    "Answer:"
)

_ACTION_RE = re.compile(
    r"\b(left|right|forward|pickup|drop|toggle|done)\b",
    re.IGNORECASE,
)


def build_prompt(mission: str, mode: str) -> str:
    if mode == "text_action":
        return PROMPT_TEXT_ACTION.format(mission=mission)
    return PROMPT_ACTION.format(mission=mission)


def parse_action_from_text(text: str, *, default: str = "forward") -> str:
    """Extract the last valid MiniGrid action token from model output."""
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        t = (text or "").strip().lower().split()
        if t:
            w = t[0].rstrip(".,;:!?")
            if w in {"left", "right", "forward", "pickup", "drop", "toggle", "done"}:
                return w
        return default
    return matches[-1].lower()
