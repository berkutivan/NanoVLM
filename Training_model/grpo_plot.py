"""Plot GRPO training curves."""

from __future__ import annotations

from typing import Any


def plot_grpo_history(history: dict[str, Any], *, title: str = "GRPO") -> None:
    import matplotlib.pyplot as plt

    updates = history.get("update", [])
    if not updates:
        print("Empty history")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    if history.get("loss"):
        axes[0, 0].plot(updates, history["loss"], label="train loss")
        axes[0, 0].set_title("GRPO loss")
        axes[0, 0].grid(True, alpha=0.3)

    if history.get("mean_return"):
        axes[0, 1].plot(updates, history["mean_return"], color="green", label="rollout return")
        axes[0, 1].set_title("Rollout mean return")
        axes[0, 1].grid(True, alpha=0.3)

    if history.get("success_rate"):
        axes[1, 0].plot(
            updates,
            [x * 100 for x in history["success_rate"]],
            color="blue",
            label="rollout success %",
        )
        axes[1, 0].set_title("Rollout success rate")
        axes[1, 0].grid(True, alpha=0.3)

    eval_u = updates[-len(history.get("eval_success_rate", [])) :]
    if history.get("eval_success_rate") and eval_u:
        axes[1, 1].plot(
            eval_u,
            [x * 100 for x in history["eval_success_rate"]],
            "o-",
            color="purple",
            label="val success %",
        )
        if history.get("eval_mean_return"):
            ax2 = axes[1, 1].twinx()
            ax2.plot(
                eval_u,
                history["eval_mean_return"],
                "s--",
                color="orange",
                alpha=0.7,
                label="val return",
            )
        axes[1, 1].set_title("Validation (BabyAI manifest)")
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
