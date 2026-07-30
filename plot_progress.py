"""
plot_progress.py
================
Visualise training progress from a saved q_table.json.
Produces two plots useful for your report / demo:

  1. Episode reward over time (shows the robot is learning)
  2. Q-table heatmap (shows what the robot has learned)

Run with:
    python plot_progress.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

QTABLE_PATH = "q_table.json"

STATE_NAMES  = ["Far-left", "Left", "On-line", "Right", "Far-right"]
ACTION_NAMES = ["Forward",  "Reverse", "Left turn", "Right turn"]


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_rewards(rewards: list[float]):
    """Smoothed episode reward curve."""
    fig, ax = plt.subplots(figsize=(9, 4))

    episodes = list(range(1, len(rewards) + 1))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.8, label="Raw reward")

    # Rolling average (window = 10 episodes)
    if len(rewards) >= 10:
        smooth = np.convolve(rewards, np.ones(10)/10, mode="valid")
        ax.plot(range(10, len(rewards) + 1), smooth, color="steelblue", linewidth=2, label="10-ep average")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Training progress — episode reward over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_rewards.png", dpi=150)
    print("Saved: training_rewards.png")
    plt.show()


def plot_qtable(q_table: list):
    """Q-table heatmap — shows which action the robot prefers in each state."""
    q = np.array(q_table)
    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(q, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Q-value")

    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=20, ha="right")
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_title("Learned Q-table (green = preferred action)")

    # Annotate cells with values
    for s in range(len(STATE_NAMES)):
        for a in range(len(ACTION_NAMES)):
            ax.text(a, s, f"{q[s, a]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")

    # Circle the best action per state
    for s in range(len(STATE_NAMES)):
        best_a = int(np.argmax(q[s]))
        ax.add_patch(plt.Rectangle(
            (best_a - 0.48, s - 0.48), 0.96, 0.96,
            fill=False, edgecolor="black", linewidth=2
        ))

    fig.tight_layout()
    fig.savefig("q_table_heatmap.png", dpi=150)
    print("Saved: q_table_heatmap.png")
    plt.show()


if __name__ == "__main__":
    data = load_data(QTABLE_PATH)
    plot_rewards(data.get("episode_rewards", []))
    plot_qtable(data["q_table"])
