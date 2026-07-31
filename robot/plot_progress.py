"""
plot_progress.py
================
Saves all plots as PNG files — no pop-up windows needed.
Works in Git Bash, PowerShell, CMD, and any terminal on Windows.

Usage:
    python plot_progress.py         <- uses q_table.json  (real robot)
    python plot_progress.py sim     <- uses q_table_sim.json (simulator)
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # <-- saves to file, never opens a window
import matplotlib.pyplot as plt

# Pick which Q-table to load
if len(sys.argv) > 1 and sys.argv[1] == "sim":
    QTABLE_PATH = "q_table_sim.json"
    TAG = "sim"
else:
    QTABLE_PATH = "q_table.json"
    TAG = "robot"

STATE_NAMES  = ["Far-left", "Left", "On-line", "Right", "Far-right"]
ACTION_NAMES = ["Forward", "Reverse", "Left turn", "Right turn"]


def load_data(path):
    with open(path) as f:
        return json.load(f)


def plot_rewards(rewards):
    fig, ax = plt.subplots(figsize=(10, 4))
    episodes = list(range(1, len(rewards) + 1))
    ax.plot(episodes, rewards, alpha=0.25, color="#5b8dd9", linewidth=0.8, label="Episode reward")
    window = 20
    if len(rewards) >= window:
        smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(rewards) + 1), smooth,
                color="#5b8dd9", linewidth=2.2, label=f"{window}-ep average")
        ax.fill_between(range(window, len(rewards) + 1), smooth, alpha=0.12, color="#5b8dd9")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Q-Learning training progress — reward per episode")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = f"training_rewards_{TAG}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_qtable(q_table):
    q = np.array(q_table)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(q, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=10)
    plt.colorbar(im, ax=ax, label="Q-value", shrink=0.85)
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, fontsize=11)
    ax.set_yticks(range(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES, fontsize=11)
    ax.set_title("Learned Q-table (boxed = best action per state)")
    for s in range(len(STATE_NAMES)):
        for a in range(len(ACTION_NAMES)):
            ax.text(a, s, f"{q[s, a]:.2f}", ha="center", va="center", fontsize=10)
    for s in range(len(STATE_NAMES)):
        best_a = int(np.argmax(q[s]))
        ax.add_patch(plt.Rectangle(
            (best_a - 0.48, s - 0.48), 0.96, 0.96,
            fill=False, edgecolor="black", linewidth=2.5
        ))
    fig.tight_layout()
    out = f"q_table_heatmap_{TAG}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_epsilon(rewards):
    n = len(rewards)
    epsilon_vals = [min(1.0, max(0.05, 1.0 * (0.995 ** i))) for i in range(n)]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, n + 1), epsilon_vals, color="#e07b39", linewidth=2)
    ax.fill_between(range(1, n + 1), epsilon_vals, alpha=0.15, color="#e07b39")
    ax.axhline(0.05, color="#888", linewidth=1, linestyle="--", label="min epsilon (0.05)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration rate decay")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = f"epsilon_decay_{TAG}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    print(f"Loading: {QTABLE_PATH}")
    data = load_data(QTABLE_PATH)
    rewards = data.get("episode_rewards", [])
    plot_rewards(rewards)
    plot_qtable(data["q_table"])
    plot_epsilon(rewards)
    print("\nAll done! Open the PNG files in your folder to see the plots.")
