"""
train_sim.py
============
Train the Q-agent using the simulator — no EV3 hardware needed.

This runs thousands of episodes in seconds.
When you get access to a robot, switch to train.py (same agent, real hardware).

Run with:
    python train_sim.py
"""

import json
import os
import signal

from q_learning import QLearningAgent, STATES, get_reward
from simulator  import LineFollowingSimulator, ACTION_NAMES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_EPISODES   = 500     # sim is fast — run more episodes than on hardware
MAX_STEPS      = 400     # steps per episode
LOG_EVERY      = 50      # print Q-table every N episodes
SAVE_EVERY     = 100     # save Q-table to disk every N episodes
QTABLE_PATH    = "q_table_rect.json"

# Hyperparameters
ALPHA         = 0.3
GAMMA         = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN   = 0.05

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

running = True
agent   = None
sim     = None

def shutdown(sig, frame):
    global running
    print("\n[Train] Interrupt — saving...")
    running = False

signal.signal(signal.SIGINT, shutdown)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    global running, agent, sim

    agent = QLearningAgent(
        alpha=ALPHA, gamma=GAMMA,
        epsilon=EPSILON_START, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN,
    )
    agent.load(QTABLE_PATH)

    sim = LineFollowingSimulator(noise_level=0.15, obstacle_prob=0.02)

    print(f"[Sim] Training for {NUM_EPISODES} episodes...")
    print(f"[Sim] Hyperparams: α={ALPHA}  γ={GAMMA}  ε₀={EPSILON_START}  decay={EPSILON_DECAY}\n")

    for episode in range(1, NUM_EPISODES + 1):
        if not running:
            break

        sim.reset()
        state        = sim.read_state()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            # Obstacle check (rule-based)
            if sim.obstacle_detected():
                sim.avoid_obstacle()
                state = sim.read_state()
                continue

            # Choose + execute action
            action = agent.choose_action(state)
            sim.execute_action(action)

            next_state = sim.read_state()
            reward     = get_reward(next_state)
            total_reward += reward

            # Re-find path if line lost
            if next_state in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
                sim.find_path()
                next_state = sim.read_state()

            # Q-update
            agent.update(state, action, reward, next_state)
            state = next_state

        agent.end_episode(total_reward)

        if episode % LOG_EVERY == 0:
            print(f"Episode {episode:>4}/{NUM_EPISODES}  "
                  f"reward={total_reward:>8.1f}  ε={agent.epsilon:.3f}")

        if episode % SAVE_EVERY == 0:
            agent.save(QTABLE_PATH)

    # Final report
    agent.save(QTABLE_PATH)
    print("\n=== Final Q-table ===")
    agent.print_q_table()

    rewards = agent.episode_rewards
    n = len(rewards)
    if n >= 10:
        early  = sum(rewards[:n//5])     / (n//5)
        late   = sum(rewards[-n//5:])    / (n//5)
        print(f"\nAvg reward first {n//5} episodes : {early:.1f}")
        print(f"Avg reward last  {n//5} episodes : {late:.1f}")
        pct = (late - early) / max(abs(early), 1) * 100
        print(f"Improvement: {pct:+.1f}%")

    print(f"\nSaved to {QTABLE_PATH}")
    print("Run  python plot_progress.py sim  to see learning curves.")
    print("Run  python visualise_sim.py       to watch the trained robot.")


if __name__ == "__main__":
    train()