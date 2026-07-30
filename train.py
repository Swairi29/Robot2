"""
train.py
========
Main training script.

Run this on your PC with the EV3 connected via USB:
    python train.py

What it does each episode
--------------------------
1. Reset the robot to the start position (you place it manually).
2. Run for MAX_STEPS steps:
   a. Read sensor → discrete state
   b. Check for obstacles (rule-based avoidance if needed)
   c. Choose action (ε-greedy)
   d. Execute action on robot
   e. Read new state, compute reward
   f. Update Q-table
3. Log Q-table and episode reward.
4. Decay ε (less exploration as training progresses).
5. Repeat for NUM_EPISODES.

After training, the best Q-table is saved and can be loaded
by deploy.py for a pure-exploitation run (no exploration).
"""

import time
import signal
import sys

from q_learning   import QLearningAgent, STATES, ACTION_NAMES, get_reward
from ev3_interface import EV3Interface

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

NUM_EPISODES = 200       # total training episodes
MAX_STEPS    = 300       # max steps per episode (prevents infinite loops)
LOG_EVERY    = 10        # print Q-table every N episodes
SAVE_EVERY   = 20        # save Q-table to disk every N episodes
QTABLE_PATH  = "q_table.json"

# Hyperparameters (see lecture slides for tuning guidance)
ALPHA         = 0.3      # learning rate
GAMMA         = 0.9      # discount factor
EPSILON_START = 1.0      # start fully exploratory
EPSILON_DECAY = 0.99     # decay per episode
EPSILON_MIN   = 0.05     # always keep a little exploration

# ---------------------------------------------------------------------------
# Graceful shutdown on Ctrl+C
# ---------------------------------------------------------------------------

robot  = None
agent  = None
running = True

def shutdown(sig, frame):
    global running
    print("\n[Train] Ctrl+C received — saving and shutting down...")
    running = False

signal.signal(signal.SIGINT, shutdown)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    global robot, agent, running

    # Initialise agent and hardware
    agent = QLearningAgent(
        alpha         = ALPHA,
        gamma         = GAMMA,
        epsilon       = EPSILON_START,
        epsilon_decay = EPSILON_DECAY,
        epsilon_min   = EPSILON_MIN,
    )
    agent.load(QTABLE_PATH)   # resume if a previous session exists

    robot = EV3Interface()

    print(f"\n[Train] Starting {NUM_EPISODES} episodes  (MAX_STEPS={MAX_STEPS})")
    print("[Train] Place robot on the line and press Enter to begin each episode.\n")

    for episode in range(1, NUM_EPISODES + 1):
        if not running:
            break

        # --- Wait for manual reset ---
        input(f"  Episode {episode}/{NUM_EPISODES}  ε={agent.epsilon:.3f} → Press Enter when robot is placed...")

        total_reward = 0.0
        state        = robot.read_state()

        for step in range(MAX_STEPS):
            if not running:
                break

            # --- Obstacle check (rule-based, not RL) ---
            if robot.obstacle_detected():
                robot.avoid_obstacle()
                state = robot.read_state()
                continue

            # --- ε-greedy action selection ---
            action = agent.choose_action(state)

            # --- Execute action on robot ---
            robot.execute_action(action)

            # --- Observe outcome ---
            next_state = robot.read_state()
            reward     = get_reward(next_state)
            total_reward += reward

            # --- If line completely lost, use rule-based re-find ---
            if next_state in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
                found = robot.find_path()
                if found:
                    next_state = robot.read_state()
                else:
                    # Line not found — end episode early
                    print(f"  [Episode {episode}] Line lost at step {step}, ending episode.")
                    break

            # --- Q-table update ---
            agent.update(state, action, reward, next_state)
            state = next_state

        # --- End of episode ---
        robot.stop()
        agent.end_episode(total_reward)

        print(f"  Episode {episode:>4}  reward={total_reward:>8.2f}  ε={agent.epsilon:.3f}")

        # Log Q-table periodically
        if episode % LOG_EVERY == 0:
            print()
            agent.print_q_table()
            print()

        # Save periodically
        if episode % SAVE_EVERY == 0:
            agent.save(QTABLE_PATH)

    # Final save
    agent.save(QTABLE_PATH)
    robot.cleanup()
    print("\n[Train] Done. Q-table saved to", QTABLE_PATH)
    print("Run  python deploy.py  to run the trained robot.")


if __name__ == "__main__":
    train()
