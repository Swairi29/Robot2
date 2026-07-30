"""
train.py  — UPDATED FOR RECTANGULAR TRACK WITH CORNERS
=======================================================
Changes from v1:
  1. Corner detection added — robot handles 90° corners rule-based
  2. T-junction detection added — handled rule-based (go straight)
  3. Inverted sensor already handled in ev3_interface.py
  4. Reduced episode steps (rectangular track = shorter episodes)
"""

import signal
import sys
from q_learning   import QLearningAgent, STATES, get_reward
from ev3_interface import EV3Interface

NUM_EPISODES = 200
MAX_STEPS    = 400
LOG_EVERY    = 10
SAVE_EVERY   = 20
QTABLE_PATH  = "q_table_robot.json"

ALPHA         = 0.3
GAMMA         = 0.9
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN   = 0.05

running = True
def shutdown(sig, frame):
    global running
    print("\n[Train] Saving and stopping...")
    running = False
signal.signal(signal.SIGINT, shutdown)

def train():
    global running
    agent = QLearningAgent(alpha=ALPHA, gamma=GAMMA,
                           epsilon=EPSILON_START,
                           epsilon_decay=EPSILON_DECAY,
                           epsilon_min=EPSILON_MIN)
    agent.load(QTABLE_PATH)
    robot = EV3Interface()

    print(f"\n[Train] {NUM_EPISODES} episodes on RECTANGULAR track")
    print("[Train] Track: LIGHT TAPE on DARK MAT")
    print("[Train] Corners and T-junctions handled by rules (not RL)\n")

    for episode in range(1, NUM_EPISODES + 1):
        if not running:
            break

        input(f"  Ep {episode}/{NUM_EPISODES}  ε={agent.epsilon:.3f} → Place robot on tape, press Enter...")

        state        = robot.read_state()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            if not running:
                break

            # ── Rule-based checks BEFORE RL action ──────────────────────

            # 1. Obstacle check
            if robot.obstacle_detected():
                robot.avoid_obstacle()
                state = robot.read_state()
                continue

            # 2. T-junction check (top-left of track)
            if robot.is_t_junction():
                robot.handle_t_junction(go="straight")
                state = robot.read_state()
                continue

            # ── RL action ────────────────────────────────────────────────
            action = agent.choose_action(state)
            robot.execute_action(action)

            next_state   = robot.read_state()
            reward       = get_reward(next_state)
            total_reward += reward

            # 3. Lost tape — rule-based re-find
            if next_state in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
                found = robot.find_path()
                if found:
                    next_state = robot.read_state()
                else:
                    print(f"  [Ep {episode}] Tape lost at step {step}, ending episode.")
                    break

            agent.update(state, action, reward, next_state)
            state = next_state

        robot.stop()
        agent.end_episode(total_reward)
        print(f"  Ep {episode:>4}  reward={total_reward:>8.2f}  ε={agent.epsilon:.3f}")

        if episode % LOG_EVERY == 0:
            print(); agent.print_q_table(); print()
        if episode % SAVE_EVERY == 0:
            agent.save(QTABLE_PATH)

    agent.save(QTABLE_PATH)
    robot.cleanup()
    print("\n[Train] Done. Run python deploy.py to test the trained robot.")

if __name__ == "__main__":
    train()