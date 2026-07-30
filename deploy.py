"""
deploy.py
=========
Run the fully-trained robot in exploitation-only mode (ε=0).
No learning happens here — it just reads the Q-table and picks
the best known action at every step.

Use this for:
  - Demo / marking day
  - Clockwise and anticlockwise runs (just flip the robot)
  - Recording smooth path-following footage for the report

Run with:
    python deploy.py
"""

import signal
import sys

from q_learning    import QLearningAgent, STATES, ACTION_NAMES
from ev3_interface import EV3Interface

QTABLE_PATH = "q_table_react.json"

running = True
robot   = None

def shutdown(sig, frame):
    global running
    print("\n[Deploy] Stopping...")
    running = False

signal.signal(signal.SIGINT, shutdown)


def deploy():
    global robot, running

    agent = QLearningAgent()
    agent.load(QTABLE_PATH)

    # Force pure exploitation — no random actions
    agent.epsilon = 0.0

    robot = EV3Interface()

    print("\n[Deploy] Loaded trained Q-table. Running in exploitation mode (ε=0).")
    agent.print_q_table()
    print("\nPress Enter to start, Ctrl+C to stop.\n")
    input()

    state = robot.read_state()

    step = 0
    while running:
        step += 1

        # Obstacle avoidance (rule-based)
        if robot.obstacle_detected():
            robot.avoid_obstacle()
            state = robot.read_state()
            continue

        # Best known action (no exploration)
        action = agent.choose_action(state)
        print(f"  Step {step:>5}  state={state}  action={ACTION_NAMES[action]}")

        robot.execute_action(action)
        next_state = robot.read_state()

        # Rule-based re-find if line lost
        if next_state in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
            robot.find_path()
            next_state = robot.read_state()

        state = next_state

    robot.cleanup()
    print("[Deploy] Stopped.")


if __name__ == "__main__":
    deploy()
