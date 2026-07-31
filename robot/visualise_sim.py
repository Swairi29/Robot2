"""
visualise_sim.py
================
Watch the trained robot navigating the simulated track in your terminal.
Uses ANSI colors — works in any modern terminal (Linux, Mac, Windows Terminal).

Run AFTER training:
    python visualise_sim.py

Controls
--------
  Press Ctrl+C to stop.
  Runs 5 demo episodes by default (change DEMO_EPISODES below).
"""

import os
import time
import sys

from q_learning import QLearningAgent, STATES, ACTION_NAMES, get_reward
from simulator  import LineFollowingSimulator, STATE_NAMES

QTABLE_PATH    = "q_table_sim.json"
DEMO_EPISODES  = 5
MAX_STEPS      = 300
STEP_DELAY     = 0.07    # seconds between steps (lower = faster)

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

def clr(text, code): return f"\033[{code}m{text}\033[0m"

RED     = lambda t: clr(t, "31")
GREEN   = lambda t: clr(t, "32")
YELLOW  = lambda t: clr(t, "33")
CYAN    = lambda t: clr(t, "36")
BOLD    = lambda t: clr(t, "1")
DIM     = lambda t: clr(t, "2")

# ---------------------------------------------------------------------------
# Track renderer
# ---------------------------------------------------------------------------

TRACK_WIDTH = 60   # characters wide

def render_frame(offset: float, track_pos: float, state: int,
                 action: int, reward: float, step: int,
                 episode: int, total_reward: float, epsilon: float):
    """
    Print a single frame showing:
    - Top-down view of the track (robot as R, line as |)
    - Current state, action, reward
    - Q-table values for current state
    """
    os.system("clear" if os.name != "nt" else "cls")

    # --- Header ---
    print(BOLD(f"  EV3 Line-Following Robot — Simulation Demo"))
    print(DIM(f"  Episode {episode}/{DEMO_EPISODES}  |  Step {step}  |  ε={epsilon:.3f}"))
    print()

    # --- Track view (top-down) ---
    # offset -1.0 (far left) to +1.0 (far right)
    # Map to character position 0..TRACK_WIDTH
    centre_char = TRACK_WIDTH // 2
    robot_char  = int((offset + 1.0) / 2.0 * TRACK_WIDTH)
    robot_char  = max(1, min(TRACK_WIDTH - 2, robot_char))

    row = [" "] * (TRACK_WIDTH + 4)

    # Walls
    row[0]              = "|"
    row[TRACK_WIDTH + 3] = "|"

    # Line (two chars wide at centre)
    row[centre_char + 1] = DIM("▌")
    row[centre_char + 2] = DIM("▐")

    # Robot
    state_colour = {
        STATES["ON_LINE"]:   GREEN,
        STATES["LEFT"]:      YELLOW,
        STATES["RIGHT"]:     YELLOW,
        STATES["FAR_LEFT"]:  RED,
        STATES["FAR_RIGHT"]: RED,
    }[state]
    row[robot_char + 1] = state_colour("R")

    print("  " + "".join(str(c) for c in row))

    # Track position bar
    tp = int(track_pos * TRACK_WIDTH)
    bar = "·" * tp + "▶" + "·" * (TRACK_WIDTH - tp)
    print(f"  [" + DIM(bar) + f"]  {track_pos*100:.0f}% around loop")
    print()

    # --- State / Action info ---
    state_str  = state_colour(f"{STATE_NAMES[state]:<12}")
    action_str = CYAN(ACTION_NAMES[action])
    reward_str = GREEN(f"+{reward:.1f}") if reward > 0 else RED(f"{reward:.1f}")

    print(f"  State  : {state_str}   Action : {action_str:<14}   Reward : {reward_str}")
    print(f"  Total reward this episode : {BOLD(f'{total_reward:.1f}')}")
    print()

    # --- Offset meter ---
    meter_width = 40
    meter_pos   = int((offset + 1.0) / 2.0 * meter_width)
    meter_pos   = max(0, min(meter_width - 1, meter_pos))
    centre_m    = meter_width // 2
    meter       = [DIM("─")] * meter_width
    meter[centre_m] = DIM("│")
    meter[meter_pos] = state_colour("◆")
    print("  FAR-L " + "".join(str(c) for c in meter) + " FAR-R")
    print()


# ---------------------------------------------------------------------------
# Main demo loop
# ---------------------------------------------------------------------------

def demo():
    agent = QLearningAgent()
    agent.load(QTABLE_PATH)
    agent.epsilon = 0.05   # near-greedy for demo (tiny exploration for variety)

    sim = LineFollowingSimulator(noise_level=0.12, obstacle_prob=0.0)

    print(f"\nLoaded Q-table from {QTABLE_PATH}")
    print("Starting demo in 2 seconds... (Ctrl+C to stop)\n")
    time.sleep(2)

    for episode in range(1, DEMO_EPISODES + 1):
        sim.reset()
        state        = sim.read_state()
        total_reward = 0.0
        last_action  = 0
        last_reward  = 0.0

        for step in range(1, MAX_STEPS + 1):
            action = agent.choose_action(state)
            sim.execute_action(action)
            next_state   = sim.read_state()
            reward       = get_reward(next_state)
            total_reward += reward

            info = sim.get_info()
            render_frame(
                offset       = info["offset"],
                track_pos    = info["track_pos"],
                state        = next_state,
                action       = action,
                reward       = reward,
                step         = step,
                episode      = episode,
                total_reward = total_reward,
                epsilon      = agent.epsilon,
            )

            # Handle line lost
            if next_state in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
                sim.find_path()
                next_state = sim.read_state()

            state = next_state
            time.sleep(STEP_DELAY)

        print(f"\n  Episode {episode} complete — total reward: {total_reward:.1f}")
        time.sleep(1.5)

    print("\nDemo finished.")


if __name__ == "__main__":
    try:
        demo()
    except KeyboardInterrupt:
        print("\nStopped.")