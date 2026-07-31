"""
q_learning.py
=============
The Q-learning brain for the EV3 line-following robot.

This module is pure Python — no EV3 hardware needed.
It can be tested on a PC without the robot connected.

State space  (5 states)  : discretized color sensor reading
Action space (4 actions) : Forward, Reverse, Left Turn, Right Turn
"""

import random
import json
import os


# ---------------------------------------------------------------------------
# State and Action definitions
# ---------------------------------------------------------------------------

STATES = {
    "FAR_LEFT":  0,
    "LEFT":      1,
    "ON_LINE":   2,
    "RIGHT":     3,
    "FAR_RIGHT": 4,
}
STATE_NAMES = {v: k for k, v in STATES.items()}   # reverse lookup

ACTIONS = {
    "FORWARD":     0,
    "REVERSE":     1,
    "LEFT_TURN":   2,
    "RIGHT_TURN":  3,
}
ACTION_NAMES = {v: k for k, v in ACTIONS.items()}

N_STATES  = len(STATES)   # 5
N_ACTIONS = len(ACTIONS)  # 4


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def get_reward(state: int) -> float:
    """
    Map a state index to a scalar reward.

    Tuning guide:
      - Increase the ON_LINE reward to encourage staying centred.
      - Make the FAR_* penalty larger to discourage leaving the track.
      - LEFT / RIGHT give a small positive reward — the robot is still near
        the line and is about to correct.
    """
    reward_table = {
        STATES["ON_LINE"]:   1.0,
        STATES["LEFT"]:      0.3,
        STATES["RIGHT"]:     0.3,
        STATES["FAR_LEFT"]: -1.0,
        STATES["FAR_RIGHT"]:-1.0,
    }
    return reward_table[state]


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    alpha : float
        Learning rate (0–1). How much new information overrides old estimates.
        Start around 0.3; tune on real hardware.
    gamma : float
        Discount factor (0–1). Close to 1 → robot values future rewards.
        Use ~0.9 for line following (long-horizon task).
    epsilon : float
        Initial exploration rate. 1.0 = fully random, 0.0 = fully greedy.
    epsilon_decay : float
        Multiply epsilon by this after each episode (e.g. 0.99).
    epsilon_min : float
        Floor for epsilon so the robot never stops exploring completely.
    """

    def __init__(
        self,
        alpha:         float = 0.3,
        gamma:         float = 0.9,
        epsilon:       float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min:   float = 0.05,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-table: rows = states, columns = actions, initialised to zero
        self.q_table = [[0.0 for _ in range(N_ACTIONS)] for _ in range(N_STATES)]

        # Training history (for logging / demo)
        self.episode_rewards = []
        self.total_episodes_completed = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: int) -> int:
        """
        ε-greedy policy.
        With probability ε → pick a random action (explore).
        Otherwise       → pick the action with the highest Q-value (exploit).
        """
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)   # explore
        row = self.q_table[state]
        return max(range(N_ACTIONS), key=lambda action: row[action])     # exploit

    # ------------------------------------------------------------------
    # Q-table update
    # ------------------------------------------------------------------

    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Apply the Q-Learning update rule:

            Q(s, a) ← Q(s, a) + α [ r + γ · max_a' Q(s', a') − Q(s, a) ]
        """
        best_next = max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------

    def end_episode(self, total_reward: float):
        """Call at the end of every episode to decay ε and record reward."""
        self.episode_rewards.append(total_reward)
        self.total_episodes_completed += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "q_table.json"):
        """Save Q-table and hyperparameters to a JSON file."""
        data = {
            "q_table":       self.q_table.tolist(),
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min":   self.epsilon_min,
            "episode_rewards": self.episode_rewards,
            "total_episodes_completed": self.total_episodes_completed,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("[Q-agent] Saved to {}".format(path))

    def load(self, path: str = "q_table.json"):
        """Load a previously saved Q-table (resume training or deploy)."""
        if not os.path.exists(path):
            print("[Q-agent] No saved table at {}, starting fresh.".format(path))
            return
        with open(path) as f:
            data = json.load(f)
        self.q_table       = data["q_table"]
        self.alpha         = data["alpha"]
        self.gamma         = data["gamma"]
        self.epsilon       = data["epsilon"]
        self.epsilon_decay = data["epsilon_decay"]
        self.epsilon_min   = data["epsilon_min"]
        self.episode_rewards = data.get("episode_rewards", [])
        self.total_episodes_completed = data.get("total_episodes_completed", len(self.episode_rewards))
        print("[Q-agent] Loaded from {}  (ε={:.3f})".format(path, self.epsilon))

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_q_table(self):
        """Pretty-print the Q-table (mirrors the lecture slide format)."""
        header = "{:<12}".format("State") + "".join("{:>12}".format(n) for n in ACTION_NAMES.values())
        print(header)
        print("-" * (12 + 12 * N_ACTIONS))
        for s_idx, s_name in STATE_NAMES.items():
            row = "{:<12}".format(s_name) + "".join("{:>12.3f}".format(self.q_table[s_idx][a]) for a in range(N_ACTIONS))
            print(row)
        print("\nε={:.3f}  Episodes={}".format(self.epsilon, len(self.episode_rewards)))
