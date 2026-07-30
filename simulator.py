"""
simulator.py
============
A physics-inspired simulator for the EV3 line-following robot.

Replaces ev3_interface.py entirely — same interface, no hardware needed.
The Q-agent cannot tell the difference; it calls read_state() and
execute_action() just like it would on the real robot.

How the simulation works
------------------------
The robot has a position along a line track (0.0 to 1.0, wraps around).
It also has a lateral offset from the centre of the line (-1.0 to +1.0).
  offset = 0.0  → perfectly on the line
  offset = +1.0 → far right of the line
  offset = -1.0 → far left of the line

Each action nudges the offset and advances the track position.
Noise is added to simulate real-world sensor and motor imperfection.

The discrete state returned by read_state() matches exactly what
the real color sensor would return after discretization.
"""

import random
import time


# ---------------------------------------------------------------------------
# State / Action constants (mirrors q_learning.py)
# ---------------------------------------------------------------------------

STATES = {
    "FAR_LEFT":  0,
    "LEFT":      1,
    "ON_LINE":   2,
    "RIGHT":     3,
    "FAR_RIGHT": 4,
}
STATE_NAMES  = {v: k for k, v in STATES.items()}

ACTIONS = {
    "FORWARD":    0,
    "REVERSE":    1,
    "LEFT_TURN":  2,
    "RIGHT_TURN": 3,
}
ACTION_NAMES = {v: k for k, v in ACTIONS.items()}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class LineFollowingSimulator:
    """
    Simulates the EV3 robot on a looped line track.

    Parameters
    ----------
    noise_level : float
        0.0 = perfect robot, 1.0 = very noisy. Start at 0.15.
    obstacle_prob : float
        Probability of an obstacle appearing each step (0.0 = never).
    action_duration : float
        Simulated seconds per action (for pacing; doesn't affect training).
    """

    def __init__(
        self,
        noise_level:      float = 0.15,
        obstacle_prob:    float = 0.02,
        action_duration:  float = 0.0,   # set to 0 for fast training
    ):
        self.noise_level     = noise_level
        self.obstacle_prob   = obstacle_prob
        self.action_duration = action_duration

        # Robot state
        self.offset    = 0.0    # lateral position: -1 (far left) to +1 (far right)
        self.track_pos = 0.0    # progress around the loop (0.0–1.0)
        self._obstacle = False

        # Statistics
        self.steps_taken      = 0
        self.obstacle_count   = 0
        self.line_lost_count  = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _noise(self) -> float:
        """Gaussian noise scaled by noise_level."""
        return random.gauss(0, self.noise_level)

    def _offset_to_state(self, offset: float) -> int:
        """
        Convert continuous lateral offset to discrete state.
        Mirrors what the color sensor discretization would produce.
        """
        if   offset < -0.6:  return STATES["FAR_LEFT"]
        elif offset < -0.2:  return STATES["LEFT"]
        elif offset <  0.2:  return STATES["ON_LINE"]
        elif offset <  0.6:  return STATES["RIGHT"]
        else:                return STATES["FAR_RIGHT"]

    # ------------------------------------------------------------------
    # Public interface (same API as EV3Interface)
    # ------------------------------------------------------------------

    def reset(self):
        """Reset robot to a random starting position on the line."""
        self.offset    = random.uniform(-0.3, 0.3)   # start near the line
        self.track_pos = random.uniform(0.0, 1.0)
        self._obstacle = False
        self.steps_taken = 0

    def read_state(self) -> int:
        """Return the current discrete state (with sensor noise)."""
        noisy_offset = self._clamp(self.offset + self._noise() * 0.3, -1.0, 1.0)
        return self._offset_to_state(noisy_offset)

    def obstacle_detected(self) -> bool:
        """Randomly inject obstacles based on obstacle_prob."""
        if random.random() < self.obstacle_prob:
            self._obstacle = True
            self.obstacle_count += 1
        return self._obstacle

    def execute_action(self, action: int):
        """
        Apply action physics to the robot's lateral offset.

        Each action has:
          - a primary effect (move toward line / turn)
          - motor noise (imperfect execution)
          - a small random drift (simulates uneven floor, motor imbalance)
        """
        noise = self._noise()
        drift = random.gauss(0, 0.05)   # slow random drift

        if action == ACTIONS["FORWARD"]:
            # Move forward: advance track position, small drift
            self.track_pos = (self.track_pos + 0.05) % 1.0
            self.offset    = self._clamp(self.offset + drift + noise * 0.1, -1.0, 1.0)

        elif action == ACTIONS["REVERSE"]:
            # Move backward: retreat, small drift
            self.track_pos = (self.track_pos - 0.03) % 1.0
            self.offset    = self._clamp(self.offset - drift + noise * 0.1, -1.0, 1.0)

        elif action == ACTIONS["LEFT_TURN"]:
            # Turn left: offset decreases (moves toward left edge → corrects if right of line)
            correction = 0.4 + noise * 0.1
            self.offset = self._clamp(self.offset - correction, -1.0, 1.0)

        elif action == ACTIONS["RIGHT_TURN"]:
            # Turn right: offset increases (moves toward right edge → corrects if left of line)
            correction = 0.4 + noise * 0.1
            self.offset = self._clamp(self.offset + correction, -1.0, 1.0)

        self.steps_taken += 1

        # Track if line is lost
        if abs(self.offset) > 0.6:
            self.line_lost_count += 1

        if self.action_duration > 0:
            time.sleep(self.action_duration)

    def avoid_obstacle(self):
        """Rule-based obstacle avoidance in simulation."""
        self._obstacle = False
        # Simulate a small detour: lateral bump then return
        self.offset    = self._clamp(self.offset + 0.3, -1.0, 1.0)
        self.track_pos = (self.track_pos + 0.08) % 1.0

    def find_path(self) -> bool:
        """Rule-based path-finding: sweep until line found (always succeeds in sim)."""
        # In simulation, always finds the line after a small correction
        self.offset = self._clamp(self.offset * 0.5, -1.0, 1.0)
        return True

    def cleanup(self):
        pass   # nothing to clean up in simulation

    def get_info(self) -> dict:
        """Return internal state for visualisation / debugging."""
        return {
            "offset":    round(self.offset, 3),
            "track_pos": round(self.track_pos, 3),
            "state":     STATE_NAMES[self._offset_to_state(self.offset)],
            "steps":     self.steps_taken,
        }