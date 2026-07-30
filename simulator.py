"""
simulator.py  — UPDATED FOR ACTUAL TRACK
=========================================
Key changes from v1:
  1. Track is now a rectangular loop with 4 corners — not an oval
  2. Sensor logic INVERTED: high reading = on light tape (good)
  3. Corners are handled as rule-based events, not RL actions
  4. T-junction at top-left is handled rule-based (go straight)
  5. Dead-end box at bottom-right: robot reverses out (rule-based)

Track layout (matches your photo):
  - Main rectangular loop
  - T-junction arm exits top-left corner
  - Small dead-end box at bottom-right corner

The robot follows the TAPE (bright) on the dark mat.
"""

import random
import time
import math

from q_learning import STATES, ACTION_NAMES

STATE_NAMES  = {v: k for k, v in STATES.items()}
ACTION_NAMES_MAP = {v: k for k, v in {
    "FORWARD":0,"REVERSE":1,"LEFT_TURN":2,"RIGHT_TURN":3}.items()}

# Track segments: list of (x1,y1, x2,y2) normalised 0..1
# Represents the actual rectangular path in your photo
TRACK_SEGMENTS = [
    # Main rectangle (clockwise: top, right, bottom, left)
    ((0.2, 0.15), (0.85, 0.15)),   # top
    ((0.85, 0.15), (0.85, 0.85)),  # right
    ((0.85, 0.85), (0.2, 0.85)),   # bottom
    ((0.2, 0.85), (0.2, 0.15)),    # left
    # T-junction arm (exits top-left corner leftward)
    ((0.05, 0.15), (0.2, 0.15)),   # T arm horizontal
    # T arm goes up slightly (the vertical bit in your photo)
    ((0.05, 0.05), (0.05, 0.20)),  # T arm vertical
]

class LineFollowingSimulator:
    """
    Rectangular track simulator matching your actual photo.

    State definition (INVERTED from original):
      offset = 0.0  → perfectly on tape edge (ON_LINE)
      offset > 0    → drifted off tape to the right (less tape under sensor)
      offset < 0    → sensor moved onto tape centre (too far left)

    High light value = on tape = good
    Low light value  = off tape (dark floor) = bad
    """

    def __init__(self, noise_level=0.12, obstacle_prob=0.01, action_duration=0.0):
        self.noise_level    = noise_level
        self.obstacle_prob  = obstacle_prob
        self.action_duration = action_duration

        # Position on track (0.0 to 1.0 = progress around the rectangle)
        self.track_pos = 0.0
        # Lateral offset from tape edge (-1 = far left/on tape, +1 = far right/off tape)
        self.offset    = 0.0
        self._obstacle = False

        self.steps_taken    = 0
        self.corner_count   = 0
        self.line_lost_count = 0

    def _noise(self):
        return random.gauss(0, self.noise_level)

    def _offset_to_state(self, offset: float) -> int:
        """
        INVERTED: negative offset = deep on tape = reads HIGH light = ON_LINE
        Positive offset = off tape = reads LOW light = FAR state
        """
        if   offset < -0.6:  return STATES["ON_LINE"]    # deep on tape — centred
        elif offset < -0.2:  return STATES["LEFT"]        # slightly on tape — veer right needed
        elif offset <  0.2:  return STATES["RIGHT"]       # on edge — slight correction
        elif offset <  0.6:  return STATES["FAR_LEFT"]    # mostly off tape
        else:                return STATES["FAR_RIGHT"]   # completely off tape

    def reset(self):
        self.track_pos = random.uniform(0.0, 1.0)
        self.offset    = random.uniform(-0.25, 0.25)
        self._obstacle = False
        self.steps_taken = 0

    def read_state(self) -> int:
        noisy = max(-1.0, min(1.0, self.offset + self._noise() * 0.3))
        return self._offset_to_state(noisy)

    def obstacle_detected(self) -> bool:
        if random.random() < self.obstacle_prob:
            self._obstacle = True
        return self._obstacle

    def is_at_corner(self) -> bool:
        """True when approaching a 90° corner of the rectangle."""
        # Corners at ~0.25, 0.50, 0.75, 1.0 of the loop
        pos = self.track_pos % 1.0
        corner_positions = [0.25, 0.50, 0.75, 0.0]
        for cp in corner_positions:
            if abs(pos - cp) < 0.03:
                return True
        return False

    def is_at_t_junction(self) -> bool:
        """T-junction is at the start of the top-left arm (~track_pos 0.0)."""
        return abs(self.track_pos % 1.0) < 0.02

    def execute_action(self, action: int):
        noise = self._noise()
        drift = random.gauss(0, 0.04)

        if action == 0:   # FORWARD — advance along track
            self.track_pos = (self.track_pos + 0.04) % 1.0
            # On straight sections: small drift off tape
            self.offset = max(-1, min(1, self.offset + drift + noise * 0.08))

        elif action == 1:  # REVERSE
            self.track_pos = (self.track_pos - 0.02 + 1) % 1.0
            self.offset = max(-1, min(1, self.offset - drift + noise * 0.08))

        elif action == 2:  # LEFT_TURN — moves sensor toward tape (negative offset)
            self.offset = max(-1, min(1, self.offset - 0.35 + noise * 0.08))

        elif action == 3:  # RIGHT_TURN — moves sensor away from tape (positive offset)
            self.offset = max(-1, min(1, self.offset + 0.35 + noise * 0.08))

        self.steps_taken += 1
        if abs(self.offset) > 0.6:
            self.line_lost_count += 1

        if self.action_duration > 0:
            time.sleep(self.action_duration)

    def navigate_corner(self, direction="right"):
        """Rule-based: robot pivots 90° at a corner."""
        self.corner_count += 1
        self.track_pos = (self.track_pos + 0.02) % 1.0
        # After turning, reset offset to near tape edge
        self.offset = random.uniform(-0.15, 0.15)

    def handle_t_junction(self, go="straight"):
        """Rule-based T-junction: just continue straight."""
        self.offset = random.uniform(-0.1, 0.1)

    def avoid_obstacle(self):
        self._obstacle = False
        self.offset = max(-1, min(1, self.offset + 0.25))
        self.track_pos = (self.track_pos + 0.05) % 1.0

    def find_path(self) -> bool:
        self.offset = self.offset * 0.4
        return True

    def cleanup(self):
        pass

    def get_info(self) -> dict:
        return {
            "offset":    round(self.offset, 3),
            "track_pos": round(self.track_pos, 3),
            "state":     STATE_NAMES[self._offset_to_state(self.offset)],
            "steps":     self.steps_taken,
            "corners":   self.corner_count,
        }