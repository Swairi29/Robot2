"""
ev3_interface.py
================
Hardware abstraction layer for the EV3 robot.

All sensor reading and motor driving lives here.
The Q-learning code never touches hardware directly —
it calls this interface, making it easy to swap in a
simulator without changing the learning code.

Requirements
------------
  pip install ev3dev2

Setup
-----
1. Flash ev3dev image to a microSD card and boot the EV3 from it.
2. Connect EV3 to PC via USB cable.
3. The EV3 appears as a network device (192.168.0.1 by default).
   You can SSH in, or use ev3dev2 directly from your PC Python.

Wiring assumptions (change port strings to match your build):
  - Left motor   → OUTPUT_B
  - Right motor  → OUTPUT_C
  - Color sensor → INPUT_3
  - Ultrasonic   → INPUT_1   (for obstacle avoidance)
"""

from ev3dev2.motor  import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_3
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
import time

from q_learning import STATES


# ---------------------------------------------------------------------------
# Tunable constants — adjust after testing on your actual track
# ---------------------------------------------------------------------------

# Motor speeds (% of max)
SPEED_FORWARD   = 30   # straight ahead
SPEED_TURN      = 25   # during turns
SPEED_REVERSE   = 20   # backward

# Duration of each action (seconds) before re-reading the sensor
ACTION_DURATION = 0.3

# Color sensor thresholds (reflected light intensity, 0–100)
# Measure these on YOUR track with YOUR lighting conditions.
# Typical black line on white: black ≈ 5–15, white ≈ 60–80, edge ≈ 30–45
THRESHOLD_FAR_LEFT   = 20   # sensor fully off line to the left
THRESHOLD_LEFT       = 35   # sensor on the left edge
THRESHOLD_ON_LINE    = 50   # sensor centred on line
THRESHOLD_RIGHT      = 65   # sensor on the right edge
# anything > THRESHOLD_RIGHT → FAR_RIGHT

# Obstacle distance (cm) — below this triggers avoidance
OBSTACLE_DISTANCE_CM = 15


# ---------------------------------------------------------------------------
# EV3Interface class
# ---------------------------------------------------------------------------

class EV3Interface:
    """
    Wraps the ev3dev2 API into clean read_state() / execute_action() calls.
    """

    def __init__(self):
        print("[EV3] Connecting to motors and sensors...")
        self.left_motor  = LargeMotor(OUTPUT_B)
        self.right_motor = LargeMotor(OUTPUT_C)
        self.color       = ColorSensor(INPUT_3)
        self.ultrasonic  = UltrasonicSensor(INPUT_1)

        # Set color sensor to reflected light intensity mode
        self.color.mode       = "COL-REFLECT"
        self.ultrasonic.mode  = "US-DIST-CM"

        print("[EV3] Connected.")

    # ------------------------------------------------------------------
    # Sensor reading → discrete state
    # ------------------------------------------------------------------

    def read_raw_light(self) -> int:
        """Return raw reflected light intensity (0–100)."""
        return self.color.reflected_light_intensity

    def read_state(self) -> int:
        """
        Read the color sensor and map it to one of the 5 discrete states.

        Adjust the thresholds above after placing your robot on the track
        and printing read_raw_light() values for each position.
        """
        light = self.read_raw_light()

        if light < THRESHOLD_FAR_LEFT:
            return STATES["FAR_LEFT"]
        elif light < THRESHOLD_LEFT:
            return STATES["LEFT"]
        elif light < THRESHOLD_ON_LINE:
            return STATES["ON_LINE"]
        elif light < THRESHOLD_RIGHT:
            return STATES["RIGHT"]
        else:
            return STATES["FAR_RIGHT"]

    def obstacle_detected(self) -> bool:
        """Return True if an obstacle is within OBSTACLE_DISTANCE_CM."""
        return self.ultrasonic.distance_centimeters < OBSTACLE_DISTANCE_CM

    # ------------------------------------------------------------------
    # Motor commands
    # ------------------------------------------------------------------

    def stop(self):
        self.left_motor.stop()
        self.right_motor.stop()

    def move_forward(self):
        self.left_motor.on(SpeedPercent(SPEED_FORWARD))
        self.right_motor.on(SpeedPercent(SPEED_FORWARD))

    def move_reverse(self):
        self.left_motor.on(SpeedPercent(-SPEED_REVERSE))
        self.right_motor.on(SpeedPercent(-SPEED_REVERSE))

    def turn_left(self):
        """Pivot left: left motor backward, right motor forward."""
        self.left_motor.on(SpeedPercent(-SPEED_TURN))
        self.right_motor.on(SpeedPercent(SPEED_TURN))

    def turn_right(self):
        """Pivot right: left motor forward, right motor backward."""
        self.left_motor.on(SpeedPercent(SPEED_TURN))
        self.right_motor.on(SpeedPercent(-SPEED_TURN))

    # ------------------------------------------------------------------
    # RL action execution
    # ------------------------------------------------------------------

    def execute_action(self, action: int):
        """
        Execute one of the 4 learned actions for ACTION_DURATION seconds,
        then stop so the sensor can take a fresh reading.

        action: 0=FORWARD, 1=REVERSE, 2=LEFT_TURN, 3=RIGHT_TURN
        """
        action_map = {
            0: self.move_forward,
            1: self.move_reverse,
            2: self.turn_left,
            3: self.turn_right,
        }
        action_map[action]()
        time.sleep(ACTION_DURATION)
        self.stop()

    # ------------------------------------------------------------------
    # Rule-based behaviours (obstacle + re-find path)
    # These are NOT learned by RL — they are hard-coded as per the brief.
    # ------------------------------------------------------------------

    def avoid_obstacle(self):
        """
        Simple rule-based obstacle avoidance.
        Backs up, turns right 90°, moves forward to clear, then turns left.
        Does NOT use RL.
        """
        print("[EV3] Obstacle detected! Avoiding...")
        self.move_reverse()
        time.sleep(0.5)
        self.stop()
        self.turn_right()
        time.sleep(0.6)      # ~90° turn — tune this for your robot
        self.stop()
        self.move_forward()
        time.sleep(0.8)
        self.stop()
        self.turn_left()
        time.sleep(0.6)
        self.stop()

    def find_path(self) -> bool:
        """
        Rule-based path-finding after the line is completely lost.
        Rotates slowly until the color sensor sees the line again.
        Returns True if line found, False if timeout.
        Does NOT use RL.
        """
        print("[EV3] Line lost — searching...")
        timeout = 5.0   # seconds to search before giving up
        start   = time.time()

        while time.time() - start < timeout:
            self.turn_right()
            time.sleep(0.1)
            self.stop()
            state = self.read_state()
            if state not in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
                print("[EV3] Line found again!")
                return True

        print("[EV3] Could not find line within timeout.")
        return False

    def cleanup(self):
        """Call on exit to stop motors safely."""
        self.stop()
        print("[EV3] Motors stopped.")
