"""
ev3_interface.py  — UPDATED FOR ACTUAL TRACK
=============================================
Key changes from v1:
  1. INVERTED sensor: light tape on dark mat
     HIGH reading = dark floor (OFF the tape)
     LOW reading  = light tape (ON the tape)
  2. Sharper turn actions for 90-degree corners
  3. T-junction detection and rule-based handling
  4. Slower forward speed for safer cornering
"""

from ev3dev2.motor  import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sensor import INPUT_1, INPUT_4
from ev3dev2.sensor.lego import ColorSensor, UltrasonicSensor
import time

from q_learning import STATES

# ── Motor ports (change if your wiring differs) ───────────────────────────
# Left motor  → OUTPUT_B
# Right motor → OUTPUT_C

# ── Speeds ────────────────────────────────────────────────────────────────
SPEED_FORWARD  = 25   # slower than before — track has sharp corners
SPEED_TURN     = 20   # pivot turn speed
SPEED_REVERSE  = 18

ACTION_DURATION = 0.3  # seconds per action step

# ── INVERTED thresholds for light-tape-on-dark-mat ────────────────────────
# Run calibrate.py first to get YOUR exact values.
# Calibrated on this track:
#   dark mat  ≈ 3
#   tape edge ≈ 18
#   full tape ≈ 30
#
# State mapping (INVERTED from original):
#   LOW reading  → ON the light tape   → ON_LINE
#   HIGH reading → ON the dark floor   → FAR from tape
#
# The robot is FAR_LEFT  when the sensor sees mostly dark (high) on the LEFT
# The robot is FAR_RIGHT when the sensor sees mostly dark (high) on the RIGHT
#
# With a single colour sensor, we discretize the reflected intensity:
THRESHOLD_ON_TAPE     = 24   # above this = on or near the tape
THRESHOLD_EDGE_NEAR   = 11   # above this = edge region
THRESHOLD_EDGE_FAR    = 6    # above this = transition away from dark mat

# Obstacle distance
OBSTACLE_DISTANCE_CM  = 15

# T-junction: very bright reading indicates wide tape area.
T_JUNCTION_LIGHT_LEVEL = 33   # observed on the T-junction during calibration


class EV3Interface:

    def __init__(self):
        print("[EV3] Connecting...")
        self.left_motor  = LargeMotor(OUTPUT_B)
        self.right_motor = LargeMotor(OUTPUT_C)
        self.color       = ColorSensor(INPUT_4)
        self.color.mode      = "COL-REFLECT"
        self.ultrasonic = None
        try:
            self.ultrasonic = UltrasonicSensor(INPUT_1)
            self.ultrasonic.mode = "US-DIST-CM"
        except Exception:
            print("[EV3] Ultrasonic sensor not detected on in1; obstacle avoidance disabled.")
        print("[EV3] Connected.")

    # ── Sensor reading → discrete state (INVERTED) ────────────────────────
    def read_raw_light(self) -> int:
        return self.color.reflected_light_intensity

    def read_state(self) -> int:
        """
        INVERTED mapping for light tape on dark mat.

        The EV3 colour sensor returns reflected light intensity (0-100).
        On dark background: low value (~3)
        On light tape:      higher value (~18-30)

        So:
          High reading → ON the tape → good (ON_LINE or near)
          Low reading  → OFF the tape → bad (FAR_LEFT or FAR_RIGHT)

        We use a single sensor positioned at the edge of the tape:
          - When perfectly centred on tape edge: mid reading
          - Drifted right (sensor over dark): low reading → state LEFT
            (need to turn right to get back)
          - Drifted left (sensor over tape centre): high reading → state RIGHT
            (need to turn left to get back)

        NOTE: After calibration you may need to swap LEFT/RIGHT here
        depending on which side of the tape the sensor is mounted.
        """
        light = self.read_raw_light()

        # Higher reading means the sensor is seeing more tape.
        if   light >= THRESHOLD_ON_TAPE:   return STATES["ON_LINE"]
        elif light >= THRESHOLD_EDGE_NEAR: return STATES["RIGHT"]
        elif light >= THRESHOLD_EDGE_FAR:  return STATES["LEFT"]
        elif light >= 3:                   return STATES["FAR_LEFT"]
        else:                              return STATES["FAR_RIGHT"]

    def is_t_junction(self) -> bool:
        """
        Detect T-junction: sensor sees very high light (wide tape area).
        Rule-based — not RL.
        """
        return self.read_raw_light() > T_JUNCTION_LIGHT_LEVEL

    def obstacle_detected(self) -> bool:
        if self.ultrasonic is None:
            return False
        return self.ultrasonic.distance_centimeters < OBSTACLE_DISTANCE_CM

    # ── Motor commands ────────────────────────────────────────────────────
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
        """Pivot: left motor backward, right forward."""
        self.left_motor.on(SpeedPercent(-SPEED_TURN))
        self.right_motor.on(SpeedPercent(SPEED_TURN))

    def turn_right(self):
        """Pivot: left motor forward, right backward."""
        self.left_motor.on(SpeedPercent(SPEED_TURN))
        self.right_motor.on(SpeedPercent(-SPEED_TURN))

    def execute_action(self, action: int):
        """Run one RL action for ACTION_DURATION seconds."""
        {0: self.move_forward,
         1: self.move_reverse,
         2: self.turn_left,
         3: self.turn_right}[action]()
        time.sleep(ACTION_DURATION)
        self.stop()

    # ── Rule-based: sharp 90° corner ─────────────────────────────────────
    def navigate_corner(self, direction: str = "right"):
        """
        Rule-based 90-degree corner pivot.
        Called when the robot reaches a corner of the rectangular track.
        direction: 'right' for clockwise, 'left' for anticlockwise.
        NOT learned by RL.
        """
        print("[EV3] Navigating {} corner...".format(direction))
        # Creep forward to centre robot on corner
        self.move_forward()
        time.sleep(0.2)
        self.stop()
        # Pivot 90 degrees — tune pivot_time on real hardware
        pivot_time = 0.65  # seconds for ~90° — adjust this!
        if direction == "right":
            self.turn_right()
        else:
            self.turn_left()
        time.sleep(pivot_time)
        self.stop()

    # ── Rule-based: T-junction ────────────────────────────────────────────
    def handle_t_junction(self, go: str = "straight"):
        """
        Rule-based T-junction handler.
        go: 'straight', 'left', or 'right'
        NOT learned by RL.
        """
        print("[EV3] T-junction - going {}".format(go))
        if go == "straight":
            self.move_forward()
            time.sleep(0.3)
            self.stop()
        elif go == "right":
            self.turn_right()
            time.sleep(0.5)
            self.stop()
        else:
            self.turn_left()
            time.sleep(0.5)
            self.stop()

    # ── Rule-based: obstacle avoidance ───────────────────────────────────
    def avoid_obstacle(self):
        """Rule-based obstacle avoidance. NOT learned by RL."""
        print("[EV3] Obstacle! Avoiding...")
        self.move_reverse(); time.sleep(0.4); self.stop()
        self.turn_right();   time.sleep(0.6); self.stop()
        self.move_forward(); time.sleep(0.8); self.stop()
        self.turn_left();    time.sleep(0.6); self.stop()

    # ── Rule-based: re-find tape ──────────────────────────────────────────
    def find_path(self) -> bool:
        """
        Sweep until tape found again.
        Returns True if found, False if timeout.
        NOT learned by RL.
        """
        print("[EV3] Lost tape - searching...")
        start = time.time()
        while time.time() - start < 5.0:
            self.turn_right()
            time.sleep(0.1)
            self.stop()
            if self.read_state() not in (STATES["FAR_LEFT"], STATES["FAR_RIGHT"]):
                print("[EV3] Tape found!")
                return True
        return False

    def cleanup(self):
        self.stop()
        print("[EV3] Motors stopped.")