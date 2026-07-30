"""
calibrate.py
============
Run this BEFORE training to measure the real light sensor values
on your specific track and lighting conditions.

Place the robot at each position (far-left, on-line, far-right, etc.)
and press Enter — the script will print the raw value.

Then update the THRESHOLD_* constants in ev3_interface.py accordingly.

Run with:
    python calibrate.py
"""

from ev3_interface import EV3Interface
import time

def calibrate():
    robot = EV3Interface()

    positions = [
        ("FAR LEFT  (sensor fully off line, left side)", "FAR_LEFT"),
        ("LEFT EDGE (sensor on left edge of line)",      "LEFT"),
        ("ON LINE   (sensor centred on line)",           "ON_LINE"),
        ("RIGHT EDGE(sensor on right edge of line)",     "RIGHT"),
        ("FAR RIGHT (sensor fully off line, right side)","FAR_RIGHT"),
    ]

    print("=== Color Sensor Calibration ===")
    print("Place the robot at each position, then press Enter.\n")

    readings = {}
    for description, name in positions:
        input(f"  Position: {description}\n  Press Enter to read...")
        samples = []
        for _ in range(10):
            samples.append(robot.read_raw_light())
            time.sleep(0.05)
        avg = sum(samples) / len(samples)
        readings[name] = avg
        print(f"  → Average reading: {avg:.1f}\n")

    print("\n=== Suggested thresholds for ev3_interface.py ===")
    print(f"THRESHOLD_FAR_LEFT  = {(readings['FAR_LEFT']  + readings['LEFT'])  / 2:.0f}")
    print(f"THRESHOLD_LEFT      = {(readings['LEFT']      + readings['ON_LINE'])/ 2:.0f}")
    print(f"THRESHOLD_ON_LINE   = {(readings['ON_LINE']   + readings['RIGHT'])  / 2:.0f}")
    print(f"THRESHOLD_RIGHT     = {(readings['RIGHT']     + readings['FAR_RIGHT'])/2:.0f}")
    print("\nUpdate THRESHOLD_* in ev3_interface.py with these values, then run train.py")

    robot.cleanup()

if __name__ == "__main__":
    calibrate()
