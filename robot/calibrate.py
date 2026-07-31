"""
calibrate.py  — UPDATED FOR LIGHT TAPE ON DARK MAT
====================================================
IMPORTANT: Your track has light (cream/yellow) tape on a dark mat.
This is the OPPOSITE of the standard black-line-on-white setup.

What to expect:
  - On the dark mat:   sensor reads LOW  (5–20)
  - On the light tape: sensor reads HIGH (60–80)
  - On the tape edge:  sensor reads MID  (30–50)

Run this before training to get YOUR exact values.
"""

from ev3dev2.sensor import INPUT_4
from ev3dev2.sensor.lego import ColorSensor
import time

def calibrate():
    color = ColorSensor(INPUT_4)
    color.mode = "COL-REFLECT"

    positions = [
        ("DARK MAT     (sensor fully OFF the tape)",     "DARK_FLOOR"),
        ("TAPE EDGE    (sensor half on tape, half off)", "TAPE_EDGE"),
        ("ON TAPE      (sensor fully ON the tape)",      "FULL_TAPE"),
        ("CORNER AREA  (wide tape at a corner)",         "CORNER"),
        ("T-JUNCTION   (where two tape lines meet)",     "T_JUNCTION"),
    ]

    print("=== Calibration for LIGHT TAPE on DARK MAT ===")
    print("Expected: dark mat = LOW value, light tape = HIGH value\n")

    readings = {}
    for desc, name in positions:
        input("  Place sensor at: {}\n  Press Enter to read...".format(desc))
        samples = [color.reflected_light_intensity for _ in range(15)]
        time.sleep(0.01)
        avg = sum(samples) / len(samples)
        readings[name] = avg
        print("  → Average: {:.1f}   (min={}, max={})\n".format(avg, min(samples), max(samples)))

    print("\n=== Suggested thresholds for ev3_interface.py ===")
    dark  = readings["DARK_FLOOR"]
    edge  = readings["TAPE_EDGE"]
    tape  = readings["FULL_TAPE"]
    print("  Dark mat reading:   {:.0f}".format(dark))
    print("  Tape edge reading:  {:.0f}".format(edge))
    print("  Full tape reading:  {:.0f}".format(tape))
    print()
    # Inverted thresholds
    t1 = (dark + edge) / 2
    t2 = (edge + tape) / 2
    t3 = tape - (tape - edge) * 0.3
    print("THRESHOLD_ON_TAPE   = {:.0f}   (above this = on/near tape)".format(t2))
    print("THRESHOLD_EDGE_NEAR = {:.0f}   (above this = edge region)".format(t1))
    print("T_JUNCTION_LIGHT_LEVEL = {:.0f}".format(readings['T_JUNCTION']))
    print()
    print("Update these values in ev3_interface.py, then run train.py")

if __name__ == "__main__":
    calibrate()