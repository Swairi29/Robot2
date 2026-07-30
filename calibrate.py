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

from ev3dev2.sensor import INPUT_3
from ev3dev2.sensor.lego import ColorSensor
import time

def calibrate():
    color = ColorSensor(INPUT_3)
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
        input(f"  Place sensor at: {desc}\n  Press Enter to read...")
        samples = [color.reflected_light_intensity for _ in range(15)]
        time.sleep(0.01)
        avg = sum(samples) / len(samples)
        readings[name] = avg
        print(f"  → Average: {avg:.1f}   (min={min(samples)}, max={max(samples)})\n")

    print("\n=== Suggested thresholds for ev3_interface.py ===")
    dark  = readings["DARK_FLOOR"]
    edge  = readings["TAPE_EDGE"]
    tape  = readings["FULL_TAPE"]
    print(f"  Dark mat reading:   {dark:.0f}")
    print(f"  Tape edge reading:  {edge:.0f}")
    print(f"  Full tape reading:  {tape:.0f}")
    print()
    # Inverted thresholds
    t1 = (dark + edge) / 2
    t2 = (edge + tape) / 2
    t3 = tape - (tape - edge) * 0.3
    print(f"THRESHOLD_ON_TAPE   = {t2:.0f}   (above this = on/near tape)")
    print(f"THRESHOLD_EDGE_NEAR = {t1:.0f}   (above this = edge region)")
    print(f"T_JUNCTION_LIGHT_LEVEL = {readings['T_JUNCTION']:.0f}")
    print()
    print("Update these values in ev3_interface.py, then run train.py")

if __name__ == "__main__":
    calibrate()