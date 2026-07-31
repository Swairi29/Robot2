# EV3 Q-Learning Line-Following Robot
### Assignment 02 — Robotics & Intelligent Systems

---

## Project structure

```
ev3_rl_robot/
├── q_learning.py       # Q-Learning agent (pure Python, no hardware)
├── ev3_interface.py    # EV3 hardware abstraction (motors + sensors)
├── calibrate.py        # Measure sensor thresholds on your track
├── train.py            # Main training loop
├── deploy.py           # Run trained robot (exploitation only)
├── plot_progress.py    # Plot reward curve + Q-table heatmap (for report)
└── q_table.json        # Saved Q-table (created after first training run)
```

---

## Setup

### 1. Flash ev3dev to microSD
Download: https://www.ev3dev.org/downloads/
Flash with Balena Etcher. Boot EV3 from the SD card.

### 2. Connect EV3 to PC via USB
The EV3 appears as a USB network device at `192.168.0.1`.
You can SSH in: `ssh robot@ev3dev.local`

### 3. Install ev3dev2 on your PC
```bash
pip install python-ev3dev2 numpy matplotlib
```

### 4. Wire your robot
| Component       | Port      |
|-----------------|-----------|
| Left motor      | OUTPUT_B  |
| Right motor     | OUTPUT_C  |
| Color sensor    | INPUT_3   |
| Ultrasonic      | INPUT_1   |

Update port constants in `ev3_interface.py` if your wiring differs.

---

## Workflow

### Step 1 — Calibrate sensor thresholds
```bash
python calibrate.py
```
Place the robot at each position when prompted.
Copy the suggested threshold values into `ev3_interface.py`.

### Step 2 — Train
```bash
python train.py
```
The script pauses before each episode so you can place the robot.
Press Ctrl+C at any time — progress is auto-saved.

### Step 3 — Review learning
```bash
python plot_progress.py
```
Generates `training_rewards.png` and `q_table_heatmap.png` for your report.

### Step 4 — Deploy / demo
```bash
python deploy.py
```
Loads the saved Q-table and runs with ε=0 (no exploration, pure exploitation).

---

## Key hyperparameters (in train.py)

| Parameter       | Value | Effect                                      |
|-----------------|-------|---------------------------------------------|
| `ALPHA`         | 0.3   | Learning rate — lower = slower, more stable |
| `GAMMA`         | 0.9   | Discount — high = values future rewards     |
| `EPSILON_START` | 1.0   | Start fully exploratory                     |
| `EPSILON_DECAY` | 0.99  | Decay per episode                           |
| `EPSILON_MIN`   | 0.05  | Always keep 5% exploration                  |

Expect to retrain and adjust these on the real hardware.

---

## Marking criteria covered

| Criterion                          | Marks | How achieved                        |
|------------------------------------|-------|-------------------------------------|
| Learning moving forward            | 10    | Q-table learns FORWARD in ON_LINE   |
| Learning moving backwards          | 20    | Q-table learns REVERSE when off     |
| Learning left turn                 | 10    | Q-table learns LEFT in FAR_RIGHT    |
| Learning right turn                | 10    | Q-table learns RIGHT in FAR_LEFT    |
| Clockwise & anticlockwise path     | 20    | deploy.py — flip robot direction    |
| Smoothness of following the line   | 05    | Tune ACTION_DURATION + motor speed  |
| Smoothness of turns                | 10    | Tune SPEED_TURN + thresholds        |
| Obstacle avoidance                 | 10    | Rule-based in ev3_interface.py      |
| Finding the path                   | 05    | Rule-based find_path() method       |
| **Total**                          | **100** |                                   |
