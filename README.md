# CereCe MI Tracking

Real-time wheelchair control from motor-imagery (MI) EEG, head-gaze tracking, and
an optional BLE ultrasonic obstacle guard.

![architecture diagram](./figs/architecture_v3.png)

---

## Quick start

```bash
conda create -n mi-tracking python==3.10
conda activate mi-tracking

pip install -r requirements.txt

python main.py                 # runs whatever config.yaml enables
```

For tests and the smoke runner:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Running any subset of modules

`main.py` can run **any combination** of the four subsystems — one at a time, a
mixture, or everything. Module aliases:

| Alias     | Config key               | What it does                                             |
|-----------|--------------------------|----------------------------------------------------------|
| `eeg`     | `eeg_receiver`           | Streams EEG, predicts MI, drives the active/inactive gate |
| `gaze`    | `gaze_receiver`          | Head-pose gaze → steering direction + blink gear toggle   |
| `sonar`   | `ultrasonic_receiver`    | BLE distance sensor; blocks configured directions         |
| `control` | `wheelchair_controller`  | Sends motor commands (Jrk controllers via `jrk2cmd`)      |

The **default is everything the config enables**, so plain `python main.py`
behaves as before. Flags override the YAML:

```bash
python main.py --list-modules                              # show aliases + current state
python main.py --modules eeg                               # EEG + MI gate only
python main.py --modules gaze --dry-run                    # gaze steering, no motor output
python main.py --modules control --dry-run --headless --duration 10
python main.py --modules eeg,sonar --disable control
python main.py --enable sonar --disable eeg --dry-run      # tweak the config's set
python main.py --modules eeg --source file                 # replay a CSV instead of the device
```

| Flag | Meaning |
|------|---------|
| `--config_path`, `--config` | Config file (default `config.yaml`) |
| `--modules A,B` | Run **only** these modules (exclusive) |
| `--enable A,B` / `--disable A,B` | Add/remove modules on top of the config's set |
| `--dry-run` | Force the controller into debug mode — prints, never drives |
| `--headless` | No window and no keyboard input (Ctrl+C or `--duration` to stop) |
| `--duration SECONDS` | Stop automatically after N seconds |
| `--source file\|device` | Override the EEG input source |
| `--warmup SECONDS` | EEG warmup before the first health check |
| `--list-modules` | Print module aliases and exit |

### Graceful degradation

A module that is requested but cannot start is **disabled for the run** and
reported at startup; the remaining modules keep working.

| Module | Unavailable because | Behaviour |
|--------|--------------------|-----------|
| `eeg` | no SDK / no device / stale data | MI gate is off; gaze and keyboard still steer. A requested-but-dead EEG never auto-arms the drive. |
| `gaze` | camera open failure | Falls back to keyboard steering |
| `sonar` | `bleak` missing / never connects | Obstacle guard off, one-time warning, UI shows `SONAR: --` |
| `control` | `jrk2cmd` missing / disabled | Commands are logged instead of sent (dry run) |

### Controls while running

| Key | Action |
|-----|--------|
| `W` `A` `S` `D` | Steer. A **held key overrides gaze** and gaze resumes when released, so lag is never a dead end. |
| `SPACE` | Toggle drive — only when no MI gate is live (otherwise the gate owns it) |
| `G` | Sticky keyboard override on/off |
| `Q` | Quit (closing the window also quits) |

Motor commands are always stopped on exit — quit, window close, Ctrl+C,
exception, `--duration`, or end of a replayed file.

---

## Design notes

**1. Sampling Rate Mismatch**
- **Problem:** The EEG hardware streams at 250 Hz, faster than inference or the
  motor loop can consume, so a naive reader accumulates lag.
- **Solution:** **Rolling FIFO buffer.** The receiver keeps the most recent data
  in a thread-safe `deque`; the predictor takes the newest window each tick
  instead of draining a backlog.
- **Note:** Turning requires strong, continuous mental concentration.

**2. Signal Instability**
- **Problem:** Raw MI predictions fluctuate with noise, causing jitter or false
  positives.
- **Solution:** **Evidence Accumulation.** A leaky-integrator style score must
  cross a threshold before the drive gate changes, with hysteresis to hold a
  state through brief dropouts.

---

## Architecture

```
main.py                # CLI, render loop, keyboard  (thin)
  └── modules/pipeline.py    # builds the requested modules, owns control state
        ├── modules/eeg_receiver.py        → MI gate
        ├── modules/mi_predictor.py
        ├── modules/evidence_accumulator.py
        ├── modules/gaze_receiver.py       → steering + blink
        ├── modules/ultrasonic_receiver.py → obstacle guard
        └── modules/wheelchair_controller.py → motors
```

`Pipeline` builds only the modules you asked for and returns a status dict per
tick. Every real module is imported lazily inside a factory, so the pipeline can
be tested with fakes without cv2, pandas, sklearn or mediapipe installed.

**Steering precedence each tick:** held key → gaze → `stop`.

---

## Modules

### 1. EEG Receiver (`modules/eeg_receiver.py`)
- **Content:** Connects to the Explore device (ExplorePy) or replays a `.csv`/`.npy`
  recording as if it were live. Background thread; thread-safe rolling buffer.
- **Input:** Device stream, or a file with `TimeStamp` + channel columns.
- **Output:** `get_buffer_data()` → `numpy.ndarray` shaped `(N, 1 + n_channels)`,
  column 0 is the timestamp, plus `is_healthy()` and `finished`.
- **Note:** device mode requires `explorepy`. On Linux a pure-Python socket shim
  replaces the vendor SDK's RFCOMM transport, which drops the link ~45 ms after
  streaming starts. If `explorepy` is unavailable but a `data_path` is set, it
  falls back to file replay.

### 2. MI Predictor (`modules/mi_predictor.py`)
- **Content:** Cz re-reference → notch (45–55 Hz) → bandpass (7–30 Hz) → Welch PSD
  → LDA. Matches the Cerebruh training pipeline (1 s Hanning window, no overlap,
  7–29 Hz bins).
- **Input:** `(N, n_channels + 1)` or `(N, n_channels)`.
- **Output:** `'active'` or `'none'`. Feature vector is 69 long
  (3 channels after Cz removal × 23 bins); a mismatch is rejected loudly rather
  than silently mis-scored. A missing model is reported explicitly.

### 3. Evidence Accumulator (`modules/evidence_accumulator.py`)
- **Content:** Leaky integrate-and-fire smoothing over per-tick predictions.
  All evidence decays each tick, a matching prediction builds its score, and a
  threshold with hysteresis decides the state.
- **Input:** `'active'` / `'none'` (anything unexpected counts as inactive).
- **Output:** `'active'` (MI intent held) or `'inactive'`.

### 4. Gaze Receiver (`modules/gaze_receiver.py`)
- **Content:** Wraps `GazeTracker` from `Cerebruh_Gaze_Tracking` (head-pose mode).
- **Input:** Camera frames; configuration from `config_gaze.json`.
- **Output:** `'forward' | 'backward' | 'left' | 'right' | 'stop'`, a blink flag
  (double-blink toggles reverse gear), and the latest UI frame.
- **Thresholds:** `horizontal_threshold`, `vertical_forward_threshold`,
  `vertical_backward_threshold` in `gaze_params`.

### 5. Ultrasonic Receiver (`modules/ultrasonic_receiver.py`)
- **Content:** Background BLE thread reading `"lhs,rhs"` distance notifications
  from the `UltrasonicBLE` Arduino (`modules/Ultrasonic_Sensor_Data_Extraction.ino`).
  Scans and reconnects so the sensor can be powered on late.
- **Input:** BLE notifications; `device_name`, `char_uuid`, `obstacle_distance`,
  `block_directions`.
- **Output:** `get_distances()`, `is_healthy()`, `is_obstacle()`,
  `filter_direction(direction, obstacle)`.
- **Safety:** a missing/stale sensor is **fail-open** by default (guard inactive)
  and warns once. Set `treat_unhealthy_as_obstacle: true` to fail safe.

### 6. Wheelchair Controller (`modules/wheelchair_controller.py`)
- **Content:** Drives the two Jrk motor controllers through `jrk2cmd`.
- **Input:** `move_forward()`, `move_backward()`, `move_left()`, `move_right()`,
  `stop()`.
- **Note:** `debug_mode: true` prints the computed targets and sends nothing.
  `main.py` de-duplicates commands and re-sends on a keepalive interval instead
  of spawning a subprocess every tick.

---

## Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration — all modules, all parameters |
| `config_eeg_only.yaml` | EEG gate + keyboard steering, no camera |
| `config_gaze_only.yaml` | Gaze steering, no EEG |
| `config_gaze.json` | GazeTracker tuning (thresholds, smoothing, blink behaviour) |

The `modules:` block turns subsystems on and off; the CLI flags override it.
Runtime keys worth knowing: `loop_interval`, `show_camera_ui`, `headless`,
`run_duration`, `exit_on_source_end`, `command_keepalive`,
`keyboard_override_hold`, `start_drive_enabled` (`auto` arms the drive only when
`eeg_receiver` is off), `eeg_warmup_seconds`, `eeg_health_timeout`.

---

## Testing

```bash
pytest                                  # unit + integration, no hardware needed
pytest -m hardware                      # opt in to hardware tests
python -m tools.smoke_run --all --duration 3     # boot all 15 module subsets
python -m tools.smoke_run --module eeg --source file
python -m tools.smoke_run --module gaze --real --dry-run
```

- `tests/fakes.py` provides stdlib-only fake receivers and a recording
  controller, so module mixtures and gating logic are testable anywhere.
- `tests/test_pipeline_modes.py` covers subsets, steering precedence, the MI
  gate, degradation, command de-duplication and shutdown safety.
- `tests/test_receivers.py` covers the individual modules (payload parsing,
  thresholds, feature layout, replay).
- `tools/smoke_run.py` boots each subset headlessly and prints a PASS/FAIL
  table; fake mode is the default, real mode stays dry unless you pass `--live`.

Each module also runs standalone:

```bash
python -m modules.eeg_receiver --mode file --data data/MItest_24-01-27_ExG.csv
python -m modules.mi_predictor
python -m modules.evidence_accumulator
python -m modules.gaze_receiver
python -m modules.ultrasonic_receiver
python -m modules.wheelchair_controller           # dry run; add --live for motors
```

`main_test.py` is the **legacy** EEG-only integration smoke test (predictor →
accumulator → controller). Use `main.py` for anything real.

---

## Gaze dashboard

The browser-based gaze dashboard and bundled Unity test game are documented in
[`Gaze_Dashboard_Import/README.md`](Gaze_Dashboard_Import/README.md).

---

## Troubleshooting

**`DeviceNotFoundError: No device found with the name: Explore_XXXX` on Linux, first connect.**
The Linux socket shim (`modules/explorepy_linux_socket_shim.py`) replaces the
vendor SDK, but its device search is a stub: explorepy normally resolves the MAC
from a cached settings file, and on a machine that has never connected that cache
is empty, so discovery fails before the shim is reached. Seed it once — the
device name in the filename must match `receiver_params.device_name`, and the
trailing bytes of the MAC must match the name (`...:84:2F` → `Explore_842F`):

```bash
bluetoothctl --timeout 15 scan on | grep -i explore      # note the MAC address
mkdir -p ~/.config/Mentalab
printf 'mac_address: 00:13:43:A1:84:2F\n' > ~/.config/Mentalab/Explore_842F.yaml
```

**Camera errors mention "Windows Privacy Settings" while running on Linux.**
That string comes from `GazeTracker`, not from this project; the usual cause is
that no camera is present (`ls /dev/video*`). The pipeline degrades to keyboard
steering and logs `DEGRADED: gaze_receiver` rather than aborting.

**`[Ultrasonic] WARNING: sensor unavailable — obstacle guard is INACTIVE`.**
Expected when the Arduino is off or out of range. The guard is fail-open by
default; set `treat_unhealthy_as_obstacle: true` in `ultrasonic_params` to fail
safe instead.

**Motor commands are noisy in the log but nothing moves.**
`control_params.debug_mode` (or `--dry-run`) is on: targets are printed, nothing
is sent. Start-up prints `*** LIVE MOTOR CONTROL ENABLED ***` when they will be.

---

## Future Plan

**1. Process Logging & Visualization**

- **Pipeline Monitoring**: High-level visual tracking of the entire process to provide clear insights into system operations.

- **Signal-Prediction Alignment**: Synchronized visualization of EEG signals (including the receiver buffer) alongside model predictions for precise behavioral analysis.

- **Comparative Analysis**: When using ground-truth labeled data, the system provides a side-by-side visualization of labels vs. model predictions, facilitating intuitive performance assessment during testing.


**2. Improving prediction accuracy**
- Integrated real-time artifact correction into the current prediction process. [post](https://www.linkedin.com/posts/victor-ferat_python-eeg-meg-activity-7415019925267853312-YJF8/?utm_source=share&utm_medium=member_android&rcm=ACoAABYU9BsBVhMTffCvRJALhNsmM7-QLtESekQ), [paper](https://www.biorxiv.org/content/10.1101/2025.10.04.680449v1)

---

## Mindmap

![mindmap](figs/mindmap_v1.png)
