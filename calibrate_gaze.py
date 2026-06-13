"""
Run gaze-tracker calibration and save results to config_gaze.json.

Usage (from project root):
    python calibrate_gaze.py

Calibration steps:
  1. A camera window opens. Press C to begin.
  2. Follow the green dot: centre → left → right.
  3. Close left eye (keep right open), then close right eye (keep left open).
  4. Values are saved automatically when the sequence finishes.
     Press Q at any time to abort without saving.
"""

import json
import os
import sys

# Make Cerebruh_Gaze_Tracking/src importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_GAZE_SRC = os.path.join(_HERE, 'Cerebruh_Gaze_Tracking', 'src')
if _GAZE_SRC not in sys.path:
    sys.path.insert(0, _GAZE_SRC)

from calibrate import do_calibration  # noqa: E402

CONFIG_PATH = os.path.join(_HERE, 'config_gaze.json')


def main():
    with open(CONFIG_PATH, 'r') as f:
        data = json.load(f)

    print(f"Loaded config from {CONFIG_PATH}")
    print("Current thresholds:")
    for key in ('center_offset', 'left_threshold', 'right_threshold',
                'left_eye_closed_threshold', 'right_eye_closed_threshold'):
        print(f"  {key}: {data.get(key)}")

    data = do_calibration(data)

    with open(CONFIG_PATH, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nCalibration saved to {CONFIG_PATH}")
    print("Updated thresholds:")
    for key in ('center_offset', 'left_threshold', 'right_threshold',
                'left_eye_closed_threshold', 'right_eye_closed_threshold'):
        print(f"  {key}: {data.get(key)}")


if __name__ == '__main__':
    main()
