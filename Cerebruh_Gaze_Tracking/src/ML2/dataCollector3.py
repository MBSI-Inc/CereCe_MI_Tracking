import cv2
import numpy as np
import pandas as pd
import time
import sys
sys.path.append("./src/")
sys.path.append("./")

from src.util import Util
from calibrate import print_text_to_screen
from statistics import median, mean
from facemesh_gazetracker import GazeTracker
from src.ML.feature_extractor import FeatureExtractor

def get_bin_index(value, bin_edges):
    """
    Same as before: clamps value to the nearest bin on the ends,
    otherwise returns the bin interval index.
    """
    if value <= bin_edges[0]:
        return 0
    if value > bin_edges[-1]:
        return len(bin_edges) - 2
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] < value <= bin_edges[i+1]:
            return i
    return None

def draw_bin_overlay(
    base_frame,
    x_center,
    y_center,
    bin_counts,
    B,
    pitch_bins,
    yaw_bins,
    p_idx_current,
    y_idx_current,
    alpha=0.4,
    grid_size=100
):
    """
    Same overlay function as before:
      - Gradual interpolation from red to green based on how many samples are collected.
      - Higher pitch at the top (inverting the pitch bin).
      - Blue rectangle border for the current bin.
    """
    overlay = base_frame.copy()
    cell_w = grid_size / yaw_bins
    cell_h = grid_size / pitch_bins

    _, _, r_bin_num = bin_counts.shape
    total_needed = B * r_bin_num

    grid_left = int(x_center - grid_size // 2)
    grid_top  = int(y_center - grid_size // 2)

    def interpolate_red_green(fraction):
        fraction = max(0.0, min(1.0, fraction))
        red_bgr   = np.array([0,   0, 255], dtype=float)
        green_bgr = np.array([0, 255,   0], dtype=float)
        interp_bgr = (1 - fraction) * red_bgr + fraction * green_bgr
        return interp_bgr.astype(int).tolist()

    # Fill each cell
    for p_idx in range(pitch_bins):
        for y_idx in range(yaw_bins):
            sum_collected = bin_counts[p_idx, y_idx, :].sum()
            fraction = sum_collected / total_needed
            color_bgr = interpolate_red_green(fraction)

            # invert pitch index
            render_row = (pitch_bins - 1) - p_idx
            x0 = int(grid_left + y_idx * cell_w)
            y0 = int(grid_top  + render_row * cell_h)
            x1 = int(x0 + cell_w)
            y1 = int(y0 + cell_h)

            cv2.rectangle(overlay, (x0, y0), (x1, y1), color_bgr, -1)

    # White grid lines
    for row in range(pitch_bins + 1):
        render_row = (pitch_bins - row)
        y_line = int(grid_top + render_row * cell_h)
        cv2.line(overlay, (grid_left, y_line),
                 (grid_left + grid_size, y_line), (255, 255, 255), 1)
    for col in range(yaw_bins + 1):
        x_line = int(grid_left + col * cell_w)
        cv2.line(overlay, (x_line, grid_top),
                 (x_line, grid_top + grid_size), (255, 255, 255), 1)

    # Blue border for current bin
    if (0 <= p_idx_current < pitch_bins) and (0 <= y_idx_current < yaw_bins):
        render_row_current = (pitch_bins - 1) - p_idx_current
        x0 = int(grid_left + y_idx_current * cell_w)
        y0 = int(grid_top  + render_row_current * cell_h)
        x1 = int(x0 + cell_w)
        y1 = int(y0 + cell_h)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)

    out_frame = cv2.addWeighted(overlay, alpha, base_frame, 1 - alpha, 0)
    return out_frame

def process_frame_for_calibration(
    gaze_tracker,
    bin_counts,
    pitch_edges,
    yaw_edges,
    roll_edges,
    B,
    df,
    x_coord,
    y_coord,
    p_bin_num,
    y_bin_num,
    acquire_data,
    countdown_text=None
):
    """
    1. Reads a frame from `gaze_tracker`.
    2. Draws the calibration dot at (x_coord, y_coord).
    3. Overlays the pitch/yaw grid via draw_bin_overlay.
    4. (LAST) If countdown_text is not None, draw it on top.
    5. If user presses ESC, return user_pressed_esc=True immediately.
    6. If acquire_data=True, record data if bin underfilled.

    Returns:
      display_frame   : The final image (with overlays + text).
      bins_filled     : True if all bins in bin_counts >= B.
      user_pressed_esc: True if user pressed ESC.
    """

    # 1) Get a frame
    raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
    if raw_data is None or not raw_data["has_data"]:
        # Return a blank frame so we can still display something.
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return blank_frame, False, False

    frame = raw_data["frame"].copy()
    landmarks = raw_data["landmarks"]
    head_yaw = raw_data["head_rotation_yaw"]
    head_pitch = raw_data["head_rotation_pitch"]
    head_roll = raw_data["head_rotation_roll"]

    # 2) Draw the calibration dot
    display_frame = frame.copy()
    cv2.circle(display_frame, (x_coord, y_coord), 12, (0, 255, 0), -1)

    # 3) Determine bin indices
    p_idx = get_bin_index(head_pitch, pitch_edges)
    y_idx = get_bin_index(head_yaw,   yaw_edges)
    r_idx = get_bin_index(head_roll,  roll_edges)

    # 4) Overlay the grid (pitch-yaw bins)
    display_frame = draw_bin_overlay(
        base_frame=display_frame,
        x_center=x_coord,
        y_center=y_coord,
        bin_counts=bin_counts,
        B=B,
        pitch_bins=p_bin_num,
        yaw_bins=y_bin_num,
        p_idx_current=p_idx if p_idx is not None else -1,
        y_idx_current=y_idx if y_idx is not None else -1,
        alpha=0.9
    )

    # 5) (LAST) If countdown_text is present, draw it now on top
    if countdown_text is not None:
        cv2.putText(
            display_frame,
            countdown_text,
            (x_coord, y_coord),  # directly on the calibration target
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 255), 2
        )

    # Show the frame, check for ESC
    cv2.imshow("Calibration", display_frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        return display_frame, False, True  # user_pressed_esc=True

    # 6) If we're not acquiring data, just return
    if not acquire_data:
        return display_frame, False, False

    # 7) Otherwise, record data if bin is underfilled
    if (p_idx is not None) and (y_idx is not None) and (r_idx is not None):
        if bin_counts[p_idx, y_idx, r_idx] < B:
            landmarks_list = [(lm.x, lm.y, lm.z) for lm in landmarks]
            row_data = {
                "landmarks": landmarks_list,
                "head_yaw": head_yaw,
                "head_pitch": head_pitch,
                "head_roll": head_roll,
                "calib_point_x": x_coord,
                "calib_point_y": y_coord
            }
            df.loc[len(df)] = row_data
            bin_counts[p_idx, y_idx, r_idx] += 1

    bins_filled = np.all(bin_counts >= B)
    return display_frame, bins_filled, False


def main():
    # -------------------------
    # 1. Configuration
    # -------------------------
    n, m = 3, 3
    screen_w = 1280
    screen_h = 720
    margin = 50

    pitch_config = [0, 9, 3]
    yaw_config   = [-8, 8, 5]
    roll_config  = [-2, 2, 1]

    B = 5
    pitch_start, pitch_end, p_bin_num = pitch_config
    yaw_start,   yaw_end,   y_bin_num = yaw_config
    roll_start,  roll_end,  r_bin_num = roll_config

    pitch_edges = np.linspace(pitch_start, pitch_end, p_bin_num + 1)
    yaw_edges   = np.linspace(yaw_start,   yaw_end,   y_bin_num + 1)
    roll_edges  = np.linspace(roll_start,  roll_end,  r_bin_num + 1)

    gaze_tracker = GazeTracker("config.json")

    columns = [
        "landmarks",
        "head_yaw",
        "head_pitch",
        "head_roll",
        "calib_point_x",
        "calib_point_y"
    ]
    df = pd.DataFrame(columns=columns)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", screen_w, screen_h)

    # 4-second countdown
    countdown_time = 4

    # -------------------------
    # 2. Iterate through each calibration target
    # -------------------------
    for i in range(n):
        for j in range(m):
            if n > 1:
                y_coord = int(margin + i * (screen_h - 2*margin) / (n - 1))
            else:
                y_coord = screen_h // 2

            if m > 1:
                x_coord = int(margin + j * (screen_w - 2*margin) / (m - 1))
            else:
                x_coord = screen_w // 2

            bin_counts = np.zeros((p_bin_num, y_bin_num, r_bin_num), dtype=int)

            # 2.1 Countdown Phase
            countdown_start = time.time()
            while True:
                elapsed = time.time() - countdown_start
                remaining = countdown_time - elapsed
                if remaining <= 0:
                    break

                # Show the UI but do NOT acquire data
                countdown_text = str(int(np.ceil(remaining)))  # e.g. "3", "2", "1"
                _, _, user_pressed_esc = process_frame_for_calibration(
                    gaze_tracker=gaze_tracker,
                    bin_counts=bin_counts,
                    pitch_edges=pitch_edges,
                    yaw_edges=yaw_edges,
                    roll_edges=roll_edges,
                    B=B,
                    df=df,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    p_bin_num=p_bin_num,
                    y_bin_num=y_bin_num,
                    acquire_data=False,  # no data recording yet
                    countdown_text=countdown_text
                )
                if user_pressed_esc:
                    print("User interrupted calibration.")
                    cv2.destroyAllWindows()
                    return

            # 2.2 Data Collection Phase
            while True:
                _, bins_filled, user_pressed_esc = process_frame_for_calibration(
                    gaze_tracker=gaze_tracker,
                    bin_counts=bin_counts,
                    pitch_edges=pitch_edges,
                    yaw_edges=yaw_edges,
                    roll_edges=roll_edges,
                    B=B,
                    df=df,
                    x_coord=x_coord,
                    y_coord=y_coord,
                    p_bin_num=p_bin_num,
                    y_bin_num=y_bin_num,
                    acquire_data=True,  # now store data
                    countdown_text=None
                )
                if user_pressed_esc:
                    print("User interrupted calibration.")
                    cv2.destroyAllWindows()
                    return

                if bins_filled:
                    print(f"Target ({j},{i}) completed.")
                    break

    cv2.destroyAllWindows()
    df.to_csv("calibration_data.csv", index=False)
    print("Calibration data saved to calibration_data.csv")

if __name__ == "__main__":
    main()
