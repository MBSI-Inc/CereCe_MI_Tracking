import cv2
import numpy as np
import pandas as pd
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
    Given a 'value' and a sorted array of bin edges (e.g. [-2, -1, 0, 1, 2]),
    return the bin index where 'value' falls. We treat intervals as:
       (bin_edges[i], bin_edges[i+1]]  for i in [0..len(bin_edges)-2].
    If the 'value' is below bin_edges[0], clamp it to bin 0.
    If the 'value' is above bin_edges[-1], clamp it to bin (len(bin_edges)-2).

    Example:
      bin_edges = [-2, -1, 0, 1, 2]  # 4 intervals
      # Indices:  0: (-2, -1]
      #           1: (-1,  0]
      #           2: ( 0,  1]
      #           3: ( 1,  2]

    Returns:
      i (int)   : bin index from 0 to len(bin_edges)-2 inclusive.
    """
    # Handle below the first bin edge => clamp to bin 0
    if value <= bin_edges[0]:
        return 0

    # Handle above the last bin edge => clamp to the final bin
    if value > bin_edges[-1]:
        return len(bin_edges) - 2

    # Otherwise, find the interval
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] < value <= bin_edges[i+1]:
            return i

    # Fallback (should never happen if bin_edges is sorted and above checks are correct)
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
    Draw a translucent grid overlay of size `grid_size` x `grid_size`.
    Rows = pitch bins, Columns = yaw bins.

    1) Each cell's fill color interpolates from RED (0% progress) to GREEN (100% progress),
       reflecting how many samples have been collected across ALL roll bins.
       - If sum_collected = 0 => color is fully RED.
       - If sum_collected = B * r_bin_num => color is fully GREEN.
       - Values in between are a linear blend.

    2) Higher pitch corresponds to the TOP of the grid. We invert the row index:
         render_row = (pitch_bins - 1) - p_idx
       so that p_idx=0 (lowest pitch) is at the bottom, p_idx=(pitch_bins-1) (highest pitch) is at the top.

    3) If (p_idx == p_idx_current) and (y_idx == y_idx_current),
       draw a BLUE border (thickness=2) around that cell, preserving its fill color.

    4) Grid lines are drawn in white.

    Parameters:
    - base_frame      : The image to draw on.
    - x_center, y_center : Center of the grid (usually the calibration target).
    - bin_counts      : Shape (pitch_bins, yaw_bins, roll_bins), tracking counts.
    - B               : How many samples per bin are needed.
    - pitch_bins      : Number of pitch bins.
    - yaw_bins        : Number of yaw bins.
    - p_idx_current   : The user's current pitch bin index (if valid).
    - y_idx_current   : The user's current yaw bin index (if valid).
    - alpha           : Blending factor for overlay (0 -> transparent, 1 -> opaque).
    - grid_size       : Size in pixels of the square overlay grid.

    Returns:
      out_frame: A copy of `base_frame` with the overlay drawn.
    """
    overlay = base_frame.copy()

    cell_w = grid_size / yaw_bins
    cell_h = grid_size / pitch_bins

    # Number of roll bins
    _, _, r_bin_num = bin_counts.shape
    total_needed = B * r_bin_num

    # Grid center at (x_center, y_center)
    grid_left = int(x_center - grid_size // 2)
    grid_top  = int(y_center - grid_size // 2)

    def interpolate_red_green(fraction):
        """
        fraction=0 => red=(0, 0, 255),
        fraction=1 => green=(0, 255, 0).
        """
        fraction = max(0.0, min(1.0, fraction))
        red_bgr   = np.array([0,   0, 255], dtype=float)
        green_bgr = np.array([0, 255,   0], dtype=float)
        interp_bgr = (1 - fraction) * red_bgr + fraction * green_bgr
        return interp_bgr.astype(int).tolist()

    # 1) Fill each cell
    for p_idx in range(pitch_bins):
        for y_idx in range(yaw_bins):
            sum_collected = bin_counts[p_idx, y_idx, :].sum()
            fraction = sum_collected / total_needed

            color_bgr = interpolate_red_green(fraction)

            # Invert pitch -> higher pitch at top
            render_row = (pitch_bins - 1) - p_idx

            x0 = int(grid_left + y_idx * cell_w)
            y0 = int(grid_top + render_row * cell_h)
            x1 = int(x0 + cell_w)
            y1 = int(y0 + cell_h)

            cv2.rectangle(overlay, (x0, y0), (x1, y1), color_bgr, -1)

    # 2) Draw white grid lines
    for row in range(pitch_bins + 1):
        render_row = (pitch_bins - row)
        y_line = int(grid_top + render_row * cell_h)
        cv2.line(overlay, (grid_left, y_line),
                 (grid_left + grid_size, y_line), (255, 255, 255), 1)

    for col in range(yaw_bins + 1):
        x_line = int(grid_left + col * cell_w)
        cv2.line(overlay, (x_line, grid_top),
                 (x_line, grid_top + grid_size), (255, 255, 255), 1)

    # 3) Draw a BLUE border for current bin if valid
    if (0 <= p_idx_current < pitch_bins) and (0 <= y_idx_current < yaw_bins):
        render_row_current = (pitch_bins - 1) - p_idx_current
        x0 = int(grid_left + y_idx_current * cell_w)
        y0 = int(grid_top + render_row_current * cell_h)
        x1 = int(x0 + cell_w)
        y1 = int(y0 + cell_h)

        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), 2)

    # 4) Blend overlay
    out_frame = cv2.addWeighted(overlay, alpha, base_frame, 1 - alpha, 0)
    return out_frame


def main():
    # -------------------------
    # 1. Configuration
    # -------------------------
    # Grid for calibration: n rows, m columns
    n, m = 3, 3  # example: 3x3 grid

    # Screen / Window dimensions
    screen_w = 1280
    screen_h = 720
    margin = 50

    # Pose bin configurations [start, end, bin_count]
    pitch_config = [0, 9, 3]   # e.g. 3 pitch bins across [0..9]
    yaw_config   = [-8, 8, 5]  # e.g. 5 yaw bins across [-8..8]
    roll_config  = [-2, 2, 1]  # e.g. 1 roll bin across [-2..2]

    # B: how many samples to collect per bin for each calibration target
    B = 5

    # Prepare bin edges
    pitch_start, pitch_end, p_bin_num = pitch_config
    yaw_start,   yaw_end,   y_bin_num = yaw_config
    roll_start,  roll_end,  r_bin_num = roll_config

    pitch_edges = np.linspace(pitch_start, pitch_end, p_bin_num + 1)
    yaw_edges   = np.linspace(yaw_start,   yaw_end,   y_bin_num + 1)
    roll_edges  = np.linspace(roll_start,  roll_end,  r_bin_num + 1)

    # Create GazeTracker instance
    gaze_tracker = GazeTracker("config.json")

    # DataFrame to store collected samples
    columns = [
        "landmarks",       # list of (x, y, z) for 478 landmarks
        "head_yaw",
        "head_pitch",
        "head_roll",
        "calib_point_x",
        "calib_point_y"
    ]
    df = pd.DataFrame(columns=columns)

    # Create the calibration window
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", screen_w, screen_h)

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

            # Reset bin counts for this target
            bin_counts = np.zeros((p_bin_num, y_bin_num, r_bin_num), dtype=int)

            while True:
                # Acquire raw data
                raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
                if raw_data is None or not raw_data["has_data"]:
                    continue

                frame = raw_data["frame"]
                landmarks = raw_data["landmarks"]
                head_yaw = raw_data["head_rotation_yaw"]
                head_pitch = raw_data["head_rotation_pitch"]
                head_roll = raw_data["head_rotation_roll"]

                # Draw the calibration target
                display_frame = frame.copy()
                cv2.circle(display_frame, (x_coord, y_coord), 12, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    f"Calib ({j},{i})",
                    (x_coord - 45, y_coord - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1
                )

                # Determine bin indices
                p_idx = get_bin_index(head_pitch, pitch_edges)
                y_idx = get_bin_index(head_yaw,   yaw_edges)
                r_idx = get_bin_index(head_roll,  roll_edges)

                # Draw the bin overlay
                display_frame = draw_bin_overlay(
                    base_frame=display_frame,
                    x_center=x_coord,
                    y_center=y_coord,
                    bin_counts=bin_counts,
                    B=B,
                    pitch_bins=p_bin_num,
                    yaw_bins=y_bin_num,
                    p_idx_current=p_idx,
                    y_idx_current=y_idx,
                    alpha=0.9
                )

                cv2.imshow("Calibration", display_frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESC to quit early
                    print("User interrupted calibration.")
                    cv2.destroyAllWindows()
                    return

                # If valid bin indices, record data if under-filled
                if (p_idx is not None and y_idx is not None and r_idx is not None):
                    if bin_counts[p_idx, y_idx, r_idx] < B:
                        # Record data
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

                    # Check if all bins are filled
                    if np.all(bin_counts >= B):
                        print(f"Target ({j},{i}) completed.")
                        break

    # Finished collecting all targets
    cv2.destroyAllWindows()

    # Save to CSV (or another format)
    df.to_csv("calibration_data.csv", index=False)
    print("Calibration data saved to calibration_data.csv")


if __name__ == "__main__":
    main()
