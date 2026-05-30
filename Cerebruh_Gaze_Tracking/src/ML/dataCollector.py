import json
import pandas as pd
import time
import cv2
import mediapipe as mp
import pandas as pd
import sys
import numpy as np
sys.path.append("./src/")

sys.path.append("./")
from src.util import Util
from calibrate import print_text_to_screen
from statistics import median, mean
from facemesh_gazetracker import GazeTracker
from src.ML.feature_extractor import FeatureExtractor

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
RECORD_INTERVAL = 0.02
RECORD_DURATION = 10
TRANSITION_DURATION = 1
GET_READY_TIME = 2
CENTER_END = 5
LEFT_END = 10
RGHT_END = 15
DIRECTIONS = [0, -1, 1]
EMA_ALPHA = 0.3
EMA_LEFT_START = 0.0
EMA_RIGHT_START = 0.0
DF_SAVE_PATH = "./src/ML/trained_models/pipeline_run/training_data.csv"
# COLUMNS = ["left_dir_ratio", "right_dir_ratio", "head_rotation", "direction"]
COLUMNS = ["left_dir_ratio", "right_dir_ratio", "left_vert_ratio", "right_vert_ratio", "head_rotation", "direction"]

def display_black_screen(duration):
                black_screen = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
                start_time = time.time()
                while time.time() - start_time < duration:
                    cv2.imshow("display", black_screen)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

def get_direction_data() -> pd.DataFrame:
    # CODE SNIPPET TAKEN FROM LAM (THANKS LAM)
    # Need this to initialize the camera settings
    data_list = []
    feature_names = []
    label_list = []
    finished = False
    start_calibration = False
    max_time = 3*GET_READY_TIME + 3*RECORD_DURATION + 2*TRANSITION_DURATION
    gaze_tracker = GazeTracker("config.json")
    ema_alpha = EMA_ALPHA
    ema_left = EMA_LEFT_START
    ema_right = EMA_RIGHT_START
    start_time = 0
    while not finished:

        # Now need to initialize the call
        frame = gaze_tracker._get_camera_frame()
        frame = cv2.flip(frame, 1)
        if not start_calibration:
            print_text_to_screen(frame, "Press C to start data collection :)")

        if cv2.waitKey(1) & 0xFF == ord("c"):
            start_calibration = True
            time_of_last_sample = 0
            start_time = time.time()
        # Need to calculate based on the time 
        if start_calibration:
            elapsed_time_since_start = time.time() - start_time
            elasepd_time_since_last_sample = time.time() - time_of_last_sample
            print(elasepd_time_since_last_sample)
            if elasepd_time_since_last_sample < RECORD_INTERVAL:
                continue
            if elapsed_time_since_start >= max_time:
                break
            
            # Display instructions
            if elapsed_time_since_start < GET_READY_TIME:
                print_text_to_screen(frame, "Get ready")
                continue
            if GET_READY_TIME< elapsed_time_since_start < GET_READY_TIME + RECORD_DURATION:
                print_text_to_screen(frame, "Look at the center")
                label = DIRECTIONS[0]
            elif GET_READY_TIME +RECORD_DURATION < elapsed_time_since_start < GET_READY_TIME  + RECORD_DURATION + TRANSITION_DURATION:
                display_black_screen(TRANSITION_DURATION)
                continue
            elif 2*GET_READY_TIME + RECORD_DURATION +TRANSITION_DURATION < elapsed_time_since_start < 2*GET_READY_TIME + 2*RECORD_DURATION+TRANSITION_DURATION: 
                print_text_to_screen(frame, "Look at left")
                label = DIRECTIONS[1]
            elif 2*GET_READY_TIME + 2*RECORD_DURATION+TRANSITION_DURATION< elapsed_time_since_start < 2*GET_READY_TIME + 2*RECORD_DURATION+ 2*TRANSITION_DURATION:
                display_black_screen(TRANSITION_DURATION)
                continue
            elif elapsed_time_since_start > 3*GET_READY_TIME + 2*RECORD_DURATION + 2*TRANSITION_DURATION:
                print_text_to_screen(frame, "Look at right")
                label = DIRECTIONS[2]
            # Now get the desired data points

            raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
            if raw_data["has_data"] == False:
                print("No Raw data available")
                continue
            feature_dict = FeatureExtractor.get_feature_vector_manual_curated(raw_data)
            feature = feature_dict.values()
            feature_names = feature_dict.keys()
            #print("1 row of feature appended")
            data_list.append(feature)
            label_list.append(label)
            time_of_last_sample = time.time()
            

        cv2.imshow("display",frame)
        if cv2.waitKey(3) & 0xFF == ord("q"):
            break
    
    # Now convert the data into a dataframe
    df = pd.DataFrame(data_list, columns=feature_names)
    df["direction"] = label_list
    cv2.destroyAllWindows()
    return df

def collect_data(data_save_path):
    data = get_direction_data()
    data.to_csv(data_save_path, index=False)

if __name__ == "__main__":
    data = get_direction_data()
    data.to_csv(DF_SAVE_PATH, index=False)


