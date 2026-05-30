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
from src.ML.gaze_prediction_model import GazePredictionModel

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
RECORD_INTERVAL = 0.02
RECORD_DURATION = 5
TRANSITION_DURATION = 1
GET_READY_TIME = 2
CENTER_END = 5
LEFT_END = 10
RGHT_END = 15
DIRECTIONS = [0, -1, 1]
EMA_ALPHA = 0.3
EMA_LEFT_START = 0.0
EMA_RIGHT_START = 0.0
DF_SAVE_PATH = "./src/ML/data/test_data.csv"
# COLUMNS = ["left_dir_ratio", "right_dir_ratio", "head_rotation", "direction"]
COLUMNS = ["left_dir_ratio", "right_dir_ratio", "left_vert_ratio", "right_vert_ratio", "head_rotation", "direction"]


def test_model_realtime(gaze_tracker, gaze_model) -> pd.DataFrame:
    # CODE SNIPPET TAKEN FROM LAM (THANKS LAM)
    # Need this to initialize the camera settings
    data_list = []
    feature_names = []
    label_list = []
    finished = False
    start_calibration = False
    max_time = 3*GET_READY_TIME + 3*RECORD_DURATION + 2*TRANSITION_DURATION
    
    ema_alpha = EMA_ALPHA
    ema_left = EMA_LEFT_START
    ema_right = EMA_RIGHT_START
    start_time = 0
    while not finished:

        # Now need to initialize the call
        frame = gaze_tracker._get_camera_frame()
        frame = cv2.flip(frame, 1)
        
        raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
        if raw_data["has_data"] == False:
            print("No Raw data available")
            continue
        feature_dict = FeatureExtractor.get_feature_vector_manual_curated(raw_data)
        feature_names = feature_dict.keys()
        feature_values = list(feature_dict.values()) 
        feature = pd.DataFrame([feature_values], columns=feature_names)
        feature = feature[gaze_model.feature_names]
        #print(feature)
        
        predictions = gaze_model.predict(feature)
        prediction = predictions[0]
        print("Prediction: "+ str(prediction))
        #time.sleep(2)
        height, width, _ = frame.shape
        height = int( height/2)
        if prediction == -1:
            cv2.rectangle(frame, (0, 0), (width // 3, height), (0, 255, 0, 0.5), -1)  # Add translucent red rectangle to left 1/3 of the frame
        elif prediction == 0:
            cv2.rectangle(frame, (width // 3, 0), (2 * width // 3, height), (0, 255, 0, 0.5), -1)  # Add translucent green rectangle to centre 1/3 of the frame
        elif prediction == 1:
            cv2.rectangle(frame, (2 * width // 3, 0), (width, height), (0, 255, 0, 0.5), -1)  # Add translucent blue rectangle to right 1/3 of the frame
        
        cv2.imshow("display",frame)
        if cv2.waitKey(3) & 0xFF == ord("q"):
            break
    
    # Now convert the data into a dataframe
    df = pd.DataFrame(data_list, columns=feature_names)
    df["direction"] = label_list
    return df

if __name__ == "__main__":
    gaze_model = GazePredictionModel("/Users/joshuachung/Coding/MBSI/Cerebruh_Gaze_Tracking/src/ML/trained_models/lr_model_manual_curated_features_3")
    gaze_tracker = GazeTracker("config.json")
    data = test_model_realtime(gaze_tracker, gaze_model)


