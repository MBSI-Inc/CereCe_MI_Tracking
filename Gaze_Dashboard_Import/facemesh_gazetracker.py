"""
facemesh_gazetracker.py
-----------------------

This module implements a gaze-tracking system using Mediapipe's FaceMesh. It estimates
eye gaze direction, blinks, and head pose, and provides outputs that can be used for
control applications (e.g., integration with big wheelchair, Unity simulator).

Usage
=====
1. Initialization:
   Create a `GazeTracker` instance by passing in the path to a configuration file:
       gaze_tracker = GazeTracker("config.json")

   - `config.json` specifies thresholds, offsets, and options (e.g., debugging,
     unity/scene camera, cruise control).
   - By default, the system initializes the primary webcam (index 0).
   - Optional parameters include:
       - `cam_index`: Which camera to open (default 0).

2. Running the tracker:
   - Synchronous mode:
       output = gaze_tracker.get_gaze()
   - Asynchronous mode:
       gaze_tracker.start()
       ...
       output = gaze_tracker.get_gaze_async()
       ...
       gaze_tracker.stop()

   In asynchronous mode, the tracker runs in a background thread and periodically samples
   gaze data.

3. Getting Gaze Data:
   The main method is `get_gaze()`, which captures a frame, processes landmarks, smooths
   gaze ratios, and detects blinks. It returns a dictionary describing gaze state.

Output Format
=============
The `get_gaze()` function returns a dictionary with the following keys:

    {
        "combined_gaze_horizontal": int,   # -100 (far left) to 100 (far right), 0 = center
        "combined_gaze_vertical": int,     # -100 (backward) to 100 (forward), 0 = idle
        "blinked": bool,                   # True if blink pattern triggered toggle/recalibration
        "face_detected": bool,             # False if no valid face landmarks were found
        "move_enabled": bool               # False if face/head position disables movement
    }

- Horizontal gaze is normalized and scaled relative to thresholds in config.
- Vertical gaze is tied to forward/backward movement or speed scaling.
- Blink detection is used for recalibration or toggling modes.

4. Getting Raw Gaze Data
    The`get_gaze_raw_data_from_camera(frame)` method processes a single video frame (BGR, OpenCV format) and extracts raw
       facial landmark and eye-gaze features using Mediapipe FaceMesh. It does
       not apply smoothing or post-processing, making it useful for debugging
       or advanced downstream processing.

       Returns a dictionary with:
         {
            "has_data": bool,                  # False if no face landmarks detected
            "frame_timestamp": float,          # Time of frame capture
            "frame": np.ndarray,               # Processed image frame with overlays
            "landmarks": list,                 # Mediapipe landmark points
            "left_dir_ratio_h": float,
            "left_dir_ratio_v": float,
            "right_dir_ratio_h": float,
            "right_dir_ratio_v": float,
            "head_rotation_yaw": float,
            "head_rotation_pitch": float,
            "head_rotation_roll": float,
            "eyeslit_h_r": float,
            "eyeslit_h_l": float,
            "nosebridge_h": float
         }

Example Usage
=======
    if __name__ == "__main__":
        gaze_tracker = GazeTracker("config.json")
        while True:
            output = gaze_tracker.get_gaze()
            if output["face_detected"]:
                print(output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        gaze_tracker.deinit()

"""

import time
import cv2
import numpy as np
import pandas as pd
import sys
import mediapipe as mp
import queue
import json
from threading import Thread
from timeit import default_timer as timer

sys.path.append('./src')
from util import Util

# from src.headPoseProcessor import getHeadRotation as get_head_rotation
try:
    from networking.unity_connect import sendToUnity_targetSpeedAndDirection as send_data_to_unity, \
        fetchFrameUnity as fetch_frame_unity, \
        sendToUnity_MoveCommand as sent_unity_move_command, \
        sendValueToUnity as send_value_to_unity, \
        is_unity_connected as is_unity_connected, \
        sendToUnity_MoveCommand as sendToUnity_MoveCommand
except ImportError as e:
    print(f"Failed to import unity packages: {e}")

class GazeTracker:
    # Class constants
    DEFAULT_SPEED = 20
    SPEEDUP_MULTIPLIER = 1.1
    OVERLAY_TRANSPARENCY = 0.1
    TURN_RATE_MULTIPLIER = 80
    TURN_THRESHOLD_LEFT = 0.3
    TURN_THRESHOLD_RIGHT = 0.7
    TURN_DEADZONE_LEFT = 0.4
    TURN_DEADZONE_RIGHT = 0.6
    EMA_ALPHA = 0.2
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    BLINK_COUNT_TO_RECALIBRATE = 3
    MAX_FACE_ANGLE_FOR_CRUISE = 20  # Cruise control only if head yaw is within ±15 degrees

    # Class variables
    cam = None
    cam_scene = None
    config = {}
    debug_flag = True
    value_buffer = {}
    smoothing_type = "EMA"
    blink_min_duration = 0.1
    right_eye_status_record = {}
    left_eye_status_record = {}
    last_recalibrate_time = 0.0
    recalibrate_time_threshold = 3.0
    

    def __init__(self, config_path, cam_index=0, smoothing_type="EMA", no_opencv=False):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        if not no_opencv:
            self.cam = self.set_up_camera(0)
            if self.config["scene_camera"]:
                self.cam = self.set_up_camera(1)
                self.cam_scene = self.set_up_camera(0)



        self.debug_flag = self.config["debug"]
        self.value_buffer = {
            "gaze_horizontal_r": [0],
            "gaze_horizontal_l": [0],
            "gaze_horizontal_all": [0],
            "gaze_vertical": [0],
            "gaze_horizontal_full": [0],
            "head_rotation": [0]
        }
        self.smoothing_type = smoothing_type
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.blink_min_duration = 0.1
        self.eye_blink_status_pd = None
        self.toggle_turn = True
        self.right_eye_status_record = {
            "open_last_frame": True,
            "last_close_time": 0.0,
            "last_blink_times": []
        }
        self.left_eye_status_record = {
            "open_last_frame": True,
            "last_close_time": 0.0,
            "last_blink_times": []
        }
        self.last_recalibrate_time = 0.0
        self.last_frame_time_stamp = 0.0
        self.recalibrate_time_threshold = 1.6
        self.blink_to_recalibrate_gaze_to_center = self.config["blink_to_recalibrate"]
        self.cruise_control_enabled = self.config["cruise_control_enabled"]
        self.current_combined_gaze_horizontal = 0.0
        self.use_boundary = self.config["use_boundary"]
        self.boundary_triggered = False
        self.running = False
        self.async_sampling_interval = self.config["async_sampling_interval"]
        self.output = None
        self.showCVWindow= True
        self.frame_queue = queue.Queue(maxsize=5)
        self.current_control_frame = np.zeros((1920, 1920, 3), dtype=np.uint8)
        self.drag_control_horizontal_gaze = 0
        return

    def deinit(self):
        cv2.destroyAllWindows()
        self.cam.release()
        if self.cam_scene is not None:
            self.cam_scene.release()
        return

    def set_debug(self, debug_flag):
        self.debug_flag = debug_flag

    def EMA_smoothing(self, old_value, new_value):
        return new_value * self.EMA_ALPHA + (1 - self.EMA_ALPHA) * old_value
    
    def adaptive_alpha(self, old_value, new_value, base_alpha=0.5, max_alpha=0.9):
        # Magnitude of change
        delta = abs(new_value - old_value)

        # Normalize change (optional: clip to [0,1])
        scaled_delta = min(delta, 1.0)

        # Adapt alpha: bigger delta → faster update
        return base_alpha + (max_alpha - base_alpha) * scaled_delta
    
    def adaptive_EMA_smoothing(self, old_value, new_value):
        alpha = self.adaptive_alpha(old_value, new_value)
        return new_value * alpha + (1 - alpha) * old_value


    
    def set_up_camera(self, cam_index):
        cam = None
        if sys.platform == "win32":
            cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(cam_index)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, 30)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return cam

    def _get_camera_frame(self):
        _, frame = self.cam.read()

        if not frame is None and self.config['with_black_border']:
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            radius = min(width, height) // 2 
            cv2.circle(mask, center, radius, 255, thickness=-1)
            frame = cv2.bitwise_and(frame, frame, mask=mask)

        return frame

    def _log(self, message, log_level=0):
        if log_level == 0:
            print(message)

    def get_gaze_raw_data_from_camera(self):
        frame = self._get_camera_frame()
        frame = cv2.flip(frame, 1)
        return self.get_gaze_raw_data(frame)

    def get_gaze_raw_data(self, frame):
        """
        Process a single camera frame and extract raw gaze-related features.

        Parameters
        ----------
        frame : np.ndarray
            A BGR image frame (as returned by OpenCV's VideoCapture).

        Returns
        -------
        dict
            Dictionary containing raw gaze and face landmark data:
            {
                "has_data": bool
                    False if no face landmarks are detected.
                "frame_timestamp": float
                    Timestamp when the frame was processed.
                "frame": np.ndarray
                    Frame with optional debug overlays.
                "landmarks": list
                    List of Mediapipe facial landmarks.
                "left_dir_ratio_h": float
                    Horizontal gaze ratio for the left eye.
                "left_dir_ratio_v": float
                    Vertical gaze ratio for the left eye.
                "right_dir_ratio_h": float
                    Horizontal gaze ratio for the right eye.
                "right_dir_ratio_v": float
                    Vertical gaze ratio for the right eye.
                "head_rotation_yaw": float
                    Estimated head yaw angle.
                "head_rotation_pitch": float
                    Estimated head pitch angle.
                "head_rotation_roll": float
                    Estimated head roll angle.
                "eyeslit_h_r": float
                    Horizontal slit size of the right eye (used in blink detection).
                "eyeslit_h_l": float
                    Horizontal slit size of the left eye.
                "nosebridge_h": float
                    Reference nose bridge measurement for normalization.
            }

        Notes
        -----
        - This function returns *raw* measurements only.
        - For higher-level gaze control (with smoothing, calibration, and blink logic),
        use `get_gaze()` instead.
        """

        raw_data = {"has_data": False}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output = self.face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        if not landmark_points:
            self._log("No landmark points detected from camera image")
            return raw_data
        landmarks = landmark_points[0].landmark

        # Get eye direction ratios (horizontal and vertical)
        frame_h, frame_w, _ = frame.shape
        frame, right_dir_ratio_h, right_dir_ratio_v = Util.get_eye_direction_ratio(frame, landmarks, True, self.debug_flag)
        frame, left_dir_ratio_h, left_dir_ratio_v = Util.get_eye_direction_ratio(frame, landmarks, False, self.debug_flag)
        
        frame, head_rotation_yaw, head_rotation_pitch, head_rotation_roll = Util.get_head_rotation(frame, landmarks, frame_w, frame_h)

        ratio, eyeslit_h_r, nosebridge_h = Util.calculate_eye_slit_h_nosebridge_ratio(frame, landmarks, True, debug=self.debug_flag, flip=True)
        ratio, eyeslit_h_l, nosebridge_h = Util.calculate_eye_slit_h_nosebridge_ratio(frame, landmarks, False, debug=self.debug_flag, flip=True)

        raw_data["has_data"] = True
        raw_data["frame_timestamp"] = timer()
        raw_data["frame"] = frame
        raw_data["landmarks"] = landmarks
        raw_data["left_dir_ratio_h"] = left_dir_ratio_h
        raw_data["left_dir_ratio_v"] = left_dir_ratio_v
        raw_data["right_dir_ratio_h"] = right_dir_ratio_h
        raw_data["right_dir_ratio_v"] = right_dir_ratio_v
        raw_data["head_rotation_yaw"] = head_rotation_yaw
        raw_data["head_rotation_pitch"] = head_rotation_pitch
        raw_data["head_rotation_roll"] = head_rotation_roll
        raw_data["eyeslit_h_r"] = eyeslit_h_r
        raw_data["eyeslit_h_l"] = eyeslit_h_l
        raw_data["nosebridge_h"] = nosebridge_h

        return raw_data
    
    def __run(self):
        print("runnnnn")
        while True:
            self.output = self.get_gaze()
            if not self.running:
                break

            time.sleep(self.async_sampling_interval)

    # to run gazetracker as background thread and get data async. Use in conjunction with get_gaze_async()
    def start(self):
        self.showCVWindow = False # can't call cv2.imshow() on a background thread on a mac
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.__run, daemon=False)
            self.thread.start()
            print("thread started")

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()
    
    def get_gaze_async(self):
        return self.output
    
    def get_frame(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            return frame
        
        return None

    def put_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
            
    def display_queued_frame(self):
        cv2.imshow("Gazetracker frame",self.current_control_frame)
        cv2.waitKey(1)
        # frame = self.get_frame()
        # if frame is not None:
        #     cv2.imshow("Gazetracker frame",frame)
        #     cv2.waitKey(1)
        # else:
        #     print("no frame queued")
    
    def default_no_data(self):
        return self.data_to_dict(0,0,False, face_detected= False, move_enabled= False)
        
    def data_to_dict(self,combined_gaze_horizontal, combined_gaze_vertical, blinked, face_detected = True, move_enabled = True):
        output = {
            "combined_gaze_horizontal": combined_gaze_horizontal,
            "combined_gaze_vertical": combined_gaze_vertical,
            "blinked": blinked,
            "face_detected":face_detected,
            "move_enabled" : move_enabled
            }
        return output
    
    def get_gaze(self):
        '''
        Returns 
          output = {
            "combined_gaze_horizontal": int,
            "combined_gaze_vertical": int,
            "blinked": bool,
            "face_detected": bool,
            "move_enabled" : bool
            }
        -100 < combined_gaze_horizontal < 100. Full left = -100, centre = 0, full right = 100
        -100 < combined_gaze_vertical < 100. Full speed forwards =100. No speed = 0. Moving back = 100.
        '''
        raw_data = self.get_gaze_raw_data_from_camera()
        # If no data is detected, reset the blink counters and return immediately.
        if not raw_data["has_data"]:
            self.eye_blink_status_pd = None
            self.right_eye_status_record["last_blink_times"] = []
            self.left_eye_status_record["last_blink_times"] = []
            return self.default_no_data()
        
        frame = raw_data["frame"]
        landmarks = raw_data["landmarks"]
        left_dir_ratio = raw_data["left_dir_ratio_h"]
        right_dir_ratio = raw_data["right_dir_ratio_h"]
        head_rotation_yaw = raw_data["head_rotation_yaw"]
        head_rotation_pitch = raw_data["head_rotation_pitch"]
        eyeslit_h_r = raw_data["eyeslit_h_r"]
        eyeslit_h_l = raw_data["eyeslit_h_l"]
        nosebridge_h = raw_data["nosebridge_h"]

        # Detect eye open/close status
        frame, right_eye_open = Util.detect_eye_open(frame, self.config["right_eye_closed_threshold"], eyeslit_h_r, nosebridge_h, True, flip=True)
        frame, left_eye_open = Util.detect_eye_open(frame, self.config["left_eye_closed_threshold"], eyeslit_h_l, nosebridge_h, False, flip=True)
        current_time = raw_data["frame_timestamp"]
        self.update_blink_status(right_eye_open, left_eye_open, current_time)

        # Update blink status dataframe and calculate derivative
        bTable = self.eye_blink_status_pd
        row_data = [current_time, eyeslit_h_l / nosebridge_h * 100, 0]
        if bTable is None:
            bTable = pd.DataFrame([row_data])
        else:
            bTable = pd.concat([bTable, pd.DataFrame([row_data])], ignore_index=True)
        if len(bTable) >= 2:
            bTable.iat[-1, -1] = (bTable.iat[-1, 1] - bTable.iat[-2, 1]) / (bTable.iat[-1, 0] - bTable.iat[-2, 0])
        if len(bTable) >= 1000:
            bTable = bTable.iloc[-500:]
        self.eye_blink_status_pd = bTable
        dESdt = bTable.iloc[-500:, -1].tolist()
        timestamps = bTable.iloc[-500:, 0].tolist()
        blinkedTrue = False
        blinkedTrue = self.check_recent_blinks(timestamps, dESdt, current_time, time_window=1, min_blinks=2)
        
        #Determines blink toggle true or not
        if blinkedTrue:
            if current_time - self.last_recalibrate_time >= self.recalibrate_time_threshold:
                self.last_recalibrate_time = current_time
                self.toggle_turn = not self.toggle_turn
        
        # Calculate FPS and display on frame
        fps = 1 / (current_time - self.last_frame_time_stamp)
        fps_text = "fps " + str(int(fps))
        cv2.putText(frame, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
        
        # Smooth the horizontal gaze ratios
        if self.smoothing_type == "EMA":
            ema_right = self.EMA_smoothing(self.value_buffer["gaze_horizontal_r"][0], right_dir_ratio)
            self.value_buffer["gaze_horizontal_r"][0] = ema_right
            right_dir_ratio = ema_right
            ema_left = self.EMA_smoothing(self.value_buffer["gaze_horizontal_l"][0], left_dir_ratio)
            self.value_buffer["gaze_horizontal_l"][0] = ema_left
            left_dir_ratio = ema_left
        
        
        # Smooth the vertical gaze ratios
        if self.smoothing_type == "EMA":
            ema_vertical = self.EMA_smoothing(self.value_buffer["gaze_vertical"][0], (raw_data["left_dir_ratio_v"] + raw_data["right_dir_ratio_v"]) / 2)
            self.value_buffer["gaze_vertical"][0] = ema_vertical
            vertical_gaze = ema_vertical
        else:
            vertical_gaze = (raw_data["left_dir_ratio_v"] + raw_data["right_dir_ratio_v"]) / 2

        horizontal_gaze = (left_dir_ratio + right_dir_ratio) / 2

        if self.smoothing_type == "EMA":
            ema_horizontal_gaze = self.adaptive_EMA_smoothing(self.value_buffer["gaze_horizontal_full"][0], horizontal_gaze)
            self.value_buffer["gaze_horizontal_full"][0] = ema_horizontal_gaze
            horizontal_gaze = ema_horizontal_gaze





        
        
        
        # Optionally recalibrate gaze when blinked
        if self.blink_to_recalibrate_gaze_to_center:
            if blinkedTrue and (current_time - self.last_recalibrate_time >= self.recalibrate_time_threshold):
                self.config["center_offset"] = -horizontal_gaze
                self.config["head_rotation_center_offset"] = -head_rotation_yaw
                print("recalibrating")
                self.last_recalibrate_time = current_time
                frame = cv2.circle(frame, (100, 100), 50, (255, 0, 255), -1)
        
        self.last_frame_time_stamp = current_time
        # Recalibrate gaze to centre using key stroke c 
        # if cv2.waitKey(3) == ord("c"):
        #     self.config["center_offset"] = -horizontal_gaze
        #     self.config["head_rotation_center_offset"] = -head_rotation_yaw
        #     print("recalibrating")
        #     self.last_recalibrate_time = current_time

        # Adjust horizontal gaze with center offset and normalize
        horizontal_gaze = horizontal_gaze + self.config["center_offset"]
        horizontal_gaze_scaled = (horizontal_gaze - self.config['left_threshold']) / (self.config['right_threshold'] - self.config['left_threshold'])
        horizontal_gaze_scaled = horizontal_gaze_scaled * (self.TURN_THRESHOLD_RIGHT - self.TURN_THRESHOLD_LEFT) + self.TURN_THRESHOLD_LEFT
        horizontal_gaze = horizontal_gaze_scaled
        
        head_rotation_yaw = head_rotation_yaw + self.config["head_rotation_center_offset"]
        head_rotation_scaled = head_rotation_yaw / 15 + 0.5
        gaze_direction_unadjustedForHeadRotation = horizontal_gaze

        if self.smoothing_type == "EMA":
            ema_head_rotation = self.adaptive_EMA_smoothing(self.value_buffer["head_rotation"][0], head_rotation_scaled)
            self.value_buffer["head_rotation"][0] = ema_head_rotation
            head_rotation_scaled = ema_head_rotation

        horizontal_gaze = horizontal_gaze + head_rotation_scaled - 0.5

        combined_gaze_horizontal = self.determine_direction(horizontal_gaze)
        combined_gaze_horizontal = combined_gaze_horizontal * self.TURN_RATE_MULTIPLIER

        combined_gaze_vertical = self.DEFAULT_SPEED
        

        ##Check if head if the face is roughly centered. If not, stop wheelchair
        if abs(head_rotation_yaw) > self.MAX_FACE_ANGLE_FOR_CRUISE:
            print("Face NOT CENTERED. Stopping wheelchair")
            combined_gaze_vertical = 0
            combined_gaze_horizontal = 0
            horizontal_gaze = 0.5  # reset to center

        
   
        ### Headpose control mode
        if self.config["controller_mode"] == 1:
            combined_gaze_horizontal = (raw_data["head_rotation_yaw"] ) * self.TURN_RATE_MULTIPLIER * 0.5
            combined_gaze_horizontal = self.determine_direction((combined_gaze_horizontal+100)/200.0)*200.0
            combined_gaze_vertical = self.DEFAULT_SPEED + 5 - head_rotation_pitch * 4.8
            horizontal_gaze = head_rotation_scaled
            frame = cv2.putText(frame, str(raw_data["head_rotation_yaw"]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,)
        
        ### Drag control mode
        if self.config["controller_mode"] == 2:
            drag_control_sensitivity = 2
            # if horizontal_gaze > self.TURN_THRESHOLD_RIGHT:
            #      self.drag_control_horizontal_gaze += drag_control_sensitivity
            # if horizontal_gaze < self.TURN_THRESHOLD_LEFT:
            #     self.drag_control_horizontal_gaze -= drag_control_sensitivity
            if combined_gaze_horizontal < -10:
                self.drag_control_horizontal_gaze -= drag_control_sensitivity
            elif combined_gaze_horizontal > 10:
                self.drag_control_horizontal_gaze += drag_control_sensitivity
            self.drag_control_horizontal_gaze = min(max(int(self.drag_control_horizontal_gaze),-100),100)
            frame = Util.draw_pointer(frame, (self.drag_control_horizontal_gaze/200)+0.5, screen_ratio_y=0.4, color=(155, 0, 100), radius = 10) 
            
            print(vertical_gaze)       
            frame = Util.draw_pointer(frame, 0.5, screen_ratio_y=vertical_gaze, color=(0, 0, 100), radius = 10) 
       
       # Update the face camera UI (for debugging)
        frame = self.update_UI(frame, combined_gaze_horizontal, horizontal_gaze, head_rotation_scaled, gaze_direction_unadjustedForHeadRotation)
              
        ### Clamping the values between -100,100
        combined_gaze_horizontal = min(max(int(combined_gaze_horizontal),-100),100)
        combined_gaze_vertical = min(max(int(combined_gaze_vertical),-100),100)
        
        
        # --- Cruise Control Block ---
        if self.cruise_control_enabled and not self.toggle_turn:
            print("toggle on. Cruise control")
            combined_gaze_horizontal = 0
            combined_gaze_vertical = self.DEFAULT_SPEED
       
        if not self.config['unity_camera'] and not self.config['scene_camera']:
            frame = self.update_UI_outputTargetVector(frame, combined_gaze_horizontal, combined_gaze_vertical)
        
        
        # Update the Unity overlay and pass both horizontal and vertical gaze.
        if (self.config['unity_camera'] or self.config['scene_camera']):
            u_frame = self.update_scene_frame(horizontal_gaze, vertical_gaze, frame)
            if u_frame is not None:
                u_frame = self.update_UI_outputTargetVector(u_frame, combined_gaze_horizontal, combined_gaze_vertical)
                self.put_frame(u_frame)
                self.current_control_frame = u_frame
            if self.use_boundary and self.boundary_triggered:
                combined_gaze_horizontal = 0
                combined_gaze_vertical = 0
        
        if not( self.config['unity_camera'] or self.config['scene_camera']):
            ###Save UI frame into a queue so the main thread that intialized the facemesh_gazetracker can retrieve the UI frame and display it from the main thread.
            ### this is done because stupid mac system wouldn't allow a worker thread to call cv2.show(frame) when facemesh_gazetracker is run as async.
            if frame is not None:
                self.put_frame(frame)   
                self.current_control_frame = frame
        
        return self.data_to_dict(combined_gaze_horizontal, combined_gaze_vertical, self.toggle_turn)

    def determine_direction(self, horizontal_gaze):
        turn_rate = 0
        if self.TURN_THRESHOLD_LEFT < horizontal_gaze < self.TURN_THRESHOLD_RIGHT:
            turn_rate = 0
        if ((self.TURN_THRESHOLD_LEFT < horizontal_gaze < self.TURN_DEADZONE_LEFT) or
            (self.TURN_DEADZONE_RIGHT < horizontal_gaze < self.TURN_THRESHOLD_RIGHT)):
            ###set turn rate as exisitng turn rate while gaze is in the dead zone
            turn_rate = self.current_combined_gaze_horizontal
            #print(f"Gaze entered DEADZONE, keeping existing turnrate: {turn_rate}")
        if horizontal_gaze > self.TURN_THRESHOLD_RIGHT:
            turn_rate = 1
        if horizontal_gaze < self.TURN_THRESHOLD_LEFT:
            turn_rate = -1
        
        self.current_combined_gaze_horizontal = turn_rate
        return turn_rate

    def update_UI(self, frame, turn_rate, horizontal_gaze, head_rotation_scaled, uncorrected_gaze_direction):
        frame = Util.write_direction_on_frame(turn_rate, frame)
        control_frame = Util.draw_steering_control_frame(frame, horizontal_gaze,
                                                         self.TURN_THRESHOLD_LEFT, self.TURN_THRESHOLD_RIGHT,
                                                         self.TURN_DEADZONE_LEFT, self.TURN_DEADZONE_RIGHT)
        control_frame = Util.draw_pointer(control_frame, horizontal_gaze)
        control_frame = Util.draw_pointer(control_frame, head_rotation_scaled, 0.6, (255, 0, 0))
        control_frame = Util.draw_pointer(control_frame, uncorrected_gaze_direction, 0.7, (0, 0, 255))
        if self.showCVWindow:
            cv2.imshow("Steering_Control", control_frame)
        return control_frame
    
    def update_UI_outputTargetVector(self, frame, turn_rate, targetSpeed):
        frame = Util.write_direction_on_frame(turn_rate, frame)
        frame_h, frame_w, _ = frame.shape
        start_point = (int(frame_w*0.5), int(frame_h*0.6))
        
        end_point = (int(start_point[0]+(turn_rate/100)*(frame_w/2)), int(start_point[1]-(targetSpeed/100)*(frame_h/2)))
        color = (255, 0, 0)  # Blue color
        thickness = 5
        frame = cv2.arrowedLine(frame, start_point, end_point, color, thickness)
        return frame
    
        

    def update_scene_frame(self, horizontal_gaze, vertical_gaze, frame):
        u_frame = None
        if self.config['unity_camera']:
            try:
                u_frame = fetch_frame_unity()
            except Exception:
                print("Unity not up")
        elif self.config['scene_camera']:
            _, u_frame = self.cam_scene.read()
        else:
            print("logic error, neither unity nor scene camera are being used but update_scene_frame is called")
            
        if u_frame is not None:
            u_frame = Util.draw_steering_control_frame(u_frame, horizontal_gaze,
                                                       self.TURN_THRESHOLD_LEFT, self.TURN_THRESHOLD_RIGHT,
                                                       self.TURN_DEADZONE_LEFT, self.TURN_DEADZONE_RIGHT)
            u_frame = Util.draw_pointer(u_frame, horizontal_gaze)
            # Draw a transparent red boundary on the Unity overlay.
            boundary_thickness = 18  # ~0.5 cm in pixels
            overlay = u_frame.copy()
            red_color = (0, 0, 255)
            cv2.rectangle(overlay, (0, 0), (u_frame.shape[1], boundary_thickness), red_color, -1)
            cv2.rectangle(overlay, (0, u_frame.shape[0] - boundary_thickness), (u_frame.shape[1], u_frame.shape[0]), red_color, -1)
            cv2.rectangle(overlay, (0, 0), (boundary_thickness, u_frame.shape[0]), red_color, -1)
            cv2.rectangle(overlay, (u_frame.shape[1] - boundary_thickness, 0), (u_frame.shape[1], u_frame.shape[0]), red_color, -1)
            alpha = 0.3
            u_frame = cv2.addWeighted(overlay, alpha, u_frame, 1 - alpha, 0)
            
            # Compute pointer position based on horizontal and vertical gaze.
            pointer_x = int(horizontal_gaze * u_frame.shape[1])
            pointer_y = int(vertical_gaze * u_frame.shape[0])
            self.boundary_triggered = False
            
            # Check horizontal boundaries
            if pointer_x < boundary_thickness:
                self.boundary_triggered = True
                center_y = u_frame.shape[0] // 2
                pts = np.array([
                    [boundary_thickness, center_y],
                    [boundary_thickness + 20, center_y - 20],
                    [boundary_thickness + 20, center_y + 20]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(u_frame, [pts], red_color)
            elif pointer_x > (u_frame.shape[1] - boundary_thickness):
                self.boundary_triggered = True
                center_y = u_frame.shape[0] // 2
                pts = np.array([
                    [u_frame.shape[1] - boundary_thickness, center_y],
                    [u_frame.shape[1] - boundary_thickness - 20, center_y - 20],
                    [u_frame.shape[1] - boundary_thickness - 20, center_y + 20]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(u_frame, [pts], red_color)
            
            # Check vertical boundaries
            if pointer_y < boundary_thickness:
                self.boundary_triggered = True
                center_x = u_frame.shape[1] // 2
                pts = np.array([
                    [center_x, boundary_thickness],
                    [center_x - 20, boundary_thickness + 20],
                    [center_x + 20, boundary_thickness + 20]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(u_frame, [pts], red_color)
            elif pointer_y > (u_frame.shape[0] - boundary_thickness):
                self.boundary_triggered = True
                center_x = u_frame.shape[1] // 2
                pts = np.array([
                    [center_x, u_frame.shape[0] - boundary_thickness],
                    [center_x - 20, u_frame.shape[0] - boundary_thickness - 20],
                    [center_x + 20, u_frame.shape[0] - boundary_thickness - 20]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(u_frame, [pts], red_color)
            
            u_frame = Util.cam_overlay(u_frame, frame)
            if self.showCVWindow:
                cv2.imshow("Unityfetchertester", u_frame)
            return u_frame

    def update_blink_status(self, right_eye_open, left_eye_open, current_time=None):
        if current_time is None:
            current_time = timer()
        self.right_eye_status_record = self.detect_blink(self.right_eye_status_record, right_eye_open, current_time)
        self.left_eye_status_record = self.detect_blink(self.left_eye_status_record, left_eye_open, current_time)
        return 
    
    def detect_blink(self, eye_status_record, current_eye_open, current_time):
        if eye_status_record["open_last_frame"] and not current_eye_open:
            eye_status_record["last_close_time"] = current_time
        if not eye_status_record["open_last_frame"] and current_eye_open:
            if (current_time - eye_status_record["last_close_time"]) > self.blink_min_duration:
                eye_status_record["last_blink_times"].append(eye_status_record["last_close_time"])
        eye_status_record["open_last_frame"] = current_eye_open
        return eye_status_record
    
    def get_blink_count_over_elasped_time(self, eye_status_record, current_time, elapsed_time):
        elapsed_time_start = current_time - elapsed_time
        blink_count = 0
        last_blink_times = eye_status_record["last_blink_times"]
        for i in range(len(last_blink_times)):
            if last_blink_times[len(last_blink_times) - i - 1] > elapsed_time_start:
                blink_count += 1
            else:
                break
        return blink_count

    def check_recent_blinks(self, timestamps, dES_dt, current_time, time_window=2, neg_thresh=-30, pos_thresh=30, min_blinks=2):
        start_index = 0
        for i in range(len(timestamps) - 1, -1, -1):
            if timestamps[i] <= current_time - time_window:
                start_index = i + 1
                break

        recent_timestamps = timestamps[start_index:]
        recent_dES_dt = dES_dt[start_index:]
        
        blinks_count = 0
        found_neg = False
        
        for i in range(len(recent_dES_dt)):
            if recent_dES_dt[i] < neg_thresh:
                found_neg = True
            elif found_neg and recent_dES_dt[i] > pos_thresh:
                blinks_count += 1
                found_neg = False
            if blinks_count >= min_blinks:
                return True

        return False
    

if __name__ == '__main__':
    unity_connected = False
    gaze_tracker = GazeTracker("config.json")
    moving = True
    while True:
        # Check for unity connection
        if gaze_tracker.config['unity_camera']:
            while not unity_connected:
                print("Checking for Unity connection...")
                if is_unity_connected():
                    print("Unity connection established!")
                    unity_connected = True
                    break
                print("Unity not connected. Retrying in 2 seconds...")
                time.sleep(2)  # Wait before retrying
        # Get gaze data 
        output = gaze_tracker.get_gaze()
        if output is None or output["face_detected"] is False:
            continue
        print(output)
        if gaze_tracker.config['unity_camera']:
            send_data_to_unity(output["combined_gaze_vertical"], output["combined_gaze_horizontal"])
            if(output["blinked"]!=moving):
                sendToUnity_MoveCommand()
                moving = not moving
                
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    gaze_tracker.deinit()