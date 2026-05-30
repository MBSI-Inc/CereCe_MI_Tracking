"""
This script demonstrates how to run gazetracker in async mode and display the queued frame.
"""
import time
import cv2 
from facemesh_gazetracker import GazeTracker
if __name__ == '__main__':
    unity_connected = False
    gaze_tracker = GazeTracker("config.json")
    gaze_tracker.start()
    while True:
        output = gaze_tracker.get_gaze_async()
        if output is None or output["face_detected"] is False:
            continue
        print(output)
        gaze_tracker.display_queued_frame()
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    gaze_tracker.deinit()