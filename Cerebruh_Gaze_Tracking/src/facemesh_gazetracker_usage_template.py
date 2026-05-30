from CereCe_Integration.Cerebruh_Gaze_Tracking.src.facemesh_gazetracker import GazeTracker
import cv2

if __name__ == '__main__':
    gaze_tracker = GazeTracker("config.json")
    while True:
        raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
        if raw_data == None or not raw_data["has_data"]:
            continue
        
        frame = raw_data["frame"]
        landmarks = raw_data["landmarks"]
        
        cv2.imshow("Face",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gaze_tracker.deinit()
