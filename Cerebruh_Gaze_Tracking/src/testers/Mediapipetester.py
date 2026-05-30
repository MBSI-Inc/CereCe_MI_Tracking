from facemesh_gazetracker import GazeTracker
import cv2
from src.util import Util


if __name__ == '__main__':
    gaze_tracker = GazeTracker("config.json")
    while True:
        raw_data = gaze_tracker.get_gaze_raw_data_from_camera()
        if raw_data == None or not raw_data["has_data"]:
            continue
        
        frame = raw_data["frame"]
        landmarks = raw_data["landmarks"]
        print(landmarks)
        print(landmarks[0])
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, [474])
        pts_3d = Util.get_raw_numpy_points_from_landmarks_3d(frame, landmarks, [474])
        for idx,point in enumerate(pts):
                x,y = point
                z = pts_3d[idx][2]
                print(z)
                cv2.circle(frame, tuple((x,y)), 2, (255, 255, 255), -1)
                cv2.putText(frame, f"{round(x, 2)} | {round(y, 2)}| {round(z, 4)}", (350, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        cv2.imshow("Face",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gaze_tracker.deinit()
