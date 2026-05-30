import json
import time
import cv2
import mediapipe as mp
from sys import platform
from src.util import Util
from statistics import median, mean
from playsound import playsound

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DURATION_EACH_STEP = 3
GET_READY_TIME = 2

play_sound_progress = -1 # To avoid same sound played multiple times

def print_text_to_screen(frame, text):
    cv2.putText(frame, text, (350, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA,)


def play_sound(filename, sound_id):
    global play_sound_progress
    if play_sound_progress < sound_id:
        start_audio_time = time.time()
        playsound(filename)
        play_sound_progress = sound_id
        elapsed_audio_time = time.time() - start_audio_time
        print("PLAY_SOUND", filename, "FOR", elapsed_audio_time, "s")
        return elapsed_audio_time
    return 0

def do_calibration(data):
    finished = False
    cam = None
    if platform == "win32":
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    ema_alpha = 0.3  # smoothing factor
    ema_left = 0
    ema_right = 0

    cv2.namedWindow("Camera")

    start_time = None
    start_calibration = False

    DIRECTION_BORDER_RATIO_LEFT = 0.1  # the x ratio of the left turn border on screen 0~1 
    DIRECTION_BORDER_RATIO_CENTER = 0.5
    DIRECTION_BORDER_RATIO_RIGHT = 0.9 # the x ratio of the right turn border on screen 0~1 

    left_closed_data = []
    right_closed_data = []
    center_offset_data = []
    left_threshold_data = []
    right_threshold_data = []

    total_audio_time = 0

    while not finished:
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            frame, right_direction_ratio = Util.get_eye_direction_ratio(frame, landmarks, True)
            ema_right = (ema_alpha * right_direction_ratio) + ((1 - ema_alpha) * ema_right)
            right_direction_ratio = ema_right

            frame, left_direction_ratio = Util.get_eye_direction_ratio(frame, landmarks, False)
            ema_left = (ema_alpha * left_direction_ratio) + ((1 - ema_alpha) * ema_left)
            left_direction_ratio = ema_left

            if not start_calibration:
                print_text_to_screen(frame, "press C to start calibration")
                play_sound("resource/audio/press_c_calibrate.mp3", 0)

            if cv2.waitKey(1) & 0xFF == ord("c"):
                start_calibration = True
                start_time = time.time()

            direction = (left_direction_ratio + right_direction_ratio )/2
            if start_calibration:
                elapsed_time = time.time() - start_time - total_audio_time
                
                # Center
                if (elapsed_time > GET_READY_TIME and
                        elapsed_time < GET_READY_TIME + DURATION_EACH_STEP):
                    print_text_to_screen(frame, "Look at center")
                    audio_time = play_sound("resource/audio/look_at_center.mp3", 1)
                    # total_audio_time += audio_time
                    # center = (round(frame_w * DIRECTION_BORDER_RATIO_CENTER), round(frame_h * 0.4))
                    # radius = 1 - ((elapsed_time - GET_READY_TIME) / DURATION_EACH_STEP)
                    # radius = round(15 * radius)
                    # cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    # data["left_offset"] = -left_direction_ratio
                    # data["right_offset"] = -right_direction_ratio
                    center_offset_data.append(-direction)
                    data["center_offset"] = median(center_offset_data)

                # Left
                if (elapsed_time > GET_READY_TIME + DURATION_EACH_STEP and
                        elapsed_time < GET_READY_TIME + DURATION_EACH_STEP * 2):
                    print_text_to_screen(frame, "Look at left")
                    audio_time = play_sound("resource/audio/look_at_left.mp3", 2)
                    total_audio_time += audio_time
                    # center = (round(frame_w * DIRECTION_BORDER_RATIO_LEFT), round(frame_h * 0.4))
                    # radius = 1 - ((elapsed_time - (GET_READY_TIME + DURATION_EACH_STEP)) / DURATION_EACH_STEP)
                    # radius = round(15 * radius)
                    # cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    # data["left_comfortable"] = left_direction_ratio
                    # data['left_comfortable_with_offset'] = left_direction_ratio + data['left_offset']
                    left_threshold_data.append(data["center_offset"] + direction)
                    data["left_threshold_screenless"] = median(left_threshold_data)
                # Right
                if (elapsed_time > GET_READY_TIME + DURATION_EACH_STEP * 2 and
                        elapsed_time < GET_READY_TIME + DURATION_EACH_STEP * 3):
                    print_text_to_screen(frame, "Look at right")
                    audio_time = play_sound("resource/audio/look_at_right.mp3", 3)
                    total_audio_time += audio_time
                    # center = (round(frame_w * DIRECTION_BORDER_RATIO_RIGHT), round(frame_h * 0.4))
                    # radius = 1 - ((elapsed_time - (GET_READY_TIME + DURATION_EACH_STEP * 2)) / DURATION_EACH_STEP)
                    # radius = round(15 * radius)
                    # cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    # data["right_comfortable"] = right_direction_ratio
                    # data['right_comfortable_with_offset'] = right_direction_ratio + data['right_offset']
                    right_threshold_data.append(data["center_offset"] + direction)
                    data["right_threshold_screenless"] = median(right_threshold_data)

                if (elapsed_time > GET_READY_TIME + DURATION_EACH_STEP * 3 and
                        elapsed_time < GET_READY_TIME * 2 + DURATION_EACH_STEP * 3):
                    audio_time = play_sound("resource/audio/look_at_center.mp3", 4)
                    total_audio_time += audio_time

                # Close left eye, open right eye
                # We also give ready time because we need some time for
                # user to read the instruction
                if (elapsed_time > GET_READY_TIME * 2 + DURATION_EACH_STEP * 3 and
                        elapsed_time < GET_READY_TIME * 2 + DURATION_EACH_STEP * 4):
                    print_text_to_screen(frame, "Close left eye, open right eye")
                    audio_time = play_sound("resource/audio/close_left_open_right.mp3", 5)
                    total_audio_time += audio_time
                    # center = (round(frame_w * DIRECTION_BORDER_RATIO_CENTER), round(frame_h * 0.4))
                    # radius = 1 - ((elapsed_time - (GET_READY_TIME * 2 + DURATION_EACH_STEP * 3)) / DURATION_EACH_STEP)
                    # radius = round(15 * radius)
                    # cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    ratio, _, _ = Util.calculate_eye_slit_h_nosebridge_ratio(frame, landmarks, right_eye=False, debug=True, flip=True)
                    ratio = round(ratio, 2)
                    left_closed_data.append(ratio)
                    # data["left_eye_closed_threshold"] = median(left_closed_data)
                    
                # Close right eye, open left eye
                if (elapsed_time > GET_READY_TIME * 3 + DURATION_EACH_STEP * 4 and
                        elapsed_time < GET_READY_TIME * 3 + DURATION_EACH_STEP * 5):
                    print_text_to_screen(frame, "Close right eye, open left eye")
                    audio_time = play_sound("resource/audio/close_right_open_left.mp3", 6)
                    total_audio_time += audio_time
                    # center = (round(frame_w * DIRECTION_BORDER_RATIO_CENTER), round(frame_h * 0.4))
                    # radius = 1 - ((elapsed_time - (GET_READY_TIME * 3 + DURATION_EACH_STEP * 4)) / DURATION_EACH_STEP)
                    # radius = round(15 * radius)
                    # cv2.circle(frame, center, radius, (0, 255, 0), -1)
                    ratio, _, _ = Util.calculate_eye_slit_h_nosebridge_ratio(frame, landmarks, right_eye=True, debug=True, flip=True)
                    ratio = round(ratio, 2)
                    right_closed_data.append(ratio)
                    # data["right_eye_closed_threshold"] = median(right_closed_data)
                    # data["closed_eye_threshold"] = (data["left_eye_closed_threshold"] + data["right_eye_closed_threshold"]) / 2
                    

                # Exit
                if (elapsed_time > GET_READY_TIME * 3 + DURATION_EACH_STEP * 5 + 1):
                    left_closed_data.sort()
                    right_closed_data.sort()
                    print("Left close eye thresh range: ", left_closed_data[0], " ", left_closed_data[-1], " median: ", median(left_closed_data), " mean: ", mean(left_closed_data))
                    print("Right close eye thresh range: ", right_closed_data[0], " ", right_closed_data[-1], " median: ", median(right_closed_data), " mean: ", mean(right_closed_data))
                    data["left_eye_closed_threshold"] = median(left_closed_data)
                    data["right_eye_closed_threshold"] = median(right_closed_data)
                    print("Done")
                    play_sound("resource/audio/finish_calibrate.mp3", 7)
                    break

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    return data


def main():
    data = {}

    # Load JSON data from file
    with open('config.json', 'r') as f:
        data = json.load(f)

    data = do_calibration(data)

    # Save the modified JSON data back to the file
    with open('config.json', 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
