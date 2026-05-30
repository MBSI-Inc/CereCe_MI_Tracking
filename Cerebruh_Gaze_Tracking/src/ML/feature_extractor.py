import numpy as np
import math
import cv2
import sys
import os, sys
# Import module from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import Util


SCREEN_SIZE = (1280, 720)

def get_numpy_points_from_landmarks(frame, landmarks):
    frame_h, frame_w, _ = frame.shape
    pts = []
    for i in range(len(landmarks)):
        x = int(landmarks[i].x * frame_w)
        y = int(landmarks[i].y * frame_h)
        pts.append((x, y))
    pts = np.array(pts, np.int32)
    return pts

def get_centroid(frame, landmarks_np):
    contour = cv2.convexHull(landmarks_np)
    face_surface_area = cv2.contourArea(contour)
    #print("Surface area of minimum contour:", face_surface_area)
    centroid = np.mean(contour, axis=0)[0] 
    #print("Centroid of contour:", centroid)
    cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, (0, 0, 255), -1)
    #cv2.imshow("asda",frame)
    return centroid, face_surface_area

def calculate_distance(pt1, pt2):
    distance = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    return distance

class FeatureExtractor:
    def get_feature_vector(self, raw_data):
        landmarks= raw_data["landmarks"]
        frame = raw_data["frame"]
        pupil_landmark_l = Util.get_iris_position(frame, landmarks, right=False) 
        pupil_landmark_r = Util.get_iris_position(frame, landmarks, right=True)
        frame_h, frame_w, _ = frame.shape

        final_feature = []
        landmarks_np = get_numpy_points_from_landmarks(frame, landmarks)

        # Calculate face height (by getting face contour)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [landmarks_np], (255, 255, 255))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(largest_contour)
        face_height = min_rect[1][1]

        #get dist_to_landmark
        dist_to_landmark = []
        for i in range(len(landmarks_np)):
            dist_to_landmark.append(calculate_distance(pupil_landmark_l, landmarks_np[i]))
        for i in range(len(landmarks_np)):
            dist_to_landmark.append(calculate_distance(pupil_landmark_r, landmarks_np[i]))
        final_feature.extend(dist_to_landmark)

        # get dist_to_landmark_h
        dist_to_landmark_h = []
        for i in range(len(landmarks_np)):
            dist_to_landmark_h.append(abs(pupil_landmark_l[0] - landmarks_np[i][0]))
        for i in range(len(landmarks_np)):
            dist_to_landmark_h.append(abs(pupil_landmark_r[0] - landmarks_np[i][0]))
        final_feature.extend(dist_to_landmark_h)

        # get dist_to_landmark_v
        dist_to_landmark_v = []
        for i in range(len(landmarks_np)):
            dist_to_landmark_v.append(abs(pupil_landmark_l[1] - landmarks_np[i][1]))
        for i in range(len(landmarks_np)):
            dist_to_landmark_v.append(abs(pupil_landmark_r[1] - landmarks_np[i][1]))
        final_feature.extend(dist_to_landmark_v)

        # Rescale these data point to ratio based on face height
        for i in range(len(final_feature)):
            final_feature[i] = final_feature[i] / face_height

        # get head rotation vector
        final_feature.append(raw_data["head_rotation_yaw"])
        final_feature.append(raw_data["head_rotation_pitch"])
        final_feature.append(raw_data["head_rotation_roll"])

        ##get face_area_to_screen_ratio, width_toScreenWidth_ratio, height_toScreenHeight_ratio
        width_ratio = min_rect[1][0] / frame_w
        height_ratio = min_rect[1][1] / frame_h
        area_ratio = width_ratio * height_ratio
        final_feature.append(area_ratio)
        final_feature.append(width_ratio)
        final_feature.append(height_ratio)

        return final_feature
    
    def get_feature_vector_manual_curated(raw_data):
        landmarks= raw_data["landmarks"]
        frame = raw_data["frame"]
        pupil_landmark_l = Util.get_iris_position(frame, landmarks, right=False) 
        pupil_landmark_r = Util.get_iris_position(frame, landmarks, right=True)
        frame_h, frame_w, _ = frame.shape
        noseTip = (int(landmarks[1].x*frame_w), int(landmarks[1].y*frame_h))  # nose
        glabella = (int(landmarks[9].x*frame_w), int(landmarks[9].y*frame_h)) 

        final_feature = []
        numpy_landmark = get_numpy_points_from_landmarks(frame, landmarks)

        # Calculate face height (by getting face contour)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [numpy_landmark], (255, 255, 255))
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        min_rect = cv2.minAreaRect(largest_contour)
        face_height = min_rect[1][1]
        face_width = min_rect[1][0]
        

        centroid, face_surface_area = get_centroid(frame, numpy_landmark)

        # final_feature.extend([
        #     raw_data["left_dir_ratio_h"],
        #     raw_data["left_dir_ratio_v"],
        #     raw_data["right_dir_ratio_h"],
        #     raw_data["right_dir_ratio_v"],
        #     raw_data["eyeslit_h_r"],
        #     raw_data["eyeslit_h_l"],
        #     raw_data["nosebridge_h"],
        #     raw_data["head_rotation_yaw"],
        #     raw_data["head_rotation_pitch"],
        #     raw_data["head_rotation_roll"],
        #     face_surface_area,
        #     face_height,
        #     face_width
        # ])
        # final_feature.extend(centroid)
        # final_feature.extend(noseTip)
        # final_feature.extend(glabella)

        final_feature = {
            "left_dir_ratio_h": raw_data["left_dir_ratio_h"],
            "left_dir_ratio_v": raw_data["left_dir_ratio_v"],
            "right_dir_ratio_h": raw_data["right_dir_ratio_h"],
            "right_dir_ratio_v": raw_data["right_dir_ratio_v"],
            "sum_dir_ratio_h": raw_data["left_dir_ratio_h"]+raw_data["right_dir_ratio_h"],
            "hvProduct_l" : raw_data["left_dir_ratio_h"]*raw_data["left_dir_ratio_v"],
            "hvProduct_r" : raw_data["right_dir_ratio_h"]*raw_data["right_dir_ratio_v"],
            "hvProduct_sum" : (raw_data["left_dir_ratio_h"]+raw_data["right_dir_ratio_h"])*(raw_data["right_dir_ratio_v"]+raw_data["left_dir_ratio_v"]),
            "eyeslit_h_r": raw_data["eyeslit_h_r"],
            "eyeslit_h_l": raw_data["eyeslit_h_l"],
            "nosebridge_h": raw_data["nosebridge_h"],
            "head_rotation_yaw": raw_data["head_rotation_yaw"],
            "head_rotation_pitch": raw_data["head_rotation_pitch"],
            "head_rotation_roll": raw_data["head_rotation_roll"],
            "face_surface_area": face_surface_area,
            "face_height": face_height,
            "face_width": face_width,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "noseTip_x": noseTip[0],
            "noseTip_y": noseTip[1],
            "glabella_x": glabella[0],
            "glabella_y": glabella[1]
        }
        
        # final_feature = {
        #      "left_dir_ratio_h": raw_data["left_dir_ratio_h"],
  
        #     "right_dir_ratio_h": raw_data["right_dir_ratio_h"],

        #     "sum_dir_ratio_h": raw_data["left_dir_ratio_h"]+raw_data["right_dir_ratio_h"],
        #     "head_rotation_yaw": raw_data["head_rotation_yaw"]
        # }

        final_feature = {
            "left_dir_ratio_h": raw_data["left_dir_ratio_h"],
            "right_dir_ratio_h": raw_data["right_dir_ratio_h"],
            "eyeslit_h_r": raw_data["eyeslit_h_r"],
            "eyeslit_h_l": raw_data["eyeslit_h_l"],
            "nosebridge_h": raw_data["nosebridge_h"],
            "head_rotation_yaw": raw_data["head_rotation_yaw"],
            "head_rotation_pitch": raw_data["head_rotation_pitch"],
            "head_rotation_roll": raw_data["head_rotation_roll"]
            # "face_height": face_height,
            # "face_width": face_width,
            # "centroid_x": centroid[0],
            # "centroid_y": centroid[1],
            # "noseTip_x": noseTip[0],
            # "noseTip_y": noseTip[1],
            # "glabella_x": glabella[0],
            # "glabella_y": glabella[1]
        }

        return final_feature
        




