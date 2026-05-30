import numpy as np
import cv2


class Util:
    # Remove some landmarks in the eye corners so the bounding box is tighter
    # RIGHT_EYE_LANDMARKS = [160, 33, 161, 163, 133, 7,
    #                        173, 144, 145, 246, 153, 154, 155, 157, 158, 159]
    RIGHT_UPPER_EYELID_LANDMARKS = [160, 161, 157, 158, 159]
    RIGHT_LOWER_EYELID_LANDMARKS = [163, 144, 145, 153, 154]
    RIGHT_EYE_LANDMARKS = RIGHT_UPPER_EYELID_LANDMARKS+RIGHT_LOWER_EYELID_LANDMARKS
    RIGHT_IRIS_LANDMARKS = [472, 469, 470, 471]
    # LEFT_EYE_LANDMARKS = [384, 385, 386, 387, 388, 390,
    #                       263, 362, 398, 466, 373, 374, 249, 380, 381, 382]
    LEFT_UPPER_EYELID_LANDMARKS = [384, 385, 386, 387, 388]
    LEFT_LOWER_EYELID_LANDMARKS = [390, 373, 374, 380, 381]
    LEFT_EYE_LANDMARKS = LEFT_UPPER_EYELID_LANDMARKS + LEFT_LOWER_EYELID_LANDMARKS

    LEFT_IRIS_LANDMARKS = [474, 475, 476, 477]
    NOSE_LANDMARKS = [1, 9]
    OVERLAY_TRANSPARENCY = 0.1
    DEFAULT_TURN_RATIO_THRESHOLD_LEFT = 0.3
    DEFAULT_TURN_RATIO_THRESHOLD_RIGHT = 0.7
    DEFAULT_TURN_RATIO_DEADZONE_LEFT = 0.4
    DEFAULT_TURN_RATIO_DEADZONE_RIGHT = 0.6

    @staticmethod
    def determine_direction(pupil_x, eye_x_low, eye_x_high):
        direction_ratio = (pupil_x - eye_x_low) / (eye_x_high - eye_x_low)

        # Convert from [0, 1] range into [-1, 1] range so it's
        # easier to use in Unity.
        direction_ratio = (2 * direction_ratio) - 1

        return direction_ratio

    def VEC_getProjectionVec(u, v):
        # return the projection of u on v
        proj = (np.dot(u, v)/np.dot(v, v)).reshape(-1, 1)*v
        return proj

    def VEC_getDistanceVec(u, v):
        # returns the distance between two points u and v. u and v are numpy arrays of size (2,)
        return int(np.linalg.norm(v - u))

    def VEC_getCentroid(points):
        # takes a numpy array of (n,2) and returns the 2D average (aka centroid) of those points.
        mean_x = int(np.mean(points[:, 0]))
        mean_y = int(np.mean(points[:, 1]))
        average_point = np.array([mean_x, mean_y])
        return average_point

    def get_raw_numpy_points_from_landmarks_3d(frame, landmarks, points_of_interest):
        # returns a (n,2) numpy array of the coordinates of the n points of interest
        frame_h, frame_w, _ = frame.shape
        pts = []
        for i in points_of_interest:
            x = landmarks[i].x 
            y = landmarks[i].y 
            z = landmarks[i].z
            print("{0}, {1}, {2}".format(landmarks[i].x,landmarks[i].y,landmarks[i].z))
            pts.append((x, y, z))
        pts = np.array(pts, np.float32)
        return pts
    def get_numpy_points_from_landmarks(frame, landmarks, points_of_interest):
        # returns a (n,2) numpy array of the coordinates of the n points of interest
        frame_h, frame_w, _ = frame.shape
        pts = []
        for i in points_of_interest:
            x = int(landmarks[i].x * frame_w)
            y = int(landmarks[i].y * frame_h)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        return pts

    def get_direction_test(frame, landmarks, right=False, debug=False):
        frame_h, frame_w, _ = frame.shape
        # get the nose bridge landmarks
        c1 = (int(landmarks[1].x*frame_w), int(landmarks[1].y*frame_h))  # nose
        c2 = (int(landmarks[9].x*frame_w), int(landmarks[9].y*frame_h))  # centre of glabella
        face_centre_x = int((c1[0]+c2[0])/2)
        face_centre_y = int((c1[1]+c2[1])/2)
        cv2.circle(frame, (face_centre_x, face_centre_y), 5, (0, 255, 255), 2)
        cv2.line(frame, c1, c2, (255, 255, 0), 3)

        # get eye corner landmarks
        eye_corner_landmarks = [263, 359, 446, 353, 342, 261, 448, 249, 466,  33, 130, 247, 25, 113, 226, 124, 35]
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, eye_corner_landmarks)

        # Calculate projection onto nose bridge
        origin = np.array(c1)
        vec_origin_to_pts = pts-origin
        vec_origin_to_head = np.array(c2)-origin
        projection_onto_centreLine = Util.VEC_getProjectionVec(vec_origin_to_pts, vec_origin_to_head) + origin
        projection_onto_centreLine = np.round(projection_onto_centreLine).astype(int)

        for i in range(pts.shape[0]):
            point_eye = pts[i]
            point_proj = projection_onto_centreLine[i]
            distance = Util.VEC_getDistanceVec(point_eye, point_proj)
            distance = eye_corner_landmarks[i]
            color = (255, 0, 0)
            cv2.line(frame, point_eye, point_proj, color, 1)
            cv2.circle(frame, point_eye, 2, (0, 0, 255), 1)
            cv2.putText(frame, str(distance), (point_eye[0]+10, point_eye[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # for i in eye_corner_landmarks:
        #     x = int(landmarks[i].x * frame_w)
        #     y = int(landmarks[i].y * frame_h)
        #     x_dist = abs(x-face_centre_x)
        #     color = (int(max(0,(255-x_dist))),122,min(255,int(x_dist)))
        #     #color = (0,0,min(255,int(x_dist)))
        #     cv2.line(frame, (face_centre_x,y), (x,y),color,1)
        #     cv2.circle(frame, (x,y), 2,(0,0,255),1)
        #     cv2.putText(frame, str(x_dist), (x+10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # iris
        iris_landmarks = Util.LEFT_IRIS_LANDMARKS
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, iris_landmarks)
        center, radius = cv2.minEnclosingCircle(pts)
        center = tuple(map(int, center))
        radius = int(radius/2)
        iris = np.array(center)
        vec_origin_to_iris = iris-origin
        projection_onto_centreLine = Util.VEC_getProjectionVec(vec_origin_to_iris, vec_origin_to_head)+origin
        projection_onto_centreLine = np.round(projection_onto_centreLine).astype(int)

        point_proj = projection_onto_centreLine[0]
        distance = Util.VEC_getDistanceVec(iris, point_proj)

        cv2.circle(frame, center, 5, (0, 255, 0), 2)
        cv2.line(frame, center, point_proj, color, 1)
        cv2.putText(frame, str(distance), (center[0]+10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return frame
    
    @staticmethod
    def get_iris_position(frame, landmarks, right = False):
        ### returns a tuple of (x,y) coordinates of the iris
        if (right):
            iris_landmarks = Util.RIGHT_IRIS_LANDMARKS
        else:
            iris_landmarks = Util.LEFT_IRIS_LANDMARKS
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, iris_landmarks)
        center, radius = cv2.minEnclosingCircle(pts)
        center = tuple(map(int, center))
        radius = int(radius/2)
        return center


    @staticmethod
    def get_eye_direction_ratio(frame, landmarks, right=False, debug=False):
        ###
        # right = is this the right eye (true) or left eye (false).
        # returns {frame, direction}, where frame is is the face cam image with eyes and iris box annotated on top.
        #
        # direction ranges -1~1, where 0 indicates iris is at the horizontal centre of the eye box,
        # -1 means iris cenre is at the eye-box's left-most boundary (as in user is looking all the way left, biologically this will never happen)
        ###
        if (right):
            eye_landmarks = Util.RIGHT_EYE_LANDMARKS
            iris_landmarks = Util.RIGHT_IRIS_LANDMARKS
        else:
            eye_landmarks = Util.LEFT_EYE_LANDMARKS
            iris_landmarks = Util.LEFT_IRIS_LANDMARKS

        # Eye
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, eye_landmarks)
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)  # Draw rectangle around eye

        if debug:
            # Draw all eye landmarks
            hull_indices = cv2.convexHull(pts, returnPoints=False)
            for point in pts:
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1)
            hull_points = [pts[index] for index in hull_indices]
            cv2.drawContours(frame, [np.array(hull_points)], 0, (255, 255, 255), 1)

        eye_x_low = x
        eye_x_high = x+w

        # Iris
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, iris_landmarks)
        center, radius = cv2.minEnclosingCircle(pts)
        center = tuple(map(int, center))
        radius = int(radius/2)
        cv2.circle(frame, center, radius, (0, 0, 255), 1)  # Draw circle around iris


        ### Vertical eye direction ratio
        frame_h, frame_w, _ = frame.shape
        # Getting information about nosebridge
        noseTip = (int(landmarks[1].x*frame_w), int(landmarks[1].y*frame_h))  # nose
        glabella = (int(landmarks[9].x*frame_w), int(landmarks[9].y*frame_h))  # centre of glabella
        if debug:
            cv2.line(frame, noseTip, glabella, (255, 255, 0), 3)
        noseTip = np.array(noseTip)
        glabella = np.array(glabella)
        n1 = Util.VEC_getDistanceVec(noseTip, glabella)

        # now incorporate with y
        curr_eye = np.array((center[0] - glabella[0], center[1] - glabella[1]))
        # Get the current vector from glabella and nose bridge
        line_vec = (noseTip[0] - glabella[0], noseTip[1] - glabella[1])
        line_vec = np.array(line_vec)
        # Now get the projection
        eye_to_vector = Util.VEC_getProjectionVec(curr_eye, line_vec)
        eye_y = Util.VEC_getDistanceVec(eye_to_vector[0], np.array((0, 0)))/n1  # [(x, y)]

        h_direction_ratio = Util.determine_direction(center[0], eye_x_low, eye_x_high)
        v_direction_ratio = eye_y
        # print(f"CURRENT DIR RATIO: {v_direction_ratio}")

        return frame, h_direction_ratio, v_direction_ratio
    
    def get_head_rotation(frame, landmarks, frame_w, frame_h):
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(landmarks):
            if idx in [33, 263, 1, 61, 291, 199]:
                if idx == 1:
                    nose_2d = (lm.x * frame_w, lm.y * frame_h)
                    nose_3d = (lm.x * frame_w, lm.y * frame_h, lm.z * 3000)
                x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        focal_length = 1 * frame_w
        cam_matrix = np.array([[focal_length, 0, frame_h / 2],
                            [0, focal_length, frame_w / 2],
                            [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x_angle = angles[0] * 360
        y_angle = angles[1] * 360
        z_angle = angles[2] * 360
        
        if y_angle < -10:
            text = "Looking Left"
        elif y_angle > 10:
            text = "Looking Right"
        elif x_angle < -10:
            text = "Looking Down"
        elif x_angle > 10:
            text = "Looking Up"
        else:
            text = "Forward"

        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
        cv2.line(frame, p1, p2, (255, 0, 0), 3)
        cv2.putText(frame, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        pitch = x_angle
        yaw = y_angle
        roll = z_angle
        return frame, yaw, pitch, roll

    def get_eye_detailed_variables(frame, landmarks, frame_w, frame_h, right=False, debug=False):
        if (right):
            eye_landmarks = Util.RIGHT_EYE_LANDMARKS
            iris_landmarks = Util.RIGHT_IRIS_LANDMARKS
        else:
            eye_landmarks = Util.LEFT_EYE_LANDMARKS
            iris_landmarks = Util.LEFT_IRIS_LANDMARKS

        # Eye
        pts = []
        for i in eye_landmarks:
            x = int(landmarks[i].x * frame_w)
            y = int(landmarks[i].y * frame_h)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)  # Draw rectangle around eye

        if debug:
            # Draw all eye landmarks
            hull_indices = cv2.convexHull(pts, returnPoints=False)
            for point in pts:
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1)
            hull_points = [pts[index] for index in hull_indices]
            cv2.drawContours(frame, [np.array(hull_points)], 0, (255, 255, 255), 1)

        eye_x_low = x
        eye_x_high = x+w

        # Iris
        pts = []
        for i in iris_landmarks:
            x = int(landmarks[i].x * frame_w)
            y = int(landmarks[i].y * frame_h)
            pts.append((x, y))
        pts = np.array(pts, np.int32)
        center, radius = cv2.minEnclosingCircle(pts)
        center = tuple(map(int, center))
        radius = int(radius/2)
        cv2.circle(frame, center, radius, (0, 0, 255), 1)  # Draw circle around iris

        direction_ratio = Util.determine_direction(center[0], eye_x_low, eye_x_high)

        return frame, direction_ratio, eye_x_low, eye_x_high, center[0]

    # Calculate openess of an eye as a percentage ratio between [height between the upper and lower eyelid] and [length of nose bridge]
    def calculate_eye_slit_h_nosebridge_ratio(frame, landmarks, right_eye=False, flip=False, debug=False):
        # This is to deal with some frame are flipped like mirror
        if flip:
            right_eye = not right_eye

        frame_h, frame_w, _ = frame.shape
        # Getting information about nosebridge
        c1 = (int(landmarks[1].x*frame_w), int(landmarks[1].y*frame_h))  # nose
        c2 = (int(landmarks[9].x*frame_w), int(landmarks[9].y*frame_h))  # centre of glabella
        if debug:
            cv2.line(frame, c1, c2, (255, 255, 0), 3)
        c1 = np.array(c1)
        c2 = np.array(c2)
        nosebridge_h = Util.VEC_getDistanceVec(c1, c2)

        # Getting indices of landmarks of interest
        if right_eye:
            eye_landmarks = Util.RIGHT_EYE_LANDMARKS
            upper_eyelid_landmarks = Util.RIGHT_UPPER_EYELID_LANDMARKS
            lower_eyelid_landmarks = Util.RIGHT_LOWER_EYELID_LANDMARKS
        else:
            eye_landmarks = Util.LEFT_EYE_LANDMARKS
            upper_eyelid_landmarks = Util.LEFT_UPPER_EYELID_LANDMARKS
            lower_eyelid_landmarks = Util.LEFT_LOWER_EYELID_LANDMARKS

        # Eye
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, eye_landmarks)
        if debug:
            # Draw all eye landmarks
            hull_indices = cv2.convexHull(pts, returnPoints=False)
            for point in pts:
                cv2.circle(frame, tuple(point), 2, (255, 255, 255), -1)
            hull_points = [pts[index] for index in hull_indices]
            cv2.drawContours(frame, [np.array(hull_points)], 0, (255, 255, 255), 1)

        ###
        # Gets the centroid of upper and lower eyelid, calculate distance between those two points.
        # This works better than just using the height of a rectangular box surrounding eye because
        # as soon as head rotates side to side the measurements would be off (since the bounding rectangle
        # is always parallel to the screen)
        ###
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, upper_eyelid_landmarks)
        upper_eyelid_centroid = Util.VEC_getCentroid(pts)
        pts = Util.get_numpy_points_from_landmarks(frame, landmarks, lower_eyelid_landmarks)
        lower_eyelid_centroid = Util.VEC_getCentroid(pts)
        if debug:
            # draws the line connecting upper eyelid to lower eyelid
            cv2.line(frame, upper_eyelid_centroid, lower_eyelid_centroid, (0, 255, 255), 3)
        eyeSlit_h = Util.VEC_getDistanceVec(upper_eyelid_centroid, lower_eyelid_centroid)
        ratio =  eyeSlit_h/nosebridge_h *100
        return ratio, eyeSlit_h, nosebridge_h

    # Detect whether an eye is open by comparing the ratio between [height between the upper and lower eyelid] and [length of nose bridge]
    def detect_eye_open(frame, threshold, eyeslit_h, nosebridge_h, right_eye=False, flip=False, debug=False):
        OFFSET = 1
        threshold += OFFSET # Make it a bit easier
        ratio = eyeslit_h / nosebridge_h * 100
        eye_open = True
        if ratio < threshold:
            eye_open = False
        if right_eye:
            cv2.putText(frame, f"{round(threshold, 2)} | {round(ratio, 2)}", (350, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"R:{eye_open}", (350, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            cv2.putText(frame, f"{round(threshold, 2)} | {round(ratio, 2)}", (200, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"L:{eye_open}", (200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame, eye_open

    # Provide a 0<screen_ratio_x<1 to display a point on the screen, where 0 is left edge and 1 is right edge
    def draw_pointer(frame, screen_ratio_x, screen_ratio_y=0.5, color=(0, 255, 0),radius = 5):
        frame_h, frame_w, _ = frame.shape
        center = (round(frame_w * screen_ratio_x), round(frame_h * screen_ratio_y))
        frame = cv2.circle(frame, center, radius, color, -1)
        return frame
    
    def cam_overlay(u_frame, frame): #Grab User Camera and Overlay it top right of Unity Simulation
        u_frame = cv2.cvtColor(u_frame, cv2.COLOR_BGRA2BGR)
        frame_h, frame_w, _= frame.shape
        cam_ratio = frame_h/frame_w
        u_frame_h, u_frame_w, _ = u_frame.shape
        x_offset = round(u_frame_w-0.25*u_frame_w)
        y_offset = 0
        frame_resized = cv2.resize(frame, (round(0.25*u_frame_w), round(0.25*u_frame_w*cam_ratio)))
        y1, y2 = y_offset, y_offset + frame_resized.shape[0]
        x1, x2 = x_offset, x_offset + frame_resized.shape[1]
        alpha = 0
        beta = 1
        blended_frame = cv2.addWeighted(u_frame[y1:y2, x1:x2], alpha, frame_resized, beta, 0)
        u_frame[y1:y2, x1:x2] = blended_frame
        return(u_frame)




    def get_cursor_position_window(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            global cursor
            cursor = [x, y]
            return

    def write_direction_on_frame(direction, frame):
        if direction < 0:
            cv2.putText(frame, "left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,)
        elif direction > 0:
            cv2.putText(frame, "right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        return frame

    # takes a left and right turn ratio threshold (0-1, 0.5 being center), and a gaze direction (0-1)
    @staticmethod
    def draw_control_panel_overlay(input_image, gaze_direction, 
                                   turn_ratio_threshold_left=DEFAULT_TURN_RATIO_THRESHOLD_LEFT, 
                                   turn_ratio_threshold_right=DEFAULT_TURN_RATIO_THRESHOLD_RIGHT,
                                   turn_ratio_deadzone_left = DEFAULT_TURN_RATIO_DEADZONE_LEFT,
                                   turn_ratio_deadzone_right = DEFAULT_TURN_RATIO_DEADZONE_RIGHT):
        # Create control panel the same size as input image to overlay on top
        control_panel = np.zeros_like(input_image, np.uint8)

        input_h, input_w, input_c = input_image.shape
        left_border_x = int(input_w*turn_ratio_threshold_left)
        right_border_x = int(input_w*turn_ratio_threshold_right)
        left_deadzone_x = int(input_w*turn_ratio_deadzone_left)
        right_deadzone_x = int(input_w*turn_ratio_deadzone_right)

        # Draw lines to divide the panel into 3
        # Left turn border line
        cv2.line(control_panel,  # Image
                 (left_border_x, 0),  # Start point
                 (left_border_x, input_h),  # End point
                 (255, 255, 255),  # color = white
                 3  # Thickness
                 )
        # Right turn border line
        cv2.line(control_panel,  # Image
                 (right_border_x, 0),  # Start point
                 (right_border_x, input_h),  # End point
                 (255, 255, 255),  # color = white
                 3  # Thickness
                 )
        
        # Deadzones
        cv2.line(control_panel,  # Image
            (left_deadzone_x, 0),  # Start point
            (left_deadzone_x, input_h),  # End point
            (255, 255, 255),  # color = white
            3  # Thickness
            )

        cv2.line(control_panel,  # Image
            (right_deadzone_x, 0),  # Start point
            (right_deadzone_x, input_h),  # End point
            (255, 255, 255),  # color = white
            3  # Thickness
            )

        # Arrows and text for UX (turn directions)
        cv2.arrowedLine(control_panel,
                        (left_border_x-100, input_h//2),
                        (left_border_x-270, input_h//2),
                        (255, 255, 255),
                        10,
                        tipLength=0.8
                        )
        cv2.arrowedLine(control_panel,
                        (right_border_x+100, input_h//2),
                        (right_border_x+270, input_h//2),
                        (255, 255, 255),
                        10,
                        tipLength=0.8
                        )
        # Add a rectangle to highlight left/right column
        if gaze_direction < turn_ratio_threshold_left:
            cv2.rectangle(control_panel,  # Image
                          (0, 0),  # Start point
                          (left_border_x, input_h),  # End point
                          (255, 255, 255),  # color = white
                          -1  # Thickness -1 to fill
                          )
        elif gaze_direction > turn_ratio_threshold_right:
            cv2.rectangle(control_panel,  # Image
                          (right_border_x, 0),  # Start point
                          (input_w, input_h),  # End point
                          (255, 255, 255),  # color = white
                          -1  # Thickness -1 to fill
                          )
        elif gaze_direction < turn_ratio_deadzone_left and gaze_direction > turn_ratio_threshold_left:
            cv2.rectangle(control_panel,  # Image
                          (left_border_x, 0),  # Start point
                          (left_deadzone_x, input_h),  # End point
                          (255, 255, 255),  # color = white
                          -1  # Thickness -1 to fill
                          )
        elif gaze_direction > turn_ratio_deadzone_right and gaze_direction < turn_ratio_threshold_right:
            cv2.rectangle(control_panel,  # Image
                          (right_deadzone_x, 0),  # Start point
                          (right_border_x, input_h),  # End point
                          (255, 255, 255),  # color = white
                          -1  # Thickness -1 to fill
                          )
        # Else, do nothing (TODO: Discuss if we want a stop panel)

        # Return overlay image
        return control_panel

    def draw_steering_control_frame(input_image, direction, 
                                    turn_threshold_left, turn_threshold_right, 
                                    turn_deadzone_left, turn_deadzone_right,
                                    transparency=OVERLAY_TRANSPARENCY):
        # Draw control panel overlay
        control_panel = Util.draw_control_panel_overlay(
            input_image, direction, turn_threshold_left, turn_threshold_right,
            turn_deadzone_left, turn_deadzone_right)

        # Piece together the steering control image
        steering_frame = cv2.addWeighted(control_panel, transparency, input_image, 1-transparency, 0)
        return steering_frame
