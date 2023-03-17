import numpy as np
from face_module.settings import LANDMARK_PREDICTOR
import dlib

predictor = dlib.shape_predictor(LANDMARK_PREDICTOR)


# Defining the mid-point
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# Defining the Euclidean distance
def euclidean_distance(leftx, lefty, rightx, righty):
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


# Defining the eye aspect ratio
def get_EAR(eye_points, facial_landmarks):
    # Defining the left point of the eye
    left_point = [facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    # Defining the right point of the eye
    right_point = [facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
    # Defining the top mid-point of the eye
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    # Defining the bottom mid-point of the eye
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    # Drawing horizontal and vertical line
    # hor_line = cv2.line(frame, (left_point[0], left_point[1]), (right_point[0], right_point[1]), (255, 0, 0), 3)
    # ver_line = cv2.line(frame, (center_top[0], center_top[1]), (center_bottom[0], center_bottom[1]), (255, 0, 0), 3)
    # Calculating length of the horizontal and vertical line
    hor_line_length = euclidean_distance(left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_length = euclidean_distance(center_top[0], center_top[1], center_bottom[0], center_bottom[1])
    # Calculating eye aspect ratio
    eye_aspect_ratio = ver_line_length / hor_line_length
    return eye_aspect_ratio
