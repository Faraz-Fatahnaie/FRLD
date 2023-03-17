import cv2
import numpy as np
import mediapipe as mp


def head_pose(frame):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    results = face_mesh.process(frame)
    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []
    result = ''

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x_c, y_c = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x_c, y_c])

                    # Get the 3D Coordinates
                    face_3d.append([x_c, y_c, lm.z])

                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x_a = angles[0] * 360
            y_a = angles[1] * 360

            # print(y)

            # See where the user's head tilting
            if y_a < -10:
                result = "Left"
            elif y_a > 10:
                result = "Right"
            elif x_a < -10:
                result = "Down"
            else:
                result = "Forward"

            # Display the nose direction
            # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
            #                                                 dist_matrix)

            # p1 = (int(nose_2d[0]), int(nose_2d[1]))
            # p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

            # cv2.line(frame, p1, p2, (255, 0, 0), 2)
    return result
