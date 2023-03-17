import cv2
import numpy as np
import dlib
import time

from settings import DIR
from recognition import Recognition
from eye_blinking import predictor, get_EAR
from head_pose import head_pose

# font
font_ = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (0, 255, 0)
# Line thickness of 2 px
thickness = 2
color_main = 'blue'


def face_recognition(instructions: list = None, eye_blink: int = None, time_limit: int = 50):

    id = '0021219958'
    image_dir = DIR.joinpath('files/' + id + '.jpeg')
    recognition = Recognition()

    embed_src = []
    try:
        im = cv2.imread(str(image_dir))
        embed_src, bbox_src = recognition.detection(im)
    except:
        print('cant read src image from local')

    # Define Flags
    flag_verification = False
    flag_liveness = False
    flag_eye_blink = False
    flag_head_pose = False

    # this is for store similarity between source and detected embed vector
    similarity = []

    # for store head positions of user
    head_poses = []

    # Capture video from API
    src = cv2.VideoCapture(str(DIR.joinpath('face_module/video/vid.mp4')))
    start = time.time()

    # parameters related to eye blink counting process
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 4
    counter = 0
    TOTAL = 0

    result = 'System Failed!'
    time_taken = 0

    while src.isOpened():
        now = time.time()

        # READ FRAMES FROM SRC
        ret, frame = src.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        colored = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detection and Calculate embed vector
        embed, bbox = recognition.detection(frame)
        if len(bbox) != 0:

            # dlib predictor works with own bounding box format, so we do this
            x, y, x1, y1 = bbox.astype(np.int32)
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            face_bbox = dlib.rectangle(x, y, x1, y1)

            # START Verification Based on Similarity between source(embed_src) and Detected(embed) embed vector
            sim = recognition.verification(embed_src, embed)

            if float(sim) > 0.5:

                flag_verification = True
                similarity.append(sim)

                # START Liveness Detection Using EyeBlinkDetector
                # Creating an obj in which we will store detected facial landmarks which calc by dlib predictor
                if eye_blink is not None and not flag_eye_blink:

                    landmarks = predictor(gray, face_bbox)

                    # Calculating left eye aspect ratio
                    left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)

                    # Calculating right eye aspect ratio
                    right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)

                    # Calculating aspect ratio for both eyes
                    blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                    # print(blinking_ratio)

                    if blinking_ratio < EYE_AR_THRESH:
                        counter += 1
                        # otherwise, the eye aspect ratio is not below the blink threshold
                    else:
                        # if the eyes were closed for a sufficient number of then increment the total number of blinks
                        if counter >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                            # reset the eye frame counter
                            counter = 0
                            if TOTAL == eye_blink:
                                flag_eye_blink = True
                    # print(TOTAL)
                    cv2.putText(frame, id, (x, y - 10), font_, fontScale, color, thickness, cv2.LINE_AA)
                    cv2.putText(frame, "number of blink: " + str(TOTAL), (x, y - 40), font_, fontScale, color,
                                thickness,
                                cv2.LINE_AA)

                # START Liveness Detection Using Head Position Detection
                if instructions is not None and not flag_head_pose:
                    head_position = head_pose(colored)
                    if len(head_poses) > 0:
                        if head_position != head_poses[-1]:
                            head_poses.append(head_position)
                            if head_poses == instructions:
                                flag_head_pose = True
                    else:
                        head_poses.append(head_position)
                    cv2.putText(frame, head_position, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                result = f'You are not {id}'
                cv2.putText(frame, result, (x, y - 10), font_, fontScale, color, thickness, cv2.LINE_AA)
                # END Liveness Detection
        else:
            TOTAL = 0
            flag_verification = False

        if (instructions is not None) and (eye_blink is None):
            flag_liveness = flag_head_pose
        if (instructions is not None) and (eye_blink is not None):
            flag_liveness = flag_head_pose & flag_eye_blink
        if (instructions is None) and (eye_blink is not None):
            flag_liveness = flag_eye_blink

        if flag_verification and flag_liveness:
            result = "Face Recognition and Liveness Detection have Done Successfully"
            break

        time_taken = round(now - start)
        if time_taken == time_limit:
            if flag_verification:
                result = 'Liveness Failed'
            else:
                result = 'Liveness and Verification Failed'
            break

        cv2.imshow('live', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print('LIVENESS:', flag_liveness)
    print('VERIFICATION:', flag_verification)
    print('HEAD POSITIONS:', head_poses)
    print('NUMBER OF BLINKS:', TOTAL)
    print('TIME TAKEN:', time_taken, ' sec')
    print('FINAL RESULT', result)
    src.release()
    cv2.destroyAllWindows()

    return flag_liveness, flag_verification


if __name__ == "__main__":
    # TODO: compute similarity of instruction and what user has done / put threshold on this similarity
    instruction = ['Forward', 'Left', 'Forward', 'Right']
    face_recognition(instructions=instruction, eye_blink=10, time_limit=50)
