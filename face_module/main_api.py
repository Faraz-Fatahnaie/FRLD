from face_module.recognition import Recognition
from face_module.eye_blinking import predictor, get_EAR
import cv2
import numpy as np
import dlib
from face_module.settings import DIR


def face_reco_api(video, image):
    image_dir = DIR.joinpath(image)
    video_dir = DIR.joinpath(video)

    recognition = Recognition()

    embed_src = []
    bbox_src = []
    try:
        im = cv2.imread(str(image_dir))
        embed_src, bbox_src = recognition.detection(im)
    except:
        print('cant read src_image from local')

    # these variables related to EyeBlinkDetector, Creating a list eye_blink_signal
    eye_blink_signal = []

    # Creating an object blink_ counter
    blink_counter = 0
    previous_ratio = 100

    # Verification and Liveness flags
    flag_verification = False
    flag_liveness = False

    # this is for store similarity between source and detected embed vector
    similarity = []

    # Capture video from API
    src = cv2.VideoCapture(str(video_dir))

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    # initialize the frame counters and the total number of blinks
    counter = 0
    TOTAL = 0

    n_frames_read = 0
    while src.isOpened():

        # read frames from src
        ret, frame = src.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detection and Calculate embed vector
        embed, bbox = recognition.detection(frame)

        if len(bbox) != 0:

            # dlib predictor works with own bounding box format, so we do this
            x, y, x1, y1 = bbox.astype(np.int32)
            face_bbox = dlib.rectangle(x, y, x1, y1)

            # START Verification Based on Similarity between source(embed_src) and Detected(embed) embed vector
            sim = recognition.verification(embed_src, embed)

            if float(sim) > 0.5:

                flag_verification = True
                similarity.append(sim)

                # START Liveness Detection Using EyeBlinkDetector
                # Creating an obj in which we will store detected facial landmarks which calc by dlib predictor
                landmarks = predictor(gray, face_bbox)

                # Calculating left eye aspect ratio
                left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)

                # Calculating right eye aspect ratio
                right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)

                # Calculating aspect ratio for both eyes
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                # print(right_eye_ratio)

                # Rounding blinking_ratio on two decimal places
                blinking_ratio_1 = blinking_ratio * 100
                blinking_ratio_2 = np.round(blinking_ratio_1)
                blinking_ratio_rounded = blinking_ratio_2 / 100
                # print(blinking_ratio_rounded)
                blinking_ratio_3 = blinking_ratio_2 ** 3
                # print(blinking_ratio_2)
                # print(blinking_ratio_3)
                #print(blinking_ratio)

                # Appending blinking ratio to a list eye_blink_signal
                eye_blink_signal.append(blinking_ratio)

                if blinking_ratio < EYE_AR_THRESH:
                    counter += 1
                    # otherwise, the eye aspect ratio is not below the blink
                    # threshold
                else:
                    # if the eyes were closed for a sufficient number of
                    # then increment the total number of blinks
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        # reset the eye frame counter
                        counter = 0
                        if TOTAL == 5:
                            flag_liveness = True
                            break
                #print(TOTAL)
                # END Liveness Detection

    return flag_liveness, flag_verification


if __name__ == '__main__':

    image_path = './files/0021219958.jpeg'
    video_path = './face_module/video/IMG_0086.MOV'
    flag_live, flag_verify = face_reco_api(video_path, image_path)
    print('live:', flag_live, '\nverify:', flag_verify)
