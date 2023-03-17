# importing libraries
import cv2
import numpy as np
import dlib
import tkinter as tk
import os
import time
from pathlib import Path

from settings import DIR
from recognition import Recognition
from eye_blinking import predictor, get_EAR

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

window = tk.Tk()
window.title("Face Recognition")
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(
    window, text="Iran Sign Face Recognition System",
    bg=color_main, fg="white", width=50,
    height=3, font=('times', 30, 'bold'))

message.place(x=100, y=20)

lbl = tk.Label(window, text="National ID Code",
               width=20, height=2, fg="black",
               bg="white", font=('times', 15, ' bold '))

lbl.place(x=350, y=200)

txt = tk.Entry(window,
               width=20, bg="white",
               fg="black", font=('times', 15, ' bold '))

txt.place(x=600, y=215)

lbl2 = tk.Label(window, text="Name",
                width=20, fg="black", bg="white",
                height=2, font=('times', 15, ' bold '))

lbl2.place(x=290, y=300)

txt2 = tk.Entry(window, width=20,
                bg="white", fg="black",
                font=('times', 15, ' bold '))

txt2.place(x=600, y=315)


def TakeImages():
    id_code = (txt.get())
    if len(str(id_code)) != 10:
        message.configure(text='your ID code is invalid!')
    else:
        cam = cv2.VideoCapture(0)
        ret, img = cam.read()
        file_name = DIR.joinpath('files/')
        ids = os.listdir(file_name)
        file_name = file_name.joinpath(id_code + '.jpeg')
        pic_name = id_code + '.jpeg'
        if pic_name in ids:
            message.configure(text='user exist!')
        else:
            cv2.imwrite(str(file_name), img)
            message.configure(text='user added')


def face_recognition():
    name = txt2.get()
    id = (txt.get())

    image_dir = DIR.joinpath('files/' + id + '.jpeg')
    recognition = Recognition()

    embed_src = []
    result = ""
    try:
        im = cv2.imread(str(image_dir))
        embed_src, bbox_src = recognition.detection(im)
    except:
        print('cant read src image from local')

    # Verification and Liveness flags
    flag_verification = False
    flag_liveness = False

    # this is for store similarity between source and detected embed vector
    similarity = []

    # Capture video from API
    src = cv2.VideoCapture(str(DIR.joinpath('face_module/video/faraz.MOV')))
    start = time.time()
    n_frames_read = 0

    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 4
    # initialize the frame counters and the total number of blinks
    counter = 0
    TOTAL = 0

    while src.isOpened():
        now = time.time()
        if (round(now - start)) == 30:
            if flag_verification:
                print(flag_verification)
                result = 'Liveness Failed'

            else:
                result = 'Liveness and Verification Failed'

            break

        # READ FRAMES FROM SRC
        ret, frame = src.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                landmarks = predictor(gray, face_bbox)

                # Calculating left eye aspect ratio
                left_eye_ratio = get_EAR([36, 37, 38, 39, 40, 41], landmarks)

                # Calculating right eye aspect ratio
                right_eye_ratio = get_EAR([42, 43, 44, 45, 46, 47], landmarks)

                # Calculating aspect ratio for both eyes
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                print(blinking_ratio)

                if blinking_ratio < EYE_AR_THRESH:
                    counter += 1
                    # otherwise, the eye aspect ratio is not below the blink threshold
                else:
                    # if the eyes were closed for a sufficient number of then increment the total number of blinks
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        # reset the eye frame counter
                        counter = 0
                        if TOTAL == 5:
                            flag_liveness = True
                            result = "Face Recognition and Liveness Detection have Done Successfully " + "\nHello " + name
                            break
                print(TOTAL)
                cv2.putText(frame, name, (x, y - 10), font_, fontScale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, "number of blink: " + str(TOTAL), (x, y - 40), font_, fontScale, color, thickness,
                            cv2.LINE_AA)
            else:
                result = f'You are not {name}'
                cv2.putText(frame, f'you are not {name}', (x, y - 10), font_, fontScale, color, thickness, cv2.LINE_AA)
                # END Liveness Detection
        else:
            TOTAL = 0
            flag_verification = False

        cv2.imshow('live', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    print('LIVENESS:', flag_liveness)
    print('VERIFICATION', flag_verification)
    src.release()
    cv2.destroyAllWindows()
    message.configure(text=result)

    return flag_liveness, flag_verification


def reset():
    message.configure(text="Face Recognition and Liveness Detection System")


takeImg = tk.Button(window, text="Sign up",
                    command=TakeImages, fg="white", bg=color_main,
                    width=20, height=3, activebackground="Red",
                    font=('times', 15, ' bold '))
takeImg.place(x=300, y=500)

trainImg = tk.Button(window, text="Recognition",
                     command=face_recognition, fg="white", bg=color_main,
                     width=20, height=3, activebackground="Red",
                     font=('times', 15, ' bold '))
trainImg.place(x=600, y=500)

reset_bt = tk.Button(window, text="reset",
                     command=reset, fg="white", bg=color_main,
                     width=20, height=3, activebackground="Red",
                     font=('times', 15, ' bold '))
reset_bt.place(x=900, y=500)

window.mainloop()
