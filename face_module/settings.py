from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import configparser
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DIR = Path(__file__).resolve().parent.parent
# print(DIR)


_conf = configparser.ConfigParser()
_conf.read(BASE_DIR.joinpath("configuration/dep.ini"))

# start, configuration
MODEL_CONF = _conf['Model']
DETECTOR_CONF = _conf['Detector']
IMAGE_CONF = _conf["Image"]
DEFAULT_CONF = _conf['Default']

# end, configuration

IMAGE_SHAPE = (
    int(IMAGE_CONF["height"]),
    int(IMAGE_CONF["width"]),
    int(IMAGE_CONF["channel"])
)

CTR_CROP_SHAPE = (
    int(IMAGE_CONF["ctr_crop_h"]),
    int(IMAGE_CONF["ctr_crop_w"])
)

RESIZE_SHAPE = (
    int(IMAGE_CONF["resize_width"]),
    int(IMAGE_CONF["resize_height"])
)

EMBED_SIZE = int(MODEL_CONF['embedding'])

MARGINS = (
    int(DETECTOR_CONF["left_margin"]),
    int(DETECTOR_CONF["right_margin"]),
    int(DETECTOR_CONF["up_margin"]),
    int(DETECTOR_CONF["down_margin"]),

)

FACENET_PATH = BASE_DIR.joinpath(MODEL_CONF["facenet"])
ARC_FACE_PATH = BASE_DIR.joinpath(MODEL_CONF["arc_face"])
OX_ARC_FACE_PATH = BASE_DIR.joinpath(MODEL_CONF["onnx_arc_face"])
FACENET_BATCH = int(MODEL_CONF["batch_size"])

USE_GPU = True if int(DEFAULT_CONF["use_gpu"]) == 1 else False

# retina face detector
RETINA_CONF = float(DETECTOR_CONF["retina_threshold"])
RETINA_NMS_CONF = float(DETECTOR_CONF["retina_nms_threshold"])
RETINA_MODEL = BASE_DIR.joinpath(MODEL_CONF["retina_face"])
OX_RETINA_MODEL = BASE_DIR.joinpath(MODEL_CONF["onnx_retina"])
ERS_GAN_MODEL = BASE_DIR.joinpath(MODEL_CONF["ers_gan"])

# dlib landmark predictor
LANDMARK_PREDICTOR_PATH = BASE_DIR.joinpath('trained_model/dlibLandmarkPredictor')
LANDMARK_PREDICTOR = str(LANDMARK_PREDICTOR_PATH.joinpath('shape_predictor_68_face_landmarks.dat'))

# face path
FACE_FOLD = DIR.joinpath('faces/faraz')
