import onnxruntime as ort
import numpy as np
from face_module.models.networks.retina.model import RetinaFace
from face_module.models.networks.arc_face.models import ArcFaceONNX
from face_module.models.utils import Face, norm_crop
from face_module.settings import (OX_RETINA_MODEL,
                                  OX_ARC_FACE_PATH,
                                  RETINA_CONF,
                                  RETINA_NMS_CONF,
                                  FACE_FOLD,
                                  RESIZE_SHAPE)

# this is about log file
ort.set_default_logger_severity(3)


class Recognition:
    def __init__(self):
        self.face_dm = RetinaFace(model_file=str(OX_RETINA_MODEL), nms_thresh=RETINA_NMS_CONF, det_thresh=RETINA_CONF)
        self.face_dm.prepare(ctx_id=0, input_size=(RESIZE_SHAPE[1], RESIZE_SHAPE[0]))

        self.arc_face = ArcFaceONNX(model_file=str(OX_ARC_FACE_PATH))
        self.arc_face.prepare(ctx_id=0)

    def detection(self, image):
        boxes, points = self.face_dm.detect(image, max_num=1)

        bbox = np.empty((1, 4))
        embed = []
        if len(boxes) != 0:

            for idx in range(boxes.shape[0]):
                bbox = boxes[idx, 0:4]
                det_score = boxes[idx, 4]

                kps = None
                if points is not None:
                    kps = points[idx]

                face = norm_crop(image, kps)
                embed = self.arc_face.get_feat(face)

            return embed, bbox
        else:
            return embed, np.empty((0, 4))

    def verification(self, emb1, emb2):
        similarity = self.arc_face.compute_sim(emb1, emb2)

        return similarity
