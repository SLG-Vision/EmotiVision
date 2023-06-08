import numpy as np
import dlib
from imutils import face_utils
import cv2


class FaceLandmarkK():
    BOX = 'box'
    EAR_DX = 'ear_dx'
    EAR_SX = 'ear_sx'
    NOSE = 'nose'
    MOUTH_DX = 'mouth_dx'
    MOUTH_SX = 'mouth_sx'
    CHIN = 'chin'
    EYE_DX_CENTER = 'eye_dx_center'
    EYE_SX_CENTER = 'eye_sx_center'
    EYE_DX_OUT = 'eye_dx_out'
    EYE_DX_IN = 'eye_dx_in'
    EYE_SX_OUT = 'eye_sx_out'
    EYE_SX_IN = 'eye_sx_in'
    EYE_DX_TOP = 'eye_dx_top'
    EYE_DX_BOTTOM = 'eye_dx_bottom'
    EYE_SX_TOP = 'eye_sx_top'
    EYE_SX_BOTTOM = 'eye_sx_bottom'


class FaceLandmarkTracking():
    def __init__(self, shape_predictor_path) -> None:
        self.landmark_predictor = dlib.shape_predictor(shape_predictor_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.k = FaceLandmarkK()

    def face_analysis(self, frame):
        faces = self.face_detector(frame,  1)
        poi = []
        for face in faces:

            points = face_utils.shape_to_np(
                self.landmark_predictor(frame, face))
            (x, y, w, h) = face_utils.rect_to_bb(face)

            poi.append({
                self.k.BOX: (x, y, w, h),
                self.k.EAR_DX: points[0],
                self.k.EAR_SX: points[16],
                self.k.NOSE: points[30],
                self.k.MOUTH_DX: points[48],
                self.k.MOUTH_SX: points[54],
                self.k.CHIN: points[8],
                self.k.EYE_DX_CENTER: np.round(np.mean(points[36:42], axis=0)).astype(int),
                self.k.EYE_SX_CENTER: np.round(np.mean(points[42:48], axis=0)).astype(int),
                self.k.EYE_DX_OUT: points[36],
                self.k.EYE_DX_IN: points[39],
                self.k.EYE_SX_OUT: points[45],
                self.k.EYE_SX_IN: points[42],
                self.k.EYE_DX_TOP: np.round(np.mean(points[37:39], axis=0)).astype(int),
                self.k.EYE_DX_BOTTOM: np.round(np.mean(points[40:42], axis=0)).astype(int),
                self.k.EYE_SX_TOP: np.round(np.mean(points[43:45], axis=0)).astype(int),
                self.k.EYE_SX_BOTTOM: np.round(np.mean(points[46:48], axis=0)).astype(int),
            })

        return poi
