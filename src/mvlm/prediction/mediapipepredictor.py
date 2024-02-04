__all__ = ["MediaPipePredictor"]

from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .predictor2d import Predictor2D

class MediaPipePredictor(Predictor2D):
    def __init__(self):
        super().__init__()
        
        base_options = python.BaseOptions(model_asset_path=str(Path(__file__).parent / "2023-07-09_face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False, 
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
    def get_lm_count(self) -> int:
        return 478
        
    def predict_landmarks_from_images(self, image_stack) -> np.ndarray:
        landmarks = np.empty((478, image_stack.shape[0], 3), dtype=np.float32)

        for idx, image in enumerate(image_stack):        
            h, w = image.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=(image[...,:3] * 255).astype(np.uint8))

            face_landmarker_result = self.detector.detect(mp_image)
            if face_landmarker_result.face_landmarks:
                face_landmarks = face_landmarker_result.face_landmarks[0]
                for ldx, lm in enumerate(face_landmarks):
                    # TODO somehow x and y are switched here...
                    landmarks[ldx, idx, 0] =  lm.y * h
                    landmarks[ldx, idx, 1] =  lm.x * w
                    landmarks[ldx, idx, 2] = -lm.z * w
        return landmarks