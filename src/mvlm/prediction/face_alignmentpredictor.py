__all__ = ["FaceAlignmentPredictor"]

import numpy as np

import torch
import face_alignment

from .predictor2d import Predictor2D

"""
@inproceedings{bulat2017far,
  title={How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)},
  author={Bulat, Adrian and Tzimiropoulos, Georgios},
  booktitle={International Conference on Computer Vision},
  year={2017}
}
"""
class FaceAlignmentPredictor(Predictor2D):
    def __init__(self):
        super().__init__()
    
        if torch.cuda.is_available():
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='blazeface', device='cuda', dtype=torch.bfloat16)
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='blazeface', device='cpu')

    def get_lm_count(self) -> int:
        return 68
        
    def predict_landmarks_from_images(self, image_stack) -> np.ndarray:
        landmarks = np.empty((self.get_lm_count(), image_stack.shape[0], 3), dtype=np.float32)
        valid = np.ones((image_stack.shape[0]), dtype=bool)
        
        for idx, image in enumerate(image_stack):        
            c_image = np.array(image[..., :3] * 255, dtype=np.uint8)
            # TODO we could force a bounding box of the whole image here... but that might descrease the landmark accuracy
            pred = self.fa.get_landmarks_from_image(c_image, return_bboxes=False, return_landmark_score=False)
            if pred is None:
                landmarks[:, idx, :] = np.nan
                valid[idx] = False
                continue
            
            # TODO also here the landmarks coordinates are flipped
            x = pred[0][:, 0]
            y = pred[0][:, 1]
            landmarks[:, idx, 0] = y
            landmarks[:, idx, 1] = x   
            landmarks[:, idx, 2] = image_stack[idx, np.clip(y, 0, 255).astype(int), np.clip(x, 0, 255).astype(int), 3]
        return landmarks, valid