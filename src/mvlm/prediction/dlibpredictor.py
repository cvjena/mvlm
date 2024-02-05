__all__ = ["DlibPredictor"]

import numpy as np
import dlib
import cv2
from pathlib import Path
import requests
import bz2

from .predictor2d import Predictor2D

class DlibPredictor(Predictor2D):
    def __init__(self):
        super().__init__()
        
        # check if the shape predictor file exists
        shape_predictor_path = Path(__file__).parent / "models/shape_predictor_68_face_landmarks.dat"
        if not shape_predictor_path.exists():
            # download the file
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            r = requests.get(url, allow_redirects=True)
            save_name = shape_predictor_path.with_suffix(".dat.bz2")
            open(save_name, 'wb').write(r.content)
        
            # extract the file
            with open(save_name, 'rb') as f:
                data = f.read()
                data = bz2.decompress(data)
                with open(shape_predictor_path, 'wb') as f:
                    f.write(data)
            # remove the bz2 file
            save_name.unlink()
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(shape_predictor_path))
        
    def get_lm_count(self) -> int:
        return 68
    
    def predict_landmarks_from_images(self, image_stack: np.ndarray) -> np.ndarray:
        """
        Predict the landmarks from a stack of images
        
        Returns:
        landmarks: np.ndarray
            The landmarks in the format (n_landmarks, n_views, 2/3)
        """
        landmarks = np.zeros((68, image_stack.shape[0], 3), dtype=np.float32)
        valid = np.ones((image_stack.shape[0]), dtype=bool)
        
        for idx, image in enumerate(image_stack):
            # convert the image to grayscale
            c_image = np.array(image[..., :3] * 255, dtype=np.uint8)
            gray = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)
            # detect the faces
            rects = self.detector(gray, 1)
            # get the landmarks
            if len(rects) == 0:
                landmarks[:, idx, :] = np.nan
                valid[idx] = False
                continue

            shape = self.predictor(c_image, rects[0])
            for j in range(0, shape.num_parts):
                x = shape.part(j).x
                y = shape.part(j).y
                z = image[int(y), int(x), 3]
                # TODO also here the x and y are swapped... there must be a switch later in the code ...
                landmarks[j, idx, :] = [y, x, z]
        return landmarks, valid