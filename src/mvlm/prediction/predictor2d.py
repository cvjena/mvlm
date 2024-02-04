__all__ = ["Predictor2D"]

import abc
import numpy as np
import matplotlib.pyplot as plt

class Predictor2D(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def predict_landmarks_from_images(self, image_stack) -> np.ndarray:
        """
        Predict the landmarks from a stack of images
        
        Returns:
        landmarks: np.ndarray
            The landmarks in the format (n_landmarks, n_views, 2/3)
        """
        pass
    
    @abc.abstractmethod
    def get_lm_count(self) -> int:
        pass
    
    def draw_image_with_landmarks(self, image, landmarks):
        image_out = image[:, :, 0:3].copy()
        
        image_marked = image_out.copy()
        for lm in landmarks:
            x = int(lm[0])
            y = int(lm[1])
            image_marked[x-1:x+1, y-1:y+1, 0] = 0
            image_marked[x-1:x+1, y-1:y+1, 1] = 0
            image_marked[x-1:x+1, y-1:y+1, 2] = 1
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_out)
        ax[1].imshow(image_marked)
        plt.show()
        