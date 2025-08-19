__all__ = ["Predictor2D"]

import abc
import numpy as np
import matplotlib.pyplot as plt


class Predictor2D(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def predict_landmarks_from_images(self, image_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict the landmarks from a stack of images

        Returns:
        landmarks: np.ndarray
            The landmarks in the format (n_landmarks, n_views, 3)
        valid: np.ndarray
            The validity of the landmarks in the format (n_views) (True/False)
        """
        pass

    @abc.abstractmethod
    def get_lm_count(self) -> int:
        pass

    def draw_image_with_landmarks(self, image, landmarks, save: bool = False):
        import cv2

        image_out = image[:, :, 0:3].copy()
        image_marked = image_out.copy()
        image_marked = (image_marked * 255).astype(np.uint8)

        for lm in landmarks:
            y = int(lm[0])
            x = int(lm[1])
            cx, cy = int(x), int(y)
            if 0 <= cx < image_marked.shape[1] and 0 <= cy < image_marked.shape[0]:
                image_marked = cv2.circle(image_marked, (cx, cy), 2, (0, 0, 255), -1)

        _, ax = plt.subplots(1, 2)
        ax[0].imshow(image_out)
        ax[1].imshow(image_marked)
        plt.show()

        from pathlib import Path

        output_dir = Path.cwd() / "visualization"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"notebook_SINGLEVIEW_{self.__class__.__name__}_landmarks.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(image_marked, cv2.COLOR_RGB2BGR))
