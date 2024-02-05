"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ['DeepMVLM']

import time
from pathlib import Path

import numpy as np
from PIL import Image

from mvlm.prediction import MediaPipePredictor, BU3DFEPredictor
from mvlm.utils import ObjVTKRenderer3D, Utils3D

class TimeMixin:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        
    def tic(self):
        self.start_time = time.time()
        
    def toc(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

    def toc_p(self):
        return self.p_time(self.toc())

    def p_time(self, t):
        return f"{t:08.6f} s"


class DeepMVLM(TimeMixin):
    def __init__(
        self, 
        config: str, # basically the path to the config file
        render_image_stack: bool = False, # if true, the image stack will be saved
        render_image_folder: Path = None, # if not None, the image stack will be saved in this folder
    ):
        self.config = config
        self.render_image_stack = render_image_stack
        self.render_image_folder = render_image_folder
        
        # loading of the renderer and the predictor
        self.renderer_3d = ObjVTKRenderer3D(image_size=(256, 256))
        # self.predictor_2d = PaulsenPredictor(self.config)
        self.predictor_2d = MediaPipePredictor()
        self.estimator_3d = Utils3D(self.config)
        
    def get_lm_count(self) -> int:
        return self.predictor_2d.get_lm_count()

    def predict_one_file(self, file_name: Path):
        full_s = time.time()
        if not file_name.exists():
            print(f"File {file_name} does not exist")
            return None
        self.tic()
        image_stack, transform_stack, pd = self.renderer_3d.multiview_render(file_name)
        print('Render [Total]: ', self.toc_p())
        
        if self.render_image_stack:
            # TODO Check if the folder exists
            self.visualize_image_stack(image_stack, file_name)
       
        self.tic()
        landmark_stack = self.predictor_2d.predict_landmarks_from_images(image_stack)
        print('Prediction [Total]: ', self.toc_p())
        
        # self.predictor_2d.draw_image_with_landmarks(image_stack[0], landmark_stack[:, 0])

        self.tic()
        self.estimator_3d.heatmap_maxima = landmark_stack
        self.estimator_3d.transformations_3d = transform_stack
        self.estimator_3d.compute_lines_from_heatmap_maxima()
        print('Landmarks [0] - From Heatmaps: ', self.toc_p())

        self.tic()
        error = self.estimator_3d.compute_all_landmarks_from_view_lines()
        print('Landmarks [1] - From View Lines: ', self.toc_p())

        self.tic()
        self.estimator_3d.project_landmarks_to_surface(pd)
        print('Landmarks [2] - Project to Surface: ', self.toc_p())
        print('Landmarks [Error]: ', f"{error:08.6f}", " mm")

        print("Landmarks 3D Total: ", self.p_time(time.time() - full_s))
        return self.estimator_3d.landmarks
    
    def visualize_image_stack(self, image_stack: np.ndarray, file_name: str):
        # save the iamge stack flattened such that it close to a square
        n, h, w, c = image_stack.shape
        # closest nrows and ncols
        nrows = int(np.sqrt(n))
        ncols = n // nrows
        if nrows * ncols < n:
            ncols += 1
        # create the image
        out_image = np.zeros((nrows * h, ncols * w, 3))
        for i in range(n):
            r = i // ncols
            c = i % ncols
            out_image[r*h:(r+1)*h, c*w:(c+1)*w, :] = image_stack[i, :, :, 0:3]
        out_image = np.uint8(out_image * 255)
        out_image = Image.fromarray(out_image)
        
        file_name = Path(file_name)
        out_image.save(f'{self.render_image_folder}/{file_name.stem}.png')

        #  # save the image stack, make the in a 2 by 4 grid
        # n, h, w, c = image_stack.shape
        # out_image = np.reshape(image_stack, (2, 4, h, w, c))
        # # remove last channel
        # out_image = out_image[:, :, :, :, 0:3]
        # out_image = np.transpose(out_image, (0, 2, 1, 3, 4))
        # out_image = np.reshape(out_image, (2*h, 4*w, 3))
        # out_image = np.uint8(out_image * 255)
        # out_image = Image.fromarray(out_image)
        
        # file_name = Path(file_name)
        # out_image.save(f'{self.render_image_folder}/{file_name.stem}_{n}.png')
