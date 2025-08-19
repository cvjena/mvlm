"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

__all__ = ["Pipeline"]

import abc
import time
from pathlib import Path

import numpy as np
from PIL import Image

from mvlm.utils import Estimator3D, ObjVTKRenderer3D


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


class Pipeline(abc.ABC, TimeMixin):
    def __init__(
        self,
        render_image_stack: bool = False,
        offscreen: bool = True,
        n_views: int = 8,
        render_image_folder: Path | None = None,
        visualize_rays: bool = False,
        screenshot_folder: Path | None = None,
    ):
        # configuration
        self.render_image_stack = render_image_stack
        self.render_image_folder = render_image_folder
        self.n_views = n_views
        self.visualize_rays = visualize_rays
        self.screenshot_folder = screenshot_folder

        # components
        self.renderer_3d = ObjVTKRenderer3D(image_size=(256, 256), offscreen=offscreen, n_views=n_views)
        self.estimator_3d = Estimator3D()
        self.predictor_2d = None  # will be assigned externally

    def get_lm_count(self) -> int:
        if self.predictor_2d is None:
            raise ValueError("Predictor2D is not initialized.")

        return self.predictor_2d.get_lm_count()

    def predict_one_file(
        self,
        file_name: Path,
        landmark_indices: list[int] | None = None,
        view_indices: list[int] | None = None,
        clip_rays_to_mesh: bool = True,
    ):
        if self.predictor_2d is None:
            raise ValueError("Predictor2D is not initialized.")

        full_s = time.time()
        if not file_name.exists():
            print(f"File {file_name} does not exist")
            return None
        self.tic()

        image_stack, transform_stack, pd = self.renderer_3d.multiview_render(file_name)
        print("Render [Total]: ", self.toc_p())

        if self.render_image_stack:
            self.visualize_image_stack(image_stack, file_name)

        self.tic()
        landmark_stack, valid = self.predictor_2d.predict_landmarks_from_images(image_stack)  # type: ignore
        print("Prediction [Total]: ", self.toc_p())

        landmark_stack = landmark_stack[:, valid, :]
        transform_stack = transform_stack[valid]
        image_stack = image_stack[valid]

        # self.predictor_2d.draw_image_with_landmarks(image_stack[0], landmark_stack[:, 0])
        self.tic()
        lines_s, lines_e = self.estimator_3d.estimate_landmark_lines(image_stack, landmark_stack, transform_stack)
        print("Landmarks [0] - From Heatmaps: ", self.toc_p())

        self.tic()
        landmarks, error = self.estimator_3d.estimate_landmarks_from_lines(landmark_stack, lines_s, lines_e)
        print("Landmarks [1] - From View Lines: ", self.toc_p())

        self.tic()
        landmarks = self.estimator_3d.project_landmarks_to_surface(pd, landmarks)
        print("Landmarks [2] - Project to Surface: ", self.toc_p())
        print("Landmarks [Error]: ", f"{error:08.6f}", " mm")

        print("Landmarks 3D Total: ", self.p_time(time.time() - full_s))
        if self.visualize_rays:
            try:
                from mvlm.visualization.ray_visualizer import RayVisualizer, RayVisualizerConfig

                viz = RayVisualizer(RayVisualizerConfig())
                screenshot_dir = self.screenshot_folder or (Path.cwd() / "visualization" / "ray_screenshots")
                viz.show(
                    pd,
                    lines_s,
                    lines_e,
                    landmarks=landmarks,
                    screenshot_dir=screenshot_dir,
                    landmark_indices=[-1],  # 4 -> mediapipe nose tip
                    view_indices=view_indices,
                    obj_path=file_name if file_name.suffix == ".obj" else None,
                    clip_to_mesh=clip_rays_to_mesh,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[Pipeline] Ray visualization failed: {e}")
        return landmarks

    def visualize_image_stack(self, image_stack: np.ndarray, file_name: Path):
        save_folder = self.render_image_folder or file_name.parent
        if not save_folder.exists():
            raise ValueError(f"Folder for --visualize-method flag [{save_folder}] does not exist.")

        for i in range(self.n_views):
            # Extract the individual image
            single_image = image_stack[i, :, :, 0:3]
            single_image = np.uint8(single_image * 255)
            single_image = Image.fromarray(single_image)

            # Save the image with leading zeros in the filename
            save_path = save_folder / f"{file_name.stem}_{i:02d}.png"
            single_image.save(save_path)
