"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

__all__ = ['MediaPipePipeline', 'BU3DFEPipeline', "DlibPipeline", "DTU3DPipeline", "FaceAlignmentPipeline", "create_pipeline"]

from mvlm.pipeline.dlib_pipeline import DlibPipeline
from mvlm.pipeline.mediapipe_pipeline import MediaPipePipeline
from mvlm.pipeline.paulsen_pipeline import BU3DFEPipeline, DTU3DPipeline
from mvlm.pipeline.face_alignment_pipeline import FaceAlignmentPipeline

def create_pipeline(name: str, render_image_stack: bool = False, render_image_folder: str = None):
    """
    Create a pipeline object based on the specified name.

    Args:
        name (str): The name of the pipeline.
        render_image_stack (bool, optional): Whether to render the image stack. Defaults to False.
        render_image_folder (str, optional): The folder to save rendered images. Defaults to None.

    Returns:
        Pipeline: An instance of the specified pipeline.

    Raises:
        ValueError: If the specified pipeline name is unknown.
    """
    name = name.lower()
    
    if name == "mediapipe":
        return MediaPipePipeline(render_image_stack, render_image_folder)
    elif name == "bu3dfe":
        return BU3DFEPipeline(render_image_stack, render_image_folder)
    elif name == "dlib":
        return DlibPipeline(render_image_stack, render_image_folder)
    elif name == "dtu3d":
        return DTU3DPipeline(render_image_stack, render_image_folder)
    elif name == "face_alignment":
        return FaceAlignmentPipeline(render_image_stack, render_image_folder)
    else:
        raise ValueError(f"Unknown pipeline: {name}")
