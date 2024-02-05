__all__ = ["FaceAlignmentPipeline"]

from mvlm.prediction.face_alignmentpredictor import FaceAlignmentPredictor


from .general_pipeline import Pipeline 

class FaceAlignmentPipeline(Pipeline):
     def __init__(self, render_image_stack: bool = False, render_image_folder: str = None):
         super().__init__(render_image_stack, render_image_folder)
         self.predictor_2d = FaceAlignmentPredictor()