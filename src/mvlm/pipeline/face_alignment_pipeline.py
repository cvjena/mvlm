__all__ = ["FaceAlignmentPipeline"]

from mvlm.prediction.face_alignmentpredictor import FaceAlignmentPredictor


from .general_pipeline import Pipeline


class FaceAlignmentPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d = FaceAlignmentPredictor()
