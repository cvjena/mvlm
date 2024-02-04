__all__ = ["MediaPipePipeline"]

from mvlm.prediction import MediaPipePredictor

from .general_pipeline import Pipeline

class MediaPipePipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d = MediaPipePredictor()
    