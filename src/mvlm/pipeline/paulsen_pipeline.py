__all__ = ["PaulsenPipeline"]

from mvlm.prediction import PaulsenPredictor

from .general_pipeline import Pipeline

class PaulsenPipeline(Pipeline):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d = PaulsenPredictor()