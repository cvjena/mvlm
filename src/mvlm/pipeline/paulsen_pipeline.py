__all__ = ["BU3DFEPipeline"]

from mvlm.prediction import BU3DFEPredictor

from .general_pipeline import Pipeline

class BU3DFEPipeline(Pipeline):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d = BU3DFEPredictor()