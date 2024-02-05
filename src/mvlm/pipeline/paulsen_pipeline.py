__all__ = ["BU3DFEPipeline", "DTU3DPipeline"]

from mvlm.prediction import BU3DFEPredictor, DTU3DPredictor

from .general_pipeline import Pipeline

class BU3DFEPipeline(Pipeline):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d = BU3DFEPredictor()

class DTU3DPipeline(Pipeline):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d = DTU3DPredictor()