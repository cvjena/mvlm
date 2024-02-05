__all__ = ["DlibPipeline"]

from mvlm.prediction import DlibPredictor 

from .general_pipeline import Pipeline

class DlibPipeline(Pipeline):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_2d =  DlibPredictor()
        self.estimator_3d.mode = "absolute"
        self.estimator_3d.threshold_absolute = 0.1