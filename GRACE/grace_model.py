import os
from tensorpotential.calculator import TPCalculator

class GRACEModel():
    def __init__(self, model_path=None, **kwargs):
        if model_path is None:
            # TPCalculator expects the directory containing saved_model.pb
            model_path = os.path.join(os.path.dirname(__file__), 'GRACE-2L-OMAT-medium-ft-AM')
        
        # Ensure model_path is a directory
        if os.path.isfile(model_path) and model_path.endswith('.pb'):
            model_path = os.path.dirname(model_path)
            
        self.calcu = TPCalculator(model_path, **kwargs)
