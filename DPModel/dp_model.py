import os
from deepmd.calculator import DP

class DPModel():
    def __init__(self, model_path=None,default_dtype='float32', device='cpu', **kwargs):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'Omat24.pth')
        self.calcu = DP(model=model_path, default_dtype=default_dtype, device=device, **kwargs)