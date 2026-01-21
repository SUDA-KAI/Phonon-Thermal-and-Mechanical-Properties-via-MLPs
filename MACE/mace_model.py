import os
from mace.calculators import MACECalculator
class MACEModel():
    def __init__(self, model_path=None, default_dtype='float32', device='cpu', head='default', **kwargs):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'mace-mpa-0-medium.model')
        self.calcu = MACECalculator(model_paths=model_path,head=head, default_dtype=default_dtype, device=device, **kwargs)  