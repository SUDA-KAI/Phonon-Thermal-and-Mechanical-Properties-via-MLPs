import os
from gptff.model.mpredict import ASECalculator
class GPTFFModel():
    def __init__(self, model_path=None, device='cpu', **kwargs):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'gptff_v1.pth')
        self.calcu = ASECalculator(model_path=model_path, device=device, **kwargs)
