import os
from pynep.calculate import NEP
class NEPModel():
    def __init__(self, model_path=None, **kwargs):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'nep89_20250409.txt')
        self.calcu = NEP(model_path)