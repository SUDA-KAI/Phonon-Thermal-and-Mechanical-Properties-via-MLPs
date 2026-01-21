import logging
logger = logging.getLogger('m3gnet.graph._converters')
logger.setLevel(logging.ERROR)
from mattersim.forcefield import MatterSimCalculator


class MatterSimModel():
    def __init__(self, model_path=None, default_dtype='float32', device='cpu', **kwargs):
        if model_path is None:
            model_path = r'/media/kai/data/MOF/potential/mattersim/mattersim-v1.0.0-1M.pth'
        self.calcu = MatterSimCalculator(load_path=model_path, default_dtype=default_dtype, device=device, **kwargs)   