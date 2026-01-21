import os
import logging
logger = logging.getLogger('m3gnet.graph._converters')
logger.setLevel(logging.ERROR)
from m3gnet.models import M3GNet, Potential, M3GNetCalculator
class M3GNetModel():
    def __init__(self, model_path=None, device='cpu', **kwargs):
        if (model_path is None) or (not os.path.exists(model_path)):
            model_path = os.path.join(os.path.dirname(__file__), 'origin_model/EFS2021')
        self.m3gnet = M3GNet.from_dir(dirname=model_path)
        self.potential = Potential(model=self.m3gnet)
        self.calcu = M3GNetCalculator(potential=self.potential)