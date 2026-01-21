import os
import logging
logger = logging.getLogger('m3gnet.graph._converters')
logger.setLevel(logging.ERROR)
from chgnet.model.model import CHGNet
from chgnet.model import CHGNetCalculator

class CHGNetModel():
    def __init__(self, model_name=None, device='cpu', **kwargs):
        if model_name is None:
            model = CHGNet.load(use_device=device)
        else:
            model = CHGNet.load(model_name=model_name, use_device=device)
        self.calcu = CHGNetCalculator(model=model, use_device=device, **kwargs)

