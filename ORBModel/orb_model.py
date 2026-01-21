import os
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

class ORBModel():
    def __init__(self, model_path=None, device='cpu', **kwargs):
        orbff = pretrained.orb_v3_conservative_inf_omat(device=device,precision='float32-high')
        self.calcu = ORBCalculator(model=orbff, device=device)
