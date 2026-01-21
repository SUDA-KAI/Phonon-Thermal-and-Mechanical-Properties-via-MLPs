import os
from metatomic.torch.ase_calculator import MetatomicCalculator

class UPETModel():
    def __init__(self, model_path=None, device='cpu', **kwargs):
        if model_path is None:
            # Default to the manually converted TorchScript model
            model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        
        # Use MetatomicCalculator directly for .pt (TorchScript) models
        # This bypasses UPETCalculator's conversion checks and HuggingFace connections
        self.calcu = MetatomicCalculator(model_path, device=device, **kwargs)   
