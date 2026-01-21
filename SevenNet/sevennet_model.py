import os
from sevenn.calculator import SevenNetCalculator

class SevenNetModel():
    def __init__(self, model_path=None, device='cpu', **kwargs):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'sevennet_omni.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SevenNet model not found at {model_path}. Please download it first.")

        self.calcu = SevenNetCalculator(model='7net-omni', modal='mpa',device=device,enable_cueq=True)
