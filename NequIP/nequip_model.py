import os
from nequip.ase import NequIPCalculator

class NequIPModel():
    def __init__(self, model_path=None, device='cpu', **kwargs):
        if model_path is None:
            # Default to the deployed torchscript model in the same directory
            model_path = os.path.join(os.path.dirname(__file__), 'deployed_model.nequip.pth')
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"NequIP compiled model not found at {model_path}. Please run compile_model.sh inside Calculators/NequIP folder first.")

        # Use from_compiled_model to load the compiled model (works for both .pth and .pt2)
        # chemical_species_to_atom_type_map=True suppresses warning for standard elements
        self.calcu = NequIPCalculator.from_compiled_model(
            compile_path=model_path, 
            device=device, 
            chemical_species_to_atom_type_map=True,
            **kwargs
        )