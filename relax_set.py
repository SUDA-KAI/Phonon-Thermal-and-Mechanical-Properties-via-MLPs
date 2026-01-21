from typing import Union
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.optimize.bfgs import BFGS
from ase.filters import ExpCellFilter
from ase.units import GPa
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
import io
import sys
import contextlib
from ase.optimize.bfgs import BFGS
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS, LBFGSLineSearch
from ase.optimize.mdmin import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG, SciPyFminPowell

OPTIMIZERS = {
    "FIRE": FIRE,
    "BFGS": BFGS,
    "LBFGS": LBFGS,
    "LBFGSLineSearch": LBFGSLineSearch,
    "MDMin": MDMin,
    "SciPyFminCG": SciPyFminCG,
    "SciPyFminBFGS": SciPyFminBFGS,
    "SciPyFminPowell": SciPyFminPowell,
    "BFGSLineSearch": BFGSLineSearch,
}

class Relaxer:

    def __init__(self, calculator, optimizer: str = "BFGS"):
        self.calculator = calculator
        self.ase_adaptor = AseAtomsAdaptor()
        
        optimizer_class = OPTIMIZERS.get(optimizer)
        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer}' not recognized. Available options: {list(OPTIMIZERS.keys())}")
        self.optimizer_class = optimizer_class
        
    def relax(self, structure: Union[Atoms, Structure, Molecule], fmax: float = 0.01, steps: int = 500, relax_cell: bool = True, is_2d: bool = False, verbose: bool = False, **kwargs):
        if isinstance(structure, (Structure, Molecule)):
            atoms = self.ase_adaptor.get_atoms(structure)
        else:
            atoms = structure
        atoms.calc = self.calculator
        if relax_cell:
            if is_2d:
                mask = [True, True, False, False, False, True]
                obj_to_optimize = ExpCellFilter(atoms, mask=mask)
            else:
                obj_to_optimize = ExpCellFilter(atoms, hydrostatic_strain=False)
            
            optimizer = self.optimizer_class(obj_to_optimize, **kwargs)
        else:
            optimizer = self.optimizer_class(atoms, **kwargs)
        stream = sys.stdout if verbose else io.StringIO()
        with contextlib.redirect_stdout(stream):
            optimizer.run(fmax=fmax, steps=steps)
        if relax_cell:
            final_atoms = obj_to_optimize.atoms
        else:
            final_atoms = atoms
        return final_atoms