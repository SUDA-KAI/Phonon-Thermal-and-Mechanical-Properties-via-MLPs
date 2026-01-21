import warnings
import multiprocessing
warnings.filterwarnings("ignore")
import os
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pymatgen.core import Structure
from phonopy import PhonopyQHA

from calculators.file_utils import check_and_new_path
from calculators.relax_set import Relaxer
from calculators.phonon_set import PhononSet

CWD = os.path.dirname(os.path.abspath(__file__))
multiprocessing.set_start_method('spawn', force=True)

class QHASet():
    def __init__(self, calculator, device='cpu', dim=[2, 2, 2], mesh=[30, 30, 30], n=11, nscale=0.003, is_rm_dir=True, **kwargs):
        self.device = device
        self.mesh = mesh
        self.n = n
        self.nscale = nscale
        self.calculator = calculator
        self.is_rm_dir = is_rm_dir
        
        self.dim = np.array(dim)
        if len(self.dim) == 3:
            self.supercell_matrix = np.diag(self.dim)
        elif len(self.dim) == 9:
            self.supercell_matrix = self.dim.reshape(3, 3)
        else:
            raise ValueError("dim must contain either 3 elements (diagonal) or 9 elements (full 3x3 matrix)")
    
    def _format_phonopy_dim(self):
        return ' '.join(map(str, self.supercell_matrix.flatten()))

    def get_gruneisen(self, struct: Structure, calcu_dir=r'./gruneisen_tmp', fmax=0.01, steps=1000, t_max=1000, **kwargs):
        calcu_dir = check_and_new_path(calcu_dir)
        calcu_dir = os.path.abspath(calcu_dir)
        
        # 1. relax structure to equilibrium
        relaxer = Relaxer(calculator=self.calculator, optimizer="BFGS")
        relaxed_atoms = relaxer.relax(structure=struct, fmax=fmax, steps=steps, relax_cell=True, verbose=False)


        # 2. phonon calculation at equilibrium volume
        phonon_set = PhononSet(calculator=self.calculator)
        central_index = (self.n - 1) // 2
        phonon_dir_central = os.path.join(calcu_dir, f'phonon_{central_index}')
        os.makedirs(phonon_dir_central, exist_ok=True)
        energy_orig = relaxed_atoms.get_potential_energy()
        volume_orig = relaxed_atoms.get_volume()
        has_imag = phonon_set.get_phonon(
            relaxed_atoms,
            calcu_dir=phonon_dir_central,
            supercell_matrix=self.supercell_matrix,
            mesh=self.mesh,
            t_max=t_max,
            if_thermal=True,
            **kwargs
        )

        # if has_imag:
        #     print("\nERROR: Imaginary phonon frequencies detected in the equilibrium structure. QHA calculation aborted.")
        #     if self.is_rm_dir:
        #         shutil.rmtree(calcu_dir)
        #     return

        # 3. Calculate energies and phonons at different volumes and immediately extract thermal properties
        e_list = []
        v_list = []
        
        for i in range(self.n):
            if i == central_index:
                e_list.append(energy_orig)
                v_list.append(volume_orig)
                continue

            scale_factor = 1 + (i - (self.n - 1) / 2) * self.nscale
            scaled_atoms = relaxed_atoms.copy()
            scaled_atoms.set_calculator(self.calculator)
            scaled_atoms.set_cell(scaled_atoms.get_cell() * scale_factor, scale_atoms=True)
            
            volume = scaled_atoms.get_volume()
            energy = scaled_atoms.get_potential_energy()
            e_list.append(energy)
            v_list.append(volume)
            
            phonon_dir = os.path.join(calcu_dir, f'phonon_{i}')
            os.makedirs(phonon_dir, exist_ok=True)

            has_imag_scaled = phonon_set.get_phonon(
                scaled_atoms, 
                calcu_dir=phonon_dir, 
                supercell_matrix=self.supercell_matrix, 
                mesh=self.mesh, 
                t_max=t_max,
                if_thermal=True,
                **kwargs
            )
            
            if has_imag_scaled:
                print(f"Warning: Imaginary phonon frequencies detected for volume {volume:.2f} Å^3.")
        
        # 4. Prepare input for phonopy-qha and run QHA analysis
        thermal_properties_dir = os.path.join(calcu_dir, 'thermal_properties')
        os.makedirs(thermal_properties_dir, exist_ok=True)            
        with open(os.path.join(thermal_properties_dir, 'v-e.dat'), 'w') as f:        
            for e, v in zip(e_list, v_list):
                f.write(f"{v:.4f} {e}\n")
                
        thermal_files = []
        for i in range(self.n):
            thermal_files.append(os.path.join(calcu_dir, f'phonon_{i}', 'thermal_properties.yaml'))
        
        os.chdir(thermal_properties_dir)
        subprocess.run(['phonopy-qha', '-p', os.path.join(thermal_properties_dir, "v-e.dat")] + thermal_files + ['-s'])