import os
import shutil
import numpy as np
from ase import Atoms
from ase.io import read
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from calculators.relax_set import Relaxer

class ElasticSet:
    def __init__(self, calculator, relaxer: Relaxer, **kwargs):
        self.calculator = calculator
        self.relax = relaxer

    def calcu_energy(self, calcu_dir, **kwargs):
        """计算单个目录的能量"""
        try:
            outcar_file = os.path.join(calcu_dir, 'OUTCAR')
            if os.path.exists(outcar_file):
                os.remove(outcar_file)
            
            struct = read(os.path.join(calcu_dir, "POSCAR"))
            
            f_max = 0.001
            Relaxed_atoms = self.relax.relax(structure=struct, fmax=f_max, steps=500, relax_cell=False, verbose=False)
            U_atom = Relaxed_atoms.get_potential_energy()
            
            text = f"  free  energy   TOTEN  = {U_atom} eV\n" + \
                   "                 Voluntary context switches:"
            with open(outcar_file, 'a+') as f:
                f.write(text)
                
        except Exception as e:
            print(f"计算能量时出错 ({calcu_dir}): {e}")
            outcar_file = os.path.join(calcu_dir, 'OUTCAR')
            with open(outcar_file, 'w') as f:
                f.write("  free  energy   TOTEN  = 0.0 eV\n")

    def gen_vaspkit_in(self, calcu_dir, in_type=1, strainstep: float = 0.005, strain_num: int = 9):
        strain_list = np.linspace(-(strain_num-1)/2*strainstep, (strain_num-1)/2*strainstep, strain_num)
        text = f'{in_type}\n' + \
               '3D\n' + \
               f'{strain_num}\n' + \
               ' '.join([f'{x:.6f}' for x in strain_list]) + '\n'
        with open(os.path.join(calcu_dir, 'VPKIT.in'), 'w+') as f:
            f.write(text)

    def get_elastic(self, struct: Structure, calcu_dir: str, strainstep: float = 0.005, strain_num: int = 9, is_rm: bool = True, **kwargs):
        os.makedirs(calcu_dir, exist_ok=True)
        struct.to(fmt='poscar', filename=os.path.join(calcu_dir, 'POSCAR'))
        self.gen_vaspkit_in(calcu_dir=calcu_dir, in_type=1, strainstep=strainstep, strain_num=strain_num)
        os.system('cd %s; vaspkit -task 201 > output.log 2>&1 ' % calcu_dir)
        _calcu_dirs = []
        for cij in os.listdir(calcu_dir):
            if 'C' not in cij:
                continue
            cij_dir = os.path.join(calcu_dir, cij)
            if not os.path.isdir(cij_dir):
                continue
            for s in os.listdir(cij_dir):
                if 'strain_' not in s:
                    continue
                s_dir = os.path.join(cij_dir, s)
                _calcu_dirs.append(s_dir)
        for _calcu_dir in _calcu_dirs:
            self.calcu_energy(calcu_dir=_calcu_dir, **kwargs)
        self.gen_vaspkit_in(calcu_dir=calcu_dir, in_type=2)
        os.system('cd %s; vaspkit -task 201 > BM_SS.log ' % calcu_dir)
        if is_rm:
            for item in os.listdir(calcu_dir):
                if item.startswith('C') and os.path.isdir(os.path.join(calcu_dir, item)):
                    shutil.rmtree(os.path.join(calcu_dir, item), ignore_errors=True)

