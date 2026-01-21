import os
from ase import Atoms
from mattersim.applications.phonon import PhononWorkflow
from phonopy.file_IO import write_FORCE_CONSTANTS, write_FORCE_SETS

class PhononSet:

    def __init__(self, calculator, **kwargs):
        self.calculator = calculator

    def get_phonon(self, atoms:Atoms, calcu_dir: str, supercell_matrix, mesh: list = [30, 30, 30], t_max: int = 1000, if_thermal: bool = False, **kwargs):

        atoms.calc = self.calculator
        ph = PhononWorkflow(atoms, amplitude = 0.01, supercell_matrix = supercell_matrix, find_prim = False, work_dir = calcu_dir, **kwargs)
        has_imag, phonons = ph.run()
        print('ph:',phonons.supercell_matrix)
        print(f"Has imaginary phonon: {has_imag}")
        
        # force_constants = phonons.force_constants
        # write_FORCE_CONSTANTS(force_constants, filename=os.path.join(calcu_dir, "FORCE_CONSTANTS"))
        # if hasattr(phonons, 'dataset') and phonons.dataset is not None:
        #     write_FORCE_SETS(phonons.dataset, filename=os.path.join(calcu_dir, "FORCE_SETS"))
        # else:
        #     if hasattr(ph, 'displacements') and hasattr(ph, 'forces'):
        #         dataset = {'displacements': ph.displacements, 'forces': ph.forces}
        #         write_FORCE_SETS(dataset, filename=os.path.join(calcu_dir, "FORCE_SETS"))
        
        if if_thermal:
            try:
                phonons.run_mesh(mesh=mesh)
                phonons.run_thermal_properties(t_max=t_max)
                phonons.write_yaml_thermal_properties(filename=os.path.join(calcu_dir, 'thermal_properties.yaml'))
                phonons.write_total_dos(filename=os.path.join(calcu_dir, 'dos.dat'))
                
                tp_dict = phonons.get_thermal_properties_dict()
                with open(os.path.join(calcu_dir, 'thermal_properties.dat'), 'w') as f:
                    f.write("# T [K], F [kJ/mol], S [J/K/mol], Cv [J/K/mol]\n")
                    for t, F, S, Cv in zip(tp_dict['temperatures'], tp_dict['free_energy'], tp_dict['entropy'], tp_dict['heat_capacity']):
                        f.write(f"{t:12.6f} {F:12.6f} {S:12.6f} {Cv:12.6f}\n")

                print(f"Thermal properties written to {os.path.join(calcu_dir, 'thermal_properties.yaml')} and thermal_properties.dat")
            except Exception as e:
                print(f"Could not calculate or write thermal properties: {e}")

        return has_imag