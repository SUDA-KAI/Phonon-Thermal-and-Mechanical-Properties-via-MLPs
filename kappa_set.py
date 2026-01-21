import os
import glob
import numpy as np
import h5py
import matplotlib
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from phono3py import Phono3py
from phono3py.file_IO import write_FORCES_FC3
from phonopy import Phonopy
from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

class KappaSet:
    def __init__(self, calculator):
        """
        Initialize the KappaSet with a calculator.
        
        Args:
            calculator: An ASE-compatible calculator object (e.g., from MatterSim, MACE, etc.)
        """
        self.calculator = calculator

    def _to_phonopy_atoms(self, atoms: Atoms):
        """Convert ASE Atoms to PhonopyAtoms."""
        return PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                            masses=atoms.get_masses(),
                            positions=atoms.get_positions(),
                            cell=atoms.get_cell())

    def _ensure_ase_atoms(self, structure):
        """Ensure input is ASE Atoms."""
        if isinstance(structure, Atoms):
            return structure
        elif isinstance(structure, Structure):
            return AseAtomsAdaptor.get_atoms(structure)
        else:
            raise ValueError(f"Unsupported structure type: {type(structure)}")

    def run_kappa(self, structure, dim_fc3=[2, 2, 2], dim_fc2=[2, 2, 2], 
                  mesh=[11, 11, 11], temp_range=np.arange(0, 1001, 10), 
                  work_dir=".", primitive_matrix='auto'):
        """
        Run the Thermal Conductivity calculation (FC3 + FC2).

        Args:
            structure: ASE Atoms or Pymatgen Structure object.
            dim_fc3: Supercell matrix for FC3 (e.g., [2, 2, 2]).
            dim_fc2: Supercell matrix for FC2.
            mesh: q-point mesh for thermal conductivity (e.g., [11, 11, 11]).
            temp_range: List or array of temperatures. Default: 0 to 1000 step 10.
            work_dir: Directory to run the calculation in.
            primitive_matrix: Primitive matrix setting for Phono3py/Phonopy.
        """
        if temp_range is None:
            temp_range = np.arange(0, 1001, 10)

        # Prepare directory
        original_dir = os.getcwd()
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        
        try:
            os.chdir(work_dir)
            
            atoms_ase = self._ensure_ase_atoms(structure)
            unitcell = self._to_phonopy_atoms(atoms_ase)

            # --- FC3 Calculation ---
            print("Initializing Phono3py for FC3...")
            ph3 = Phono3py(unitcell,
                           supercell_matrix=dim_fc3,
                           primitive_matrix=primitive_matrix)
            
            ph3.generate_displacements()
            supercells = ph3.supercells_with_displacements
            ph3.save("phono3py_disp.yaml")

            forces_fc3 = []
            for sc in tqdm(supercells, desc="Calculating FC3 Forces"):
                if sc is None:
                    forces_fc3.append(None)
                    continue
                
                # Convert Phonopy supercell to ASE Atoms
                atoms_sc = Atoms(
                    symbols=sc.symbols,
                    positions=sc.positions,
                    cell=sc.cell,
                    pbc=True
                )
                
                atoms_sc.calc = self.calculator
                forces = atoms_sc.get_forces()
                forces_fc3.append(forces)

            write_FORCES_FC3(ph3.dataset, forces_fc3, filename="FORCES_FC3")
            ph3.forces = np.array(forces_fc3, dtype='double', order='C')
            ph3.produce_fc3()
            ph3.save("fc3.hdf5")

            # --- FC2 Calculation ---
            print("Calculating FC2...")
            ph2 = Phonopy(unitcell,
                          supercell_matrix=dim_fc2,
                          primitive_matrix=primitive_matrix)
            ph2.generate_displacements(distance=0.01)
            supercells_fc2 = ph2.supercells_with_displacements
            forces_fc2 = []
            for sc in tqdm(supercells_fc2, desc="Calculating FC2 Forces"):
                if sc is None:
                    forces_fc2.append(None)
                    continue
                
                atoms_sc = Atoms(
                    symbols=sc.symbols,
                    positions=sc.positions,
                    cell=sc.cell,
                    pbc=True
                )
                atoms_sc.calc = self.calculator
                forces = atoms_sc.get_forces()
                forces_fc2.append(forces)

            ph2.forces = forces_fc2
            ph2.produce_force_constants()
            write_force_constants_to_hdf5(ph2.force_constants, filename='fc2.hdf5')

            # --- Thermal Conductivity ---
            print("Calculating Thermal Conductivity...")
            ph3.fc2 = ph2.force_constants
            ph3.mesh_numbers = mesh
            ph3.init_phph_interaction()
            ph3.run_thermal_conductivity(temperatures=temp_range, write_kappa=True)
            
            output_file = f"kappa-m{''.join(map(str, mesh))}.hdf5"
            print(f"Thermal conductivity calculation finished. Check {output_file}")
            
            # Auto-plot
            self.plot_kappa(output_file)

        except Exception as e:
            print(f"An error occurred during execution: {e}")
            raise
        finally:
            os.chdir(original_dir)

    def plot_kappa(self, filename: str = None):
        """
        Plot thermal conductivity from hdf5 file.
        
        Args:
            filename: Path to kappa-m*.hdf5 file. If None, searches in current dir.
        """
        if filename is None:
            files = glob.glob("kappa-m*.hdf5")
            if not files:
                print("No kappa-m*.hdf5 file found.")
                return
            filename = files[0]
        
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return

        print(f"Reading file: {filename}")

        with h5py.File(filename, 'r') as f:
            temperatures = f['temperature'][:]
            if 'kappa' in f:
                kappa = f['kappa'][:]
            else:
                print("Dataset 'kappa' not found in file.")
                return

        if kappa.ndim == 3:
            print("Warning: kappa data dimension is 3. Summing over axis 1.")
            kappa = kappa.sum(axis=1)

        # Plotting
        plt.figure(figsize=(10, 6))
        
        # xx, yy, zz components (indices 0, 1, 2)
        plt.plot(temperatures, kappa[:, 0], label=r'$\kappa_{xx}$', marker='o', markersize=4)
        plt.plot(temperatures, kappa[:, 1], label=r'$\kappa_{yy}$', marker='^', markersize=4)
        plt.plot(temperatures, kappa[:, 2], label=r'$\kappa_{zz}$', marker='s', markersize=4)
        
        # Isotropic average
        kappa_iso = (kappa[:, 0] + kappa[:, 1] + kappa[:, 2]) / 3
        plt.plot(temperatures, kappa_iso, label=r'$\kappa_{iso}$', linestyle='--', color='black')

        plt.xlabel('Temperature (K)', fontsize=12)
        plt.ylabel('Thermal Conductivity (W/m-K)', fontsize=12)
        plt.title(f'Thermal Conductivity ({os.path.basename(filename)})', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        try:
            plt.xlim(0, max(temperatures) + 10)
        except:
             pass
        
        output_filename = 'kappa_plot.png'
        plt.savefig(output_filename, dpi=300)
        plt.close() # Close figure to free memory
        print(f"Plot saved to {output_filename}")
