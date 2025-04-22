import os
import sys
from os.path import exists
from numpy import pi, sqrt, exp
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.core import Spin
from pathlib import Path
from lib_DOS_lcfo import DOSCAR_LCFO

# --- Helper: Parse POSCAR for lattice and atomic positions ---
def parse_poscar(ifile):
    """
    Parses the POSCAR file and extracts lattice vectors, atomic positions, and atom types.
    """
    with open(ifile, 'r') as file:
        lines = file.readlines()
        sf = float(lines[1])
        latticevectors = [float(lines[i].split()[j]) * sf for i in range(2, 5) for j in range(3)]
        latticevectors = np.array(latticevectors).reshape(3, 3)
        atomtypes = lines[5].split()
        atomnums = [int(i) for i in lines[6].split()]
        if 'Direct' in lines[7] or 'Cartesian' in lines[7]:
            start = 8
            mode = lines[7].split()[0]
        else:
            mode = lines[8].split()[0]
            start = 9
        coord = np.array([[float(lines[i].split()[j]) for j in range(3)]
                           for i in range(start, sum(atomnums) + start)])
        if mode != 'Cartesian':
            for i in range(sum(atomnums)):
                for j in range(3):
                    while coord[i][j] > 1.0 or coord[i][j] < 0.0:
                        if coord[i][j] > 1.0:
                            coord[i][j] -= 1.0
                        elif coord[i][j] < 0.0:
                            coord[i][j] += 1.0
                coord[i] = np.dot(coord[i], latticevectors)
    return latticevectors, coord, atomtypes, atomnums

# --- Helper: Compute Tunneling Factor (unchanged) ---
def tunneling_factor(V, E, phi):
    """
    Calculate the tunneling factor based on the applied voltage (V),
    energy relative to Fermi level (E), and the workfunction (phi).
    """
    V *= 1.60218e-19   # Convert from eV to Joules
    E *= 1.60218e-19
    phi *= 1.60218e-19

    m_e = 9.11e-31     # Electron mass (kg)
    hbar = 6.626e-34   # Planck's constant (J·s)
    prefactor = (8 / (3 * V)) * np.pi * np.sqrt(2 * m_e) / hbar
    barrier = (phi - E + V)**(3/2) - (phi - E)**(3/2)
    return prefactor * barrier

# --- Main LDOS Calculator Class (MO diagram logic removed) ---
class ldos_single_point:
    def __init__(self, filepath):
        """
        Initialize the single-point LDOS calculator.

        Parameters:
            filepath (str): Path to the folder containing the output files.
              Expected files: DOSCAR.LCFO.lobster, LCFO_Fragments.lobster, and either POSCAR or CONTCAR.
        """
        self.filepath = filepath
        self.lv = None            # Lattice vectors from POSCAR
        self.coord = None         # Cartesian coordinates for fragments
        self.atomtypes = None     # Atom types from POSCAR
        self.atomnums = None      # Atom numbers from POSCAR
        self.energies = None      # Energies from DOSCAR.LCFO
        self.ef = None            # Fermi energy from DOSCAR.LCFO
        self.pdos = None          # pMODOS data from DOSCAR.LCFO
        self.tip_disp = 15.0      # Default tip displacement
        self.estart = None        # Start index for energy range
        self.eend = None          # End index for energy range
        self._fragments = None    # Fragment mappings (from LCFO_Fragments file)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Specified filepath does not exist: {filepath}")

        # Files are loaded automatically on instantiation.
        self.load_files()

    def load_files(self):
        """
        Load and parse the DOSCAR.LCFO, LCFO_Fragments, and structure file.
        All necessary data is obtained directly from the DOSCAR_LCFO parser.
        """
        doscar_path = f"{self.filepath}/DOSCAR.LCFO.lobster"
        fragments_path = f"{self.filepath}/LCFO_Fragments.lobster"
        poscar_path = f"{self.filepath}/POSCAR"

        if not os.path.exists(doscar_path):
            raise FileNotFoundError(f"DOSCAR.LCFO file not found in: {doscar_path}")
        if not os.path.exists(fragments_path):
            raise FileNotFoundError(f"LCFO_Fragments.lobster file not found in: {fragments_path}")
        if not os.path.exists(poscar_path):
            if os.path.exists(f"{self.filepath}/CONTCAR"):
                poscar_path = f"{self.filepath}/CONTCAR"
            else:
                raise FileNotFoundError(f"Neither POSCAR nor CONTCAR file found in: {self.filepath}")

        # Parse structure from POSCAR.
        self.lv, atom_coords, self.atomtypes, self.atomnums = parse_poscar(poscar_path)

        # Initialize DOSCAR_LCFO. Note that no MO diagram is passed.
        lcfo_obj = DOSCAR_LCFO(
            Path(doscar_path),
            Path(fragments_path),
            Path(poscar_path)  # structure_file argument: use POSCAR
        )

        # Debug information.
        '''
        print("Debug: LCFO Data Loaded")
        print("Energies:", lcfo_obj.energies)
        print("Fermi Energy:", lcfo_obj.fermi_energy)
        print("pMODOS:", lcfo_obj.pmodos)
        print("Fragments:", lcfo_obj._fragments)
        '''

        if lcfo_obj.energies is None or not lcfo_obj.energies.size:
            raise ValueError("DOSCAR_LCFO did not load valid energies data. Check file contents.")
        if lcfo_obj.pmodos is None or len(lcfo_obj.pmodos) == 0:
            raise ValueError("DOSCAR_LCFO did not load valid pMODOS data. Check file contents.")

        self.energies = np.array(lcfo_obj.energies)
        self.ef = lcfo_obj.fermi_energy

        # IMPORTANT: Instead of indexing pMODOS using fragment keys from LCFO_Fragments,
        # we use all of the pMODOS values. This avoids mismatches between keys.
        self._fragments = lcfo_obj._fragments
        self.coord = np.array([frag["coordinates"] for frag in self._fragments.values()])
        self.pdos = list(lcfo_obj.pmodos.values())

    def calculate_single_point_ldos(self, position, emin, emax, phi, V):
        """
        Calculates the local density of states (LDOS) at a single spatial position,
        restricted to an energy window, given a workfunction (phi) and applied voltage (V).
        Tunneling weights (exponential decay) are applied based on the distance between
        a tip position (derived from the mean fragment z-coordinate plus a tip displacement)
        and each fragment's representative coordinate.
        """
        tip_pos = np.array([
            position[0],
            position[1],
            np.mean(self.coord[:, 2]) + self.tip_disp
        ])

        if emax > max(self.energies):
            emax = max(self.energies)
        if emin < min(self.energies):
            emin = min(self.energies)

        self.estart = np.where(self.energies >= emin)[0][0]
        self.eend = np.where(self.energies <= emax)[0][-1] + 1
        energy_range = self.energies[self.estart:self.eend]

        ldos = {frag_idx: {} for frag_idx in range(len(self.coord))}

        for frag_idx, fragment_dos in enumerate(self.pdos):
            fragment_coords = self.coord[frag_idx]
            # Loop over each orbital registered in the pMODOS section.
            for orbital, spin_data in fragment_dos.items():
                if orbital not in ldos[frag_idx]:
                    ldos[frag_idx][orbital] = {Spin.up: np.zeros_like(energy_range)}
                    if Spin.down in spin_data:
                        ldos[frag_idx][orbital][Spin.down] = np.zeros_like(energy_range)
                for spin, dos_values in spin_data.items():
                    energy_filtered_dos = np.array(dos_values)[self.estart:self.eend]
                    distance = np.linalg.norm(tip_pos - fragment_coords)
                    tunneling_weights = np.array([
                        exp(tunneling_factor(abs(V), abs(E), phi) * (-1) * distance * 1e-10)
                        for E in energy_range
                    ])
                    ldos_contrib = energy_filtered_dos * tunneling_weights
                    ldos[frag_idx][orbital][spin] += ldos_contrib

        return ldos

    def plot_ldos_curve(self, ldos, emin, emax):
        """
        Plots the total and orbital‐resolved LDOS for each fragment over the specified energy window.
        Each orbital's curve is assigned a distinct color so that you can tell them apart.
        The legend uses the actual fragment names and orbital labels.
        """
        import matplotlib.pyplot as plt
    
        energy_range = self.energies[self.estart:self.eend]
        total_ldos = np.zeros_like(energy_range)
        areas = {}
        
        # First, calculate the integrated area for each orbital contribution.
        for frag_name, orbital_data in ldos.items():
            for orbital, spin_data in orbital_data.items():
                orbital_ldos = np.zeros_like(energy_range)
                for spin, ldos_values in spin_data.items():
                    orbital_ldos += ldos_values
                area = np.trapz(orbital_ldos, energy_range)
                areas[(frag_name, orbital)] = area
                total_ldos += orbital_ldos
    
        if not areas:
            print("No orbital contributions found in the LDOS data. " +
                  "Please verify that your LCFO DOSCAR file and pMODOS data are valid.")
            return
    
        max_area = max(areas.values())
        normalized_areas = {key: area / max_area for key, area in areas.items()}
        # Filter out contributions below threshold (you can adjust the threshold here)
        filtered_areas = {key: area for key, area in normalized_areas.items() if area > 0.1}
        significant_area_sum = sum(areas[key] for key in filtered_areas.keys())
        total_area = np.trapz(total_ldos, energy_range)
        if total_area > 0:
            total_ldos *= significant_area_sum / total_area
    
        plt.figure(figsize=(12, 8))
        plt.plot(energy_range, total_ldos, label="Total LDOS (Normalized)",
                 color="black", linewidth=2, linestyle="--")
    
        # Create a color map for the number of orbital curves in the filtered list.
        unique_curves = list(filtered_areas.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_curves)))
    
        # Plot each orbital curve with its distinct color.
        for i, ((frag_name, orbital), area) in enumerate(filtered_areas.items()):
            orbital_ldos = np.zeros_like(energy_range)
            for spin, ldos_values in ldos[frag_name][orbital].items():
                orbital_ldos += ldos_values
            plt.plot(energy_range, orbital_ldos,
                     label=f"{frag_name} - {orbital} (Norm. Area: {normalized_areas[(frag_name, orbital)]:.2f})",
                     alpha=0.7, color=colors[i])
    
        plt.xlabel("Energy (eV)")
        plt.ylabel("LDOS (states/eV)")
        plt.title("Filtered and Normalized LDOS (Total LDOS Scaled)")
        plt.legend(fontsize="small", loc="best", ncol=2, frameon=True)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    filepath = 'C:/directory'
    spatial_position = np.array([5.99343,  10.38093, 24.53685])
    emin, emax = -2.0, 1.5  # Energy window (eV)
    phi = 5.0885            # Workfunction (eV)
    V = 1.5                 # Applied voltage (eV)

    try:
        # The files are loaded automatically on instantiation.
        ldos_calc = ldos_single_point(filepath)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    ldos = ldos_calc.calculate_single_point_ldos(spatial_position, emin, emax, phi, V)
    ldos_calc.plot_ldos_curve(ldos, emin, emax)
