import os
from os.path import exists
import numpy as np
from numpy import exp
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
        latticevectors = [float(lines[i].split()[j]) * sf 
                          for i in range(2, 5) for j in range(3)]
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
    Calculate the tunneling factor based on applied voltage (V),
    energy (E) relative to the Fermi level and the workfunction (phi).
    """
    V *= 1.60218e-19   # from eV to Joules
    E *= 1.60218e-19
    phi *= 1.60218e-19
    m_e = 9.11e-31     # electron mass in kg
    hbar = 6.626e-34   # Planck's constant in J·s
    prefactor = (8 / (3 * V)) * np.pi * np.sqrt(2 * m_e) / hbar
    barrier = (phi - E + V)**(3/2) - (phi - E)**(3/2)
    return prefactor * barrier

# --- Main LDOS Calculator Class ---
class ldos_single_point:
    def __init__(self, filepath):
        """
        Initialize the single-point LDOS calculator.

        Parameters:
            filepath (str): Path to the folder containing output files.
              Expected files: DOSCAR.LCFO.lobster, LCFO_Fragments.lobster, and either POSCAR or CONTCAR.
        """
        self.filepath = filepath
        self.lv = None               # Lattice vectors from POSCAR
        self.coord = {}              # Mapping: fragment key (string) -> Cartesian coordinate (np.array)
        self.atom_coords = None      # All atomic coordinates from POSCAR
        self.atomtypes = None        # Atom types from POSCAR
        self.atomnums = None         # Atom numbers from POSCAR
        self.energies = None         # Energies from DOSCAR.LCFO
        self.ef = None               # Fermi energy from DOSCAR.LCFO
        self.pdos = None             # pMODOS dictionary (mapping fragment key -> PDOS dict)
        self.estart = None           # Start index for energy window
        self.eend = None             # End index for energy window
        self.fragment_names = None   # List of fragment keys (used for labeling)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Specified filepath does not exist: {filepath}")
        self.load_files()

    def load_files(self):
        """
        Load and parse the DOSCAR.LCFO, LCFO_Fragments, and structure (POSCAR/CONTCAR) files.
        Data is obtained via the DOSCAR_LCFO parser.
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
        self.lv, self.atom_coords, self.atomtypes, self.atomnums = parse_poscar(poscar_path)

        # Initialize DOSCAR_LCFO; note no MO diagram is used.
        lcfo_obj = DOSCAR_LCFO(Path(doscar_path), Path(fragments_path), Path(poscar_path))

        # (Optional) Debug prints:
        # print("Debug: LCFO Data Loaded")
        # print("Energies:", lcfo_obj.energies)
        # print("Fermi Energy:", lcfo_obj.fermi_energy)
        # print("pMODOS keys:", list(lcfo_obj.pmodos.keys()))
        # print("LCFO Fragments:", lcfo_obj._fragments)

        if lcfo_obj.energies is None or not lcfo_obj.energies.size:
            raise ValueError("DOSCAR_LCFO did not load valid energies data.")
        if lcfo_obj.pmodos is None or len(lcfo_obj.pmodos) == 0:
            raise ValueError("DOSCAR_LCFO did not load valid pMODOS data.")

        self.energies = np.array(lcfo_obj.energies)
        self.ef = lcfo_obj.fermi_energy
        self.pdos = lcfo_obj.pmodos  # Keep this as a dictionary

        # Build a mapping from the pMODOS keys (fragment names from DOSCAR metadata)
        # to representative Cartesian coordinates from the LCFO_Fragments parsing.
        frag_keys = list(lcfo_obj._fragments.keys())
        pmodos_keys = list(self.pdos.keys())
        if set(pmodos_keys).issubset(set(frag_keys)):
            mapping = {k: lcfo_obj._fragments[k]["coordinates"] for k in pmodos_keys}
        else:
            # Assume the ordering is the same
            ordered_coords = [lcfo_obj._fragments[k]["coordinates"] for k in frag_keys]
            n = min(len(pmodos_keys), len(ordered_coords))
            mapping = {pmodos_keys[i]: ordered_coords[i] for i in range(n)}
            # If there are extra keys, assign the last available coordinate.
            for i in range(n, len(pmodos_keys)):
                mapping[pmodos_keys[i]] = ordered_coords[-1]
        self.coord = mapping
        self.fragment_names = list(self.pdos.keys())

    def calculate_single_point_ldos(self, position, emin, emax, phi, V):
        """
        Calculates the LDOS at the provided spatial tip position (exactly as given)
        within an energy window [emin, emax]. Tunneling weights are computed from the distance
        between the tip position and each fragment's coordinate.
        """
        # Use the provided tip position.
        tip_pos = np.array(position)

        if emax > max(self.energies):
            emax = max(self.energies)
        if emin < min(self.energies):
            emin = min(self.energies)
        self.estart = np.where(self.energies >= emin)[0][0]
        self.eend = np.where(self.energies <= emax)[0][-1] + 1
        energy_range = self.energies[self.estart:self.eend]

        ldos = {}
        # Iterate over PDOS dictionary keys (actual fragment names)
        for frag_name, fragment_dos in self.pdos.items():
            frag_coord = self.coord.get(frag_name)
            if frag_coord is None:
                print(f"Warning: No coordinate for fragment {frag_name}. Skipping.")
                continue
            ldos[frag_name] = {}
            for orbital, spin_data in fragment_dos.items():
                ldos[frag_name][orbital] = {Spin.up: np.zeros_like(energy_range)}
                if Spin.down in spin_data:
                    ldos[frag_name][orbital][Spin.down] = np.zeros_like(energy_range)
                for spin, dos_values in spin_data.items():
                    energy_filtered_dos = np.array(dos_values)[self.estart:self.eend]
                    distance = np.linalg.norm(tip_pos - frag_coord)
                    tunneling_weights = np.array([
                        exp(tunneling_factor(abs(V), abs(E), phi) * (-1) * distance * 1e-10)
                        for E in energy_range
                    ])
                    ldos_contrib = energy_filtered_dos * tunneling_weights
                    ldos[frag_name][orbital][spin] += ldos_contrib
        return ldos

    def plot_ldos_curve(self, ldos, emin, emax):
        """
        Plots the total and orbital-resolved LDOS for each fragment over the energy window.
        The legend uses the actual fragment names (the keys from the pMODOS dictionary).
        """
        import matplotlib.pyplot as plt
        energy_range = self.energies[self.estart:self.eend]
        total_ldos = np.zeros_like(energy_range)
        areas = {}
        atom_colors = plt.cm.tab10(np.linspace(0, 1, len(self.fragment_names)))

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
        # Filter out contributions below a threshold.
        filtered_areas = {key: area for key, area in normalized_areas.items() if area > 0.25}
        significant_area_sum = sum(areas[key] for key in filtered_areas.keys())
        total_area = np.trapz(total_ldos, energy_range)
        if total_area > 0:
            total_ldos *= significant_area_sum / total_area

        plt.figure(figsize=(12, 8))
        plt.plot(energy_range, total_ldos, label="Total LDOS (Normalized)", 
                 color="black", linewidth=2, linestyle="--")

        for (frag_name, orbital), area in filtered_areas.items():
            orbital_ldos = np.zeros_like(energy_range)
            for spin, ldos_values in ldos[frag_name][orbital].items():
                orbital_ldos += ldos_values
            color_idx = self.fragment_names.index(frag_name)
            plt.plot(energy_range, orbital_ldos,
                     label=f"{frag_name} - {orbital} (Norm. Area: {normalized_areas[(frag_name, orbital)]:.2f})",
                     alpha=0.7, color=atom_colors[color_idx])
        plt.xlabel("Energy (eV)")
        plt.ylabel("LDOS (states/eV)")
        plt.title("Filtered and Normalized LDOS (Total LDOS Scaled)")
        plt.legend(fontsize="small", loc="best", ncol=2, frameon=True)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    filepath = 'directory/'
    spatial_position = np.array([5.99343, 10.38093, 24.53685])
    emin, emax = -2.0, 2.0   # Energy window in eV
    phi = 5.0885             # Workfunction in eV
    V = 2.0                  # Applied voltage in eV

    try:
        ldos_calc = ldos_single_point(filepath)
    except FileNotFoundError as e:
        print(e)
        import sys
        sys.exit()

    ldos = ldos_calc.calculate_single_point_ldos(spatial_position, emin, emax, phi, V)
    ldos_calc.plot_ldos_curve(ldos, emin, emax)
