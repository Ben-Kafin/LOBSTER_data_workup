from pathlib import Path
import numpy as np
from collections import defaultdict
from pymatgen.electronic_structure.dos import Dos
from pymatgen.electronic_structure.core import Spin
from pymatgen.core.structure import Structure

class DOSCAR_LCFO:
    """
    Parses DOSCAR.LCFO files for TDOS, pMODOS, and optionally LDOS, while following Pymatgen's LobsterDoscar logic.
    Includes MO mapping from the MO diagram and fragment-to-structure mapping.
    """
    def __init__(self, doscar: Path, lcfo_fragments_path: Path, mo_diagram: dict, structure_file: Path = None):
        """
        Args:
            doscar (Path): Path to the DOSCAR.LCFO file.
            lcfo_fragments_path (Path): Path to the LCFO_Fragments.lobster file.
            mo_diagram (dict): MO diagram mapping molecular orbital names to keys.
            structure_file (Path): Optional path to a structure file (e.g., POSCAR) for LDOS.
        """

        # Assign attributes
        self._doscar = doscar
        self._lcfo_fragments_path = lcfo_fragments_path
        self._mo_diagram = mo_diagram
        self._structure = Structure.from_file(structure_file) if structure_file else None
    
        # Attributes to store parsed data
        self._energies = None
        self._tdos = None
        self._pmodos = None
        self._ldos = {}  # Initialize LDOS as an empty dictionary
        self._is_spin_polarized = None
    
        # Log progress
        print(f"Parsing DOSCAR.LCFO from: {doscar}")
        print(f"Parsing LCFO_Fragments from: {lcfo_fragments_path}")
        if structure_file:
            print(f"Using structure file (POSCAR) from: {structure_file}")
    
        # Parse the files
        self._fragments = self._parse_lcfo_fragments()
        self._parse_doscar()

    def _parse_lcfo_fragments(self):
        """
        Parses the LCFO_Fragments.lobster file to extract fragment and orbital mappings.
        Returns:
            dict: A dictionary mapping fragment names to orbital names and atomic indices.
        """
        fragments = {}
        current_fragment = None

        with open(self._lcfo_fragments_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Atoms forming fragment"):
                    current_fragment = line.split("on")[0].strip().replace("Atoms forming ", "")
                    fragments[current_fragment] = {"orbitals": [], "atoms": []}
                    continue

                if "_" in line and any(char.isdigit() for char in line):  # Parse atomic indices
                    atom_name = line.split()[0].strip()
                    atomic_index = int(atom_name.split("_")[1]) - 1  # Convert to 0-based index
                    fragments[current_fragment]["atoms"].append(atomic_index)
                    continue

                if ";" in line:  # Parse orbitals
                    parts = line.split(";")
                    fragment_name = parts[0].strip()
                    orbitals = parts[1].strip().split()
                    # Append molecular orbitals if the fragment is in the MO diagram
                    if fragment_name in self._mo_diagram:
                        detailed_orbitals = [
                            f"{fragment_name}_1_{orb}" for orb in orbitals
                        ]
                        fragments[fragment_name]["orbitals"] = detailed_orbitals
                    else:
                        fragments[fragment_name]["orbitals"] = orbitals

        return fragments

    def _parse_doscar(self):
        """
        Parses the DOSCAR.LCFO file to extract TDOS and pMODOS data.
        Renames 'Au' fragments dynamically based on their corresponding atom number from the POSCAR file.
        """
        with open(self._doscar, "r") as file:
            # Skip the first 5 lines (headers and metadata)
            for _ in range(5):
                file.readline()
    
            # Parse the 6th line for Fermi energy
            fermi_line = file.readline().strip()
            try:
                efermi = float(fermi_line.split()[3])  # Fermi energy is the 4th value
            except (IndexError, ValueError):
                raise ValueError("Could not parse Fermi energy from the 6th line of DOSCAR.LCFO.")
    
            # Initialize arrays for TDOS
            energies = []
            spin_up_densities = []
            spin_down_densities = []
    
            # Parse TDOS while skipping metadata lines
            while True:
                line = file.readline().strip()
                if ";" in line:  # Start of pMODOS metadata
                    break
                if not line or any(char.isalpha() for char in line.split()):  # Skip metadata lines
                    continue
                columns = [float(x) for x in line.split()]
                energies.append(columns[0])  # Energy column
                spin_up_densities.append(columns[1])  # Spin-up density
                if len(columns) > 2:  # Spin-down density (if spin-polarized)
                    spin_down_densities.append(columns[2])
    
            # Determine spin polarization
            self._is_spin_polarized = len(spin_down_densities) > 0
            tdensities = {Spin.up: np.array(spin_up_densities)}
            if self._is_spin_polarized:
                tdensities[Spin.down] = np.array(spin_down_densities)
    
            # Assign TDOS and energies
            self._tdos = Dos(efermi, np.array(energies), tdensities)
            self._energies = np.array(energies)
    
            # Parse fragments and orbitals sequentially for pMODOS
            pmodos = defaultdict(lambda: defaultdict(lambda: {Spin.up: [], Spin.down: []}))
            fragments = list(self._fragments.keys())  # Get fragments in sequential order
    
            fragment_index = 0  # Start with the first fragment
            while line:
                if ";" in line:  # Fragment and orbital metadata
                    # Split the metadata line by semicolons
                    metadata = line.split(";")
                    if len(metadata) < 3:
                        raise ValueError(f"Invalid metadata format: {line}")
    
                    fragment_name = metadata[1].strip()  # Fragment name comes after the first semicolon
                    orbital_section = metadata[2].strip()  # Orbital names are after the second semicolon
    
                    # Skip "X" and extract only actual orbital names
                    orbital_parts = orbital_section.split()
                    orbitals = [orb for orb in orbital_parts if orb != "X"]  # Exclude "X"
                    num_orbitals = len(orbitals)  # Dynamically determine number of orbitals
    
                    # Map fragment to structure sequentially and rename fragment if it's 'Au'
                    fragment_atoms = self._fragments.get(fragments[fragment_index], {}).get("atoms", [])
                    if fragment_name == 'Au':
                        fragment_name = f"Au_{fragment_atoms[0]}"  # Rename fragment using atom number
                    print(f"Processed fragment '{fragment_name}' with atoms: {fragment_atoms} and orbitals: {orbitals}")
                    fragment_index += 1  # Move to the next fragment in sequential order
                    line = file.readline()  # Move to numeric rows
                    continue
    
                # Parse numeric rows for pMODOS densities
                rows_read = 0  # Track the number of rows processed
                while line and (";" not in line or fragment_index == len(fragments)):  # Keep reading numeric rows
                    columns = [float(x) for x in line.split()]
                    expected_columns = 1 + 2 * num_orbitals  # Calculate expected columns dynamically
    
                    if len(columns) != expected_columns:  # Validate column count
                        raise ValueError(f"Column mismatch for fragment '{fragment_name}'. "
                                         f"Expected {expected_columns} columns, but got {len(columns)}.")
    
                    energy = columns[0]
                    for i, orbital in enumerate(orbitals):
                        spin_up_col = (i * 2) + 1
                        spin_down_col = (i * 2) + 2
                        pmodos[fragment_name][orbital][Spin.up].append(columns[spin_up_col])
                        if self._is_spin_polarized:
                            pmodos[fragment_name][orbital][Spin.down].append(columns[spin_down_col])
                    rows_read += 1
    
                    # Read the next line or stop processing
                    line = file.readline().strip()
    
                    # If the end of file is reached, terminate numeric row processing
                    if not line:
                        print(f"End of file reached for fragment '{fragment_name}'")
                        break
    
                # Debug output for each pMODOS row
                print(f"Processed {rows_read} rows for fragment '{fragment_name}'")
    
            # Validate energy-density alignment for pMODOS
            for fragment, orbital_data in pmodos.items():
                for orbital, spin_data in orbital_data.items():
                    if len(spin_data[Spin.up]) != len(self._energies):
                        raise ValueError(f"Mismatch for fragment '{fragment}', orbital '{orbital}': "
                                         f"{len(spin_data[Spin.up])} densities vs {len(self._energies)} energies.")
                    if self._is_spin_polarized and len(spin_data[Spin.down]) != len(self._energies):
                        raise ValueError(f"Spin Down mismatch for fragment '{fragment}', orbital '{orbital}': "
                                         f"{len(spin_data[Spin.down])} densities vs {len(self._energies)} energies.")
    
            # Assign parsed pMODOS
            self._pmodos = pmodos
        
        
    @property
    def tdos(self) -> Dos:
        """Returns the Total Density of States (TDOS)."""
        return self._tdos

    @property
    def pmodos(self) -> dict:
        """Returns the Projected Molecular Orbital DOS (pMODOS)."""
        return self._pmodos

    @property
    def energies(self) -> np.ndarray:
        """Returns the energies associated with the DOS."""
        return self._energies

    @property
    def is_spin_polarized(self) -> bool:
        """Returns whether the system is spin-polarized."""
        return self._is_spin_polarized