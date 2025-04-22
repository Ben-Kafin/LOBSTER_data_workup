# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:52:32 2025

@author: nazin_lab
"""
from pathlib import Path
import numpy as np
from collections import defaultdict
from pymatgen.electronic_structure.dos import Dos
from pymatgen.electronic_structure.core import Spin
from pymatgen.core.structure import Structure

class DOSCAR_LCFO:
    """
    Parses DOSCAR.LCFO files for TDOS, pMODOS, and optionally LDOS,
    following Pymatgen's LobsterDoscar logic. It handles fragment-to-structure
    mapping using the LCFO_Fragments file.
    """
    def __init__(self, doscar: Path, lcfo_fragments_path: Path, structure_file: Path = None):
        """
        Args:
            doscar (Path): Path to the DOSCAR.LCFO file.
            lcfo_fragments_path (Path): Path to the LCFO_Fragments.lobster file.
            structure_file (Path): Optional path to a structure file (e.g., POSCAR) for LDOS.
        """
        # Assign attributes
        self._doscar = doscar
        self._lcfo_fragments_path = lcfo_fragments_path
        self._structure = Structure.from_file(structure_file) if structure_file else None

        # Attributes to store parsed data
        self._energies = None
        self._tdos = None
        self._pmodos = None
        self._ldos = {}  # Initialize LDOS as an empty dictionary
        self._is_spin_polarized = None

        # Parse the files
        self._fragments = self._parse_lcfo_fragments()
        self._parse_doscar()

    def _parse_lcfo_fragments(self):
        """
        Parses the LCFO_Fragments.lobster file to extract fragment mappings.
        For each fragment it extracts the fragment name, atomic indices, and
        representative Cartesian coordinates (converted from fractional coordinates
        using the POSCAR lattice vectors). This method does not attempt to load
        orbital names from the fragments file.
        
        Returns:
            dict: A dictionary mapping fragment names to a dictionary containing:
                  - "atoms": list of atomic indices (0-based)
                  - "coordinates": representative Cartesian coordinate for the fragment.
                  - "orbitals": an empty list (to be filled from the DOSCAR pMODOS metadata).
        """
        fragments = {}
        current_fragment = None
        lattice_vectors = self._structure.lattice.matrix if self._structure else None
        atom_coords = np.array(self._structure.cart_coords) if self._structure else None

        with open(self._lcfo_fragments_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                # Parse fragment name and fractional coordinates from header line.
                if line.startswith("Atoms forming fragment"):
                    parts = line.split("on")
                    current_fragment = parts[0].strip().replace("Atoms forming ", "")
                    frag_coords = np.array([float(x) for x in parts[1].split()[:3]])
                    fragments[current_fragment] = {"orbitals": [], "atoms": [], "coordinates": None}
                    if lattice_vectors is not None:
                        cartesian_coords = np.dot(frag_coords, lattice_vectors)
                        fragments[current_fragment]["coordinates"] = cartesian_coords
                    continue

                # Parse atomic indices for the fragment.
                if "_" in line and any(char.isdigit() for char in line):
                    atom_name = line.split()[0].strip()
                    atomic_index = int(atom_name.split("_")[1]) - 1  # convert to 0-based index
                    fragments[current_fragment]["atoms"].append(atomic_index)
                    continue

                # (Remove processing of lines with ";" for orbital metadata.)
        # For single-atom fragments, use POSCAR coordinates directly.
        for fragment_name, fragment_data in fragments.items():
            atom_indices = fragment_data["atoms"]
            if len(atom_indices) == 1 and atom_coords is not None:
                fragments[fragment_name]["coordinates"] = atom_coords[atom_indices[0]]
    
        return fragments

    def _parse_doscar(self):
        """
        Parses the DOSCAR.LCFO file to extract TDOS and pMODOS data.
        Handles spin polarization, integrated densities, and assigns global energies.
        Also, it parses the pMODOS block from the file, which contains the orbital names (as provided in the DOSCAR).
        """
        with open(self._doscar, "r") as file:
            # Skip the first 5 lines (headers and metadata).
            for _ in range(5):
                file.readline()

            # Parse the 6th line for Fermi energy.
            fermi_line = file.readline().strip()
            try:
                efermi = float(fermi_line.split()[3])
            except (IndexError, ValueError):
                raise ValueError("Could not parse Fermi energy from the 6th line of DOSCAR.LCFO.")
            self.fermi_energy = efermi

            # Initialize arrays for TDOS.
            energies = []
            spin_up_densities = []
            spin_down_densities = []

            # Parse TDOS while skipping metadata lines.
            while True:
                line = file.readline().strip()
                if ";" in line:  # Reaching start of pMODOS metadata.
                    break
                if not line or any(char.isalpha() for char in line.split()):
                    continue
                columns = [float(x) for x in line.split()]

                # Determine spin polarization: 3 columns (non-spin) or 5 columns (spin-polarized).
                if len(columns) == 3:
                    self._is_spin_polarized = False
                elif len(columns) == 5:
                    self._is_spin_polarized = True
                else:
                    raise ValueError(f"Unexpected number of columns in TDOS row: {len(columns)}.")

                energies.append(columns[0])
                spin_up_densities.append(columns[1])
                if self._is_spin_polarized:
                    spin_down_densities.append(columns[2])

            # Assign TDOS densities.
            tdensities = {Spin.up: np.array(spin_up_densities)}
            if self._is_spin_polarized:
                tdensities[Spin.down] = np.array(spin_down_densities)
            self._tdos = Dos(efermi, np.array(energies), tdensities)
            self._energies = np.array(energies)

            # Parse the pMODOS block.
            pmodos = defaultdict(lambda: defaultdict(lambda: {Spin.up: [], Spin.down: []}))
            fragments = list(self._fragments.keys())
            fragment_index = 0

            while line:
                if ";" in line:
                    # Split metadata by semicolon.
                    metadata = line.split(";")
                    if len(metadata) < 3:
                        raise ValueError(f"Invalid metadata format: {line}")

                    fragment_name = metadata[1].strip()  # Fragment name from metadata.
                    orbital_section = metadata[2].strip()  # Orbital names from metadata.

                    # Here, orbital names come directly from the DOSCAR.
                    orbital_parts = orbital_section.split()
                    orbitals = [orb for orb in orbital_parts if orb != "X"]
                    num_orbitals = len(orbitals)

                    # Rename fragment for specific cases (e.g., Au) if necessary.
                    fragment_atoms = self._fragments.get(fragments[fragment_index], {}).get("atoms", [])
                    if fragment_name == 'Au':
                        fragment_name = f"Au_{fragment_atoms[0]}"
                    fragment_index += 1
                    line = file.readline()
                    continue

                rows_read = 0
                while line and (";" not in line or fragment_index == len(fragments)):
                    columns = [float(x) for x in line.split()]
                    if self._is_spin_polarized:
                        expected_columns = 1 + 2 * num_orbitals
                    else:
                        expected_columns = 1 + num_orbitals

                    if len(columns) != expected_columns:
                        raise ValueError(f"Column mismatch for fragment '{fragment_name}'. "
                                         f"Expected {expected_columns} columns, but got {len(columns)}.")

                    energy = columns[0]
                    for i, orbital in enumerate(orbitals):
                        if not self._is_spin_polarized:
                            spin_up_col = i + 1
                            pmodos[fragment_name][orbital][Spin.up].append(columns[spin_up_col])
                        else:
                            spin_up_col = (i * 2) + 1
                            spin_down_col = (i * 2) + 2
                            pmodos[fragment_name][orbital][Spin.up].append(columns[spin_up_col])
                            pmodos[fragment_name][orbital][Spin.down].append(columns[spin_down_col])
                    rows_read += 1
                    line = file.readline().strip()
                # End numeric block for one fragment.
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