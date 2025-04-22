# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:22:57 2025

@author: Benjamin Kafin
Revised to remove iMOFE logic.
"""
import numpy as np

class MOPM:
    def __init__(self, mo_diagram_simple_path, mo_diagram_complex_path, simple_atoms, complex_atoms, output_file_path, criteria):
        """
        Initialize the MOPM class by loading the MO diagrams for both the
        simple and complex systems. Complex atoms and AOs are filtered to match
        the simple system.
        
        Args:
            mo_diagram_simple_path (str): File path for the simple system MO diagram.
            mo_diagram_complex_path (str): File path for the complex system MO diagram.
            simple_atoms (list): List or shorthand of atoms in the simple system.
            complex_atoms (list): List or shorthand of atoms in the complex system.
            output_file_path (str): Path to write the output match file.
            criteria (str): Occupation filtering criteria, e.g., "simple:occupied, complex:unoccupied".
        """
        # Expand shorthand atom input, if necessary.
        self.simple_atoms = MOPM.expand_atom_list(simple_atoms) if isinstance(simple_atoms[0], tuple) else simple_atoms
        self.complex_atoms = MOPM.expand_atom_list(complex_atoms) if isinstance(complex_atoms[0], tuple) else complex_atoms
    
        # Filter complex atoms to only those present in the simple system.
        self.complex_atoms = self.filter_complex_atoms()
    
        # Load the MO diagrams for both systems.
        self.mo_diagram_simple_spin_1, self.mo_diagram_simple_spin_2 = self.load_mo_diagram(mo_diagram_simple_path, "simple")
        self.mo_diagram_complex_spin_1, self.mo_diagram_complex_spin_2 = self.load_mo_diagram(mo_diagram_complex_path, "complex")
    
        # Validate AO list lengths for the MO diagrams.
        if len(self.mo_diagram_simple_spin_1[0]['ao_contributions']) != len(self.mo_diagram_complex_spin_1[0]['ao_contributions']):
            raise ValueError("Filtered AO lists in simple and complex systems do not match in length.")
    
        # Perform the MO match comparison (excluding any iMOFE logic).
        self.final_filtered_matches = self.compare_mo_contributions(output_file_path)
    
        # Filter matches based on occupation criteria (e.g., "simple:occupied, complex:unoccupied").
        self.filtered_occupation_matches = self.filter_matches_occupation(self.final_filtered_matches, criteria)
    
    @staticmethod
    def expand_atom_list(atom_counts):
        """
        Expand shorthand atomic input into a full atom list.
    
        Args:
            atom_counts (list of tuples): Each tuple contains the count and the atomic type,
                                          e.g., [(6, 'N'), (4, 'C'), (10, 'H')].
    
        Returns:
            list: Expanded atom list.
        """
        expanded_list = []
        for count, atom_type in atom_counts:
            expanded_list.extend([atom_type] * count)
        return expanded_list
    
    def filter_complex_atoms(self):
        """
        Filter complex atoms to include only those present in the simple atom list.
    
        Returns:
            list: Filtered complex atom list.
        """
        return [atom for atom in self.complex_atoms if atom in self.simple_atoms]
    
    def filter_complex_ao_contributions(self, complex_spin_data):
        """
        Filter complex AO contributions to exclude AOs from atoms not present
        in the simple system.
    
        Args:
            complex_spin_data (list): MO data for one spin channel.
    
        Returns:
            list: Filtered MO data for the complex system.
        """
        filtered_data = []
        for mo in complex_spin_data:
            try:
                filtered_contributions = []
                filtered_identifiers = []
                for i, ao_id in enumerate(mo['ao_identifiers']):
                    # Extract the atom type (letters before the first digit or '_').
                    atom_type = ''.join([char for char in ao_id if char.isalpha() and char.isupper()])
                    if atom_type in self.complex_atoms:
                        filtered_contributions.append(mo['ao_contributions'][i])
                        filtered_identifiers.append(ao_id)
                if filtered_contributions:
                    filtered_data.append({
                        'index': mo['index'],
                        'name': mo['name'],
                        'energy': mo['energy'],
                        'ao_contributions': np.array(filtered_contributions),
                        'ao_identifiers': filtered_identifiers,
                    })
            except Exception as e:
                print(f"Error filtering AO contributions for MO '{mo['name']}': {e}")
                continue
        if not filtered_data:
            raise ValueError("No valid AOs remain after filtering. Verify atom type consistency.")
        return filtered_data
    
    def filter_complex_spin(self, complex_spin_data, simple_atoms):
        """
        Filter the complex spin channel based on simple atom types.
    
        Args:
            complex_spin_data (list): MO data for one spin channel.
            simple_atoms (list): Atom types present in the simple system.
    
        Returns:
            list: Filtered complex spin channel data.
        """
        filtered_data = []
        for mo in complex_spin_data:
            try:
                filtered_contributions = []
                filtered_identifiers = []
                for i, ao_id in enumerate(mo['ao_identifiers']):
                    atom_type = ao_id.split('_')[0].rstrip('0123456789')
                    if atom_type in simple_atoms:
                        filtered_contributions.append(mo['ao_contributions'][i])
                        filtered_identifiers.append(ao_id)
                if filtered_contributions:
                    filtered_data.append({
                        'index': mo['index'],
                        'name': mo['name'],
                        'energy': mo['energy'],
                        'ao_contributions': np.array(filtered_contributions),
                        'ao_identifiers': filtered_identifiers,
                    })
            except Exception as e:
                print(f"Error filtering MO data for '{mo['name']}': {e}")
                continue
            if not filtered_data:
                print("Filtered Complex Atom List:", simple_atoms)
                raise ValueError("No valid complex spin data remain after filtering.")
        return filtered_data
    
    @staticmethod
    def load_mo_diagram(file_path, system_type):
        """
        Load a molecular orbital diagram (MO index, name, energy, AO contributions)
        from a file. Automatically detects a second spin channel by locating the
        second non-indented line.
    
        Args:
            file_path (str): Path to the MO diagram file.
            system_type (str): 'simple' or 'complex'.
    
        Returns:
            tuple: MO data for Spin 1 and Spin 2 channels.
        """
        mo_data_spin_1 = []
        mo_data_spin_2 = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
    
        # Determine where the second spin channel begins.
        spin_1_start = 0
        spin_2_start = None
        non_indented_count = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                non_indented_count += 1
                if non_indented_count == 2:
                    spin_2_start = i
                    break
        if spin_2_start is None:
            spin_2_start = len(lines)
    
        # Parse the first spin channel.
        spin_1_lines = lines[spin_1_start:spin_2_start]
        mo_data_spin_1 = MOPM.parse_spin_dataset(spin_1_lines, system_type, "Spin 1")
    
        # Parse the second spin channel, if it exists.
        if spin_2_start < len(lines):
            spin_2_lines = lines[spin_2_start:]
            mo_data_spin_2 = MOPM.parse_spin_dataset(spin_2_lines, system_type, "Spin 2")
    
        return mo_data_spin_1, mo_data_spin_2
    
    @staticmethod
    def parse_spin_dataset(lines, system_type, spin_channel):
        """
        Parse MO data for a single spin channel. This function extracts MO names,
        energies, AO identifiers, and AO contributions from the given text.
    
        Args:
            lines (list): Lines of text from the MO diagram file.
            system_type (str): 'simple' or 'complex'.
            spin_channel (str): Identifier for the spin channel (e.g., "Spin 1").
    
        Returns:
            list: Structured MO data.
        """
        mo_data = []
        ao_identifiers = []
        ao_contributions = []
    
        # Extract MO names from the second line.
        try:
            mo_names = lines[1].strip().split()
        except IndexError:
            raise ValueError("MO names line is missing or improperly formatted.")
    
        # Extract energies from the third line (after 'Energy (eV)').
        try:
            energies = list(map(float, lines[2].strip().split()[2:]))
        except (IndexError, ValueError):
            raise ValueError("Energy values line is missing or improperly formatted.")
    
        # Parse AO identifiers and contributions from subsequent lines.
        for line in lines[3:]:
            row = line.strip().split()
            if len(row) >= 2:  # Now only one value is skipped (the identifier); the rest are contributions.
                ao_identifiers.append(row[0])
                try:
                    ao_contributions.append(list(map(float, row[1:])))
                except ValueError as e:
                    print(f"Skipping malformed line: {line.strip()} ({e})")
                    ao_contributions.append([0.0] * (len(row) - 1))
    
        if ao_contributions:
            ao_contributions = np.array(ao_contributions)
        else:
            raise ValueError("No valid AO contributions found in the file.")
    
        # Combine MO names, energies, and AO data into a structured format.
        for i, name in enumerate(mo_names):
            mo_data.append({
                'index': i,
                'name': name,
                'energy': energies[i],
                'ao_contributions': ao_contributions[:, i] if ao_contributions.size else np.array([]),
                'ao_identifiers': ao_identifiers,
            })
    
        return mo_data
    
    def compare_mo_contributions(self, output_file_path=None):
        """
        Compare each complex MO to all simple MOs to find the best match based solely
        on the normalized AO contributions. The best match is determined by the smallest 
        distance of the AO overlap from ±1.
    
        Args:
            output_file_path (str, optional): Path to output the match results.
    
        Returns:
            list: A list of best-matched MO pairs.
        """
        matches = []
    
        # For each MO in the complex system's first spin channel...
        for complex_mo in self.mo_diagram_complex_spin_1:
            best_match = None
            best_overlap_distance = float('inf')
    
            # ...compare with every MO in the simple system's first spin channel.
            for simple_mo in self.mo_diagram_simple_spin_1:
                if len(complex_mo['ao_contributions']) != len(simple_mo['ao_contributions']):
                    raise ValueError("Mismatch in AO contributions array length between complex and simple MO.")
    
                complex_contributions = np.array(complex_mo['ao_contributions'])
                simple_contributions = np.array(simple_mo['ao_contributions'])
    
                complex_normalized = complex_contributions / np.linalg.norm(complex_contributions)
                simple_normalized = simple_contributions / np.linalg.norm(simple_contributions)
    
                # Compute the dot product (AO overlap).
                ao_projection = np.dot(complex_normalized, simple_normalized)
    
                # Determine the “distance” from perfect overlap.
                projection_distance = min(abs(1 - ao_projection), abs(-1 - ao_projection))
    
                if projection_distance < best_overlap_distance:
                    best_overlap_distance = projection_distance
                    best_match = {
                        'complex_mo': complex_mo['name'],
                        'complex_mo_energy': complex_mo['energy'],
                        'simple_mo': simple_mo['name'],
                        'simple_mo_energy': simple_mo['energy'],
                        'ao_overlap': ao_projection,
                        'energy_shift': complex_mo['energy'] - simple_mo['energy'],
                    }
    
            if best_match:
                matches.append(best_match)
            else:
                print(f"No match found for Complex MO: {complex_mo['name']}")
    
        # In this version, no additional filtering is done.
        filtered_matches = matches
    
        if output_file_path:
            with open(output_file_path, 'w') as f:
                f.write("Complex MO\tComplex Energy (eV)\tSimple MO\tSimple Energy (eV)\tAO Overlap\tEnergy Shift (eV)\n")
                for match in filtered_matches:
                    f.write(f"{match['complex_mo']}\t{match['complex_mo_energy']:.4f}\t"
                            f"{match['simple_mo']}\t{match['simple_mo_energy']:.4f}\t"
                            f"{match['ao_overlap']:.4f}\t{match['energy_shift']:.4f}\n")
    
        return filtered_matches
    
    def filter_matches_occupation(self, filtered_matches, criteria="simple:both, complex:both"):
        """
        Filter the list of matches based on the occupation status of the simple and complex systems.
    
        Args:
            filtered_matches (list): List of matched MO pairs.
            criteria (str): Filtering criteria in the format "simple:X, complex:Y", where
                            X and Y can be "occupied", "unoccupied", or "both".
    
        Returns:
            list: Matches filtered according to the occupation criteria.
        """
        if not filtered_matches:
            raise ValueError("No matches provided to filter. Ensure the filtered_matches list is valid.")
    
        try:
            simple_filter = criteria.split(",")[0].split(":")[1].strip().lower()
            complex_filter = criteria.split(",")[1].split(":")[1].strip().lower()
        except IndexError:
            raise ValueError("Invalid criteria format. Use 'simple:X, complex:Y'.")
    
        # Filter based on the simple system's MO energy.
        if simple_filter == "occupied":
            filtered = [match for match in filtered_matches if match['simple_mo_energy'] < 0]
        elif simple_filter == "unoccupied":
            filtered = [match for match in filtered_matches if match['simple_mo_energy'] >= 0]
        elif simple_filter == "both":
            filtered = filtered_matches
        else:
            raise ValueError("Invalid simple system filter. Use 'occupied', 'unoccupied', or 'both'.")
    
        # Filter based on the complex system's MO energy.
        if complex_filter == "occupied":
            filtered = [match for match in filtered if match['complex_mo_energy'] < 0]
        elif complex_filter == "unoccupied":
            filtered = [match for match in filtered if match['complex_mo_energy'] >= 0]
        elif complex_filter == "both":
            pass  # No change needed.
        else:
            raise ValueError("Invalid complex system filter. Use 'occupied', 'unoccupied', or 'both'.")
    
        return filtered
    
'''
# Example usage:

# File paths for the simple and complex systems.
mo_diagram_simple_path = 'path/to/simple_system/MO_Diagram.lobster'
mo_diagram_complex_path = 'path/to/complex_system/MO_Diagram.lobster'

# Atom lists for the simple and complex systems.
simple_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]
complex_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]

# Define the output file path and occupation criteria.
output_file_path = 'path/to/output/matches.txt'
criteria = "simple:both, complex:both"

# Initialize the MOPM instance.
MOPM_instance = MOPM(mo_diagram_simple_path, mo_diagram_complex_path, simple_atoms, complex_atoms, output_file_path, criteria)

# Retrieve and display the matches.
matches = MOPM_instance.final_filtered_matches
print("\nFiltered Matches Based on AO Overlap:")
for match in matches:
    print(f"Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
          f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
          f"AO Overlap: {match['ao_overlap']:.4f}, "
          f"Energy Shift: {match['energy_shift']:.4f} eV")
'''