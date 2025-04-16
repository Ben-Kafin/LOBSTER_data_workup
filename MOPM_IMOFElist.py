# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:22:57 2025

@author: Benjamin Kafin
"""
import numpy as np

class MOPM:
    def __init__(self, mo_diagram_simple_path, mo_diagram_complex_path, imofelist_simple_path, imofelist_complex_path, simple_atoms, complex_atoms):
        """
        Initialize the MOPM class by loading both MO diagrams and iMOFE lists for both systems.
        Filter complex atoms and AOs to match the simple system.
        """
        # Expand shorthand atom input
        self.simple_atoms = MOPM.expand_atom_list(simple_atoms) if isinstance(simple_atoms[0], tuple) else simple_atoms
        self.complex_atoms = MOPM.expand_atom_list(complex_atoms) if isinstance(complex_atoms[0], tuple) else complex_atoms
    
        # Filter complex atoms and AOs
        self.complex_atoms = self.filter_complex_atoms()
    
        # Load the MO diagrams for both systems
        self.mo_diagram_simple_spin_1, self.mo_diagram_simple_spin_2 = self.load_mo_diagram(mo_diagram_simple_path, "simple")
        self.mo_diagram_complex_spin_1, self.mo_diagram_complex_spin_2 = self.load_mo_diagram(mo_diagram_complex_path, "complex")
    
        # Load the iMOFE lists for both systems
        self.imofelist_simple_spin_1, self.imofelist_simple_spin_2 = self.load_imofelist(imofelist_simple_path, "simple")
        self.imofelist_complex_spin_1, self.imofelist_complex_spin_2 = self.load_imofelist(imofelist_complex_path, "complex")
    
        # Validate AO list lengths for MO diagrams
        if len(self.mo_diagram_simple_spin_1[0]['ao_contributions']) != len(self.mo_diagram_complex_spin_1[0]['ao_contributions']):
            raise ValueError("Filtered AO lists in simple and complex systems do not match in length.")
    
        # Validate AO list lengths for iMOFE lists (optional validation)
        if len(self.imofelist_simple_spin_1[0]['ao_contributions']) != len(self.imofelist_complex_spin_1[0]['ao_contributions']):
            raise ValueError("Filtered AO lists in simple and complex systems do not match in length for iMOFE lists.")

    @staticmethod
    def expand_atom_list(atom_counts):
        """
        Expand shorthand atomic input into a full atom list.
    
        Args:
            atom_counts (list of tuples): Each tuple contains the number of atoms and the atomic type,
                                          e.g., [(6, 'N'), (4, 'C'), (10, 'H')].
    
        Returns:
            list: Expanded atom list, e.g., ['N', 'N', 'N', 'N', 'N', 'N', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'].
        """
        expanded_list = []
        for count, atom_type in atom_counts:
            expanded_list.extend([atom_type] * count)  # Repeat the atom type 'count' times
        return expanded_list

    def filter_complex_atoms(self):
        """
        Filter complex atoms to only include those present in the simple atom list.
    
        Returns:
            list: Filtered complex atom list.
        """
        filtered_complex_atoms = [atom for atom in self.complex_atoms if atom in self.simple_atoms]
    
        # Debugging: Check the filtered atom list
        #print("Filtered Complex Atom List:", filtered_complex_atoms)
    
        return filtered_complex_atoms

    def filter_complex_ao_contributions(self, complex_spin_data):
        """
        Filter complex AO contributions to exclude AOs from removed atoms.
    
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
                    # Extract the atom type as letters before the first number or '_'
                    atom_type = ''.join([char for char in ao_id if char.isalpha() and char.isupper()])  # Extract the main atom type
    
                    # Match AO identifiers by atom type
                    if atom_type in self.complex_atoms:
                        filtered_contributions.append(mo['ao_contributions'][i])
                        filtered_identifiers.append(ao_id)
    
                # Debugging: Check the filtering results
                
    
                if filtered_contributions:
                    filtered_mo = {
                        'index': mo['index'],
                        'name': mo['name'],
                        'energy': mo['energy'],
                        'ao_contributions': np.array(filtered_contributions),
                        'ao_identifiers': filtered_identifiers,
                    }
                    filtered_data.append(filtered_mo)
            except Exception as e:
                print(f"Error filtering AO contributions for MO '{mo['name']}': {e}")
                continue
    
        # Raise an error if no valid AOs remain
        if not filtered_data:
            raise ValueError("No valid AOs remain after filtering. Verify atom type consistency.")
    
        return filtered_data
    
    def filter_complex_spin(self, complex_spin_data, simple_atoms):
        """
        Filter the complex spin channel data based on simple atom types.
    
        Args:
            complex_spin_data (list): MO data for one spin channel in the complex system.
            simple_atoms (list): List of atom types present in the simple system.
    
        Returns:
            list: Filtered complex spin channel data.
        """
        filtered_data = []
        for mo in complex_spin_data:
            try:
                filtered_contributions = []
                filtered_identifiers = []
    
                for i, ao_id in enumerate(mo['ao_identifiers']):
                    atom_type = ao_id.split('_')[0].rstrip('0123456789')  # Extract atom type up to the first number or '_'
                    #print(f"AO Identifier: {ao_id}, Extracted Atom Type: {atom_type}, Match: {atom_type in self.complex_atoms}")

    
                    # Check if the atom type is in the simple system
                    if atom_type in simple_atoms:
                        filtered_contributions.append(mo['ao_contributions'][i])
                        filtered_identifiers.append(ao_id)
    
                # Debugging: Show filtered AO identifiers
                #print(f"Original AO Identifiers: {mo['ao_identifiers']}")
                #print(f"Filtered AO Identifiers: {filtered_identifiers}")
    
                # Create filtered MO structure if there are valid contributions
                if filtered_contributions:
                    filtered_mo = {
                        'index': mo['index'],
                        'name': mo['name'],
                        'energy': mo['energy'],
                        'ao_contributions': np.array(filtered_contributions),
                        'ao_identifiers': filtered_identifiers,
                    }
                    filtered_data.append(filtered_mo)
            except Exception as e:
                print(f"Error filtering MO data for '{mo['name']}': {e}")
                continue
    
        # Handle case where no valid data remains
            if not filtered_data:
                print("Filtered Complex Atom List:", simple_atoms)
                raise ValueError("No valid complex spin data remain after filtering.")
        
            return filtered_data
    
    @staticmethod
    def load_mo_diagram(file_path, system_type):
        """
        Load molecular orbital diagram with MO index, name, energy, and atomic orbital (AO) contributions.
        Automatically detects the second spin channel by finding the second non-indented line.
        """
        mo_data_spin_1 = []
        mo_data_spin_2 = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
    
        # Initialize spin channel parsing logic
        spin_1_start = 0  # Spin 1 starts at the beginning (line 0)
        spin_2_start = None
        non_indented_count = 0
    
        # Detect the start of the second spin channel (second non-indented line)
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                non_indented_count += 1
                if non_indented_count == 2:  # Found the second non-indented line
                    spin_2_start = i
                    break
    
        # If no spin 2 is found, assume it's a single spin dataset
        if spin_2_start is None:
            spin_2_start = len(lines)
    
        # Parse Spin 1 data first
        spin_1_lines = lines[spin_1_start:spin_2_start]
        mo_data_spin_1 = MOPM.parse_spin_dataset_with_imofe(spin_1_lines, system_type, "Spin 1")
    
        # Then parse Spin 2 data
        if spin_2_start < len(lines):  # Ensure spin 2 exists
            spin_2_lines = lines[spin_2_start:]
            mo_data_spin_2 = MOPM.parse_spin_dataset_with_imofe(spin_2_lines, system_type, "Spin 2")
    
        return mo_data_spin_1, mo_data_spin_2
    
    @staticmethod
    def load_imofelist(file_path, system_type):
        """
        Load iMOFE list with iMOFE index, name, energy, and atomic orbital (AO) contributions.
        Automatically detects the second spin channel by finding the second non-indented line.
        """
        imofe_data_spin_1 = []
        imofe_data_spin_2 = []
    
        # Read file and skip the first two lines
        with open(file_path, 'r') as f:
            lines = f.readlines()[2:]  # Skip the first two lines
    
        # Initialize spin channel parsing logic
        spin_1_start = 0  # Spin 1 starts at the beginning (line 0, after skipping the first two lines)
        spin_2_start = None
        non_indented_count = 0
    
        # Detect the start of the second spin channel (second non-indented line)
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                non_indented_count += 1
                if non_indented_count == 2:  # Found the second non-indented line
                    spin_2_start = i
                    break
    
        # If no spin 2 is found, assume it's a single spin dataset
        if spin_2_start is None:
            spin_2_start = len(lines)
    
        # Parse Spin 1 data first
        spin_1_lines = lines[spin_1_start:spin_2_start]
        imofe_data_spin_1 = MOPM.parse_spin_dataset_with_imofe(spin_1_lines, system_type, "Spin 1")
    
        # Then parse Spin 2 data
        if spin_2_start < len(lines):  # Ensure spin 2 exists
            spin_2_lines = lines[spin_2_start:]
            imofe_data_spin_2 = MOPM.parse_spin_dataset_with_imofe(spin_2_lines, system_type, "Spin 2")
    
        return imofe_data_spin_1, imofe_data_spin_2
    
    @staticmethod
    def parse_spin_dataset_with_imofe(lines, system_type, spin_channel):
        """
        Parse molecular orbital data for a single spin dataset with iMOFE values.
        Associates iMOFE values with their corresponding AOs.
        """
        mo_data = []
        ao_identifiers = []
        ao_contributions = []
        ao_imofes = []
    
        # Parse MO names (second line)
        try:
            mo_names = lines[1].strip().split()  # Extract MO names from the second line
        except IndexError:
            raise ValueError("MO names line is missing or improperly formatted.")
    
        # Parse energies (third line)
        try:
            energies = list(map(float, lines[2].strip().split()[2:]))  # Extract energies after 'Energy (eV)'
        except (IndexError, ValueError):
            raise ValueError("Energy values line is missing or improperly formatted.")
    
        # Output the system type and spin channel when parsing begins
        #print(f"Parsing iMOFE values and their corresponding AO identifiers from the {system_type} system, {spin_channel}:")
    
        # Parse AO identifiers, iMOFE values, and contributions from subsequent rows
        for line in lines[3:]:
            row = line.strip().split()
            if len(row) >= 3:  # Ensure sufficient columns for AO data
                ao_identifiers.append(row[0])  # AO identifier
                try:
                    # Parse iMOFE value and AO contributions
                    imofer_value = float(row[1])  # iMOFE value from second column
                    ao_imofes.append(imofer_value)
                    ao_contributions.append(list(map(float, row[2:])))  # AO contributions from third column onward
    
                    # Print AO name and iMOFE value
                    #print(f"AO Identifier: {row[0]}, iMOFE Value: {imofer_value}")
                except ValueError as e:
                    print(f"Skipping malformed line: {line.strip()} ({e})")
                    ao_imofes.append(0.0)  # Default iMOFE value for malformed data
                    ao_contributions.append([0.0] * (len(row) - 2))  # Fill with zeros for contributions
    
        # Convert AO contributions into a NumPy array
        if ao_contributions:
            ao_contributions = np.array(ao_contributions)
        else:
            raise ValueError("No valid AO contributions found in the file.")
    
        # Combine MO names, energies, and AO data into a structured format
        for i, name in enumerate(mo_names):
            mo_data.append({
                'index': i,
                'name': name,
                'energy': energies[i],
                'ao_contributions': ao_contributions[:, i] if ao_contributions.size else np.array([]),
                'ao_imofes': ao_imofes,  # Store iMOFE values for each AO
                'ao_identifiers': ao_identifiers,
            })
    
        return mo_data

    def compare_mo_contributions(self, output_file_path=None, imofe_overlap_threshold=None):
        """
        Compare each complex MO to all simple MOs to find the best match.
        Filter results exclusively based on the iMOFE overlap threshold.
        """
        matches = []
    
        # Step 1: Find the best match for each complex MO
        for complex_mo in self.mo_diagram_complex_spin_1:
            best_match = None
            best_overlap_distance = float('inf')  # Initialize to the largest possible distance
    
            #print(f"Processing Complex MO: {complex_mo['name']}")  # Debugging
    
            # Loop through all simple MOs
            for simple_mo in self.mo_diagram_simple_spin_1:
                # Ensure AO contribution arrays have the same length
                if len(complex_mo['ao_contributions']) != len(simple_mo['ao_contributions']):
                    raise ValueError("Mismatch in AO contributions array length between complex and simple MO.")
    
                # Normalize AO contributions by their respective iMOFE values
                complex_contributions = np.array(complex_mo['ao_contributions']) / np.array(complex_mo['ao_imofes'])
                simple_contributions = np.array(simple_mo['ao_contributions']) / np.array(simple_mo['ao_imofes'])
    
                # Normalize the contributions for comparison
                complex_normalized = complex_contributions / np.linalg.norm(complex_contributions)
                simple_normalized = simple_contributions / np.linalg.norm(simple_contributions)
    
                # Calculate the dot product (signed overlap)
                ao_projection = np.dot(complex_normalized, simple_normalized)
    
                # Calculate the distance to ±1 (closer to ±1 is better)
                projection_distance = min(abs(1 - ao_projection), abs(-1 - ao_projection))
    
                # Determine if this is the best match
                if projection_distance < best_overlap_distance:
                    best_overlap_distance = projection_distance
                    best_match = {
                        'complex_mo': complex_mo['name'],
                        'complex_mo_energy': complex_mo['energy'],
                        'simple_mo': simple_mo['name'],
                        'simple_mo_energy': simple_mo['energy'],
                        'ao_overlap': ao_projection,
                        'energy_shift': complex_mo['energy'] - simple_mo['energy'],  # Calculate energy shift
                    }
    
            # Add the best match for this complex MO
            if best_match:
                matches.append(best_match)
            else:
                print(f"No match found for Complex MO: {complex_mo['name']}")  # Debugging
    
        # Step 2: Filter matches using the iMOFE overlap threshold
        filtered_matches = []
        
        for match in matches:
            # Find the corresponding complex MO and simple MO in the iMOFE data
            complex_imofe = next((imofe for imofe in self.imofelist_complex_spin_1 if imofe['name'] == match['complex_mo']), None)
            simple_imofe = next((imofe for imofe in self.imofelist_simple_spin_1 if imofe['name'] == match['simple_mo']), None)
        
            if complex_imofe and simple_imofe:
                complex_contributions = np.array(complex_imofe['ao_contributions']) / np.array(complex_imofe['ao_imofes'])
                simple_contributions = np.array(simple_imofe['ao_contributions']) / np.array(simple_imofe['ao_imofes'])
        
                complex_norm = np.linalg.norm(complex_contributions)
                simple_norm = np.linalg.norm(simple_contributions)
        
                # Handle near-zero norms and set flags accordingly
                anomaly_note = ""
                if complex_norm < 1e-6 and simple_norm < 1e-6:
                    anomaly_note = " [Flagged: Near-zero iMOFE contributions in both systems]"
                elif complex_norm < 1e-6:
                    anomaly_note = " [Flagged: Near-zero iMOFE contributions in complex system]"
                elif simple_norm < 1e-6:
                    anomaly_note = " [Flagged: Near-zero iMOFE contributions in simple system]"
        
                # Proceed with normalization even for flagged cases
                complex_normalized = complex_contributions / (complex_norm if complex_norm > 0 else 1e-6)  # Avoid division by zero
                simple_normalized = simple_contributions / (simple_norm if simple_norm > 0 else 1e-6)  # Avoid division by zero
        
                # Calculate the iMOFE AO overlap
                imofe_ao_overlap = np.dot(complex_normalized, simple_normalized)
        
                # Apply the iMOFE overlap threshold
                if abs(imofe_ao_overlap) < imofe_overlap_threshold:
                    match['imofe_ao_overlap'] = imofe_ao_overlap
                    match['anomaly_note'] = anomaly_note  # Add system-specific flag to the match
                    filtered_matches.append(match)
    
        # Step 3: Write filtered matches to the output file if specified
        if output_file_path:
            with open(output_file_path, 'w') as f:
                f.write("Complex MO\tComplex Energy (eV)\tSimple MO\tSimple Energy (eV)\tAO Overlap\tEnergy Shift (eV)\tiMOFE AO Overlap\tNotes\n")
                for match in filtered_matches:
                    f.write(f"{match['complex_mo']}\t{match['complex_mo_energy']:.4f}\t"
                            f"{match['simple_mo']}\t{match['simple_mo_energy']:.4f}\t"
                            f"{match['ao_overlap']:.4f}\t{match['energy_shift']:.4f}\t"
                            f"{match['imofe_ao_overlap']:.4f}\t{match.get('anomaly_note', '')}\n")
    
        # Return only the filtered matches
        return filtered_matches
    
    
# File paths for simple and complex systems
mo_diagram_simple_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/NHC/C13N2H18_1.MO_Diagram.lobster'
mo_diagram_complex_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/Adatom/NHC_frag/C13N2H18_1.MO_Diagram.lobster'

imofelist_simple_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/NHC/IMOFElist.lobster'
imofelist_complex_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/Adatom/NHC_frag/IMOFElist.lobster'

# Atom lists for simple and complex systems
simple_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]
complex_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]

# Initialize the MOPM instance with file paths for MO diagrams and iMOFE lists
MOPM_instance = MOPM(mo_diagram_simple_path, mo_diagram_complex_path, imofelist_simple_path, imofelist_complex_path, simple_atoms, complex_atoms)

# Generate matches using iMOFE overlap threshold
imofe_overlap_threshold = 0.01  # Example threshold
matches = MOPM_instance.compare_mo_contributions(output_file_path='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/matches.txt', imofe_overlap_threshold=imofe_overlap_threshold)

# Print matches
# Print matches
print("\nFiltered Matches Based on iMOFE Overlap:")
for match in matches:
    print(f"Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
          f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
          f"AO Overlap: {match['ao_overlap']:.4f}, iMOFE Overlap: {match['imofe_ao_overlap']:.4f}, "
          f"Energy Shift: {match['energy_shift']:.4f} eV{match.get('anomaly_note', '')}")