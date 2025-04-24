import os
import numpy as np

class MOPM:
    def __init__(self, mo_diagram_simple_path, mo_diagram_complex_path,align_energy=True):
        """
        Initialize the MOComparison class by loading MO diagrams for both systems.
        Save both the unfiltered and filtered complex system MO data.
        """
        # Load the MO diagram for the simple system
        self.mo_diagram_simple_spin_1, self.mo_diagram_simple_spin_2 = self.load_mo_diagram(mo_diagram_simple_path)
    
        # Extract AO identifiers from spin 1 of the simple system
        simple_ao_identifiers = set()
        for mo in self.mo_diagram_simple_spin_1:
            simple_ao_identifiers.update(mo['ao_identifiers'])
    
        # Load the full complex MO diagram
        self.full_complex_MO_spin_1, self.full_complex_MO_spin_2 = self.load_mo_diagram(mo_diagram_complex_path)
    
        # Create filtered versions of the complex MO diagrams
        self.mo_diagram_complex_spin_1 = self.filter_complex_spin(self.full_complex_MO_spin_1, simple_ao_identifiers)
        self.mo_diagram_complex_spin_2 = self.filter_complex_spin(self.full_complex_MO_spin_2, simple_ao_identifiers)
    
        # Handle cases where filtered data might be empty
        if not self.mo_diagram_complex_spin_1:
            self.mo_diagram_complex_spin_1 = []
        if not self.mo_diagram_complex_spin_2:
            self.mo_diagram_complex_spin_2 = []
        # Initialize the attribute to store the alignment shift.
        self.alignment_shift = None
        if align_energy:
            self.align_simple_system_energies()
    
    def filter_complex_spin(self, complex_spin_data, simple_ao_identifiers):
        """
        Filter the complex spin channel data based on simple AO identifiers.
        """
        filtered_data = []
        for mo in complex_spin_data:
            try:
                filtered_contributions = [
                    contribution for i, contribution in enumerate(mo['ao_contributions'])
                    if mo['ao_identifiers'][i] in simple_ao_identifiers
                ]
                filtered_identifiers = [
                    ao_id for ao_id in mo['ao_identifiers']
                    if ao_id in simple_ao_identifiers
                ]
                filtered_mo = {
                    'index': mo['index'],
                    'name': mo['name'],
                    'energy': mo['energy'],
                    'ao_contributions': np.array(filtered_contributions),
                    'ao_identifiers': filtered_identifiers,
                }
                filtered_data.append(filtered_mo)
            except Exception as e:
                print(f"Error filtering MO data: {e}")
                continue
        return filtered_data
            
    @staticmethod
    def load_mo_diagram(file_path):
        """
        Load molecular orbital diagram with MO index, name, energy, and atomic orbital (AO) contributions.
        Automatically detects the second spin channel by finding the second non-indented line.
        Treats the second spin channel exactly like the first.
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
    
        # Parse Spin 1 data
        spin_1_lines = lines[spin_1_start:spin_2_start]
        mo_data_spin_1 = MOPM.parse_spin_dataset(spin_1_lines)
    
        # Parse Spin 2 data if present
        if spin_2_start < len(lines):  # Ensure spin 2 exists
            spin_2_lines = lines[spin_2_start:]
            mo_data_spin_2 = MOPM.parse_spin_dataset(spin_2_lines)
    
        return mo_data_spin_1, mo_data_spin_2
    
    
    @staticmethod
    def parse_spin_dataset(lines):
        """
        Parse molecular orbital data for a single spin dataset.
        Treats each spin channel using consistent parsing logic.
        """
        mo_data = []
        ao_identifiers = []
        ao_contributions = []
    
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
    
        # Parse atomic orbital (AO) contributions from the fourth line onward
        for line in lines[3:]:
            row = line.strip().split()
            if len(row) >= 3:  # Only process lines with sufficient columns
                ao_identifiers.append(row[0])  # AO identifier (e.g., atom/orbital name)
                try:
                    ao_contributions.append(list(map(float, row[2:])))  # AO contributions by MO
                except ValueError as e:
                    print(f"Skipping malformed line: {line.strip()} ({e})")
                    ao_contributions.append([0.0] * (len(row) - 2))  # Fill with zeros for malformed data
    
        # Convert AO contributions into a NumPy array
        if ao_contributions:
            ao_contributions = np.array(ao_contributions)
        else:
            raise ValueError("No valid AO contributions found in the file.")
    
        # Combine MO names, energies, and AO contributions into a structured format
        for i, name in enumerate(mo_names):
            mo_data.append({
                'index': i,
                'name': name,
                'energy': energies[i],
                'ao_contributions': ao_contributions[:, i] if ao_contributions.size else np.array([]),
                'ao_identifiers': ao_identifiers,
            })
    
        return mo_data

    def align_simple_system_energies(self):
        """
        Align the energies of the simple system to the complex system by shifting all simple MO energies.
        Assumes that the lowest-energy MO (MO1) is at index 0 in both systems.
        """
        # If already applied, just return the stored shift.
        if hasattr(self, 'alignment_shift') and self.alignment_shift is not None:
            print(f"Alignment already applied. Using stored shift of {self.alignment_shift:.4f} eV.")
            return self.alignment_shift
    
        # Ensure that we have at least one MO for reference in the simple and complex systems.
        if not self.full_complex_MO_spin_1 or not self.mo_diagram_simple_spin_1:
            raise ValueError("Missing MO1 in one or both systems for energy alignment.")
        
        # Retrieve the reference MO (MO1) from the complex and simple systems.
        # (Assuming index 0 corresponds to MO1 based on file ordering.)
        complex_mo1 = self.full_complex_MO_spin_1[0]
        simple_mo1 = self.mo_diagram_simple_spin_1[0]
        
        # Calculate the energy shift required so that simple MO1 aligns with complex MO1.
        energy_offset = complex_mo1['energy'] - simple_mo1['energy']
        
        # Apply this shift to every MO in the simple system (spin 1).
        for mo in self.mo_diagram_simple_spin_1:
            mo['energy'] += energy_offset
    
        # If there is spin 2 data for the simple system, shift those energies too.
        if self.mo_diagram_simple_spin_2:
            for mo in self.mo_diagram_simple_spin_2:
                mo['energy'] += energy_offset
    
        # Store the alignment shift.
        self.alignment_shift = energy_offset
        print(f"Applied an energy shift of {energy_offset:.4f} eV to the simple system.")
        return energy_offset
    
    def compare_mo_contributions(self, output_file_path=None, energy_shift_threshold=2.0):
        """
        Compare all MOs between the simple and complex systems based on AO contributions.
        Compare simple spin 1 with complex spin 1 and simple spin 2 with complex spin 2.
        Select the simple MO with the projection closest to ±1 for each complex MO.
        Optionally write the results to an output file.
        """
        matches = []
    
        # Compare Spin 1 of the simple and complex systems
        for complex_mo in self.mo_diagram_complex_spin_1:
            best_match = None
            closest_projection_distance = float('inf')  # Initialize the closest distance
            complex_normalized = complex_mo['ao_contributions'] / np.linalg.norm(complex_mo['ao_contributions'])
    
            for simple_mo in self.mo_diagram_simple_spin_1:
                # Normalize AO contributions
                simple_normalized = simple_mo['ao_contributions'] / np.linalg.norm(simple_mo['ao_contributions'])
    
                # Calculate the dot product (signed overlap)
                ao_projection = np.dot(simple_normalized, complex_normalized)
    
                # Calculate the distance from ±1 (closest match to perfect projection)
                projection_distance = min(abs(1 - ao_projection), abs(-1 - ao_projection))
    
                if projection_distance < closest_projection_distance:  # Check for smaller distance
                    closest_projection_distance = projection_distance
                    best_match = {
                        'complex_mo': complex_mo['name'],
                        'complex_mo_energy': complex_mo['energy'],
                        'simple_mo': simple_mo['name'],
                        'simple_mo_energy': simple_mo['energy'],
                        'ao_overlap': ao_projection,  # Signed overlap
                        'energy_shift': complex_mo['energy'] - simple_mo['energy'],
                    }
    
            # Save the best match for the current complex MO
            if best_match and abs(best_match['energy_shift']) >= energy_shift_threshold:
                matches.append(best_match)
    
        # Compare Spin 2 of the simple and complex systems
        for complex_mo in self.mo_diagram_complex_spin_2:
            best_match = None
            closest_projection_distance = float('inf')  # Initialize the closest distance
            complex_normalized = complex_mo['ao_contributions'] / np.linalg.norm(complex_mo['ao_contributions'])
    
            for simple_mo in self.mo_diagram_simple_spin_2:
                # Normalize AO contributions
                simple_normalized = simple_mo['ao_contributions'] / np.linalg.norm(simple_mo['ao_contributions'])
    
                # Calculate the dot product (signed overlap)
                ao_projection = np.dot(simple_normalized, complex_normalized)
    
                # Calculate the distance from ±1 (closest match to perfect projection)
                projection_distance = min(abs(1 - ao_projection), abs(-1 - ao_projection))
    
                if projection_distance < closest_projection_distance:  # Check for smaller distance
                    closest_projection_distance = projection_distance
                    best_match = {
                        'complex_mo': complex_mo['name'],
                        'complex_mo_energy': complex_mo['energy'],
                        'simple_mo': simple_mo['name'],
                        'simple_mo_energy': simple_mo['energy'],
                        'ao_overlap': ao_projection,  # Signed overlap
                        'energy_shift': complex_mo['energy'] - simple_mo['energy'],
                    }
    
            # Save the best match for the current complex MO
            if best_match and abs(best_match['energy_shift']) >= energy_shift_threshold:
                matches.append(best_match)
    
        # Write matches to output file if specified
        if output_file_path:
            with open(output_file_path, 'w') as f:
                f.write("Complex MO\tComplex Energy (eV)\tSimple MO\tSimple Energy (eV)\tAO Overlap\tEnergy Shift (eV)\n")
                for match in matches:
                    f.write(f"{match['complex_mo']}\t{match['complex_mo_energy']:.4f}\t"
                            f"{match['simple_mo']}\t{match['simple_mo_energy']:.4f}\t"
                            f"{match['ao_overlap']:.4f}\t{match['energy_shift']:.4f}\n")
    
        return matches

    def compare_gold_molecule_contributions(self, matches, full_MO, output_file_path=None, system="complex"):
        """
        Compare molecular orbitals and analyze gold and molecule contributions for either the simple or complex system.
        Works seamlessly for both spin channels and avoids duplicating results.
        Express totals as percentages based on their combined raw sum as 100%.
        Optionally write the results to an output file.
    
        Parameters:
        matches (list): List of matched MOs from compare_mo_contributions.
        full_MO (list): Full MO data (simple or complex system).
        output_file_path (str): Path to the output file (optional).
        system (str): Either "simple" or "complex" to indicate which system is being processed.
        """
        results = []
    
        for match in matches:
            # Locate the full MO data for the given match
            mo = next(mo for mo in full_MO if mo['name'] == match[f"{system}_mo"])
    
            # Initialize contributions
            contributions = {'gold': {'s': 0, 'd': 0, 'total': 0}, 'molecule': {'total': 0}}
    
            for i, ao_id in enumerate(mo['ao_identifiers']):
                contribution = abs(mo['ao_contributions'][i])  # Use absolute value of contributions
                if "Au" in ao_id:  # Gold AO identifiers
                    contributions['gold']['total'] += contribution
                    if 's' in ao_id.lower():
                        contributions['gold']['s'] += contribution
                    elif 'd' in ao_id.lower():
                        contributions['gold']['d'] += contribution
                else:  # Everything else is molecule AOs
                    contributions['molecule']['total'] += contribution
    
            # Calculate combined raw sum
            combined_total = contributions['gold']['total'] + contributions['molecule']['total']
    
            # Safeguard against very small combined total
            if abs(combined_total) < 1e-2:
                combined_total = 1e-2  # Prevent extreme percentage calculations
    
            # Express gold and molecule totals as percentages
            contributions['gold']['percent'] = (contributions['gold']['total'] / combined_total) * 100
            contributions['molecule']['percent'] = (contributions['molecule']['total'] / combined_total) * 100
    
            # Calculate percentages for gold contributions (s- vs. d-states relative to gold total)
            if abs(contributions['gold']['total']) > 1e-2:
                contributions['gold']['s'] = (contributions['gold']['s'] / contributions['gold']['total']) * 100
                contributions['gold']['d'] = (contributions['gold']['d'] / contributions['gold']['total']) * 100
    
            # Append the results
            results.append({
                f"{system}_mo": match[f"{system}_mo"],
                f"{system}_energy": match[f"{system}_mo_energy"],
                'ao_overlap': match['ao_overlap'],
                'energy_shift': match['energy_shift'],
                'gold_contributions': contributions['gold'],
                'molecule_contributions': contributions['molecule'],
            })
    
        # Write results to file if output_file_path is specified
        if output_file_path:
            with open(output_file_path, 'w') as f:
                # Write header row
                f.write(f"{system.capitalize()} MO\tEnergy (eV)\tAO Overlap\tEnergy Shift (eV)\t"
                        "Gold: s-states (%)\tGold: d-states (%)\tGold: Total (%)\tMolecule: Total (%)\n")
    
                # Write results row by row
                for result in results:
                    gold = result['gold_contributions']
                    molecule = result['molecule_contributions']
                    f.write(f"{result[f'{system}_mo']}\t{result[f'{system}_energy']:.4f}\t"
                            f"{result['ao_overlap']:.4f}\t{result['energy_shift']:.4f}\t"
                            f"{gold['s']:.2f}\t{gold['d']:.2f}\t{gold['percent']:.2f}\t"
                            f"{molecule['percent']:.2f}\n")
    
        return results
'''
# File paths for the MO diagrams and output files
mo_diagram_simple_path = 'C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/AuC13N2H18_1.MO_Diagram.lobster'
mo_diagram_complex_path = 'C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/Au4C13N2H18_1.MO_Diagram.lobster'
matches_output_path = 'C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/matches_output2eV_spinpol.txt'
gold_molecule_output_path = 'C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/gold_molecule_contributions2eV_spinpol.txt'
simple_gold_molecule_output_path = 'C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/simple_gold_molecule_contributions_matches2eV_spinpol.txt'

# Initialize the MOPM instance
MOPM_instance = MOPM(mo_diagram_simple_path, mo_diagram_complex_path)
# Access Spin 1 and Spin 2 data for the simple and complex systems
'''
'''
print("Simple System Spin 1:")
for mo in MOPM_instance.mo_diagram_simple_spin_1:
    print(mo)

if MOPM_instance.mo_diagram_simple_spin_2:
    print("\nSimple System Spin 2:")
    for mo in MOPM_instance.mo_diagram_simple_spin_2:
        print(mo)
else:
    print("\nNo Spin 2 Data for Simple System.")

print("\nComplex System Spin 1:")
for mo in MOPM_instance.full_complex_MO_spin_1:
    print(mo)

if MOPM_instance.full_complex_MO_spin_2:
    print("\nComplex System Spin 2:")
    for mo in MOPM_instance.full_complex_MO_spin_2:
        print(mo)
else:
    print("\nNo Spin 2 Data for Complex System.")
    '''
'''

# Generate matches and write them to the output file
matches = MOPM_instance.compare_mo_contributions(matches_output_path)

# Print the matches
print("\nMatches:")
for match in matches:
    print(f"Complex MO: {match['complex_mo']}, {match['complex_mo_energy']}eV, "
          f"Simple MO: {match['simple_mo']}, {match['simple_mo_energy']}eV, "
          f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.4f}eV")

# Analyze gold/molecule contributions for the complex system
complex_results = MOPM_instance.compare_gold_molecule_contributions(matches, MOPM_instance.full_complex_MO_spin_1, gold_molecule_output_path, system="complex")

# Analyze gold/molecule contributions for the simple system
simple_results = MOPM_instance.compare_gold_molecule_contributions(matches, MOPM_instance.mo_diagram_simple_spin_1, simple_gold_molecule_output_path, system="simple")

# Print results for both systems
print("\nComplex System Contributions:")
for result in complex_results:
    gold = result['gold_contributions']
    molecule = result['molecule_contributions']
    print(f"Complex MO: {result['complex_mo']}, Energy: {result['complex_energy']} eV, "
          f"AO Overlap: {result['ao_overlap']:.4f}, Energy Shift: {result['energy_shift']:.4f} eV, "
          f"Gold contributions: s-states = {gold['s']:.2f}%, d-states = {gold['d']:.2f}%, total = {gold['percent']:.2f}%, "
          f"Molecule contributions: total = {molecule['percent']:.2f}%")

print("\nSimple System Contributions:")
for result in simple_results:
    gold = result['gold_contributions']
    molecule = result['molecule_contributions']
    print(f"Simple MO: {result['simple_mo']}, Energy: {result['simple_energy']} eV, "
          f"AO Overlap: {result['ao_overlap']:.4f}, Energy Shift: {result['energy_shift']:.4f} eV, "
          f"Gold contributions: s-states = {gold['s']:.2f}%, d-states = {gold['d']:.2f}%, total = {gold['percent']:.2f}%, "
          f"Molecule contributions: total = {molecule['percent']:.2f}%")
'''