import os
import numpy as np

class MOPM:
    def __init__(self, mo_diagram_simple_path, mo_diagram_complex_path):
        """
        Initialize the MOComparison class by loading MO diagrams for both systems.
        Save both the unfiltered and filtered complex system MO data.
        """
        # Load the MO diagram for the simple system
        self.mo_diagram_simple = self.load_mo_diagram(mo_diagram_simple_path)

        # Extract AO identifiers from the simple system
        simple_ao_identifiers = set()
        for mo in self.mo_diagram_simple:
            simple_ao_identifiers.update(mo['ao_identifiers'])

        # Load the full complex MO diagram
        self.full_complex_MO = self.load_mo_diagram(mo_diagram_complex_path)

        # Create a filtered version of the complex MO diagram
        self.mo_diagram_complex = []
        for mo in self.full_complex_MO:
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
            self.mo_diagram_complex.append(filtered_mo)
            
    @staticmethod
    def load_mo_diagram(file_path):
        """
        Load molecular orbital diagram with MO index, name, energy, and atomic orbital (AO) contributions.
        """
        mo_data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse the file for MO data
        mo_names = lines[1].strip().split()  # MO names are in the second line
        energies = list(map(float, lines[2].strip().split()[2:]))  # Energies start after 'Energy (eV)'
        ao_identifiers = []
        ao_contributions = []

        # Parse atomic orbital (AO) contributions from the fourth line onward
        for line in lines[3:]:
            row = line.strip().split()
            ao_identifiers.append(row[0])  # AO identifier (e.g., atom/orbital name)
            ao_contributions.append(list(map(float, row[2:])))  # AO contributions by MO

        # Convert AO contributions into a NumPy array
        ao_contributions = np.array(ao_contributions)

        # Combine MO names, energies, and AO contributions into a structured format
        for i, name in enumerate(mo_names):
            mo_data.append({
                'index': i,
                'name': name,
                'energy': energies[i],
                'ao_contributions': ao_contributions[:, i],
                'ao_identifiers': ao_identifiers,
            })

        return mo_data
    def compare_mo_contributions(self, output_file_path=None, energy_shift_threshold=2.0):
        """
        Compare all MOs between the simple and complex systems based on AO contributions.
        Select the simple MO with the projection closest to ±1 for each complex MO.
        Optionally write the results to an output file.
        """
        matches = []
    
        for complex_mo in self.mo_diagram_complex:
            best_match = None
            closest_projection_distance = float('inf')  # Use infinity to initialize the closest distance
            complex_normalized = complex_mo['ao_contributions'] / np.linalg.norm(complex_mo['ao_contributions'])
            
            for simple_mo in self.mo_diagram_simple:
                # Normalize AO contributions
                simple_normalized = simple_mo['ao_contributions'] / np.linalg.norm(simple_mo['ao_contributions'])
                
                # Calculate the dot product (signed overlap)
                ao_projection = np.dot(simple_normalized, complex_normalized)
    
                # Calculate distance from ±1 (closest match to perfect projection)
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

    def compare_gold_molecule_contributions(self, matches, full_complex_MO, output_file_path=None):
        """
        Analyze gold and molecule atomic orbital contributions for each matched MO.
        Express totals as percentages based on their combined raw sum as 100%.
        Optionally write the results to an output file.
    
        Parameters:
        matches (list): List of matched MOs from compare_mo_contributions.
        full_complex_MO (list): Full complex MO data.
        output_file_path (str): Path to the output file (optional).
        """
        results = []
    
        for match in matches:
            # Locate the full complex MO data
            complex_mo = next(mo for mo in full_complex_MO if mo['name'] == match['complex_mo'])
    
            # Initialize contributions
            contributions = {'gold': {'s': 0, 'd': 0, 'total': 0}, 'molecule': {'total': 0}}
    
            for i, ao_id in enumerate(complex_mo['ao_identifiers']):
                contribution = abs(complex_mo['ao_contributions'][i])  # Use absolute value of contributions
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
                'complex_mo': match['complex_mo'],
                'complex_energy': match['complex_mo_energy'],
                'ao_overlap': match['ao_overlap'],
                'energy_shift': match['energy_shift'],
                'gold_contributions': contributions['gold'],
                'molecule_contributions': contributions['molecule'],
            })
    
        # Write results to file if output_file_path is specified
        if output_file_path:
            with open(output_file_path, 'w') as f:
                # Write header row
                f.write("Complex MO\tEnergy (eV)\tAO Overlap\tEnergy Shift (eV)\t"
                        "Gold: s-states (%)\tGold: d-states (%)\tGold: Total (%)\tMolecule: Total (%)\n")
                
                # Write results row by row
                for result in results:
                    gold = result['gold_contributions']
                    molecule = result['molecule_contributions']
                    f.write(f"{result['complex_mo']}\t{result['complex_energy']:.4f}\t"
                            f"{result['ao_overlap']:.4f}\t{result['energy_shift']:.4f}\t"
                            f"{gold['s']:.2f}\t{gold['d']:.2f}\t{gold['percent']:.2f}\t"
                            f"{molecule['percent']:.2f}\n")
    
        return results

# Example usage
mo_diagram_simple_path = 'C:/Users/nazin_lab/Downloads/C13N2H18_1.MO_Diagram.lobster'
mo_diagram_complex_path = 'C:/Users/nazin_lab/Downloads/Au10C13N2H18_1.MO_Diagram.lobster'
output_file_path = 'C:/Users/nazin_lab/Downloads/matches_output.txt'

# Load diagrams and generate matches using the provided class
MOPM_instance = MOPM(mo_diagram_simple_path, mo_diagram_complex_path)
matches = MOPM_instance.compare_mo_contributions(output_file_path)
for match in matches:
    print(f"Complex MO: {match['complex_mo']}, {match['complex_mo_energy']}eV, Simple MO: {match['simple_mo']}, {match['simple_mo_energy']}eV "
          f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.4f}eV")

output_file_path = 'C:/Users/nazin_lab/Downloads/gold_molecule_contributions.txt'
results = MOPM_instance.compare_gold_molecule_contributions(matches, MOPM_instance.full_complex_MO, output_file_path)

print(f"Results have been written to {output_file_path}")

# Print results
for result in results:
    gold = result['gold_contributions']
    molecule = result['molecule_contributions']
    print(f"Complex MO: {result['complex_mo']}, Energy: {result['complex_energy']} eV, "
          f"AO Overlap: {result['ao_overlap']:.4f}, Energy Shift: {result['energy_shift']:.4f} eV, "
          f"Gold contributions: s-states = {gold['s']:.2f}%, d-states = {gold['d']:.2f}%, total = {gold['percent']:.2f}%, "
          f"Molecule contributions: total = {molecule['percent']:.2f}%")