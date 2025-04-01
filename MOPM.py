import os
import numpy as np

class MOPM:
    def __init__(self, mo_diagram_simple_path, mo_diagram_complex_path):
        """
        Initialize the MOComparison class by loading MO diagrams for both systems.
        Filter complex system AOs based on the simple system's AOs.
        """
        # Load the MO diagram for the simple system
        self.mo_diagram_simple = self.load_mo_diagram(mo_diagram_simple_path)

        # Extract AO identifiers from the simple system
        simple_ao_identifiers = set()
        for mo in self.mo_diagram_simple:
            simple_ao_identifiers.update(mo['ao_identifiers'])

        # Load the MO diagram for the complex system
        self.mo_diagram_complex = self.load_mo_diagram(mo_diagram_complex_path)

        # Filter the complex system's AOs based on the simple system's AO identifiers
        for mo in self.mo_diagram_complex:
            filtered_contributions = [
                contribution for i, contribution in enumerate(mo['ao_contributions'])
                if mo['ao_identifiers'][i] in simple_ao_identifiers
            ]
            filtered_identifiers = [
                ao_id for ao_id in mo['ao_identifiers']
                if ao_id in simple_ao_identifiers
            ]
            mo['ao_contributions'] = np.array(filtered_contributions)
            mo['ao_identifiers'] = filtered_identifiers
            
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

    def compare_mo_contributions(self, output_file_path=None):
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

# Example usage:
mo_diagram_simple_path = "simple_MO_diagram.lobster"
mo_diagram_complex_path = "complex_MO_diagram.lobster"
output_file_path = 'matches_output.txt'

MOPM = MOPM(mo_diagram_simple_path, mo_diagram_complex_path)
matches = MOPM.compare_mo_contributions(output_file_path)

for match in matches:
    print(f"Complex MO: {match['complex_mo']}, {match['complex_mo_energy']}eV, Simple MO: {match['simple_mo']}, {match['simple_mo_energy']}eV "
          f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.4f}eV")
