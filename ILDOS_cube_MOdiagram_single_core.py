from ase.io.cube import read_cube_data
import os
import numpy as np

class LobsterAnalysis:
    def __init__(self, simple_dir, complex_dir, mo_diagram_simple_path, mo_diagram_complex_path):
        self.simple_dir = simple_dir
        self.complex_dir = complex_dir
        self.mo_diagram_simple = self.load_mo_diagram(mo_diagram_simple_path)
        self.mo_diagram_complex = self.load_mo_diagram(mo_diagram_complex_path)

        # Extract AO identifiers from the simple system
        simple_ao_identifiers = set(self.mo_diagram_simple[0]['ao_identifiers'])

        # Filter AO contributions in the complex system to include only those matching the simple system
        for mo in self.mo_diagram_complex:
            # Keep only the AO contributions with identifiers that match the simple system
            filtered_contributions = [
                ao for i, ao in enumerate(mo['ao_contributions'])
                if mo['ao_identifiers'][i] in simple_ao_identifiers
            ]
            filtered_identifiers = [
                ao_id for ao_id in mo['ao_identifiers']
                if ao_id in simple_ao_identifiers
            ]
            mo['ao_contributions'] = np.array(filtered_contributions)  # Update contributions
            mo['ao_identifiers'] = filtered_identifiers  # Update identifiers

        # Filter MO diagram for the simple system based on cube file names
        simple_cube_files = [os.path.splitext(f)[0].replace("_1_1_", "_1_") for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
        self.mo_diagram_simple = [mo for mo in self.mo_diagram_simple if mo['name'] in simple_cube_files]

        # Debug: Print the filtered simple system data
        #print("Filtered Simple Adjusted Cube File Names:")
        #print(simple_cube_files)
        #print("Filtered Simple MO Diagram Names:")
        #print([mo['name'] for mo in self.mo_diagram_simple])

        # Filter MO diagram for the complex system based on cube file names
        complex_cube_files = [os.path.splitext(f)[0].replace("_1_1_", "_1_") for f in os.listdir(self.complex_dir) if f.endswith('.cube')]
        self.mo_diagram_complex = [mo for mo in self.mo_diagram_complex if mo['name'] in complex_cube_files]

        # Debug: Print the filtered complex system data
        #print("Filtered Complex Adjusted Cube File Names:")
        #print(complex_cube_files)
        #print("Filtered Complex MO Diagram Names:")
        #print([mo['name'] for mo in self.mo_diagram_complex])
        
    @staticmethod
    def load_cube_file_ase(file_path):
        """Load molecular orbital cube file using ASE and extract data."""
        data, metadata = read_cube_data(file_path)  # Correct unpacking for two outputs
        return data  # Return only the data grid for further processing

    @staticmethod
    def load_mo_diagram(file_path):
        """Load molecular orbital diagram with MO index/name, energy, and atomic orbital contributions."""
        mo_data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
    
        # Extract MO names from the second line
        mo_names = lines[1].strip().split()  # Include all MO names
    
        # Parse the third line for energies, skipping 'Energy' and '(eV)'
        energy_line = lines[2].strip().split()[2:]  # Skip only the first two fields: 'Energy' and '(eV)'
        try:
            energies = list(map(float, energy_line))
        except ValueError as e:
            raise ValueError(f"Error parsing energies: {e}. Ensure that 'Energy' and '(eV)' labels are correctly skipped.")
    
        # Parse the AO rows starting from the fourth line
        ao_contributions = []
        ao_identifiers = []
        for line in lines[3:]:
            row = line.strip().split()
            ao_identifiers.append(row[0])  # First column is the AO identifier
            ao_contributions.append(list(map(float, row[2:])))  # Contributions start from the third column
    
        # Convert AO contributions into a NumPy array for easier slicing
        ao_contributions = np.array(ao_contributions)
    
        # Debugging: Print AO identifiers for verification
        #print("AO Identifiers (Names):")
        #print(ao_identifiers)
    
        # Combine MO names, energies, and AO contributions
        for i, name in enumerate(mo_names):
            mo_data.append({
                'index': i,  # MO index (0-based)
                'name': name,    # MO name
                'energy': energies[i],  # MO energy
                'ao_contributions': ao_contributions[:, i],  # Atomic orbital contributions for this MO
                'ao_identifiers': ao_identifiers  # List of AO identifiers
            })
    
        return mo_data


    def compare_mo_cube_and_orbital_contributions(self):
        """Find MO matches based on volumetric data and AO contributions."""
        simple_files = [f for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
        complex_files = [f for f in os.listdir(self.complex_dir) if f.endswith('.cube')]
        
        matches = []
        for s_file in simple_files:
            # Load data for the simple cube file
            s_path = os.path.join(self.simple_dir, s_file)
            s_data = self.load_cube_file_ase(s_path)
    
            # Adjust for the extra '_1' in the simple cube file name
            simple_mo_name = os.path.splitext(s_file)[0].replace("_1_1_", "_1_")
            simple_mo_info = next((mo for mo in self.mo_diagram_simple if mo['name'] == simple_mo_name), None)
    
            if not simple_mo_info:
                print(f"Warning: No MO information found for {s_file}")
                continue
    
            best_match = None
            best_correlation = -1
            best_sum_difference = float('inf')  # Initialize to a large value
    
            for c_file in complex_files:
                # Load data for the complex cube file
                c_path = os.path.join(self.complex_dir, c_file)
                c_data = self.load_cube_file_ase(c_path)
    
                # Adjust for the extra '_1' in the complex cube file name
                complex_mo_name = os.path.splitext(c_file)[0].replace("_1_1_", "_1_")
                complex_mo_info = next((mo for mo in self.mo_diagram_complex if mo['name'] == complex_mo_name), None)
    
                if not complex_mo_info:
                    continue
    
                # Compare cube file volumetric data
                correlation = np.corrcoef(s_data.flatten(), c_data.flatten())[0, 1]
    
                # Compute AO contribution differences
                ao_contribution_differences = simple_mo_info['ao_contributions'] - complex_mo_info['ao_contributions']
                sum_difference = np.sum(np.abs(ao_contribution_differences))
    
                # Track the best match based on volumetric correlation and AO contribution differences
                if correlation > best_correlation and sum_difference < best_sum_difference:
                    best_correlation = correlation
                    best_sum_difference = sum_difference
                    best_match = {
                        'complex_file': c_file,
                        'complex_mo_info': complex_mo_info,
                        'correlation': correlation,
                        'energy_shift': complex_mo_info['energy'] - simple_mo_info['energy'],
                        'ao_differences': ao_contribution_differences,
                        'sum_difference': sum_difference
                    }
    
            # Only append a match if both correlation and AO sum differences align
            if best_match and best_match['sum_difference'] < float('inf'):
                matches.append({
                    'simple_file': s_file,
                    'simple_mo_info': simple_mo_info,
                    'best_match': best_match
                })
    
        return matches
            
    def run_analysis(self):
        """Run the complete analysis, finding the best matches for all cube files."""
        matches = self.compare_mo_cube_and_orbital_contributions()
    
        # Print the best matches and AO contribution differences for each simple cube file
        print("\nMatched MOs and Energy Shifts:")
        for match in matches:
            simple_mo = match['simple_mo_info']
            best_match = match['best_match']
            complex_mo = best_match['complex_mo_info']
            ao_differences = best_match['ao_differences']
            sum_difference = best_match['sum_difference']
    
            print(f"Simple MO: {simple_mo['name']} (Energy: {simple_mo['energy']:.4f}) "
                  f"-> Complex MO: {complex_mo['name']} (Energy: {complex_mo['energy']:.4f}), "
                  f"Energy Shift: {best_match['energy_shift']:.4f}, "
                  f"Volumetric Correlation: {best_match['correlation']:.4f}, "
                  f"Sum of AO Contribution Differences: {sum_difference:.4f}")
            #print(f"AO Contribution Differences: {ao_differences}")
    
# Set up directories and paths
simple_calc_dir = "C:/Users/benka/Documents/LOBSTER_run4/"
complex_calc_dir = "C:/Users/benka/Documents/LOBSTER_run2_AuNHC/"
mo_diagram_simple_path = os.path.join(simple_calc_dir, "C13N2H18_1.MO_Diagram.lobster")
mo_diagram_complex_path = os.path.join(complex_calc_dir, "AuC13N2H18_1.MO_Diagram.lobster")

# Create the LobsterAnalysis object
analysis = LobsterAnalysis(simple_calc_dir, complex_calc_dir, mo_diagram_simple_path, mo_diagram_complex_path)

# Run the analysis
analysis.run_analysis()
'''
# Example usage
file_path = 'C:/Users/benka/Documents/LOBSTER_run4/C13N2H18_1.MO_Diagram.lobster'
mo_data = LobsterAnalysis.load_mo_diagram(file_path)

# Debug: Verify parsed data
print("Extracted MO Names:", [mo['name'] for mo in mo_data])
print("Extracted MO Energies:", [mo['energy'] for mo in mo_data])
print("Sample AO Contributions for First MO:", mo_data[0]['ao_contributions'][:10])
print("AO Identifiers:", mo_data[0]['ao_identifiers'][:10])
'''