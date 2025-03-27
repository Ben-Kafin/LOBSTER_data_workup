from ase.io.cube import read_cube_data
import os
import numpy as np

class LobsterAnalysis:
    def __init__(self, simple_dir, complex_dir, mo_diagram_simple_path, mo_diagram_complex_path):
        self.simple_dir = simple_dir
        self.complex_dir = complex_dir

        # Load the MO diagram for the simple system first to extract AO identifiers
        self.mo_diagram_simple = self.load_mo_diagram(mo_diagram_simple_path)
        simple_ao_identifiers = set(self.mo_diagram_simple[0]['ao_identifiers'])

        # Load the MO diagram for the complex system and filter based on the simple system's AOs
        self.mo_diagram_complex = self.load_mo_diagram(mo_diagram_complex_path)
        for mo in self.mo_diagram_complex:
            filtered_contributions = [
                ao for i, ao in enumerate(mo['ao_contributions'])
                if mo['ao_identifiers'][i] in simple_ao_identifiers
            ]
            filtered_identifiers = [
                ao_id for ao_id in mo['ao_identifiers']
                if ao_id in simple_ao_identifiers
            ]
            mo['ao_contributions'] = np.array(filtered_contributions)
            mo['ao_identifiers'] = filtered_identifiers

        # Filter MO diagrams based on cube file names
        simple_cube_files = [os.path.splitext(f)[0].replace("_1_1_", "_1_") for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
        complex_cube_files = [os.path.splitext(f)[0].replace("_1_1_", "_1_") for f in os.listdir(self.complex_dir) if f.endswith('.cube')]

        self.mo_diagram_simple = [mo for mo in self.mo_diagram_simple if mo['name'] in simple_cube_files]
        self.mo_diagram_complex = [mo for mo in self.mo_diagram_complex if mo['name'] in complex_cube_files]
        
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


    def compare_mo_cube_and_orbital_contributions(self, output_path="results.txt"):
        """Compare MOs with volumetric and AO matching logic, include energy shift, and write results to a text file."""
        complex_files = [f for f in os.listdir(self.complex_dir) if f.endswith('.cube')]
        simple_files = [f for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
        matches = []
    
        # Open a file for writing results
        with open(output_path, "w") as output_file:
            for c_file in complex_files:  # Iterate over complex system files first
                # Load complex cube data
                c_path = os.path.join(self.complex_dir, c_file)
                c_data = self.load_cube_file_ase(c_path)
    
                # Match complex MO name
                complex_mo_name = os.path.splitext(c_file)[0].replace("_1_1_", "_1_")
                complex_mo_info = next((mo for mo in self.mo_diagram_complex if mo['name'] == complex_mo_name), None)
    
                if not complex_mo_info:
                    print(f"Warning: No MO information found for {c_file}")
                    output_file.write(f"Warning: No MO information found for {c_file}\n")
                    continue
    
                # Initialize variables to track matches
                best_volumetric_match = None
                best_ao_match = None
                best_volumetric_correlation = -1  # Initialize for volumetric matching
                lowest_ao_difference = float('inf')  # Initialize for AO matching
    
                for s_file in simple_files:  # Compare against all simple system files
                    # Load simple cube data
                    s_path = os.path.join(self.simple_dir, s_file)
                    s_data = self.load_cube_file_ase(s_path)
    
                    # Match simple MO name
                    simple_mo_name = os.path.splitext(s_file)[0].replace("_1_1_", "_1_")
                    simple_mo_info = next((mo for mo in self.mo_diagram_simple if mo['name'] == simple_mo_name), None)
    
                    if not simple_mo_info:
                        continue
    
                    # Calculate energy shift
                    energy_shift = complex_mo_info['energy'] - simple_mo_info['energy']
    
                    # Volumetric comparison
                    correlation = np.corrcoef(s_data.flatten(), c_data.flatten())[0, 1]
                    if correlation > best_volumetric_correlation:
                        best_volumetric_correlation = correlation
                        best_volumetric_match = {
                            'simple_file': s_file,
                            'simple_mo_info': simple_mo_info,
                            'correlation': correlation,
                            'complex_file': c_file,
                            'complex_mo_info': complex_mo_info,
                            'energy_shift': energy_shift
                        }
    
                    # AO contribution comparison
                    ao_contribution_differences = simple_mo_info['ao_contributions'] - complex_mo_info['ao_contributions']
                    ao_difference_sum = np.sum(np.abs(ao_contribution_differences))
                    if ao_difference_sum < lowest_ao_difference:
                        lowest_ao_difference = ao_difference_sum
                        best_ao_match = {
                            'simple_file': s_file,
                            'simple_mo_info': simple_mo_info,
                            'ao_differences': ao_contribution_differences,
                            'sum_difference': ao_difference_sum,
                            'complex_file': c_file,
                            'complex_mo_info': complex_mo_info,
                            'energy_shift': energy_shift
                        }
    
                # Combine or separate matches based on pairing
                if best_volumetric_match and best_ao_match and \
                   best_volumetric_match['simple_file'] == best_ao_match['simple_file']:
                    # If both matches are for the same pair, print a combined match
                    simple_mo = best_volumetric_match['simple_mo_info']
                    complex_mo = best_volumetric_match['complex_mo_info']
                    result = (f"Combined Match: Complex MO: {complex_mo['name']} (Energy: {complex_mo['energy']:.4f}) "
                              f"-> Simple MO: {simple_mo['name']} (Energy: {simple_mo['energy']:.4f})\n"
                              f"  Volumetric Correlation: {best_volumetric_match['correlation']:.4f}, "
                              f"Sum of AO Differences: {best_ao_match['sum_difference']:.4f}, "
                              f"Energy Shift: {best_volumetric_match['energy_shift']:.4f}\n")
                    print(result)
                    output_file.write(result)
                    matches.append({
                        'match_type': 'combined',
                        'complex_file': c_file,
                        'simple_file': best_volumetric_match['simple_file'],
                        'correlation': best_volumetric_match['correlation'],
                        'sum_difference': best_ao_match['sum_difference'],
                        'energy_shift': best_volumetric_match['energy_shift']
                    })
                else:
                    # Print volumetric match, if it exists
                    if best_volumetric_match:
                        simple_mo = best_volumetric_match['simple_mo_info']
                        complex_mo = best_volumetric_match['complex_mo_info']
                        result = (f"Volumetric Match: Complex MO: {complex_mo['name']} (Energy: {complex_mo['energy']:.4f}) "
                                  f"-> Simple MO: {simple_mo['name']} (Energy: {simple_mo['energy']:.4f})\n"
                                  f"  Volumetric Correlation: {best_volumetric_match['correlation']:.4f}, "
                                  f"Energy Shift: {best_volumetric_match['energy_shift']:.4f}\n")
                        print(result)
                        output_file.write(result)
                        matches.append({
                            'match_type': 'volumetric',
                            'complex_file': c_file,
                            'simple_file': best_volumetric_match['simple_file'],
                            'correlation': best_volumetric_match['correlation'],
                            'energy_shift': best_volumetric_match['energy_shift']
                        })
    
                    # Print AO match, if it exists
                    if best_ao_match:
                        simple_mo = best_ao_match['simple_mo_info']
                        complex_mo = best_ao_match['complex_mo_info']
                        result = (f"AO Match: Complex MO: {complex_mo['name']} (Energy: {complex_mo['energy']:.4f}) "
                                  f"-> Simple MO: {simple_mo['name']} (Energy: {simple_mo['energy']:.4f})\n"
                                  f"  Sum of AO Differences: {best_ao_match['sum_difference']:.4f}, "
                                  f"Energy Shift: {best_ao_match['energy_shift']:.4f}\n")
                        print(result)
                        output_file.write(result)
                        matches.append({
                            'match_type': 'ao',
                            'complex_file': c_file,
                            'simple_file': best_ao_match['simple_file'],
                            'sum_difference': best_ao_match['sum_difference'],
                            'energy_shift': best_ao_match['energy_shift']
                        })
    
        return matches
    
# Set up directories and paths
simple_calc_dir = "simple_filepath/"
complex_calc_dir = "complex_filepath"
mo_diagram_simple_path = os.path.join(simple_calc_dir, "simple.MO_Diagram.lobster")
mo_diagram_complex_path = os.path.join(complex_calc_dir, "simple.MO_Diagram.lobster")

# Create the LobsterAnalysis object
analysis = LobsterAnalysis(simple_calc_dir, complex_calc_dir, mo_diagram_simple_path, mo_diagram_complex_path)

# Run the comparison and write the results to a text file
output_path = 'filepath/complex_simple_ILDOS_analysis_results.txt'  # Specify the output file path
analysis.compare_mo_cube_and_orbital_contributions(output_path=output_path)

print(f"Results have been written to {output_path}.")
