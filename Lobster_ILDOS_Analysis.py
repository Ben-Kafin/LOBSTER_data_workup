from ase.io.cube import read_cube_data
import os
import numpy as np

class Lobster_ILDOS_Analysis:
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


    def compare_mo_cube_and_orbital_contributions(self, output_dir="."):
        """Compare MOs with volumetric and AO matching logic, print matches immediately, and save results to a directory."""
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Initialize lists for storing data
        all_pairs = []  # List to store all intermediate pairs
        matches = []
    
        # Retrieve all cube files from the complex and simple directories
        complex_files = [f for f in os.listdir(self.complex_dir) if f.endswith('.cube')]
        simple_files = [f for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
    
        for c_file in complex_files:  # Iterate over all complex cube files
            # Load data for the complex cube file
            c_path = os.path.join(self.complex_dir, c_file)
            c_data = self.load_cube_file_ase(c_path)
    
            # Match the complex MO name
            complex_mo_name = os.path.splitext(c_file)[0].replace("_1_1_", "_1_")
            complex_mo_info = next((mo for mo in self.mo_diagram_complex if mo['name'] == complex_mo_name), None)
    
            if not complex_mo_info:
                print(f"Warning: No MO information found for {c_file}")
                continue
    
            best_volumetric_match = None
            best_volumetric_correlation = -1
            best_ao_match = None
            best_ao_std_dev = float('inf')
    
            for s_file in simple_files:  # Compare against all simple system cube files
                # Load data for the simple cube file
                s_path = os.path.join(self.simple_dir, s_file)
                s_data = self.load_cube_file_ase(s_path)
    
                # Match the simple MO name
                simple_mo_name = os.path.splitext(s_file)[0].replace("_1_1_", "_1_")
                simple_mo_info = next((mo for mo in self.mo_diagram_simple if mo['name'] == simple_mo_name), None)
    
                if not simple_mo_info:
                    continue
    
                energy_shift = complex_mo_info['energy'] - simple_mo_info['energy']
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
    
                ao_contribution_differences = simple_mo_info['ao_contributions'] - complex_mo_info['ao_contributions']
                ao_difference_std_dev = np.std(ao_contribution_differences)
    
                if ao_difference_std_dev < best_ao_std_dev:
                    best_ao_std_dev = ao_difference_std_dev
                    best_ao_match = {
                        'simple_file': s_file,
                        'simple_mo_info': simple_mo_info,
                        'ao_differences': ao_contribution_differences,
                        'std_dev_difference': ao_difference_std_dev,
                        'complex_file': c_file,
                        'complex_mo_info': complex_mo_info,
                        'energy_shift': energy_shift
                    }
    
                # Save all intermediate results and print immediately
                all_pair = {
                    'complex_file': c_file,
                    'complex_mo_info': complex_mo_info,
                    'simple_file': s_file,
                    'simple_mo_info': simple_mo_info,
                    'volumetric_correlation': correlation,
                    'ao_std_dev': ao_difference_std_dev,
                    'energy_shift': energy_shift
                }
                all_pairs.append(all_pair)
    
                print(
                    f"Complex MO: {complex_mo_name} -> Simple MO: {simple_mo_name}, "
                    f"Volumetric Correlation: {correlation:.4f}, AO Std Dev: {ao_difference_std_dev:.4f}, "
                    f"Energy Shift: {energy_shift:.4f}"
                )
    
            if best_volumetric_match and best_ao_match and best_volumetric_match['simple_file'] == best_ao_match['simple_file']:
                match = {
                    'match_type': 'combined',
                    **best_volumetric_match,
                    'ao_std_dev': best_ao_match['std_dev_difference']
                }
            else:
                if best_volumetric_match:
                    match = {
                        'match_type': 'volumetric',
                        **best_volumetric_match,
                        'ao_std_dev': best_ao_std_dev
                    }
                if best_ao_match:
                    match = {
                        'match_type': 'ao',
                        **best_ao_match,
                        'volumetric_correlation': best_volumetric_correlation
                    }
            matches.append(match)
    
            # Print the match immediately after determining it
            if match['match_type'] == 'combined':
                print(
                    f"Combined Match: Complex MO: {match['complex_mo_info']['name']} -> Simple MO: {match['simple_mo_info']['name']}, "
                    f"Volumetric Correlation: {match['correlation']:.4f}, AO Std Dev: {match['ao_std_dev']:.4f}, "
                    f"Energy Shift: {match['energy_shift']:.4f}"
                )
            elif match['match_type'] == 'volumetric':
                print(
                    f"Volumetric Match: Complex MO: {match['complex_mo_info']['name']} -> Simple MO: {match['simple_mo_info']['name']}, "
                    f"Volumetric Correlation: {match['correlation']:.4f}, AO Std Dev: {match['ao_std_dev']:.4f}, "
                    f"Energy Shift: {match['energy_shift']:.4f}"
                )
            elif match['match_type'] == 'ao':
                print(
                    f"AO Match: Complex MO: {match['complex_mo_info']['name']} -> Simple MO: {match['simple_mo_info']['name']}, "
                    f"AO Std Dev: {match['std_dev_difference']:.4f}, Volumetric Correlation: {match['volumetric_correlation']:.4f}, "
                    f"Energy Shift: {match['energy_shift']:.4f}"
                )
    
        # Write all pairs to a file
        all_pairs_filepath = os.path.join(output_dir, "allpairs.txt")
        with open(all_pairs_filepath, "w") as all_pairs_file:
            all_pairs_file.write("All Pairs:\n")
            for pair in all_pairs:
                all_pairs_file.write(
                    f"Complex File: {pair['complex_file']}, Simple File: {pair['simple_file']}, "
                    f"Volumetric Correlation: {pair['volumetric_correlation']:.4f}, "
                    f"AO Std Dev: {pair['ao_std_dev']:.4f}, Energy Shift: {pair['energy_shift']:.4f}\n"
                )
    
        # Write matches to a separate file
        matches_filepath = os.path.join(output_dir, "matches.txt")
        with open(matches_filepath, "w") as matches_file:
            matches_file.write("Matches:\n")
            for match in matches:
                if match['match_type'] == 'combined':
                    matches_file.write(
                        f"Combined Match: Complex MO: {match['complex_mo_info']['name']} -> Simple MO: {match['simple_mo_info']['name']}, "
                        f"Volumetric Correlation: {match['correlation']:.4f}, AO Std Dev: {match['ao_std_dev']:.4f}, "
                        f"Energy Shift: {match['energy_shift']:.4f}\n"
                    )
                elif match['match_type'] == 'volumetric':
                    matches_file.write(
                        f"Volumetric Match: Complex MO: {match['complex_mo_info']['name']} -> Simple MO: {match['simple_mo_info']['name']}, "
                        f"Volumetric Correlation: {match['correlation']:.4f}, AO Std Dev: {match['ao_std_dev']:.4f}, "
                        f"Energy Shift: {match['energy_shift']:.4f}\n"
                    )
                elif match['match_type'] == 'ao':
                    matches_file.write(
                        f"AO Match: Complex MO: {match['complex_mo_info']['name']} -> Simple MO: {match['simple_mo_info']['name']}, "
                        f"AO Std Dev: {match['std_dev_difference']:.4f}, Volumetric Correlation: {match['volumetric_correlation']:.4f}, "
                        f"Energy Shift: {match['energy_shift']:.4f}\n"
                    )
    
        # Return the matches and all pairs
        return matches, all_pairs
        
# Set up directories and paths
simple_calc_dir = "C:/Users/benka/Documents/LOBSTER_run4/"
complex_calc_dir = "C:/Users/benka/Documents/LOBSTER_run2_AuNHC/"
mo_diagram_simple_path = os.path.join(simple_calc_dir, "C13N2H18_1.MO_Diagram.lobster")
mo_diagram_complex_path = os.path.join(complex_calc_dir, "AuC13N2H18_1.MO_Diagram.lobster")

# Create the LobsterAnalysis object
analysis = LobsterAnalysis(simple_calc_dir, complex_calc_dir, mo_diagram_simple_path, mo_diagram_complex_path)

# Run the comparison and write the results to a text file
output_dir = 'C:/Users/benka/Documents/AuC13N2H18_C13N2H18_ILDOS'  # Specify the output file path
analysis.compare_mo_cube_and_orbital_contributions(output_dir=output_dir)

print(f"Results have been written to {output_dir}.")
