from ase.io.cube import read_cube_data
import os
import re
import numpy as np

class Lobster_MOPM_Analysis:
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

    @staticmethod
    def extract_state_index(state_name):
        """Extract the index value from a state name."""
        match = re.search(r"_(\d+)[a-zA-Z]$", state_name)
        if match:
            return int(match.group(1))
        return None

    def compare_mo_cube_and_orbital_contributions(self, output_dir="."):
        """Compare MOs, focusing on matches with the highest magnitude, and save results efficiently."""
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Define filenames
        matches_file_name = "matches.txt"
    
        # Open the matches file for writing
        with open(os.path.join(output_dir, matches_file_name), "w") as matches_file:
            # Write header to the file
            matches_file.write("Matches:\n")
            matches_file.flush()
    
            # Retrieve all cube files
            complex_files = [f for f in os.listdir(self.complex_dir) if f.endswith('.cube')]
            simple_files = [f for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
    
            # Initialize matches list **outside** the loop
            matches = []
    
            for c_file in complex_files:  # Iterate over all complex cube files
                c_path = os.path.join(self.complex_dir, c_file)
                c_data = self.load_cube_file_ase(c_path)
            
                # Extract the complex file name and index
                complex_mo_name = os.path.splitext(c_file)[0].replace("_1_1_", "_1_")
                complex_index = self.extract_state_index(complex_mo_name)
                complex_mo_info = next((mo for mo in self.mo_diagram_complex if mo['name'] == complex_mo_name), None)
            
                if not complex_mo_info:
                    print(f"Warning: No MO information found for {c_file}")
                    continue
                best_volumetric_match = None
                best_ao_match = None
                found_volumetric_match = False
                found_ao_match = False

                # Sort simple files by proximity to the complex index
                simple_files_sorted = sorted(
                    simple_files,
                    key=lambda s_file: abs(
                        self.extract_state_index(os.path.splitext(s_file)[0].replace("_1_1_", "_1_")) - complex_index
                    )
                )
    
            
                for s_file in simple_files_sorted:  # Compare against all simple system cube files
                
                    s_path = os.path.join(self.simple_dir, s_file)
                    s_data = self.load_cube_file_ase(s_path)
            
                    simple_mo_name = os.path.splitext(s_file)[0].replace("_1_1_", "_1_")
                    simple_mo_info = next((mo for mo in self.mo_diagram_simple if mo['name'] == simple_mo_name), None)
            
                    if not simple_mo_info:
                        continue
            
                    energy_shift = complex_mo_info['energy'] - simple_mo_info['energy']
                    correlation = np.corrcoef(s_data.flatten(), c_data.flatten())[0, 1]
            
                    # Normalize AO contributions and compute overlap
                    simple_normalized = simple_mo_info['ao_contributions'] / np.linalg.norm(simple_mo_info['ao_contributions'])
                    complex_normalized = complex_mo_info['ao_contributions'] / np.linalg.norm(complex_mo_info['ao_contributions'])
                    ao_overlap = np.dot(simple_normalized, complex_normalized)
            
                    # Print the pair details immediately
                    print(
                        f"Complex File: {c_file}, Simple File: {s_file}, "
                        f"Volumetric Correlation: {correlation:.4f}, AO Overlap: {ao_overlap:.4f}, "
                        f"Energy Shift: {energy_shift:.4f}"
                    )
                    # Update the best volumetric match regardless of thresholds
                    if best_volumetric_match is None or abs(correlation) > abs(best_volumetric_match['volumetric_correlation']):
                        best_volumetric_match = {
                            'complex_file': c_file,
                            'complex_mo_info': complex_mo_info,
                            'simple_file': s_file,
                            'simple_mo_info': simple_mo_info,
                            'volumetric_correlation': correlation,
                            'ao_overlap': ao_overlap,
                            'energy_shift': energy_shift
                        }
            
                    # Update the best AO match regardless of thresholds
                    if best_ao_match is None or abs(ao_overlap) > abs(best_ao_match['ao_overlap']):
                        best_ao_match = {
                            'complex_file': c_file,
                            'complex_mo_info': complex_mo_info,
                            'simple_file': s_file,
                            'simple_mo_info': simple_mo_info,
                            'volumetric_correlation': correlation,
                            'ao_overlap': ao_overlap,
                            'energy_shift': energy_shift
                        }
                                    
                    # Check if the thresholds are met
                    if abs(correlation) >= 0.9:
                        found_volumetric_match = True
                        print(f"Volumetric threshold met for Complex MO: {complex_mo_name} -> Simple MO: {simple_mo_name}")
            
                    if abs(ao_overlap) >= 0.9:
                        found_ao_match = True
                        print(f"AO overlap threshold met for Complex MO: {complex_mo_name} -> Simple MO: {simple_mo_name}")
            
                    # If both thresholds are met, save the match and stop further comparisons for this complex file
                    if found_volumetric_match and found_ao_match:
                        matches.append({
                            'complex_file': c_file,
                            'complex_mo_info': complex_mo_info,
                            'simple_file': s_file,
                            'simple_mo_info': simple_mo_info,
                            'volumetric_correlation': correlation,
                            'ao_overlap': ao_overlap,
                            'energy_shift': energy_shift
                        })
                        matches_file.write(
                            f"Threshold Match: Complex MO: {complex_mo_name} -> Simple MO: {simple_mo_name}, "
                            f"Volumetric Correlation: {correlation:.4f}, AO Overlap: {ao_overlap:.4f}, "
                            f"Energy Shift: {energy_shift:.4f}\n"
                        )
                        matches_file.flush()
                        print(f"Both thresholds met! Stopping further comparisons for Complex File: {c_file}\n")
                        break  # Exit the inner loop
                        break  # Exit the inner loop
            
            
                # If no pair met the thresholds, save the best matches
                if not found_volumetric_match and best_volumetric_match:
                    print(
                        f"No volumetric correlation above threshold. Best Volumetric Match: Complex MO: {best_volumetric_match['complex_mo_info']['name']} -> "
                        f"Simple MO: {best_volumetric_match['simple_mo_info']['name']}, "
                        f"Volumetric Correlation: {best_volumetric_match['volumetric_correlation']:.4f}, AO Overlap: {best_volumetric_match['ao_overlap']:.4f}, "
                        f"Energy Shift: {best_volumetric_match['energy_shift']:.4f}\n"
                    )
                    matches.append({
                        'match_type': 'volumetric_best',
                        **best_volumetric_match
                    })
                if not found_ao_match and best_ao_match:
                    print(
                        f"No AO overlap above threshold. Best AO Match: Complex MO: {best_ao_match['complex_mo_info']['name']} -> "
                        f"Simple MO: {best_ao_match['simple_mo_info']['name']}, "
                        f"AO Overlap: {best_ao_match['ao_overlap']:.4f}, Volumetric Correlation: {best_ao_match['volumetric_correlation']:.4f}, "
                        f"Energy Shift: {best_ao_match['energy_shift']:.4f}\n"
                    )
                    matches.append({
                        'match_type': 'ao_best',
                        **best_ao_match
                    })
                                    
                                    
                        # Return the matches as the final result
        return matches
         
# Set up directories and paths
simple_calc_dir = 'path'
complex_calc_dir = 'path'
mo_diagram_simple_path = os.path.join(simple_calc_dir, 1.MO_Diagram.lobster")
mo_diagram_complex_path = os.path.join(complex_calc_dir, "2..MO_Diagram.lobster")

# Create the LobsterAnalysis object
analysis = Lobster_MOPM_Analysis(simple_calc_dir, complex_calc_dir, mo_diagram_simple_path, mo_diagram_complex_path)

# Run the comparison and write the results to a text file
output_dir = 'path'  # Specify the output file path
analysis.compare_mo_cube_and_orbital_contributions(output_dir=output_dir)

print(f"Results have been written to {output_dir}.")
