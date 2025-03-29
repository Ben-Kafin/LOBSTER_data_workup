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
    
        # Define filenames
        all_pairs_file_name = "allpairs.txt"
        matches_file_name = "matches.txt"
    
        # Open the files once for writing
        with open(os.path.join(output_dir, all_pairs_file_name), "w") as all_pairs_file, \
             open(os.path.join(output_dir, matches_file_name), "w") as matches_file:
    
            # Write headers to the files
            all_pairs_file.write("All Pairs:\n")
            all_pairs_file.flush()
            matches_file.write("Matches:\n")
            matches_file.flush()
    
            # Initialize lists for storing data
            all_pairs = []  # List to store all intermediate pairs
            matches = []
    
            # Retrieve all cube files from the complex and simple directories
            complex_files = [f for f in os.listdir(self.complex_dir) if f.endswith('.cube')]
            simple_files = [f for f in os.listdir(self.simple_dir) if f.endswith('.cube')]
            #print(f"Complex files: {complex_files}")  # Debugging: Confirm complex files
            #print(f"Simple files: {simple_files}")    # Debugging: Confirm simple files
    
            for c_file in complex_files:  # Iterate over all complex cube files
                c_path = os.path.join(self.complex_dir, c_file)
                c_data = self.load_cube_file_ase(c_path)
            
                complex_mo_name = os.path.splitext(c_file)[0].replace("_1_1_", "_1_")
                complex_mo_info = next((mo for mo in self.mo_diagram_complex if mo['name'] == complex_mo_name), None)
            
                if not complex_mo_info:
                    print(f"Warning: No MO information found for {c_file}")
                    continue
            
                # Initialize a temporary list for pairs associated with this complex file
                all_simple_pairs = []
            
                for s_file in simple_files:  # Compare against all simple system cube files
                    s_path = os.path.join(self.simple_dir, s_file)
                    s_data = self.load_cube_file_ase(s_path)
            
                    simple_mo_name = os.path.splitext(s_file)[0].replace("_1_1_", "_1_")
                    simple_mo_info = next((mo for mo in self.mo_diagram_simple if mo['name'] == simple_mo_name), None)
            
                    if not simple_mo_info:
                        continue
            
                    energy_shift = complex_mo_info['energy'] - simple_mo_info['energy']
                    correlation = np.corrcoef(s_data.flatten(), c_data.flatten())[0, 1]
                    
                    # Step 1: Normalize AO contributions
                    simple_normalized = simple_mo_info['ao_contributions'] / np.linalg.norm(simple_mo_info['ao_contributions'])
                    complex_normalized = complex_mo_info['ao_contributions'] / np.linalg.norm(complex_mo_info['ao_contributions'])
                    
                    # Step 2: Compute cosine similarity (or overlap)
                    ao_overlap = np.dot(simple_normalized, complex_normalized)
                    
                    # Create the pair for this specific combination
                    pair = {
                        'complex_file': c_file,
                        'complex_mo_info': complex_mo_info,
                        'simple_file': s_file,
                        'simple_mo_info': simple_mo_info,
                        'volumetric_correlation': correlation,
                        'ao_overlap': ao_overlap,
                        'energy_shift': energy_shift
                    }
            
                    # Add pair to the temporary list and the global list
                    all_simple_pairs.append(pair)
                    all_pairs.append(pair)

                     # Print the pair to the console immediately
                   # '''
                    print(
                        f"Complex File: {pair['complex_file']}, Simple File: {pair['simple_file']}, "
                        f"Volumetric Correlation: {pair['volumetric_correlation']:.4f}, "
                        f"AO Overlap: {pair['ao_overlap']:.4f}, Energy Shift: {pair['energy_shift']:.4f}\n"
                    )
                   # '''
                    # Write the pair immediately to allpairs.txt
                    all_pairs_file.write(
                        f"Complex File: {pair['complex_file']}, Simple File: {pair['simple_file']}, "
                        f"Volumetric Correlation: {pair['volumetric_correlation']:.4f}, "
                        f"AO Overlap: {pair['ao_overlap']:.4f}, Energy Shift: {pair['energy_shift']:.4f}\n"
                    )
                    all_pairs_file.flush()  # Force the buffer to flush
                    
                # Determine best matches using the temporary list
                best_volumetric_match = max(
                    all_simple_pairs,
                    key=lambda pair: pair['volumetric_correlation'],
                    default=None
                )
                
                best_ao_match = max(
                    all_simple_pairs,
                    key=lambda pair: pair['ao_overlap'],
                    default=None
                )
                
                # Check if the matches align (combined match)
                if best_volumetric_match and best_ao_match and best_volumetric_match['simple_file'] == best_ao_match['simple_file']:
                    combined_match = {
                        'match_type': 'combined',
                        **best_volumetric_match,
                        'ao_overlap': best_ao_match['ao_overlap']
                    }
                    print(
                        f"Combined Match: Complex MO: {combined_match['complex_mo_info']['name']} -> Simple MO: {combined_match['simple_mo_info']['name']}, "
                        f"Volumetric Correlation: {combined_match['volumetric_correlation']:.4f}, AO Overlap: {combined_match['ao_overlap']:.4f}, "
                        f"Energy Shift: {combined_match['energy_shift']:.4f}\n"
                    )
                    matches.append(combined_match)
                
                    # Write the combined match immediately to matches.txt
                    matches_file.write(
                        f"Combined Match: Complex MO: {combined_match['complex_mo_info']['name']} -> Simple MO: {combined_match['simple_mo_info']['name']}, "
                        f"Volumetric Correlation: {combined_match['volumetric_correlation']:.4f}, AO Overlap: {combined_match['ao_overlap']:.4f}, "
                        f"Energy Shift: {combined_match['energy_shift']:.4f}\n"
                    )
                    matches_file.flush()  # Flush the file buffer
                else:
                    # If no combined match exists, write individual matches
                    if best_volumetric_match:
                        print(
                            f"Volumetric Match: Complex MO: {best_volumetric_match['complex_mo_info']['name']} -> Simple MO: {best_volumetric_match['simple_mo_info']['name']}, "
                            f"Volumetric Correlation: {best_volumetric_match['volumetric_correlation']:.4f}, AO Overlap: {best_volumetric_match['ao_overlap']:.4f}, "
                            f"Energy Shift: {best_volumetric_match['energy_shift']:.4f}\n"
                        )
                        matches.append({
                            'match_type': 'volumetric',
                            **best_volumetric_match
                        })
                        matches_file.write(
                            f"Volumetric Match: Complex MO: {best_volumetric_match['complex_mo_info']['name']} -> Simple MO: {best_volumetric_match['simple_mo_info']['name']}, "
                            f"Volumetric Correlation: {best_volumetric_match['volumetric_correlation']:.4f}, AO Overlap: {best_volumetric_match['ao_overlap']:.4f}, "
                            f"Energy Shift: {best_volumetric_match['energy_shift']:.4f}\n"
                        )
                        matches_file.flush()  # Flush the file buffer
                
                    if best_ao_match:
                        print(
                            f"AO Match: Complex MO: {best_ao_match['complex_mo_info']['name']} -> Simple MO: {best_ao_match['simple_mo_info']['name']}, "
                            f"AO Overlap: {best_ao_match['ao_overlap']:.4f}, Volumetric Correlation: {best_ao_match['volumetric_correlation']:.4f}, "
                            f"Energy Shift: {best_ao_match['energy_shift']:.4f}\n"
                        )
                        matches.append({
                            'match_type': 'ao',
                            **best_ao_match
                        })
                        matches_file.write(
                            f"AO Match: Complex MO: {best_ao_match['complex_mo_info']['name']} -> Simple MO: {best_ao_match['simple_mo_info']['name']}, "
                            f"AO Overlap: {best_ao_match['ao_overlap']:.4f}, Volumetric Correlation: {best_ao_match['volumetric_correlation']:.4f}, "
                            f"Energy Shift: {best_ao_match['energy_shift']:.4f}\n"
                        )
                        matches_file.flush()  # Flush the file buffer
        
            # Return the matches and all pairs
            return matches, all_pairs
         
# Set up directories and paths
simple_calc_dir = "simple_filepath"
complex_calc_dir = "complex_filepath"
mo_diagram_simple_path = os.path.join(simple_calc_dir, "simple.MO_Diagram.lobster")
mo_diagram_complex_path = os.path.join(complex_calc_dir, "copmlicated.MO_Diagram.lobster")

# Create the LobsterAnalysis object
analysis = Lobster_ILDOS_Analysis(simple_calc_dir, complex_calc_dir, mo_diagram_simple_path, mo_diagram_complex_path)

# Run the comparison and write the results to a text file
output_dir = 'Output_Directory'  # Specify the output file path
analysis.compare_mo_cube_and_orbital_contributions(output_dir=output_dir)

print(f"Results have been written to {output_dir}.")
