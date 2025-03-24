import os
import numpy as np
from scipy.spatial.distance import cosine
import pyvista as pv
from pymatgen.io.vasp.outputs import Chgcar
from scipy.interpolate import RegularGridInterpolator


class CubeFileAnalyzer:
    def __init__(self):
        self.fragments = {}
        self.match_results = {}

    def load_cube(self, file_path):
        """
        Parses a .cube file to extract 3D volumetric data and grid dimensions.
        """
        with open(file_path, 'r') as file:
            lines = file.readlines()
    
        # Read header lines (first few lines of the file)
        header_lines = lines[:6]  # First 6 lines should generally include metadata
    
        try:
            # Extract grid dimensions (first values in the 2nd, 3rd, and 4th relevant lines)
            nx = int(header_lines[2].split()[0])  # Grid points along X
            ny = int(header_lines[3].split()[0])  # Grid points along Y
            nz = int(header_lines[4].split()[0])  # Grid points along Z
            grid_dims = (nx, ny, nz)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing grid dimensions in file {file_path}: {e}")
    
        # Parse volumetric data starting from the 6th line
        data_lines = lines[6:]  # The actual volumetric data starts after the header
        data = []
        for line in data_lines:
            try:
                data.extend([float(x) for x in line.split()])
            except ValueError as e:
                raise ValueError(f"Error parsing volumetric data in file {file_path}: {e}")
    
        # Reshape the data into a 3D grid based on the parsed dimensions
        try:
            data = np.array(data).reshape(grid_dims)
        except ValueError as e:
            raise ValueError(f"Error reshaping data into grid dimensions {grid_dims} in file {file_path}: {e}")
    
        return data, header_lines





    def load_all_cube_files(self, directory):
        """
        Loads all cube files from a given directory and stores them with their 3D structure.
        """
        self.fragments = {}
        for file_name in os.listdir(directory):
            if file_name.endswith(".cube"):
                file_path = os.path.join(directory, file_name)
                data, header = self.load_cube(file_path)
                self.fragments[file_name] = {"data": data, "header": header}
        print(f"Loaded {len(self.fragments)} cube files from {directory}.")
   
    
    def interpolate_to_match(source_data, target_shape):
        """
        Interpolates source_data to match the dimensions of target_shape.
    
        Args:
            source_data (numpy.ndarray): The data to be interpolated.
            target_shape (tuple): The target shape for the interpolation.
    
        Returns:
            numpy.ndarray: Interpolated data with the target shape.
        """
        # Create a grid for the source data
        source_shape = source_data.shape
        source_grid = [np.linspace(0, 1, dim) for dim in source_shape]
    
        # Create the interpolator object
        interpolator = RegularGridInterpolator(source_grid, source_data)
    
        # Create a target grid based on target dimensions
        target_grid = [np.linspace(0, 1, dim) for dim in target_shape]
        target_points = np.meshgrid(*target_grid, indexing="ij")
    
        # Flatten the target grid and interpolate to the target shape
        target_flat_points = np.column_stack([g.flatten() for g in target_points])
        interpolated_data = interpolator(target_flat_points).reshape(target_shape)

        return interpolated_data

    
    def compare_fragments(self, smaller_data, larger_data, larger_dims):
        """
        Interpolates smaller fragment data onto the grid of the larger fragment,
        then compares molecular orbital contributions.
    
        Args:
            smaller_data (np.ndarray): 3D volumetric data from the smaller fragment.
            larger_data (np.ndarray): 3D volumetric data from the larger fragment.
            larger_dims (tuple): Grid dimensions of the larger fragment.
    
        Returns:
            float: Overlap metric between the fragments.
        """
        # Interpolate smaller fragment data to match larger fragment's grid
        interpolated_smaller = self.interpolate_to_match(smaller_data, larger_dims)
    
        # Normalize both datasets
        smaller_normalized = interpolated_smaller / np.linalg.norm(interpolated_smaller)
        larger_normalized = larger_data / np.linalg.norm(larger_data)
    
        # Calculate overlap (dot product as a similarity measure)
        overlap_metric = np.sum(smaller_normalized * larger_normalized)
    
        return overlap_metric


    def find_closest_match(self, smaller_data, fragments):
        """
        Finds the closest matching cube file based on volumetric data similarity.
    
        Args:
            smaller_data (numpy.ndarray): Volumetric data of the smaller fragment.
            fragments (dict): Dictionary of loaded fragments with keys as filenames
                              and values as volumetric data.
    
        Returns:
            tuple: Closest match filename and its similarity score.
        """
        results = {}
    
        for file_name, larger_fragment in fragments.items():
            larger_data = larger_fragment["data"]
            larger_shape = larger_data.shape
    
            # Interpolate smaller data to match the larger fragment's grid dimensions
            interpolated_smaller = self.interpolate_to_match(smaller_data, larger_shape)
    
            # Normalize both datasets for fair comparison
            smaller_normalized = interpolated_smaller / np.linalg.norm(interpolated_smaller)
            larger_normalized = larger_data / np.linalg.norm(larger_data)
    
            # Compute similarity using dot product
            similarity_score = np.sum(smaller_normalized * larger_normalized)
            results[file_name] = similarity_score
    
        # Find the fragment with the highest similarity score
        closest_match = max(results, key=results.get)
        return closest_match, results[closest_match]


    def visualize_comparison(self, fragment_1, fragment_2):
        """
        Visualizes the comparison between two fragments.
        """
        data_1 = self.fragments[fragment_1]["data"]
        data_2 = self.fragments[fragment_2]["data"]

        # Overlay the two fragments
        combined_data = data_1 + data_2
        grid = pv.UniformGrid()
        grid.dimensions = combined_data.shape
        grid.spacing = (0.1, 0.1, 0.1)
        grid.cell_data["combined_density"] = combined_data.flatten(order="C")

        plotter = pv.Plotter()
        plotter.add_mesh(grid, cmap="plasma")
        plotter.show(title=f"Comparison of {fragment_1} and {fragment_2}")


if __name__ == "__main__":
    # Step 1: Load MO Data (Cube File)
    mo_cube_file = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/LOBSTER_run4/C13N2H18_1_1_40a.cube'
    mo_data, _ = load_cube(mo_cube_file)
    print("MO Data Loaded:", mo_data.shape)

    # Step 2: Load Charge Density Data (CHGCAR File)
    chgcar_file = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/LOBSTER_run4/CHGCAR' 
    density_data = load_vasp_chgcar(chgcar_file)
    print("Density Data Loaded:", density_data.shape)

    # Step 3: Interpolate Density Data to Match MO Data
    mo_shape = mo_data.shape
    density_data_interpolated = interpolate_to_match(density_data, mo_shape)
    print("Density Data Interpolated:", density_data_interpolated.shape)

    # Step 4: Visualize the Combined Data
    visualize_mo_and_ildos(mo_data, density_data_interpolated, title="Overlay of MO and System ILDOS")

