from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from lib import Cohpcar, Icohplist
import matplotlib.pyplot as plt
import numpy as np

class CohpPlotterClass:
    def __init__(self, file_path, energy_range, subplot_adjustments, threshold=None, bond_length_range=None):
        """
        Initializes the plotter.

        :param file_path: Path to the COHPCAR file.
        :param energy_range: Y-axis range for energy levels.
        :param subplot_adjustments: Adjustments for subplot spacing.
        :param threshold: Tuple (positive_threshold, negative_threshold) for filtering curves based on ICOHP values.
        :param bond_length_range: Tuple (min_length, max_length) to filter based on bond lengths.
        """
        self.file_path = file_path
        self.energy_range = energy_range
        self.subplot_adjustments = subplot_adjustments
        self.threshold = threshold  # (positive_threshold, negative_threshold)
        self.bond_length_range = bond_length_range  # (min_length, max_length)
        self.cohr = None
        self.icohp_data = None
        self.cdata_processed = {}

    def load_cohpcar(self):
        """
        Load COHPCAR data and initialize atomic and orbital-wise COHP data.
        """
        # Load COHPCAR data
        cohpcar = Cohpcar(filename=self.file_path)
        self.cohr = cohpcar
    
        # Extract atomic and orbital-wise COHP data
        self.atomic_cohp_data = self.cohr.cohp_data  # Atomic-level COHP data
        self.orbital_cohp_data = self.cohr.orb_res_cohp  # Orbital-wise COHP data


    def load_icohplist(self):
        # Derive the ICOHPLIST file path from the COHPCAR file path
        icohplist_path = self.file_path.replace("COHPCAR", "ICOHPLIST")
        # Load ICOHPLIST data
        icohplist = Icohplist(filename=icohplist_path)
        self.icohp_data = icohplist.icohplist  # Dictionary with ICOHP values and labels
        
    def process_cohp_data(self, bond_length_min=None, bond_length_max=None):
        """
        Process both atomic and orbital-wise COHP data for plotting.
        """
        if not hasattr(self, "cohr") or not self.cohr:
            raise ValueError("COHPCAR data not loaded. Call load_cohpcar first.")
        
        # Normalize keys
        self.atomic_cohp_data = {str(k): v for k, v in self.cohr.cohp_data.items()}
        self.orbital_cohp_data = {str(k): v for k, v in self.cohr.orb_res_cohp.items()} if self.cohr.orb_res_cohp else {}
    
        for key, cohp_entry in self.atomic_cohp_data.items():
            # Extract bond length and filter
            bond_length = cohp_entry.get("length", None)
            if bond_length is None:
                print(f"Skipping bond {key} due to missing bond length.")
                continue
            if bond_length_min is not None and bond_length < bond_length_min:
                print(f"Skipping bond {key} with length {bond_length} < {bond_length_min}")
                continue
            if bond_length_max is not None and bond_length > bond_length_max:
                print(f"Skipping bond {key} with length {bond_length} > {bond_length_max}")
                continue
    
            # Retrieve spin-specific ICOHP values
            icohp = cohp_entry.get("ICOHP", {})
            spin1 = icohp.get(Spin.up, None)
            spin2 = icohp.get(Spin.down, None)
    
            # Process only atomic-level data
            valid_icohp_values = [v for v in [spin1, spin2] if v is not None]
            avg_icohp = sum(valid_icohp_values) / len(valid_icohp_values) if valid_icohp_values else 0.0
    
            # Initialize COHP object for atomic data
            c = Cohp(
                efermi=self.cohr.efermi,
                energies=self.cohr.energies,
                cohp=cohp_entry.get("COHP", {}),
                icohp=None,
                are_coops=False,
            )
    
            interaction_label = f"{key}: Atomic Level (Length: {bond_length:.2f} Å, Average ICOHP: {avg_icohp:.4f})"
            self.cdata_processed[interaction_label] = c
            print(f"Processed atomic COHP data for {interaction_label}")
    
            # Process orbital-wise COHP data separately
            if key in self.orbital_cohp_data:
                orbital_contributions = self.orbital_cohp_data[key]
                total_orb_icohp = []  # To store all orbital ICOHP values for averaging
                for orb_label, orb_data in orbital_contributions.items():
                    orb_icohp = orb_data.get("ICOHP", None)
                    if isinstance(orb_icohp, (list, np.ndarray)):
                        valid_orb_icohp_values = [v for v in orb_icohp if v is not None]
                    else:
                        valid_orb_icohp_values = [v for v in orb_icohp.values() if v is not None] if orb_icohp else []
                    
                    if not valid_orb_icohp_values:
                        print(f"Skipping orbital {orb_label} due to missing ICOHP data.")
                        continue
                    
                    orb_avg_icohp = sum(valid_orb_icohp_values) / len(valid_orb_icohp_values)
                    total_orb_icohp.extend(valid_orb_icohp_values)  # Add valid values for global averaging
                    
                    interaction_label_orbital = (
                        f"{key}: Orbital Contribution (Length: {bond_length:.2f} Å, "
                        f"Orbital: {orb_label}, Average ICOHP: {orb_avg_icohp:.4f})"
                    )
                    
                    orb_c = Cohp(
                        efermi=self.cohr.efermi,
                        energies=self.cohr.energies,
                        cohp=orb_data.get("COHP", {}),
                        icohp=None,
                        are_coops=False,
                    )
                    self.cdata_processed[interaction_label_orbital] = orb_c
                    print(f"Processed orbital-wise COHP data for {interaction_label_orbital}")
    
                # Calculate overall orbital average for the bond
                if total_orb_icohp:
                    global_orb_avg_icohp = sum(total_orb_icohp) / len(total_orb_icohp)
                    print(f"Overall average ICOHP for bond {key} across all orbitals: {global_orb_avg_icohp:.4f}")


    def plot_cohp(self):
        if not self.cdata_processed:
            raise ValueError("No data available to plot. Call process_cohp_data first.")
    
        from pymatgen.electronic_structure.plotter import CohpPlotter
        import matplotlib.pyplot as plt
    
        cp = CohpPlotter()
        cp.add_cohp_dict(self.cdata_processed)
    
        # Generate a custom legend from interaction labels
        custom_legend_labels = list(self.cdata_processed.keys())
    
        try:
            ax = cp.get_plot()
            ax.set_ylim(self.energy_range)
            ax.set_xlabel("-COHP")
            ax.set_ylabel("Energy (eV)")
    
            # Replace the legend with interaction labels
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, custom_legend_labels, fontsize=8, loc="upper right")
    
            plt.gcf().set_size_inches(10, 6)
            plt.show()
        except ValueError as e:
            print(f"Error during plotting: {e}")
            raise





# Usage Example
file_path = 'filepath/COHPCAR.lobster'
energy_range = [-24, 6]
subplot_adjustments = {'bottom': 0.1, 'top': 0.9, 'left': 0.1, 'right': 0.9}
#threshold = (0.0000, -0.0000)  # Set the threshold for ICOHP values
bond_length_range = (0.5, 5.0)  # Set the range for bond lengths (min_length, max_length)

# Initialize the plotter class
cohp_plotter = CohpPlotterClass(file_path, energy_range, subplot_adjustments, bond_length_range)

cohp_plotter.load_cohpcar()  # First, load COHPCAR
cohp_plotter.load_icohplist()  # Then, load ICOHPLIST

# Pass bond length filtering parameters to process_cohp_data
cohp_plotter.process_cohp_data(bond_length_min=bond_length_range[0], bond_length_max=bond_length_range[1])

# Plot the filtered COHP data
cohp_plotter.plot_cohp()

