from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.electronic_structure.plotter import CohpPlotter
from lib import Cohpcar, Icohplist
import matplotlib.pyplot as plt

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
        
    def process_cohp_data(self, bond_length_min=None, bond_length_max=None, top_contributors=3):
        """
        Process atomic and orbital-wise COHP data directly from the COHPCAR file.
        
        :param bond_length_min: Minimum bond length for filtering.
        :param bond_length_max: Maximum bond length for filtering.
        :param top_contributors: Number of top orbital contributors to include.
        """
        if not hasattr(self, "cohr") or not self.cohr:
            raise ValueError("COHPCAR data not loaded. Call load_cohpcar first.")
    
        # Normalize atomic and orbital COHP data
        self.atomic_cohp_data = {str(k): v for k, v in self.cohr.cohp_data.items()}
        self.orbital_cohp_data = (
            {str(k): v for k, v in self.cohr.orb_res_cohp.items()} if self.cohr.orb_res_cohp else {}
        )
    
        for key, atomic_entry in self.atomic_cohp_data.items():
            # Extract atomic bond data
            fragment_alpha, fragment_beta = atomic_entry["sites"]
            bond_length = atomic_entry["length"]
    
            # Filter based on bond length
            if bond_length_min is not None and bond_length < bond_length_min:
                print(f"Skipping bond {key} with length {bond_length} < {bond_length_min}")
                continue
            if bond_length_max is not None and bond_length > bond_length_max:
                print(f"Skipping bond {key} with length {bond_length} > {bond_length_max}")
                continue
    
            from pymatgen.electronic_structure.cohp import Cohp
    
            # Process atomic COHP data
            c = Cohp(
                efermi=self.cohr.efermi,
                energies=self.cohr.energies,
                cohp=atomic_entry["COHP"],
                icohp=atomic_entry["ICOHP"],
                are_coops=False,
            )
    
            avg_icohp = sum(v for v in atomic_entry["ICOHP"].values() if v is not None) / len(atomic_entry["ICOHP"])
            interaction_label = f"{fragment_alpha} - {fragment_beta} ({bond_length:.2f} Å, ICOHP: {avg_icohp:.4f})"
            self.cdata_processed[interaction_label] = c
            print(f"Processed atomic data for {interaction_label}")
    
            # Process orbital-wise COHP data (if available)
            if key in self.orbital_cohp_data:
                orbital_contributions = self.orbital_cohp_data[key]
                for orb_label, orb_data in orbital_contributions.items():
                    # Calculate average orbital ICOHP
                    orb_icohp = orb_data.get("ICOHP", {})
                    orb_avg_icohp = sum(v for v in orb_icohp.values() if v is not None) / len(orb_icohp) if orb_icohp else 0.0
    
                    # Create orbital-specific label
                    interaction_label_orbital = (
                        f"{fragment_alpha} - {fragment_beta} ({bond_length:.2f} Å, "
                        f"Orbital: {orb_label}, ICOHP: {orb_avg_icohp:.4f})"
                    )
    
                    orb_c = Cohp(
                        efermi=self.cohr.efermi,
                        energies=self.cohr.energies,
                        cohp=orb_data.get("COHP", {}),
                        icohp=None,
                        are_coops=False,
                    )
                    self.cdata_processed[interaction_label_orbital] = orb_c
                    print(f"Processed orbital-wise data for {interaction_label_orbital}")


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
file_path = 'C:/Users/Benjamin Kafin/Documents/VASP/LOBSTER/COHPCAR.lobster'
energy_range = [-24, 6]
subplot_adjustments = {'bottom': 0.1, 'top': 0.9, 'left': 0.1, 'right': 0.9}
#threshold = (0.0000, -0.0000)  # Set the threshold for ICOHP values
bond_length_range = (0.5, 5.0)  # Set the range for bond lengths (min_length, max_length)

# Initialize the plotter class
cohp_plotter = CohpPlotterClass(file_path, energy_range, subplot_adjustments, bond_length_range)

cohp_plotter.load_cohpcar()  # First, load COHPCAR


# Pass bond length filtering parameters to process_cohp_data
cohp_plotter.process_cohp_data(bond_length_min=bond_length_range[0], bond_length_max=bond_length_range[1])

# Plot the filtered COHP data
cohp_plotter.plot_cohp()

