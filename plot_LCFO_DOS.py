from matplotlib import pyplot as plt
from pymatgen.electronic_structure.core import Spin
from lib_DOSCAR_LCFO import DOSCAR_LCFO  # Import the DOSCAR_LCFO class

class DosPlotterLCFO:
    """
    A class to plot the TDOS and each individual pMODOS on the same graph.
    """

    def __init__(self, doscar_file, lcfo_fragments_file, mo_diagram, structure_file=None):
        """
        Args:
            doscar_file (str): Path to the DOSCAR.LCFO file.
            lcfo_fragments_file (str): Path to the LCFO_Fragments.lobster file.
            mo_diagram (dict): Molecular orbital diagram mapping MOs to names.
            structure_file (str): Optional structure file (e.g., POSCAR).
        """
        # Initialize the DOSCAR_LCFO object
        self.doscar_lcfo = DOSCAR_LCFO(
            doscar=doscar_file,
            lcfo_fragments_path=lcfo_fragments_file,
            mo_diagram=mo_diagram,
            structure_file=structure_file,
        )

    def plot(self, save_path=None, plot_tdos=True, fragment=None, orbital=None):
        """
        Plots the Total DOS (TDOS) and/or specific pMODOS on the same graph.
    
        Args:
            save_path (str): Optional path to save the plot as a file (e.g., 'plot.png').
            plot_tdos (bool): Whether to plot the Total DOS (TDOS).
            fragment (str): Fragment name to plot specific pMODOS (default is all fragments).
            orbital (str): Orbital name within the fragment to plot specific data (default is all orbitals).
        """
        # Extract data from the DOSCAR_LCFO object
        energies = self.doscar_lcfo.energies
        tdos = self.doscar_lcfo.tdos
        pmodos = self.doscar_lcfo.pmodos
    
        # Create the plot
        plt.figure(figsize=(12, 8))
    
        # Plot TDOS if requested
        if plot_tdos:
            plt.plot(energies, tdos.densities[Spin.up], label="TDOS Spin Up", color="black", linewidth=2)
            if self.doscar_lcfo.is_spin_polarized:
                plt.plot(energies, tdos.densities[Spin.down], label="TDOS Spin Down", color="gray", linestyle="--", linewidth=2)
    
        # Plot specific pMODOS data
        if fragment:
            if fragment in pmodos:
                print(f"Plotting pMODOS for fragment '{fragment}'...")
                if orbital:
                    # Specific orbital within the fragment
                    if orbital in pmodos[fragment]:
                        spin_data = pmodos[fragment][orbital]
                        plt.plot(energies, spin_data[Spin.up], label=f"{fragment} - {orbital} (Spin Up)", alpha=0.8)
                        if self.doscar_lcfo.is_spin_polarized:
                            plt.plot(energies, spin_data[Spin.down], label=f"{fragment} - {orbital} (Spin Down)", alpha=0.8, linestyle="--")
                    else:
                        print(f"Orbital '{orbital}' not found in fragment '{fragment}'. No data plotted for this orbital.")
                else:
                    # All orbitals within the fragment
                    for orbital, spin_data in pmodos[fragment].items():
                        plt.plot(energies, spin_data[Spin.up], label=f"{fragment} - {orbital} (Spin Up)", alpha=0.8)
                        if self.doscar_lcfo.is_spin_polarized:
                            plt.plot(energies, spin_data[Spin.down], label=f"{fragment} - {orbital} (Spin Down)", alpha=0.8, linestyle="--")
            else:
                print(f"Fragment '{fragment}' not found. No data plotted for this fragment.")
        else:
            # Plot all fragments if no specific fragment is requested
            print("Plotting pMODOS for all fragments...")
            for fragment, orbital_data in pmodos.items():
                for orbital, spin_data in orbital_data.items():
                    plt.plot(energies, spin_data[Spin.up], label=f"{fragment} - {orbital} (Spin Up)", alpha=0.8)
                    if self.doscar_lcfo.is_spin_polarized:
                        plt.plot(energies, spin_data[Spin.down], label=f"{fragment} - {orbital} (Spin Down)", alpha=0.8, linestyle="--")
    
        # Customize the plot
        plt.title("Density of States", fontsize=16)
        plt.xlabel("Energy (eV)", fontsize=14)
        plt.ylabel("Density of States", fontsize=14)
        plt.axvline(0, color="k", linestyle="--", linewidth=1, label="Fermi Level")
        #plt.legend(fontsize=10, loc="best", ncol=1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
    
        # Save the plot or show it
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def run(self, save_path=None):
        """
        Runs the plotting function.

        Args:
            save_path (str): Optional path to save the plot as a file.
        """
        print("Starting the DOS plotting...")
        self.plot(save_path=save_path)
        print("Plotting complete.")

# Example Usage
if __name__ == "__main__":


    # Initialize the DosPlotterLCFO class
    plotter = DosPlotterLCFO(
        doscar_file='C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/DOSCAR.LCFO.lobster',
        lcfo_fragments_file='C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/LCFO_Fragments.lobster',
        mo_diagram='C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/AuC13N2H18_1.MO_Diagram.lobster',
        structure_file='C:/Users/nazin_lab/Documents/VASP_files/NHCs/iPr/lone_adatoms/NHC_iPr_fcc/spinpol/kpoints551/POSCAR'
    )

    # Run the plotter
    #plotter.run(save_path="tdos_pmodos_plot.png")  # Save the plot
    plotter.plot(save_path="NHC_pmodos_lcfo_plot.png", plot_tdos=False, fragment="AuC13N2H18")