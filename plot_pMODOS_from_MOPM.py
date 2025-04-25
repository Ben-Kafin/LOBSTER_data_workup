from matplotlib import pyplot as plt
import numpy as np
from pymatgen.electronic_structure.core import Spin
from MOPM_Surfaces_vs_molecule_spinpol_mo1 import MOPM  # Updated import; assumes your working MOPM class is in MOPM.py
from lib_DOS_lcfo import DOSCAR_LCFO  # Import the DOSCAR_LCFO class
import itertools


class IntegratedPlotter:
    def __init__(self, simple_doscar_file, simple_lcfo_file, simple_mo_diagram,
                 complex_doscar_file, complex_lcfo_file, complex_mo_diagram,
                 matches_output_path):
        """
        Initialize the IntegratedPlotter with file paths for DOS, LCFO, and MO diagrams,
        as well as a path where the match output will be written.
    
        Args:
            simple_doscar_file (str): DOSCAR file for the simple system.
            simple_lcfo_file (str): LCFO fragments file for the simple system.
            simple_mo_diagram (str): MO diagram file for the simple system.
            complex_doscar_file (str): DOSCAR file for the complex system.
            complex_lcfo_file (str): LCFO fragments file for the complex system.
            complex_mo_diagram (str): MO diagram file for the complex system.
            matches_output_path (str): Path to write match output.
        """
        # Initialize the MOPM instance.
        self.mopm = MOPM(simple_mo_diagram, complex_mo_diagram, align_energy=True)
        self.alignment_shift = self.mopm.alignment_shift
    
        # Load the DOSCAR/LCFO instances.
        self.simple_dos = DOSCAR_LCFO(simple_doscar_file, simple_lcfo_file)
        self.complex_dos = DOSCAR_LCFO(complex_doscar_file, complex_lcfo_file)
    
        # Generate match results using the MOPM class.
        # Note: compare_mo_contributions writes its output to the provided file.
        self.matches = self.mopm.compare_mo_contributions(matches_output_path)
    
        # (Optional:) If you want to filter matches by additional criteria (e.g., occupation change), 
        # you can do that right here. For now, we simply pass along all computed matches.
        self.filtered_matches = self.matches  # No extra filtering method in the new MOPM.
    
        # Take the first set of PMODOS data from the complex DOS instance.
        self.complex_pmodos = next(iter(self.complex_dos.pmodos.values()), {})

    
    def plot_aggregated_pmodos(self, energy_shift_lower_bound, energy_shift_upper_bound,
                                show_occupation_changes=False,
                                user_defined_matches=None,
                                user_defined_complex_mos=None,
                                save_path=None):
        """
        Plots the PMODOS curves for matches (both computed and user-specified)
        that have an energy shift within the provided bounds.
        
        If user_defined_complex_mos is provided, only matches with those complex MO names are plotted.
    
        The simple system's energy axis is shifted using the MO1 alignment shift (from MOPM),
        ensuring both DOS curves are on the same energy scale.
        """
        # Retrieve energy grids and Fermi energies.
        simple_energies = self.simple_dos.energies
        complex_energies = self.complex_dos.energies
    
        # Get the alignment shift computed in MOPM.
        alignment_shift = self.alignment_shift
    
        # Shift the simple system’s energy grid by the alignment shift.
        adjusted_simple_energies = simple_energies + alignment_shift
        # Adjust the simple-system Fermi energy relative to the complex.
        adjusted_simple_fermi = (self.simple_dos.fermi_energy - self.complex_dos.fermi_energy) - alignment_shift
    
        # Start with computed matches.
        all_matches = list(self.filtered_matches)
    
        # If the user has specified complex MO names, restrict the computed matches to those.
        if user_defined_complex_mos is not None:
            all_matches = [match for match in all_matches if match["complex_mo"] in user_defined_complex_mos]
            if not all_matches:
                print("Warning: None of the computed matches correspond to the specified complex MO names.")
    
        # Optionally append user-defined full match dictionaries.
        if user_defined_matches is not None:
            for match in user_defined_matches:
                match["user_defined"] = True
                all_matches.append(match)
    
        # --- Further filter matches by energy shift bounds and, optionally, occupation change.
        energy_filtered_matches = []
        if show_occupation_changes:
            print(f"Matches with occupation change and |ΔE| between {energy_shift_lower_bound:.2f} and {energy_shift_upper_bound:.2f} eV:")
        else:
            print(f"All matches with |ΔE| between {energy_shift_lower_bound:.2f} and {energy_shift_upper_bound:.2f} eV:")
    
        for match in all_matches:
            energy_shift = match["energy_shift"]
            # For computed matches, check occupation-change if required.
            if not match.get("user_defined", False):
                changes_occupation = (match["simple_mo_energy"] * match["complex_mo_energy"]) < 0
                if show_occupation_changes and not changes_occupation:
                    continue
            if energy_shift_lower_bound <= abs(energy_shift) <= energy_shift_upper_bound:
                energy_filtered_matches.append(match)
                print(f"Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
                      f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
                      f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {energy_shift:.4f} eV")
    
        # --- Create the plot: two subplots for simple and complex PMODOS.
        # Assume energy_filtered_matches is your list of computed matches.
        # Build a color mapping (one color per unique match)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle = itertools.cycle(colors)
        match_colors = {}
        for match in energy_filtered_matches:
            key = match['complex_mo']  # unique key for the match
            if key not in match_colors:
                match_colors[key] = next(color_cycle)
        
        # Group matches by unique complex MO name so that each match is plotted only once.
        unique_matches = {}
        for match in energy_filtered_matches:
            key = match['complex_mo']
            if key not in unique_matches:
                unique_matches[key] = match
        # Now use unique_matches.values() in place of energy_filtered_matches, so that you don't plot duplicates
        
        # --- Create the plot: two subplots for simple and complex PMODOS.
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        axes[0].tick_params(labelbottom=True)
        
        # Plot the simple system PMODOS (flipped vertically) on the adjusted energy axis.
        ax_simple = axes[0]
        for match in unique_matches.values():
            # Derive the orbital key from simple MO name (assuming its format contains an underscore)
            orbital_key = match['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors.get(match['complex_mo'], 'black')
                # Plot Spin Up with a solid line.
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {match['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid',
                    alpha=0.7
                )
                # Plot Spin Down (if available) with a dashed line.
                if Spin.down in simple_dos_data:
                    ax_simple.plot(
                        adjusted_simple_energies,
                        -np.array(simple_dos_data[Spin.down]),
                        label=f"MO: {orbital_key} Spin Down (ΔE: {match['energy_shift']:.2f} eV)",
                        color=col,
                        linestyle='dashed',
                        alpha=0.7
                    )
        ax_simple.axvline(0, color="red", linestyle="--", linewidth=1, label="Surface Bound NHC Fermi Energy")
        ax_simple.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,
                          label=f"Adj. NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
        ax_simple.set_title("Lone NHC PMODOS (MO1 Aligned)")
        ax_simple.set_ylabel("Density of States (Flipped)")
        ax_simple.grid(alpha=0.3)
        ax_simple.legend(fontsize=8)
        
        # Plot the complex system PMODOS on its original energy axis.
        ax_complex = axes[1]
        for match in unique_matches.values():
            orbital_key = match['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors.get(match['complex_mo'], 'black')
                # Plot Spin Up with a solid line.
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {match['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid',
                    alpha=0.7
                )
                # Plot Spin Down with a dashed line.
                if Spin.down in complex_dos_data:
                    ax_complex.plot(
                        complex_energies,
                        np.array(complex_dos_data[Spin.down]),
                        label=f"MO: {orbital_key} Spin Down (ΔE: {match['energy_shift']:.2f} eV)",
                        color=col,
                        linestyle='dashed',
                        alpha=0.7
                    )
        ax_complex.axvline(0, color="red", linestyle="--", linewidth=1, label="Surface Bound NHC Fermi Energy")
        ax_complex.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,
                          label=f"Adj. NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
        ax_complex.set_title("Surface Bound NHC PMODOS (Reference at 0 eV)")
        ax_complex.set_xlabel("Energy (eV)")
        ax_complex.set_ylabel("Density of States")
        ax_complex.grid(alpha=0.3)
        ax_complex.legend(fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # File paths (update these paths as needed)
    simple_doscar_file = 'C:/directory/DOSCAR.LCFO.lobster'
    simple_lcfo_file   = 'C:/directory/LCFO_Fragments.lobster'
    simple_mo_diagram  = 'C:/directory/MO_Diagram.lobster'
    
    complex_doscar_file = 'C:/directory2/DOSCAR.LCFO.lobster'
    complex_lcfo_file   = 'C:/directory2/LCFO_Fragments.lobster'
    complex_mo_diagram  = 'C:/directory2/MO_Diagram_adjusted.lobster'
    
    matches_output_path = 'C:/directory2/matches_important.txt'
    
    # Energy shift bounds (in eV)
    energy_shift_lower_bound = 0
    energy_shift_upper_bound = 15.0
    
    # Initialize the IntegratedPlotter.
    integrated_plotter = IntegratedPlotter(
        simple_doscar_file,
        simple_lcfo_file,
        simple_mo_diagram,
        complex_doscar_file,
        complex_lcfo_file,
        complex_mo_diagram,
        matches_output_path
    )
    
    # Specify additional matches via complex MO names (optional).
    user_defined_complex_mos = ['37a','38a','39a','40a']
    
    #user_defined_complex_mos = ["C13N2H18_1_39a", "C13N2H18_1_40a"]
    integrated_plotter.plot_aggregated_pmodos(
        energy_shift_lower_bound=0,
        energy_shift_upper_bound=15.0,
        show_occupation_changes=False,
        user_defined_complex_mos=user_defined_complex_mos,
        save_path='C:/directory2/pmodos_plot.png'
    )
