# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:42:19 2025

@author: Benjamin Kafin
"""

from matplotlib import pyplot as plt
import numpy as np
from pymatgen.electronic_structure.core import Spin
from MOPM_occ_unocc import MOPM
from lib_DOSCAR_LCFO import DOSCAR_LCFO  # Import the DOSCAR_LCFO class


class IntegratedPlotter:
    def __init__(self, simple_doscar_file, simple_lcfo_file, simple_mo_diagram,
                 complex_doscar_file, complex_lcfo_file, complex_mo_diagram,
                 matches_output_path, simple_atoms, complex_atoms,
                 criteria):
        """
        Initialize the IntegratedPlotter with file paths, matching logic, and filtering.
    
        Args:
            (Same as before)
            criteria (str): Filtering criteria in the format "simple:X, complex:Y", 
                            where X and Y can be "occupied", "unoccupied", or "both".
                            Default is "simple:both, complex:both".
        """
        # Initialize the MOPM instance
        self.mopm = MOPM(simple_mo_diagram, complex_mo_diagram,
                         simple_atoms, complex_atoms, matches_output_path, criteria)
    
        # Initialize the DOSCAR_LCFO instances
        self.simple_dos = DOSCAR_LCFO(simple_doscar_file, simple_lcfo_file, simple_mo_diagram)
        self.complex_dos = DOSCAR_LCFO(complex_doscar_file, complex_lcfo_file, complex_mo_diagram)
        
        self.simple_atoms = simple_atoms
        self.complex_atoms = complex_atoms
        
        # Generate matches and write to output
        self.matches = self.mopm.compare_mo_contributions(matches_output_path)
    
        # Apply occupation-based filtering using the method in the MOPM class
        self.filtered_matches = self.mopm.filter_matches_occupation(self.matches, criteria=criteria)
    
        # Filter complex PMODOS to use only the first instance
        self.complex_pmodos = next(iter(self.complex_dos.pmodos.values()), {})

    def plot_aggregated_pmodos(self, criteria, energy_shift_lower_bound, energy_shift_upper_bound, save_path=None):
        """
        Plots the PMODOS curves for all filtered matches within the provided energy shift bounds,
        but now aligns the energy axes using the lowest MO energies from the first match.
        
        The simple system’s energy axis is shifted so that its lowest MO (from the first match) aligns
        with that of the complex system. In addition, the simple Fermi energy (from self.simple_dos.fermi_energy)
        is plotted relative to the simple system's lowest MO.
        
        Args:
            criteria (str): Filtering criteria string.
            energy_shift_lower_bound (float): Lower bound for the absolute adjusted energy shift (in eV).
            energy_shift_upper_bound (float): Upper bound for the absolute adjusted energy shift (in eV).
            save_path (str): Optional file path to save the resulting plot.
        """
        # Retrieve the original energies from your DOSCAR_LCFO instances.
        simple_energies = self.simple_dos.energies
        complex_energies = self.complex_dos.energies
    
        # Ensure at least one filtered match is available:
        if not self.filtered_matches:
            raise ValueError("No filtered matches available for determining reference MO energies.")
    
        # Use the first match as reference.
        first_match = self.filtered_matches[0]
        simple_ref_energy = first_match["simple_mo_energy"]
        complex_ref_energy = first_match["complex_mo_energy"]
    
        # Compute the overall shift.
        # This shifts the simple system so that its lowest MO aligns with the complex system's.
        overall_shift = simple_ref_energy - complex_ref_energy
    
        # Adjust the simple energies.
        adjusted_simple_energies = simple_energies - overall_shift
    
        # Compute the simple system's Fermi energy on the new energy scale.
        adjusted_simple_fermi = (self.simple_dos.fermi_energy-self.complex_dos.fermi_energy)+ overall_shift
    
        # Filter matches based on the adjusted energy difference relative to the reference.
        energy_filtered_matches = []
        print(f"Matches with an adjusted energy shift (|ΔE|) between {energy_shift_lower_bound:.2f} and {energy_shift_upper_bound:.2f} eV:")
        for match in self.filtered_matches:
            # Calculate the raw difference for this match.
            current_diff = match["simple_mo_energy"] - match["complex_mo_energy"]
            # Adjust relative to the overall shift so that the first match gives a 0 difference.
            adjusted_energy_shift = current_diff - overall_shift
            if energy_shift_lower_bound <= abs(adjusted_energy_shift) <= energy_shift_upper_bound:
                energy_filtered_matches.append(match)
                print(
                    f"Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
                    f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
                    f"AO Overlap: {match['ao_overlap']:.4f}, "
                    f"Adjusted Energy Shift: {adjusted_energy_shift:.4f} eV"
                )
    
        # Create the plot with two subplots.
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        axes[0].tick_params(labelbottom=True)
    
        # --- Plot the simple system (displayed upside down) ---
        ax_simple = axes[0]
        for match in energy_filtered_matches:
            # Extract the orbital key (e.g., "1a") from the name.
            orbital_key = match['simple_mo'].rsplit('_', 1)[-1]
            # Retrieve the DOS data for that orbital; adjust the key if needed.
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                current_diff = match["simple_mo_energy"] - match["complex_mo_energy"]
                adjusted_energy_shift = current_diff + overall_shift
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.up]),  # flip the simple DOS curve
                    label=f"MO: {orbital_key} (ΔE: {adjusted_energy_shift:.2f} eV)",
                    alpha=0.7
                )
        # Mark the reference line: at 0 eV the simple system's lowest MO (first match) appears.
        ax_simple.axvline(0, color="red", linestyle="--", linewidth=1, label="Surface Bound NHC Fermi Energy")
    
        # Plot the adjusted Fermi energy of the simple system.
        ax_simple.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,
                          label=f"Lone NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
    
        ax_simple.set_title("Simple System PMODOS (Lowest MO Aligned)")
        ax_simple.set_ylabel("Density of States (Flipped)")
        ax_simple.grid(alpha=0.3)
        ax_simple.text(0.05, 0.9, f"Lone NHC Fermi Energy: {adjusted_simple_fermi:.2f} eV",
                       transform=ax_simple.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"))
        ax_simple.legend(fontsize=8)
    
        # --- Plot the complex system (no shift is applied) ---
        ax_complex = axes[1]
        for match in energy_filtered_matches:
            orbital_key = match['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                current_diff = match["simple_mo_energy"] - match["complex_mo_energy"]
                adjusted_energy_shift = current_diff + overall_shift
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} (ΔE: {adjusted_energy_shift:.2f} eV)",
                    alpha=0.7
                )
        # The complex system's reference remains at 0 eV.
        ax_complex.axvline(0, color="red", linestyle="--", linewidth=1, label="Surface Bound NHC Fermi Energy")
        ax_complex.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,
                          label=f"Lone NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
        ax_complex.set_title("Complex System PMODOS (Reference at 0 eV)")
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
    # Atom lists for the simple and complex systems.
    simple_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]
    complex_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]
    criteria = "simple:both, complex:both"
    
    # Set the lower and upper energy shift bounds in eV.
    energy_shift_lower_bound = 0.0  # Lower bound in eV.
    energy_shift_upper_bound = 15.0 # Upper bound in eV.
    
    # Initialize the IntegratedPlotter with detailed filtering criteria.
    integrated_plotter = IntegratedPlotter(
        simple_doscar_file='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/DOSCAR.LCFO.lobster',
        simple_lcfo_file='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/LCFO_Fragments.lobster',
        simple_mo_diagram='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/C13N2H18_1.MO_Diagram.lobster',
        complex_doscar_file='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/DOSCAR.LCFO.lobster',
        complex_lcfo_file='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/LCFO_Fragments.lobster',
        complex_mo_diagram='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/C13N2H18_1.MO_Diagram.lobster',
        matches_output_path='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/matches.txt',
        simple_atoms=simple_atoms,
        complex_atoms=complex_atoms,
        criteria=criteria  # Specify filtering criteria for both systems.
    )
    
    # Plot PMODOS curves for matches with an adjusted energy shift within the specified bounds.
    integrated_plotter.plot_aggregated_pmodos(criteria=criteria, 
                                                energy_shift_lower_bound=energy_shift_lower_bound,
                                                energy_shift_upper_bound=energy_shift_upper_bound,
                                                save_path='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/pmodos_plot.png')