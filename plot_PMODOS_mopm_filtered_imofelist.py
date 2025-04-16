# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:42:19 2025

@author: Benjamin Kafin
"""

from matplotlib import pyplot as plt
import numpy as np
from pymatgen.electronic_structure.core import Spin
from MOPM_IMOFElist import MOPM
from lib_DOSCAR_LCFO import DOSCAR_LCFO  # Import the DOSCAR_LCFO class


class IntegratedPlotter:
    def __init__(self, simple_doscar_file, simple_lcfo_file, simple_mo_diagram,
                 complex_doscar_file, complex_lcfo_file, complex_mo_diagram, matches_output_path, simple_atoms, complex_atoms):
        """
        Initialize the IntegratedPlotter with file paths and MOPM matching logic.

        Args:
            simple_doscar_file (str): DOSCAR.LCFO file path for the simple system.
            simple_lcfo_file (str): LCFO_Fragments.lobster file path for the simple system.
            simple_mo_diagram (str): MO diagram file path for the simple system.
            complex_doscar_file (str): DOSCAR.LCFO file path for the complex system.
            complex_lcfo_file (str): LCFO_Fragments.lobster file path for the complex system.
            complex_mo_diagram (str): MO diagram file path for the complex system.
            matches_output_path (str): Path to output the matching results.
            simple_atoms (list): List of tuples for simple atom types.
            complex_atoms (list): List of tuples for complex atom types.
        """
        # Update the MOPM initialization to include the iMOFE list file paths
        imofelist_simple_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/NHC/IMOFElist.lobster'
        imofelist_complex_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/adatom/NHC_iPr_adatom_fcc/kpoints551/Adatom/NHC_frag/IMOFElist.lobster'

        # Initialize the MOPM instance
        self.mopm = MOPM(simple_mo_diagram, complex_mo_diagram, imofelist_simple_path, imofelist_complex_path, simple_atoms, complex_atoms)

        # Initialize the DOSCAR_LCFO instances
        self.simple_dos = DOSCAR_LCFO(simple_doscar_file, simple_lcfo_file, simple_mo_diagram)
        self.complex_dos = DOSCAR_LCFO(complex_doscar_file, complex_lcfo_file, complex_mo_diagram)
        
        self.simple_atoms = simple_atoms
        self.complex_atoms = complex_atoms
        
        # Generate matches and write to output
        self.matches = self.mopm.compare_mo_contributions(matches_output_path,imofe_overlap_threshold=0.01) 

        # Filter complex PMODOS to use only the first instance
        self.complex_pmodos = next(iter(self.complex_dos.pmodos.values()), {})

    def plot_aggregated_pmodos(self, save_path=None):
        """
        Plots the PMODOS curves for all matches from the filtered MOPM_IMOFElist instance.
        With Fermi energy values dynamically placed as text boxes, legends aligned below them,
        and x-axis labels restored on the top graph. Each MO's energy shift is displayed in the legend.

        Args:
            save_path (str): Optional path to save the resulting plot.
        """
        # Fetch the energy and Fermi level data
        simple_energies = self.simple_dos.energies
        simple_fermi_energy = self.simple_dos.fermi_energy
    
        complex_energies = self.complex_dos.energies
        complex_fermi_energy = self.complex_dos.fermi_energy
    
        # Calculate the energy shift for the simple system
        simple_energy_shift = complex_fermi_energy - simple_fermi_energy
        adjusted_simple_energies = simple_energies - simple_energy_shift
        shifted_simple_fermi = -simple_energy_shift
    
        # Debug: Print all matches
        print("Matches from MOPM_IMOFElist:")
        for match in self.matches:
            adjusted_energy_shift = match['energy_shift'] + simple_energy_shift
            print(
                f"Simple MO: {match['simple_mo']}, Complex MO: {match['complex_mo']}, "
                f"AO Overlap: {match['ao_overlap']:.4f}, iMOFE AO Overlap: {match['imofe_ao_overlap']:.4f}, "
                f"Energy Shift: {adjusted_energy_shift:.4f} eV"
            )
    
        # Create the plot
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
        # Plot simple PMODOS curves (upside down)
        ax_simple = axes[0]
        for match in self.matches:
            orbital_key = match['simple_mo'].rsplit('_', 1)[-1]  # Extract key like '1a'
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                adjusted_energy_shift = match['energy_shift'] + simple_energy_shift
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.up]),  # Flip the curve upside down
                    label=f"Lone NHC MO: {orbital_key} (ΔE: {adjusted_energy_shift:.2f} eV)",
                    alpha=0.7
                )
        # Add the shifted Fermi level for the simple system
        ax_simple.axvline(shifted_simple_fermi, color="blue", linestyle="--", linewidth=1, label="Lone NHC Fermi Level")
        ax_simple.set_title("Lone NHC PMODOS (All Matches from MOPM_IMOFElist)")
        ax_simple.set_xlabel("Energy (eV)")  # Restored x-axis label for the top graph
        ax_simple.set_ylabel("Density of States (Flipped)")
        ax_simple.grid(alpha=0.3)
    
        ax_simple.text(0.05, 0.9, f"Lone NHC Fermi Energy: {simple_fermi_energy:.2f} eV",
                       transform=ax_simple.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"))
    
        ax_simple.legend(fontsize=8)
    
        # Plot complex PMODOS curves (normally)
        ax_complex = axes[1]
        for match in self.matches:
            orbital_key = match['complex_mo'].rsplit('_', 1)[-1]  # Extract key like '1a'
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                adjusted_energy_shift = match['energy_shift'] + simple_energy_shift
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.up]),  # Normal curve
                    label=f"Lone Adatom NHC Fragment MO: {orbital_key} (ΔE: {adjusted_energy_shift:.2f} eV)",
                    alpha=0.7
                )
        # Add the Fermi level for the complex system
        ax_complex.axvline(0, color="red", linestyle="--", linewidth=1, label="Lone Adatom NHC Fermi Level")
        ax_complex.set_title("Lone Adatom NHC Fragment PMODOS (All Matches from MOPM_IMOFElist)")
        ax_complex.set_xlabel("Energy (eV)")  # Maintains x-axis label for the bottom graph
        ax_complex.set_ylabel("Density of States")
        ax_complex.grid(alpha=0.3)
    
        ax_complex.text(0.05, 0.9, f"Lone Adatom NHC Fragment Fermi Energy: {complex_fermi_energy:.2f} eV",
                        transform=ax_complex.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"))
    
        ax_complex.legend(fontsize=8)
    
        # Adjust layout to prevent overlap
        plt.tight_layout()
    
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Atom lists for simple and complex systems
    simple_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]
    complex_atoms = [(2, 'N'), (13, 'C'), (18, 'H')]

    # Initialize the IntegratedPlotter with updated MOPM class initialization
    integrated_plotter = IntegratedPlotter(
        simple_doscar_file='filepath/DOSCAR.LCFO.lobster',
        simple_lcfo_file='filepath/LCFO_Fragments.lobster',
        simple_mo_diagram='filepath/MO_Diagram.lobster',
        complex_doscar_file='filepath/DOSCAR.LCFO.lobster',
        complex_lcfo_file='filepath/LCFO_Fragments.lobster',
        complex_mo_diagram='filepath/MO_Diagram.lobster',
        matches_output_path='filepath/matches.txt',
        simple_atoms=simple_atoms,
        complex_atoms=complex_atoms
    )

    # Plot PMODOS curves for matches
    integrated_plotter.plot_aggregated_pmodos(save_path='filepath/filtered_pmodos_plot.png')
