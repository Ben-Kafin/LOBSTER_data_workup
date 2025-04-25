# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:05:16 2025

@author: nazin_lab
"""
from matplotlib import pyplot as plt
import numpy as np
from pymatgen.electronic_structure.core import Spin
from MOPM_Surfaces_vs_molecule_spinpol_mo1 import MOPM  # assumes new MOPM class with two match lists
from lib_DOS_lcfo import DOSCAR_LCFO  # Import the DOSCAR_LCFO class
import itertools


class IntegratedPlotter:
    def __init__(self, simple_doscar_file, simple_lcfo_file, simple_mo_diagram,
                 complex_doscar_file, complex_lcfo_file, complex_mo_diagram,
                 matches_output_path):
        """
        Initialize the IntegratedPlotter with file paths for DOS, LCFO, and MO diagrams,
        as well as a path where the match output will be written.
        """
        # Initialize the MOPM instance and align energies.
        self.mopm = MOPM(simple_mo_diagram, complex_mo_diagram, align_energy=True)
        self.alignment_shift = self.mopm.alignment_shift

        # Load the DOSCAR/LCFO instances.
        self.simple_dos = DOSCAR_LCFO(simple_doscar_file, simple_lcfo_file)
        self.complex_dos = DOSCAR_LCFO(complex_doscar_file, complex_lcfo_file)

        # Get the two match lists directly.
        # (Assumes compare_mo_contributions returns (matches_spin_up, matches_spin_down))
        self.matches_spin_up, self.matches_spin_down = self.mopm.compare_mo_contributions(
            matches_output_path, energy_shift_threshold=0.0
        )

        # Get PMODOS data (from the complex system) from the first available key.
        self.complex_pmodos = next(iter(self.complex_dos.pmodos.values()), {})


    def plot_aggregated_pmodos(self, energy_shift_lower_bound, energy_shift_upper_bound,
                                show_occupation_changes=False,
                                user_defined_complex_mos=None,
                                plot_all_matches=True,
                                save_path=None):
        """
        Plots the PMODOS curves for the two spin match lists so that:
          - It always plots the matches specified in user_defined_complex_mos (regardless of energy bounds).
          - Optionally, it also plots all matches that pass the energy shift threshold if plot_all_matches is True.
        
        The final curves are the union of (all matches within the bounds) and (user-specified matches).
        For spin-up, the DOS is taken from the Spin.up channel and drawn as a solid line.
        For spin-down, the DOS is taken from the Spin.down channel and drawn as a dotted line.
        
        The lines are drawn with the same style (line type, width, and color) across the board.
        The simple system's energy axis is shifted by the alignment shift.
        """
        # Retrieve energy grids and adjust energies.
        simple_energies = self.simple_dos.energies
        complex_energies = self.complex_dos.energies
        adjusted_simple_energies = simple_energies + self.alignment_shift
        adjusted_simple_fermi = (self.simple_dos.fermi_energy - self.complex_dos.fermi_energy) - self.alignment_shift
    
        # --- 1. Filter ALL matches (by energy shift threshold) for each spin.
        all_matches_spin1 = []
        for match in self.matches_spin_up:
            eshift = match["energy_shift"]
            if energy_shift_lower_bound <= abs(eshift) <= energy_shift_upper_bound:
                all_matches_spin1.append(match)
        all_matches_spin2 = []
        for match in self.matches_spin_down:
            eshift = match["energy_shift"]
            if energy_shift_lower_bound <= abs(eshift) <= energy_shift_upper_bound:
                all_matches_spin2.append(match)
    
        print("All matches (within energy shift bounds):")
        for match in all_matches_spin1:
            print(f"Spin Up: Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
                  f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
                  f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.2f} eV")
        for match in all_matches_spin2:
            print(f"Spin Down: Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
                  f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
                  f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.2f} eV")
    
        # --- 2. Get user-defined matches (ignoring energy bounds).
        if user_defined_complex_mos is not None:
            user_specified_spin1 = [m for m in self.matches_spin_up if m["complex_mo"] in user_defined_complex_mos]
            user_specified_spin2 = [m for m in self.matches_spin_down if m["complex_mo"] in user_defined_complex_mos]
        else:
            user_specified_spin1 = []
            user_specified_spin2 = []
    
        print("\nUser-specified matches (ignoring energy bounds):")
        for match in user_specified_spin1:
            print(f"Spin Up: Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
                  f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
                  f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.2f} eV")
        for match in user_specified_spin2:
            print(f"Spin Down: Complex MO: {match['complex_mo']} ({match['complex_mo_energy']} eV), "
                  f"Simple MO: {match['simple_mo']} ({match['simple_mo_energy']} eV), "
                  f"AO Overlap: {match['ao_overlap']:.4f}, Energy Shift: {match['energy_shift']:.2f} eV")
    
        # --- 3. Form the final union of matches.
        # If plot_all_matches is True, use union of all matches and user-specified.
        # Otherwise, use only the user-specified matches.
        def union_matches(all_matches, user_specified):
            final = all_matches.copy()
            for m in user_specified:
                if m not in final:
                    final.append(m)
            return final
    
        if plot_all_matches:
            final_spin1 = union_matches(all_matches_spin1, user_specified_spin1)
            final_spin2 = union_matches(all_matches_spin2, user_specified_spin2)
        else:
            final_spin1 = user_specified_spin1
            final_spin2 = user_specified_spin2
    
        # --- 4. Build color mappings for each spin from the final sets.
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle1 = itertools.cycle(colors)
        match_colors_spin1 = {}
        for match in final_spin1:
            key = match['complex_mo']
            if key not in match_colors_spin1:
                match_colors_spin1[key] = next(color_cycle1)
    
        color_cycle2 = itertools.cycle(colors)
        match_colors_spin2 = {}
        for match in final_spin2:
            key = match['complex_mo']
            if key not in match_colors_spin2:
                match_colors_spin2[key] = next(color_cycle2)
    
        # --- 5. Create subplots.
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        axes[0].tick_params(labelbottom=True)
    
        # --- Plot for the simple system.
        ax_simple = axes[0]
        # For spin-up: Use DOS data from Spin.up (solid line)
        for match in final_spin1:
            orbital_key = match['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors_spin1.get(match['complex_mo'], 'black')
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {match['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid'
                )
        # For spin-down: Use DOS data from Spin.down (dotted line)
        for match in final_spin2:
            orbital_key = match['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors_spin2.get(match['complex_mo'], 'black')
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.down]),
                    label=f"MO: {orbital_key} Spin Down (ΔE: {match['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='dotted'
                )
        ax_simple.axvline(0, color="red", linestyle="--", linewidth=1,
                          label="Surface Bound NHC Fermi Energy")
        #ax_simple.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,label=f"Adj. NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
        ax_simple.set_title("Lone NHC PMODOS (MO1 Aligned) - Simple System")
        ax_simple.set_ylabel("Density of States (Flipped)")
        ax_simple.grid(alpha=0.3)
        ax_simple.legend(fontsize=8)
    
        # --- Plot for the complex system.
        ax_complex = axes[1]
        # For spin-up: Use DOS data from Spin.up (solid line)
        for match in final_spin1:
            orbital_key = match['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors_spin1.get(match['complex_mo'], 'black')
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {match['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid'
                )
        # For spin-down: Use DOS data from Spin.down (dotted line)
        for match in final_spin2:
            orbital_key = match['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors_spin2.get(match['complex_mo'], 'black')
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.down]),
                    label=f"MO: {orbital_key} Spin Down (ΔE: {match['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='dotted'
                )
        ax_complex.axvline(0, color="red", linestyle="--", linewidth=1,
                           label="Surface Bound NHC Fermi Energy")
        #ax_complex.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,label=f"Adj. NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
        ax_complex.set_title("Surface Bound NHC PMODOS (Reference at 0 eV) - Complex System")
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
    # Update these paths as needed.
    simple_doscar_file = 'C:/directory/DOSCAR.LCFO.lobster'
    simple_lcfo_file   = 'C:/directory/LCFO_Fragments.lobster'
    simple_mo_diagram  = 'C:/directory/MO_Diagram.lobster'
    
    complex_doscar_file = 'C:/directory2/DOSCAR.LCFO.lobster'
    complex_lcfo_file   = 'C:/directory2/LCFO_Fragments.lobster'
    complex_mo_diagram  = 'C:/directory2/MO_Diagram_adjusted.lobster'
    
    matches_output_path = 'C:/directory2/matches_important.txt'

    energy_shift_lower_bound = 0.75
    energy_shift_upper_bound = 1.0

    user_defined_complex_mos = ['37a','38a','39a','40a']

    integrated_plotter = IntegratedPlotter(
        simple_doscar_file,
        simple_lcfo_file,
        simple_mo_diagram,
        complex_doscar_file,
        complex_lcfo_file,
        complex_mo_diagram,
        matches_output_path
    )


    integrated_plotter.plot_aggregated_pmodos(
        energy_shift_lower_bound=energy_shift_lower_bound,
        energy_shift_upper_bound=energy_shift_upper_bound,
        show_occupation_changes=False,
        user_defined_complex_mos=user_defined_complex_mos,
        plot_all_matches=True,
        save_path='C:/directory2/pmodos_plot.png'
    )
