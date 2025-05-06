# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:39:08 2025

@author: Benjamin Kafin
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

    def _has_opposite_sign_nonzero_ref(self, energies, dos, ref_energy, threshold=1e-8):
        """
        Determines whether the DOS curve has any nonzero points at energies with
        the opposite sign relative to the given reference energy (MO energy from the MO diagram).
        """
        if abs(ref_energy) < 1e-12:
            return False
        required_sign = -np.sign(ref_energy)
        for energy, dos_val in zip(energies, dos):
            if np.sign(energy) == required_sign and abs(dos_val) > threshold:
                return True
        return False

    def filter_mo_energy_opposite_matches(self, match_list, spin, energies_complex, threshold=1e-8):
        """
        Filters the provided match_list for the specified spin channel by checking whether,
        in the complex system's PMODOS curve, there is any nonzero DOS at energies with
        an opposite sign relative to the MO energy from the MO diagram.
        """
        filtered = []
        for match in match_list:
            ref_energy = match['complex_mo_energy']
            orbital_key_complex = match['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key_complex, None)
            if complex_dos_data is None:
                continue
            complex_curve = np.array(complex_dos_data.get(spin, []))
            if self._has_opposite_sign_nonzero_ref(energies_complex, complex_curve, ref_energy, threshold):
                filtered.append(match)
        return filtered

    def sum_pmodos_curves(self, matches, energy_filter="all", sum_spins=False):
        """
        Sums the entire PMODOS curves for the provided matches. (Used for plotting.)
        """
        energy_grid = self.complex_dos.energies
        
        if energy_filter == "both":
            positive_sum = np.zeros_like(energy_grid)
            negative_sum = np.zeros_like(energy_grid)
            for match in matches:
                mo_energy = match['complex_mo_energy']
                orbital_key = match['complex_mo'].rsplit('_', 1)[-1]
                dos_data = self.complex_pmodos.get(orbital_key, None)
                if dos_data is None:
                    continue
                if sum_spins:
                    # Simply sum the channels (without normalization) for plotting.
                    if Spin.up in dos_data and Spin.down in dos_data:
                        curve = np.array(dos_data[Spin.up]) + np.array(dos_data[Spin.down])
                    elif Spin.up in dos_data:
                        curve = np.array(dos_data[Spin.up])
                    else:
                        continue
                else:
                    curve = np.array(dos_data.get(Spin.up, []))
    
                if mo_energy > 0:
                    positive_sum += curve
                elif mo_energy < 0:
                    negative_sum += curve
            return energy_grid, positive_sum, negative_sum
        else:
            total_sum = np.zeros_like(energy_grid)
            for match in matches:
                if energy_filter == "positive" and match['complex_mo_energy'] < 0:
                    continue
                elif energy_filter == "negative" and match['complex_mo_energy'] > 0:
                    continue
    
                orbital_key = match['complex_mo'].rsplit('_', 1)[-1]
                dos_data = self.complex_pmodos.get(orbital_key, None)
                if dos_data is None:
                    continue
                if sum_spins:
                    if Spin.up in dos_data and Spin.down in dos_data:
                        curve = np.array(dos_data[Spin.up]) + np.array(dos_data[Spin.down])
                    elif Spin.up in dos_data:
                        curve = np.array(dos_data[Spin.up])
                    else:
                        continue
                else:
                    curve = np.array(dos_data.get(Spin.up, []))
                total_sum += curve
            return energy_grid, total_sum

    def integrate_delocalized_crossing_curves(self, final_spin1, final_spin2, sum_spins=False):
        """
        For every plotted PMODOS curve (both complex and simple), this function integrates
        only the delocalized part – that is, only the density on the opposite side of the Fermi level
        relative to the MO energy. In addition, each individual curve is normalized (per spin channel)
        so that its area is 1; the raw integrated area is recorded as the normalization factor.
        
        For a MO with positive energy (above E_F), only the DOS for energies E < 0 is integrated.
        For a MO with negative energy, only the DOS for energies E > 0 is integrated.
        
        In the spin-summed case the normalization is performed individually for each channel (normalize
        before summing) and then the normalized curves are summed and averaged. In the unsummed case,
        the spin-up and spin-down results are kept separate.
        """
        # Get the energy grids for the complex and simple systems.
        complex_energies = self.complex_dos.energies
        simple_energies = self.simple_dos.energies + self.alignment_shift

        # Normalization helper:
        def normalize_curve(curve, energies):
            total = np.trapz(curve, energies)
            if abs(total) < 1e-12:
                return curve, 1.0
            return curve / total, total

        if sum_spins:
            # Process each match individually for the complex system.
            sum_curve_complex_pos = np.zeros_like(complex_energies)
            sum_curve_complex_neg = np.zeros_like(complex_energies)
            norm_factors_complex_pos = []
            norm_factors_complex_neg = []
            for m in final_spin1:
                orbital_key_complex = m['complex_mo'].rsplit('_', 1)[-1]
                dos_data_complex = self.complex_pmodos.get(orbital_key_complex, None)
                if dos_data_complex is None:
                    continue
                norm_curve_total = np.zeros_like(complex_energies)
                norm_factor_list = []
                if Spin.up in dos_data_complex:
                    norm_curve_up, norm_factor_up = normalize_curve(np.array(dos_data_complex[Spin.up]), complex_energies)
                    norm_curve_total += norm_curve_up
                    norm_factor_list.append(norm_factor_up)
                if Spin.down in dos_data_complex:
                    norm_curve_down, norm_factor_down = normalize_curve(np.array(dos_data_complex[Spin.down]), complex_energies)
                    norm_curve_total += norm_curve_down
                    norm_factor_list.append(norm_factor_down)
                if norm_factor_list:
                    avg_norm_factor = np.mean(norm_factor_list)
                else:
                    avg_norm_factor = 0.0
                if m['complex_mo_energy'] > 0:
                    sum_curve_complex_pos += norm_curve_total
                    norm_factors_complex_pos.append(avg_norm_factor)
                elif m['complex_mo_energy'] < 0:
                    sum_curve_complex_neg += norm_curve_total
                    norm_factors_complex_neg.append(avg_norm_factor)
            # Integration:
            complex_integ_pos = np.trapz(sum_curve_complex_pos[complex_energies < 0],
                                          complex_energies[complex_energies < 0])
            complex_integ_neg = np.trapz(sum_curve_complex_neg[complex_energies > 0],
                                          complex_energies[complex_energies > 0])
            avg_norm_factor_pos = np.mean(norm_factors_complex_pos) if norm_factors_complex_pos else 0.0
            avg_norm_factor_neg = np.mean(norm_factors_complex_neg) if norm_factors_complex_neg else 0.0

            print("\nIntegrated Delocalized PMODOS (sum_spins=True):")
            print("Complex System:")
            print(f"  MOs positive delocalized area: {complex_integ_pos:.3f}")
            print(f"  MOs negative delocalized area: {complex_integ_neg:.3f}")
            print(f"  Avg Norm Factor (Positive MOs): {avg_norm_factor_pos:.3f}")
            print(f"  Avg Norm Factor (Negative MOs): {avg_norm_factor_neg:.3f}")

            return {"complex_positive": complex_integ_pos,
                    "complex_negative": complex_integ_neg,
                    "norm_complex_positive": avg_norm_factor_pos,
                    "norm_complex_negative": avg_norm_factor_neg}
        else:
            # Process the spin-up channel.
            sum_curve_complex_pos_up = np.zeros_like(complex_energies)
            sum_curve_complex_neg_up = np.zeros_like(complex_energies)
            norm_factors_complex_pos_up = []
            norm_factors_complex_neg_up = []
            for m in final_spin1:
                orbital_key_complex = m['complex_mo'].rsplit('_', 1)[-1]
                dos_data_complex = self.complex_pmodos.get(orbital_key_complex, None)
                if dos_data_complex is None or Spin.up not in dos_data_complex:
                    continue
                norm_curve, norm_factor = normalize_curve(np.array(dos_data_complex[Spin.up]), complex_energies)
                if m['complex_mo_energy'] > 0:
                    sum_curve_complex_pos_up += norm_curve
                    norm_factors_complex_pos_up.append(norm_factor)
                elif m['complex_mo_energy'] < 0:
                    sum_curve_complex_neg_up += norm_curve
                    norm_factors_complex_neg_up.append(norm_factor)

            # Process the spin-down channel.
            sum_curve_complex_pos_down = np.zeros_like(complex_energies)
            sum_curve_complex_neg_down = np.zeros_like(complex_energies)
            norm_factors_complex_pos_down = []
            norm_factors_complex_neg_down = []
            for m in final_spin2:
                orbital_key_complex = m['complex_mo'].rsplit('_', 1)[-1]
                dos_data_complex = self.complex_pmodos.get(orbital_key_complex, None)
                if dos_data_complex is None or Spin.down not in dos_data_complex:
                    continue
                norm_curve, norm_factor = normalize_curve(np.array(dos_data_complex[Spin.down]), complex_energies)
                if m['complex_mo_energy'] > 0:
                    sum_curve_complex_pos_down += norm_curve
                    norm_factors_complex_pos_down.append(norm_factor)
                elif m['complex_mo_energy'] < 0:
                    sum_curve_complex_neg_down += norm_curve
                    norm_factors_complex_neg_down.append(norm_factor)

            # Now integrate each channel separately.
            complex_integ_pos_up = np.trapz(sum_curve_complex_pos_up[complex_energies < 0],
                                             complex_energies[complex_energies < 0])
            complex_integ_neg_up = np.trapz(sum_curve_complex_neg_up[complex_energies > 0],
                                             complex_energies[complex_energies > 0])
            complex_integ_pos_down = np.trapz(sum_curve_complex_pos_down[complex_energies < 0],
                                               complex_energies[complex_energies < 0])
            complex_integ_neg_down = np.trapz(sum_curve_complex_neg_down[complex_energies > 0],
                                               complex_energies[complex_energies > 0])
            avg_norm_factor_pos_up = np.mean(norm_factors_complex_pos_up) if norm_factors_complex_pos_up else 0.0
            avg_norm_factor_neg_up = np.mean(norm_factors_complex_neg_up) if norm_factors_complex_neg_up else 0.0
            avg_norm_factor_pos_down = np.mean(norm_factors_complex_pos_down) if norm_factors_complex_pos_down else 0.0
            avg_norm_factor_neg_down = np.mean(norm_factors_complex_neg_down) if norm_factors_complex_neg_down else 0.0

            print("\nIntegrated Delocalized PMODOS (sum_spins=False):")
            print("Spin-Up:")
            print(f"  Positive delocalized area: {complex_integ_pos_up:.3f}")
            print(f"  Negative delocalized area: {complex_integ_neg_up:.3f}")
            print(f"  Avg Norm Factor (Positive): {avg_norm_factor_pos_up:.3f}")
            print(f"  Avg Norm Factor (Negative): {avg_norm_factor_neg_up:.3f}")
            print("Spin-Down:")
            print(f"  Positive delocalized area: {complex_integ_pos_down:.3f}")
            print(f"  Negative delocalized area: {complex_integ_neg_down:.3f}")
            print(f"  Avg Norm Factor (Positive): {avg_norm_factor_pos_down:.3f}")
            print(f"  Avg Norm Factor (Negative): {avg_norm_factor_neg_down:.3f}")
            
            return {"complex_positive_up": complex_integ_pos_up,
                    "complex_negative_up": complex_integ_neg_up,
                    "norm_complex_positive_up": avg_norm_factor_pos_up,
                    "norm_complex_negative_up": avg_norm_factor_neg_up,
                    "complex_positive_down": complex_integ_pos_down,
                    "complex_negative_down": complex_integ_neg_down,
                    "norm_complex_positive_down": avg_norm_factor_pos_down,
                    "norm_complex_negative_down": avg_norm_factor_neg_down}

    def plot_aggregated_pmodos(self, energy_shift_lower_bound, energy_shift_upper_bound,
                               show_occupation_changes=False,
                               user_defined_complex_mos=None,
                               plot_all_matches=True,
                               only_opposite_nonzero=False,
                               nonzero_threshold=1e-8,
                               save_path=None,
                               sum_energy_filter="all",  # Options: "positive", "negative", "all", "both".
                               sum_spins=False):         # Control whether to sum spins.
        """
        Plots the PMODOS curves for the two spin match lists. After plotting the curves,
        it calls the integration function to compute the integrated delocalized area and then
        prints the integration results (separately for spin-up and spin-down if sum_spins is False)
        in a text box.
        """
        simple_energies = self.simple_dos.energies
        complex_energies = self.complex_dos.energies
        adjusted_simple_energies = simple_energies + self.alignment_shift

        # --- 1. Filter ALL matches (by energy shift bounds) for each spin.
        all_matches_spin1 = [m for m in self.matches_spin_up
                             if energy_shift_lower_bound <= abs(m["energy_shift"]) <= energy_shift_upper_bound]
        all_matches_spin2 = [m for m in self.matches_spin_down
                             if energy_shift_lower_bound <= abs(m["energy_shift"]) <= energy_shift_upper_bound]

        print("All matches (within energy shift bounds):")
        for m in all_matches_spin1:
            print(f"Spin Up: Complex MO: {m['complex_mo']} ({m['complex_mo_energy']} eV), "
                  f"Simple MO: {m['simple_mo']} ({m['simple_mo_energy']} eV), "
                  f"AO Overlap: {m['ao_overlap']:.4f}, Energy Shift: {m['energy_shift']:.2f} eV")
        for m in all_matches_spin2:
            print(f"Spin Down: Complex MO: {m['complex_mo']} ({m['complex_mo_energy']} eV), "
                  f"Simple MO: {m['simple_mo']} ({m['simple_mo_energy']} eV), "
                  f"AO Overlap: {m['ao_overlap']:.4f}, Energy Shift: {m['energy_shift']:.2f} eV")

        # --- 2. Get user-specified matches (ignoring energy bounds).
        if user_defined_complex_mos is not None:
            user_specified_spin1 = [m for m in self.matches_spin_up if m["complex_mo"] in user_defined_complex_mos]
            user_specified_spin2 = [m for m in self.matches_spin_down if m["complex_mo"] in user_defined_complex_mos]
        else:
            user_specified_spin1 = []
            user_specified_spin2 = []

        print("\nUser-specified matches (ignoring energy bounds):")
        for m in user_specified_spin1:
            print(f"Spin Up: Complex MO: {m['complex_mo']} ({m['complex_mo_energy']} eV), "
                  f"Simple MO: {m['simple_mo']} ({m['simple_mo_energy']} eV), "
                  f"AO Overlap: {m['ao_overlap']:.4f}, Energy Shift: {m['energy_shift']:.2f} eV")
        for m in user_specified_spin2:
            print(f"Spin Down: Complex MO: {m['complex_mo']} ({m['complex_mo_energy']} eV), "
                  f"Simple MO: {m['simple_mo']} ({m['simple_mo_energy']} eV), "
                  f"AO Overlap: {m['ao_overlap']:.4f}, Energy Shift: {m['energy_shift']:.2f} eV")

        # --- 3. Optionally filter the "all matches" by the opposite-sign nonzero criteria.
        if only_opposite_nonzero:
            filtered_all_spin1 = self.filter_mo_energy_opposite_matches(all_matches_spin1, Spin.up, complex_energies, nonzero_threshold)
            filtered_all_spin2 = self.filter_mo_energy_opposite_matches(all_matches_spin2, Spin.down, complex_energies, nonzero_threshold)
        else:
            filtered_all_spin1 = all_matches_spin1
            filtered_all_spin2 = all_matches_spin2

        # --- 4. Form the final union.
        def union_matches(all_list, user_list):
            union = all_list.copy()
            for m in user_list:
                if m not in union:
                    union.append(m)
            return union

        if plot_all_matches:
            final_spin1 = union_matches(filtered_all_spin1, user_specified_spin1)
            final_spin2 = union_matches(filtered_all_spin2, user_specified_spin2)
        else:
            final_spin1 = user_specified_spin1
            final_spin2 = user_specified_spin2

        # --- 5. Build color mappings.
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_cycle1 = itertools.cycle(colors)
        match_colors_spin1 = {}
        for m in final_spin1:
            key = m['complex_mo']
            if key not in match_colors_spin1:
                match_colors_spin1[key] = next(color_cycle1)
        color_cycle2 = itertools.cycle(colors)
        match_colors_spin2 = {}
        for m in final_spin2:
            key = m['complex_mo']
            if key not in match_colors_spin2:
                match_colors_spin2[key] = next(color_cycle2)

        # --- 6. Create subplots.
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
        axes[0].tick_params(labelbottom=True)

        # Plot for the simple system.
        ax_simple = axes[0]
        for m in final_spin1:
            orbital_key = m['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors_spin1.get(m['complex_mo'], 'black')
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data.get(Spin.up, [])),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid'
                )
        for m in final_spin2:
            orbital_key = m['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors_spin2.get(m['complex_mo'], 'black')
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data.get(Spin.down, [])),
                    label=f"MO: {orbital_key} Spin Down (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='dotted'
                )
        ax_simple.axvline(0, color="red", linestyle="--", linewidth=1,
                          label="Surface Bound NHC Fermi Energy")
        ax_simple.set_title("Lone NHC PMODOS (MO1 Aligned) - Simple System")
        ax_simple.set_ylabel("Density of States (Flipped)")
        ax_simple.grid(alpha=0.3)
    
        # Plot for the complex system.
        ax_complex = axes[1]
        for m in final_spin1:
            orbital_key = m['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors_spin1.get(m['complex_mo'], 'black')
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data.get(Spin.up, [])),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid'
                )
        for m in final_spin2:
            orbital_key = m['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors_spin2.get(m['complex_mo'], 'black')
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data.get(Spin.down, [])),
                    label=f"MO: {orbital_key} Spin Down (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='dotted'
                )
        ax_complex.axvline(0, color="red", linestyle="--", linewidth=1,
                           label="Surface Bound NHC Fermi Energy")
        ax_complex.set_title("Surface Bound NHC PMODOS (Reference at 0 eV) - Complex System")
        ax_complex.set_xlabel("Energy (eV)")
        ax_complex.set_ylabel("Density of States")
        ax_complex.grid(alpha=0.3)
    
        if sum_energy_filter == "both":
            if sum_spins:
                energy_all, pos_sum, neg_sum = self.sum_pmodos_curves(final_spin1,
                                                                      energy_filter="both",
                                                                      sum_spins=True)
                ax_complex.plot(energy_all, pos_sum, label="Sum Positive MOs", 
                                color="blue", linestyle='solid', linewidth=1)
                ax_complex.plot(energy_all, neg_sum, label="Sum Negative MOs", 
                                color="red", linestyle='solid', linewidth=1)
                
                idx_fermi = np.argmin(np.abs(energy_all))
                integrated_neg = np.trapz(neg_sum[idx_fermi:], energy_all[idx_fermi:])
                integrated_pos = np.trapz(pos_sum[:idx_fermi+1], energy_all[:idx_fermi+1])
                integration_text = (f"Combined (Spins Summed):\n"
                                    f"Integrated Occupied Density: {integrated_pos:.3f}\n"
                                    f"Integrated Unoccupied Density: {integrated_neg:.3f}\n"
                                    f"Total Electron Transfer: {integrated_neg-integrated_pos:.3f}")
            else:
                # In unsummed mode, retrieve separate integration per spin.
                integration_results = self.integrate_delocalized_crossing_curves(final_spin1, final_spin2, sum_spins=False)
                integration_text = (
                    "Spin-Up:\n"
                    f"  Positive delocalized area: {integration_results['complex_positive_up']:.3f}\n"
                    f"  Negative delocalized area: {integration_results['complex_negative_up']:.3f}\n"
                    f"  Norm Factor (Pos): {integration_results['norm_complex_positive_up']:.3f}\n"
                    f"  Norm Factor (Neg): {integration_results['norm_complex_negative_up']:.3f}\n\n"
                    "Spin-Down:\n"
                    f"  Positive delocalized area: {integration_results['complex_positive_down']:.3f}\n"
                    f"  Negative delocalized area: {integration_results['complex_negative_down']:.3f}\n"
                    f"  Norm Factor (Pos): {integration_results['norm_complex_positive_down']:.3f}\n"
                    f"  Norm Factor (Neg): {integration_results['norm_complex_negative_down']:.3f}"
                )
                ax_complex.text(0.05, 0.95, integration_text, transform=ax_complex.transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            if sum_spins:
                energy_all, total_sum_all_spins = self.sum_pmodos_curves(final_spin1,
                                                                         energy_filter=sum_energy_filter,
                                                                         sum_spins=True)
                ax_complex.plot(energy_all, total_sum_all_spins, label="Sum All Spins", 
                                color="magenta", linestyle='solid', linewidth=1)
            else:
                energy_up, total_sum_up = self.sum_pmodos_curves(final_spin1,
                                                                 energy_filter=sum_energy_filter,
                                                                 sum_spins=False)
                energy_down, total_sum_down = self.sum_pmodos_curves(final_spin2,
                                                                     energy_filter=sum_energy_filter,
                                                                     sum_spins=False)
                ax_complex.plot(energy_up, total_sum_up, label="Spin Up", 
                                color="black", linestyle='solid', linewidth=1)
                ax_complex.plot(energy_down, total_sum_down, label="Spin Down", 
                                color="black", linestyle='dotted', linewidth=1)
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        # --- Finally, call the integration routine (if not already called above).
        if sum_spins:
            integration_results = self.integrate_delocalized_crossing_curves(final_spin1, final_spin2, sum_spins=True)
            info_text = (
                f"Complex (Spin-Summed):\n"
                f"  Positive delocalized area: {integration_results['complex_positive']:.3f}\n"
                f"  Negative delocalized area: {integration_results['complex_negative']:.3f}\n"
                f"  Norm Factor (Pos): {integration_results['norm_complex_positive']:.3f}\n"
                f"  Norm Factor (Neg): {integration_results['norm_complex_negative']:.3f}"
            )
        else:
            integration_results = self.integrate_delocalized_crossing_curves(final_spin1, final_spin2, sum_spins=False)
            info_text = (
                f"Complex Spin-Up:\n"
                f"  Positive: {integration_results['complex_positive_up']:.3f}\n"
                f"  Negative: {integration_results['complex_negative_up']:.3f}\n"
                f"Complex Spin-Down:\n"
                f"  Positive: {integration_results['complex_positive_down']:.3f}\n"
                f"  Negative: {integration_results['complex_negative_down']:.3f}"
            )
        # Add an additional text box at the bottom of the complex plot.
        props2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax_complex.text(0.05, 0.05, info_text, transform=ax_complex.transAxes, fontsize=10,
                        verticalalignment='bottom', bbox=props2)

if __name__ == "__main__":
    simple_doscar_file = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/DOSCAR.LCFO.lobster'
    simple_lcfo_file   = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/LCFO_Fragments.lobster'
    simple_mo_diagram  = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/NHC/C13N2H18_1.MO_Diagram.lobster'

    complex_doscar_file = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/dipole_correction/efield/kpoints551/DOSCAR.LCFO.lobster'
    complex_lcfo_file   = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/dipole_correction/efield/kpoints551/LCFO_Fragments.lobster'
    complex_mo_diagram  = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/dipole_correction/efield/kpoints551/C13N2H18_1.MO_Diagram_adjusted.lobster'

    matches_output_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/dipole_correction/efield/kpoints551/matches_important.txt'

    energy_shift_lower_bound = 0
    energy_shift_upper_bound = 10.0

    user_defined_complex_mos = []  # e.g. ['C13N2H18_1_37a', 'C13N2H18_1_38a', ...]

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
        only_opposite_nonzero=True,
        nonzero_threshold=5e-2,
        save_path='C:/Users/Benjamin Kafin/Documents/VASP/NHC/IPR/lone/NHC/NHC_iPr/4layers/freegold1/freegold2/kpoints551/dipole_correction/efield/kpoints551/pmodos_plot.png',
        sum_energy_filter="all",
        sum_spins=False  # Set to False to see separate integration for spin-up and spin-down.
    )