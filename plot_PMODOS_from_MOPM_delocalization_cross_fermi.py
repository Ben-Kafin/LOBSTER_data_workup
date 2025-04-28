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
        
        Parameters:
          energies: 1D numpy array of energies (for the complex system)
          dos: 1D numpy array of DOS values corresponding to these energies
          ref_energy: Reference energy (from the MO diagram)
          threshold: Minimum absolute DOS value to consider as nonzero
        
        Returns:
          True if at least one point in the DOS curve (with value > threshold) is found
          at an energy whose sign is opposite that of ref_energy.
          Otherwise, returns False.
        """
        # If the reference energy is essentially zero, no opposite-sign check is made.
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
        
        Parameters:
          match_list: List of match dictionaries.
          spin: Spin channel (Spin.up or Spin.down) to consider.
          energies_complex: Energy grid for the complex system.
          threshold: Minimum DOS value to consider as nonzero.
        
        Returns:
          A filtered list of matches that satisfy the opposite-sign condition.
        """
        filtered = []
        for match in match_list:
            ref_energy = match['complex_mo_energy']
            orbital_key_complex = match['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key_complex, None)
            if complex_dos_data is None:
                continue
            complex_curve = np.array(complex_dos_data[spin])
            if self._has_opposite_sign_nonzero_ref(energies_complex, complex_curve, ref_energy, threshold):
                filtered.append(match)
        return filtered

    def plot_aggregated_pmodos(self, energy_shift_lower_bound, energy_shift_upper_bound,
                                show_occupation_changes=False,
                                user_defined_complex_mos=None,
                                plot_all_matches=True,
                                only_opposite_nonzero=False,
                                nonzero_threshold=1e-8,
                                save_path=None):
        """
        Plots the PMODOS curves for the two spin match lists.
        
        The function does the following:
          1. Filters “all matches” (for each spin) using the energy shift bounds.
          2. Extracts user-specified matches (ignoring energy bounds).
          3. Optionally filters the “all matches” using the opposite-sign nonzero criteria.
          4. Forms the final union of matches as:
                 final = (filtered-all-matches) ∪ (user-specified matches)
             if plot_all_matches is True; otherwise final = user-specified matches.
          5. Plots the final union using the same styling (solid for spin-up,
             dotted for spin-down) and using consistent color mapping.
        
        The user-specified matches are always included – regardless of the opposite-sign filter.
        The simple system’s energy axis is shifted by alignment_shift.
        """
        # Retrieve energy grids and adjust energies.
        simple_energies = self.simple_dos.energies
        complex_energies = self.complex_dos.energies
        adjusted_simple_energies = simple_energies + self.alignment_shift
        adjusted_simple_fermi = (self.simple_dos.fermi_energy - self.complex_dos.fermi_energy) - self.alignment_shift
    
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
        # We'll make a union so that any match that is user-specified is included.
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
    
        # --- 5. Build color mappings for each spin from the final sets.
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
    
        # --- Plot for the simple system.
        ax_simple = axes[0]
        # For spin-up matches: Use the DOS data from Spin.up (solid line).
        for m in final_spin1:
            orbital_key = m['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors_spin1.get(m['complex_mo'], 'black')
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid'
                )
        # For spin-down matches: Use the DOS data from Spin.down (dotted line).
        for m in final_spin2:
            orbital_key = m['simple_mo'].rsplit('_', 1)[-1]
            simple_dos_data = self.simple_dos.pmodos.get('C13N2H18', {}).get(orbital_key, None)
            if simple_dos_data is not None:
                col = match_colors_spin2.get(m['complex_mo'], 'black')
                ax_simple.plot(
                    adjusted_simple_energies,
                    -np.array(simple_dos_data[Spin.down]),
                    label=f"MO: {orbital_key} Spin Down (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='dotted'
                )
        ax_simple.axvline(0, color="red", linestyle="--", linewidth=1,
                          label="Surface Bound NHC Fermi Energy")
        ax_simple.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,
                          label=f"Adj. NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
        ax_simple.set_title("Lone NHC PMODOS (MO1 Aligned) - Simple System")
        ax_simple.set_ylabel("Density of States (Flipped)")
        ax_simple.grid(alpha=0.3)
        ax_simple.legend(fontsize=8)
    
        # --- Plot for the complex system.
        ax_complex = axes[1]
        # For spin-up: Use the DOS data from Spin.up (solid line).
        for m in final_spin1:
            orbital_key = m['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors_spin1.get(m['complex_mo'], 'black')
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.up]),
                    label=f"MO: {orbital_key} Spin Up (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='solid'
                )
        # For spin-down: Use the DOS data from Spin.down (dotted line).
        for m in final_spin2:
            orbital_key = m['complex_mo'].rsplit('_', 1)[-1]
            complex_dos_data = self.complex_pmodos.get(orbital_key, None)
            if complex_dos_data is not None:
                col = match_colors_spin2.get(m['complex_mo'], 'black')
                ax_complex.plot(
                    complex_energies,
                    np.array(complex_dos_data[Spin.down]),
                    label=f"MO: {orbital_key} Spin Down (ΔE: {m['energy_shift']:.2f} eV)",
                    color=col,
                    linestyle='dotted'
                )
        ax_complex.axvline(0, color="red", linestyle="--", linewidth=1,
                           label="Surface Bound NHC Fermi Energy")
        ax_complex.axvline(adjusted_simple_fermi, color="blue", linestyle="--", linewidth=1,
                           label=f"Adj. NHC Fermi ({adjusted_simple_fermi:.2f} eV)")
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
    simple_doscar_file = 'C:/directory1/DOSCAR.LCFO.lobster'
    simple_lcfo_file   = 'C:/directory1/LCFO_Fragments.lobster'
    simple_mo_diagram  = 'C:/directory1/MO_Diagram.lobster'

    complex_doscar_file = 'C:/directory2/DOSCAR.LCFO.lobster'
    complex_lcfo_file   = 'C:/directory2/LCFO_Fragments.lobster'
    complex_mo_diagram  = 'C:/directory2/MO_Diagram_adjusted.lobster'

    matches_output_path = 'C:/directory2/matches_important.txt'


    energy_shift_lower_bound = 0
    energy_shift_upper_bound = 10.0


    user_defined_complex_mos = ['C13N2H18_1_37a', 'C13N2H18_1_38a', 'C13N2H18_1_39a', 'C13N2H18_1_40a']

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
        plot_all_matches=False,
        only_opposite_nonzero=True,  # Enable filtering using the MO energy as reference
        nonzero_threshold=5e-2,      
        save_path='C:/directory2/pmodos_plot.png'
    )
