from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.electronic_structure.plotter import CohpPlotter
from lib import Cohpcar
import matplotlib.pyplot as plt

class CohpPlotterClass:
    def __init__(self, file_path, energy_range, subplot_adjustments, specific_curves=None):
        self.file_path = file_path
        self.energy_range = energy_range
        self.subplot_adjustments = subplot_adjustments
        self.specific_curves = specific_curves
        self.cohr = None
        self.cdata_processed = {}

    def load_cohpcar(self):
        cohpcar = Cohpcar(filename=self.file_path)
        self.cohr = cohpcar
        self.process_cohp_data()

    def process_cohp_data(self):
        cdata = self.cohr.cohp_data
        for key, value in cdata.items():
            if self.specific_curves is None or key in self.specific_curves:
                c = value
                c["efermi"] = 0
                c["energies"] = self.cohr.energies
                c["are_coops"] = False
                self.cdata_processed[key] = Cohp.from_dict(c)

    def plot_cohp(self):
        cp = CohpPlotter()
        cp.add_cohp_dict(self.cdata_processed)
        ax = cp.get_plot()  # Only get the Axes object
        ax.set_ylim(self.energy_range)
        ax.set_xlabel("-COHP")
        ax.set_ylabel("Energy (eV)")

        plt.gcf().set_size_inches(10, 6)  # Set figure size using plt.gcf()
        plt.subplots_adjust(**self.subplot_adjustments)
        plt.show()

# Usage
file_path = 'C:/Users/Benjamin Kafin/Documents/VASP/NHC_Cu/freeCu1/freeCu2/kpoints551/COHPCAR.lobster'
energy_range = [-24, 6]
subplot_adjustments = {'bottom': 0.1, 'top': 0.9, 'left':0.1, 'right':0.9}
specific_curves = None  # Replace with actual keys or set to None to plot all curves

cohp_plotter = CohpPlotterClass(file_path, energy_range, subplot_adjustments, specific_curves)
cohp_plotter.load_cohpcar()
cohp_plotter.plot_cohp()
