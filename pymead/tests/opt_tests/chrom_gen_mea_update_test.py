import random

import numpy as np
import matplotlib.pyplot as plt

from pymead.core.mea import MEA
from pymead.optimization.pop_chrom import Chromosome
from pymead.optimization.sampling import ConstrictedRandomSampling
from pymead.utils.read_write_files import load_data


def run():
    x0 = np.loadtxt("test_export3.dat")
    jmea_dict = load_data("kink_initial_inboard_1.jmea")
    param_dict = load_data("param_dict.json")
    mea = MEA.generate_from_param_dict(jmea_dict)
    mea.remove_airfoil_graphs()

    np.random.seed(param_dict['seed'])
    random.seed(param_dict['seed'])
    parameter_list, _ = mea.extract_parameters()
    sampling = ConstrictedRandomSampling(n_samples=100, norm_param_list=parameter_list,
                                         max_sampling_width=0.05)
    X_list = sampling.sample()

    parents = [Chromosome(param_dict=param_dict, generation=0, population_idx=idx, mea=jmea_dict, genes=individual)
               for idx, individual in enumerate(X_list)]

    x1 = X_list[1]

    # Make plot
    fig, ax = plt.subplots()

    mea.update_parameters(x1)
    for a in mea.airfoils.values():
        a.plot_airfoil(ax, color="cornflowerblue", ls=":")
    print(f"{mea.airfoils['A0'].free_points['te_1']['FP1'].xy.value = }")
    thickness_data = mea.airfoils["A0"].compute_thickness(return_max_thickness_loc=True)
    tc_max = thickness_data["t/c_max"]
    print(f"{tc_max = }")

    c = Chromosome(param_dict, 0, 0, jmea_dict, x1)
    c.generate(deactivate_airfoil_graphs=True)
    for a in c.coords:
        a_arr = np.array(a)
        ax.plot(a_arr[:, 0], a_arr[:, 1], color="gold", ls="--")

    plt.show()
    pass


if __name__ == "__main__":
    run()
