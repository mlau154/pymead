import numpy as np
import os
from matplotlib.pyplot import show
from matplotlib.lines import Line2D
from copy import deepcopy

from pymead.utils.airfoil_matching import match_airfoil
from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from pymead.core.free_point import FreePoint


def repair_example_sd7062(xy: np.ndarray):
    r"""
    ### Description:

    An example of how to repair a poorly-defined set of discrete airfoil coordinates.

    ### Args:

    `xy`: `np.ndarray` of `shape=(N, 2)` describing the set of "discrete" airfoil coordinates, where `N` is the number
    of coordinates, and the columns represent \(x\) and \(y\)

    ### Returns:

    The modified xy array
    """
    xy[-1, 0] = 1
    xy[-1, 1] = 0
    return xy


def run():
    r"""
    ### Description:

    An example of an implementation of `pymead.utils.airfoil_matching.match_airfoil()` using the sd7062-il airfoil.
    Note that the match is imperfect because the set of discrete airfoil coordinates does not have its leading edge
    at exactly \((0,0)\), and the trailing edge had to be repaired because of a self-intersection.
    """
    airfoil_to_match = 'sd7062-il'
    airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match, repair=repair_example_sd7062)
    base_airfoil_params = BaseAirfoilParams(c=Param(1.0, active=False),
                                            alf=Param(np.deg2rad(0.0), active=False),
                                            R_le=Param(0.06, 'length', bounds=np.array([1e-5, 0.3])),
                                            L_le=Param(0.08, 'length', bounds=np.array([1e-5, 0.2])),
                                            r_le=Param(0.6, bounds=np.array([0.01, 0.99])),
                                            phi_le=Param(np.deg2rad(5.0), bounds=np.array([-np.pi / 4, np.pi / 4])),
                                            psi1_le=Param(np.deg2rad(10.0),
                                                          bounds=np.array([-np.pi / 2.1, np.pi / 2.1])),
                                            psi2_le=Param(np.deg2rad(15.0),
                                                          bounds=np.array([-np.pi / 2.1, np.pi / 2.1])),
                                            L1_te=Param(0.25, 'length', bounds=np.array([1e-5, 1.0])),
                                            L2_te=Param(0.3, 'length', bounds=np.array([1e-5, 1.0])),
                                            theta1_te=Param(np.deg2rad(14.0),
                                                            bounds=np.array([-np.pi / 2.1, np.pi / 2.1])),
                                            theta2_te=Param(np.deg2rad(1.0),
                                                            bounds=np.array([-np.pi / 2.1, np.pi / 2.1])),
                                            t_te=Param(0.0, 'length', active=False),
                                            r_te=Param(0.5, active=False),
                                            phi_te=Param(np.deg2rad(0.0), active=False),
                                            non_dim_by_chord=True
                                            )

    free_point1 = FreePoint(x=Param(0.35, 'length', bounds=np.array([0.05, 0.95])),
                            y=Param(0.2, 'length', bounds=np.array([0.05, 0.95])), previous_anchor_point='te_1',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point2 = FreePoint(x=Param(0.5, 'length', bounds=np.array([0.05, 0.95])),
                            y=Param(-0.05, 'length', bounds=np.array([0.05, 0.95])), previous_anchor_point='le',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point_tuple = (free_point1, free_point2)

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params,
                      anchor_point_tuple=(),
                      free_point_tuple=free_point_tuple)

    starting_airfoil = deepcopy(airfoil)

    fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False,
                            plot_kwargs=[{'color': 'indianred', 'ls': '-.'}] * 2)
    starting_airfoil_proxy = Line2D([], [], color='indianred', ls='-.')

    axs.plot(airfoil_to_match_xy[:, 0], airfoil_to_match_xy[:, 1], color='black', ls='', marker='o', fillstyle='none')
    airfoil_to_match_proxy = Line2D([], [], color='black', ls='', marker='o', fillstyle='none')

    print(f"Showing initial guess. Close the plot to continue.")
    show()

    print(f"Matching airfoil shape...")

    match_airfoil(airfoil, airfoil_to_match=airfoil_to_match, repair=repair_example_sd7062)

    fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False,
                            plot_kwargs=[{'color': 'cornflowerblue', 'ls': '--'}] * 2)
    final_airfoil_proxy = Line2D([], [], color='cornflowerblue', ls='--')

    starting_airfoil.plot(('airfoil',), fig=fig, axs=axs, show_plot=False, show_legend=False,
                          plot_kwargs=[{'color': 'indianred', 'ls': '-.'}] * 2)

    axs.plot(airfoil_to_match_xy[:, 0], airfoil_to_match_xy[:, 1], color='black', ls='', marker='o', fillstyle='none')

    fig.legend([starting_airfoil_proxy, airfoil_to_match_proxy, final_airfoil_proxy], ['starting airfoil',
                                                                                       'airfoil to match',
                                                                                       'final airfoil'], fontsize=12)
    axs.set_xlabel(r'$x/c$', fontsize=14)
    axs.set_ylabel(r'$y/c$', fontsize=14)

    fig.tight_layout()

    show()

    save_plot = False
    if save_plot:
        fig.suptitle('')
        fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                 'docs', 'images', 'sd7062_matching.png'), dpi=600)
        fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                 'docs', 'images', 'sd7062_matching.pdf'))


if __name__ == '__main__':
    run()
