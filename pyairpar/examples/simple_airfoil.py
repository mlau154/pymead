import numpy as np
import os
from matplotlib.pyplot import show
from pyairpar.core.param import Param
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams


def run():
    """
    ### Description:

    Generates and plots the airfoil shape, control point skeleton, anchor point skeleton, chordline, and a circle
    depicting the leading edge radius of curvature for the most basic `pyairpar.core.airfoil.Airfoil` design: an
    airfoil with no `pyairpar.core.free_point.FreePoint`s or `pyairpar.core.anchor_point.AnchorPoint`s.
    """
    base_airfoil_params = BaseAirfoilParams(c=Param(4.0),
                                            alf=Param(np.deg2rad(3.0)),
                                            R_le=Param(0.03, 'length'),
                                            L_le=Param(0.08, 'length'),
                                            r_le=Param(0.6),
                                            psi1_le=Param(np.deg2rad(10.0)),
                                            psi2_le=Param(np.deg2rad(15.0)),
                                            L1_te=Param(0.25, 'length'),
                                            L2_te=Param(0.3, 'length'),
                                            theta1_te=Param(np.deg2rad(2.0)),
                                            theta2_te=Param(np.deg2rad(3.0)),
                                            )

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params)

    fig, axs = airfoil.plot(('airfoil', 'control-point-skeleton', 'anchor-point-skeleton', 'chordline',
                                                     'R-circles'), show_plot=False)
    axs.set_xlabel(r'$x$', fontsize=14)
    axs.set_ylabel(r'$y$', fontsize=14)
    fig.suptitle('')
    fig.tight_layout()

    show_flag = True
    save_flag = False

    if save_flag:
        save_name = os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(os.getcwd()))), 'docs', 'images', 'simple_1.png')
        fig.savefig(save_name, dpi=600)
        save_name_pdf = os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(os.getcwd()))), 'docs', 'images', 'simple_1.pdf')
        fig.savefig(save_name_pdf)
    if show_flag:
        show()


if __name__ == '__main__':
    run()
