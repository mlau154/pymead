from pyairpar.core.bezier import Bezier
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.core.param import Param
from matplotlib.pyplot import subplots, show
import numpy as np


def main():
    C = Bezier(np.array([[0, 0], [1, 1], [2, 1], [3, 0]]), nt=500)
    scale_factor = 0.1 / np.max(abs(C.k))
    fig, axs = subplots(2, 1)
    C.plot_curve(axs[0], color='cornflowerblue', lw=1.8)
    C.plot_curvature_comb_normals(axs[0], scale_factor, color='mediumaquamarine', lw=0.8, interval=3)
    C.plot_curvature_comb_curve(axs[0], scale_factor, color='indianred', lw=0.8, interval=3)

    A = Airfoil(base_airfoil_params=BaseAirfoilParams(L_le=Param(0.2), psi1_le=Param(np.deg2rad(10)),
                                                      psi2_le=Param(np.deg2rad(20)), theta1_te=Param(np.deg2rad(1.0))))
    scale_factor_airfoil = 0.07 / np.max([np.max(abs(curve.k)) for curve in A.curve_list])
    for curve in A.curve_list:
        curve.plot_curve(axs[1], color='cornflowerblue', lw=2)
        curve.plot_curvature_comb_normals(axs[1], scale_factor_airfoil, color='mediumaquamarine', lw=0.8)
        curve.plot_curvature_comb_curve(axs[1], scale_factor_airfoil, color='indianred', lw=0.8)
    for ax in axs:
        ax.set_aspect('equal')
    show()


if __name__ == '__main__':
    main()
