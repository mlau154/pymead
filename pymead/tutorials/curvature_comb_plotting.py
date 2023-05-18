"""
Examples of plotting curvature combs for Bézier curves and airfoils

.. figure:: ../images/curvature_comb_light.*
    :class: only-light
    :width: 600
    :align: center

.. figure:: ../images/curvature_comb_dark.*
    :class: only-dark
    :width: 600
    :align: center

"""
from pymead.core.bezier import Bezier
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from matplotlib.pyplot import subplots, show, rcParams
from matplotlib.lines import Line2D
import numpy as np


def main(dark: bool = False):
    # Some plot settings:
    rcParams["font.family"] = "serif"
    curve_props = dict(color='cornflowerblue', lw=1.8)
    normal_props = dict(color='mediumaquamarine', lw=0.8)
    comb_props = dict(color='indianred', lw=0.8)
    skeleton_props = dict(color='gray', lw=0.7, marker='x', mec='grey', mfc='grey')

    # Plot the curvature comb for a single Bezier curve:
    C = Bezier(np.array([[0, 0], [1, 1], [2, 1], [3, 0]]), nt=500)
    scale_factor = 0.1 / np.max(abs(C.k))
    fig, axs = subplots(2, 1)
    C.plot_curve(axs[0], **curve_props)
    C.plot_curvature_comb_normals(axs[0], scale_factor, **normal_props, interval=3)
    C.plot_curvature_comb_curve(axs[0], scale_factor, **comb_props, interval=3)
    C.plot_control_point_skeleton(axs[0], **skeleton_props)
    axs[0].set_title("A single Bézier curve", size=14)

    # Create an airfoil and plot the curvature comb for all the airfoil curves:
    A = Airfoil(base_airfoil_params=BaseAirfoilParams(L_le=Param(0.2),
                                                      psi1_le=Param(np.deg2rad(10)),
                                                      psi2_le=Param(np.deg2rad(20)),
                                                      theta1_te=Param(np.deg2rad(1.0))))
    scale_factor_airfoil = 0.07 / np.max([np.max(abs(curve.k)) for curve in A.curve_list])
    for curve in A.curve_list:
        curve.plot_curve(axs[1], **curve_props)
        curve.plot_curvature_comb_normals(axs[1], scale_factor_airfoil, **normal_props)
        curve.plot_curvature_comb_curve(axs[1], scale_factor_airfoil, **comb_props)
    for ax in axs:
        ax.set_aspect('equal')
    axs[1].set_title("A pymead-generated Bézier airfoil", size=14)

    # Create a legend on top:
    labels = ["Bézier curve", "Curve normals", "Curvature comb", "Control point skeleton"]
    prop_list = [curve_props, normal_props, comb_props, skeleton_props]
    line_proxies = [Line2D([], [], **props) for props in prop_list]
    box = axs[0].get_position()
    axs[0].set_position([box.x0, box.y0, box.width, box.height * 0.8])

    extra_leg_props = dict(facecolor="black", labelcolor="white") if dark else {}
    fig.legend(line_proxies, labels, ncol=2, fancybox=True, shadow=True, loc="upper center", **extra_leg_props)

    # Show the plot
    show()

    return fig, axs


if __name__ == '__main__':
    main()
