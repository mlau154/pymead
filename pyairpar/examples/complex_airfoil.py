import numpy as np
import os
from matplotlib.pyplot import show
from pyairpar.core.param import Param
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.free_point import FreePoint
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams


def run():
    base_airfoil_params = BaseAirfoilParams(c=Param(10.0),
                                            alf=Param(np.deg2rad(5.0)),
                                            R_le=Param(0.06, 'length'),
                                            L_le=Param(0.08, 'length'),
                                            r_le=Param(0.6),
                                            phi_le=Param(np.deg2rad(12.0)),
                                            psi1_le=Param(np.deg2rad(10.0)),
                                            psi2_le=Param(np.deg2rad(15.0)),
                                            L1_te=Param(0.25, 'length'),
                                            L2_te=Param(0.3, 'length'),
                                            theta1_te=Param(np.deg2rad(2.0)),
                                            theta2_te=Param(np.deg2rad(2.0)),
                                            t_te=Param(0.0, 'length'),
                                            r_te=Param(0.5),
                                            phi_te=Param(np.deg2rad(0.0)),
                                            non_dim_by_chord=True
                                            )

    anchor_point1 = AnchorPoint(x=Param(0.55, units='length'),
                                y=Param(0.04, units='length'),
                                name='anchor-top',
                                previous_anchor_point='te_1',
                                L=Param(0.1, units='length'),
                                R=Param(0.15, units='length'),
                                r=Param(0.55),
                                phi=Param(np.deg2rad(14.0)),
                                psi1=Param(np.deg2rad(80.0)),
                                psi2=Param(np.deg2rad(40.0)),
                                length_scale_dimension=base_airfoil_params.c.value
                                )

    anchor_point2 = AnchorPoint(x=Param(0.2, units='length'),
                                y=Param(0.05, units='length'),
                                name='anchor-top2',
                                previous_anchor_point='anchor-top',
                                L=Param(0.04, units='length'),
                                R=Param(0.1, units='length'),
                                r=Param(0.5),
                                phi=Param(np.deg2rad(0.0)),
                                psi1=Param(np.deg2rad(60.0)),
                                psi2=Param(np.deg2rad(60.0)),
                                length_scale_dimension=base_airfoil_params.c.value
                                )

    anchor_point3 = AnchorPoint(x=Param(0.35, units='length'),
                                y=Param(-0.02, units='length'),
                                name='anchor-bottom',
                                previous_anchor_point='le',
                                L=Param(0.13, units='length'),
                                R=Param(0.2, units='length'),
                                r=Param(0.7),
                                phi=Param(np.deg2rad(8.0)),
                                psi1=Param(np.deg2rad(100.0)),
                                psi2=Param(np.deg2rad(100.0)),
                                length_scale_dimension=base_airfoil_params.c.value
                                )

    anchor_point_tuple = (anchor_point1, anchor_point2, anchor_point3)

    free_point1 = FreePoint(x=Param(0.15, units='length'),
                            y=Param(0.015, units='length'),
                            previous_anchor_point='le',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point2 = FreePoint(x=Param(0.58, units='length'),
                            y=Param(0.0, units='length'),
                            previous_anchor_point='anchor-bottom',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point3 = FreePoint(x=Param(0.3, units='length'),
                            y=Param(0.07, units='length'),
                            previous_anchor_point='anchor-top',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point_tuple = (free_point1, free_point2, free_point3)

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params,
                      anchor_point_tuple=anchor_point_tuple,
                      free_point_tuple=free_point_tuple)

    self_intersecting = airfoil.check_self_intersection()
    print(f"Self-intersecting? {self_intersecting}")

    fig, axs = airfoil.plot(('airfoil', 'control-point-skeleton', 'anchor-point-skeleton', 'chordline',
                             'R-circles'), show_plot=False)

    axs.set_xlabel(r'$x/c$', fontsize=14)
    axs.set_ylabel(r'$y/c$', fontsize=14)
    fig.suptitle('')
    fig.set_figwidth(9)
    fig.set_figheight(6)
    fig.tight_layout()

    show_flag = False
    save_flag = True

    if save_flag:
        save_name = os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(os.getcwd()))), 'docs', 'images', 'complex_1.png')
        fig.savefig(save_name, dpi=600)
    if show_flag:
        show()


if __name__ == '__main__':
    run()
