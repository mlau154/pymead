import numpy as np
from matplotlib.pyplot import show
from pymead.core.param import Param
from pymead.core.anchor_point import AnchorPoint
from pymead.core.free_point import FreePoint
from pymead.symmetric.symmetric_airfoil import SymmetricAirfoil
from pymead.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams


def run():
    """
    ### Description:

    An example implementation of a `pymead.symmetric.symmetric_airfoil.SymmetricAirfoil`.
    """
    c = 10.0

    base_airfoil_params = SymmetricBaseAirfoilParams(c=Param(c),
                                                     alf=Param(np.deg2rad(-2.5)),
                                                     R_le=Param(0.04, 'length'),
                                                     L_le=Param(0.08, 'length'),
                                                     psi1_le=Param(np.deg2rad(10.0)),
                                                     L1_te=Param(0.3, 'length'),
                                                     theta1_te=Param(np.deg2rad(5.0)),
                                                     t_te=Param(0.0, 'length'),
                                                     non_dim_by_chord=True
                                                     )

    anchor_point1 = AnchorPoint(x=Param(0.55, units='length'),
                                y=Param(0.04, units='length'),
                                tag='anchor-top',
                                previous_anchor_point='te_1',
                                L=Param(0.12, units='length'),
                                R=Param(0.2, units='length'),
                                r=Param(0.55),
                                phi=Param(np.deg2rad(0.0)),
                                psi1=Param(np.deg2rad(55.0)),
                                psi2=Param(np.deg2rad(45.0)),
                                length_scale_dimension=c
                                )

    anchor_point_tuple = (anchor_point1,)

    free_point1 = FreePoint(x=Param(0.4, units='length'), y=Param(0.1, units='length'),
                            previous_anchor_point='anchor-top', length_scale_dimension=c)

    free_point2 = FreePoint(x=Param(0.25, units='length'), y=Param(0.075, units='length'),
                            previous_anchor_point='anchor-top', length_scale_dimension=c)

    free_point_tuple = (free_point1, free_point2)

    airfoil = SymmetricAirfoil(number_coordinates=100,
                               base_airfoil_params=base_airfoil_params,
                               anchor_point_tuple=anchor_point_tuple,
                               free_point_tuple=free_point_tuple)

    self_intersecting = airfoil.check_self_intersection()
    print(f"Self-intersecting? {self_intersecting}")

    fig, axs = airfoil.plot(('chordline', 'anchor-point-skeleton', 'control-point-skeleton', 'airfoil', 'R-circles'),
                            show_plot=False)
    fig.set_figheight(6)
    fig.set_figwidth(9)
    fig.tight_layout()
    show()


if __name__ == '__main__':
    run()
