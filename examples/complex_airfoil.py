import numpy as np
from core.param import Param
from core.anchor_point import AnchorPoint
from core.free_point import FreePoint
from core.airfoil import Airfoil
from core.base_airfoil_params import BaseAirfoilParams


def main():
    c = 10.0

    base_airfoil_params = BaseAirfoilParams(c=Param(c),
                                            alf=Param(np.deg2rad(5.0)),
                                            R_le=Param(0.06, 'length', c=c),
                                            L_le=Param(0.08, 'length', c=c),
                                            r_le=Param(0.6),
                                            phi_le=Param(np.deg2rad(5.0)),
                                            psi1_le=Param(np.deg2rad(10.0)),
                                            psi2_le=Param(np.deg2rad(15.0)),
                                            L1_te=Param(0.25, 'length', c=c),
                                            L2_te=Param(0.3, 'length', c=c),
                                            theta1_te=Param(np.deg2rad(2.0)),
                                            theta2_te=Param(np.deg2rad(2.0)),
                                            t_te=Param(0.0, 'length', c=c),
                                            r_te=Param(0.5),
                                            phi_te=Param(np.deg2rad(0.0))
                                            )

    anchor_point1 = AnchorPoint(x=Param(0.55, units='length', c=c),
                                y=Param(0.04, units='length', c=c),
                                name='anchor-top',
                                previous_anchor_point='te_1',
                                L=Param(0.1, units='length', c=c),
                                R=Param(-0.3, units='length', c=c),
                                r=Param(0.5),
                                phi=Param(np.deg2rad(0.0)),
                                psi1=Param(np.deg2rad(45.0)),
                                psi2=Param(np.deg2rad(45.0)),
                                )

    anchor_point2 = AnchorPoint(x=Param(0.35, units='length', c=c),
                                y=Param(-0.02, units='length', c=c),
                                name='anchor-bottom',
                                previous_anchor_point='le',
                                L=Param(0.13, units='length', c=c),
                                R=Param(0.4, units='length', c=c),
                                r=Param(0.7),
                                phi=Param(np.deg2rad(0.0)),
                                psi1=Param(np.deg2rad(75.0)),
                                psi2=Param(np.deg2rad(75.0)),
                                )

    anchor_point_tuple = (anchor_point1, anchor_point2)

    free_point1 = FreePoint(x=Param(0.15, units='length', c=c),
                            y=Param(0.015, units='length', c=c),
                            previous_anchor_point='le')

    free_point2 = FreePoint(x=Param(0.58, units='length', c=c),
                            y=Param(0.0, units='length', c=c),
                            previous_anchor_point='anchor-bottom')

    free_point3 = FreePoint(x=Param(0.3, units='length', c=c),
                            y=Param(0.07, units='length', c=c),
                            previous_anchor_point='anchor-top')

    free_point_tuple = (free_point1, free_point2, free_point3)

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params,
                      anchor_point_tuple=anchor_point_tuple,
                      free_point_tuple=free_point_tuple)

    self_intersecting = airfoil.check_self_intersection()
    print(f"Self-intersecting? {self_intersecting}")

    airfoil.plot(('curvature',), axis_equal=False)
    airfoil.plot(('chordline', 'anchor-point-skeleton', 'control-point-skeleton', 'airfoil', 'R-circles'))


if __name__ == '__main__':
    main()
