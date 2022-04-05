import numpy as np
from pyairpar.core.param import Param
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams


def main():
    """
    ### Description:

    Generates and plots the airfoil shape, control point skeleton, anchor point skeleton, chordline, and a circle
    depicting the leading edge radius of curvature for the most basic `pyairpar.core.airfoil.Airfoil` design: an
    airfoil with no `pyairpar.core.free_point.FreePoint`s or `pyairpar.core.anchor_point.AnchorPoints`s. Here the
    trailing edge thickness is set to a non-zero value to show the full capability of the
    `pyairpar.core.base_airfoil_params.BaseAirfoilParams` class. A simpler airfoil could be created by setting the
    trailing edge thickness to `0.0`. In doing so, the `pyairpar.core.airfoil.Airfoil` class ignores the values set
    for `r_te` and `phi_te` and sets their `active` attributes to `False`.
    """

    base_airfoil_params = BaseAirfoilParams(c=Param(4.0),
                                            alf=Param(np.deg2rad(5.0)),
                                            R_le=Param(0.03, 'length'),
                                            L_le=Param(0.08, 'length'),
                                            r_le=Param(0.6),
                                            phi_le=Param(np.deg2rad(10.0)),
                                            psi1_le=Param(np.deg2rad(10.0)),
                                            psi2_le=Param(np.deg2rad(15.0)),
                                            L1_te=Param(0.25, 'length'),
                                            L2_te=Param(0.3, 'length'),
                                            theta1_te=Param(np.deg2rad(2.0)),
                                            theta2_te=Param(np.deg2rad(2.0)),
                                            t_te=Param(0.01, 'length'),
                                            r_te=Param(0.4),
                                            phi_te=Param(np.deg2rad(20.0))
                                            )

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params)

    print(airfoil.params)
    print(airfoil.bounds)

    airfoil.plot(('chordline', 'anchor-point-skeleton', 'control-point-skeleton', 'airfoil', 'R-circles'))


if __name__ == '__main__':
    main()
