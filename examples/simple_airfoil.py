import numpy as np
from core.param import Param
from core.airfoil import Airfoil
from core.base_airfoil_params import BaseAirfoilParams


def main():

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
                                            phi_te=Param(np.deg2rad(3.0))
                                            )

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params)

    print(airfoil.params)
    print(airfoil.bounds)

    airfoil.plot(('chordline', 'anchor-point-skeleton', 'control-point-skeleton', 'airfoil', 'R-circles'))


if __name__ == '__main__':
    main()
