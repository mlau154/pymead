import numpy as np

from pyairpar.core.free_point import FreePoint
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.param import Param
from pyairpar.core.param_setup import ParamSetup
from pyairpar.core.parametrization import AirfoilParametrization


def _generate_unlinked_param_dict():
    param_dict = {
        'c_main': Param(10.0),
        'alf_main': Param(np.deg2rad(5.0)),
        'R_le_main': Param(0.03, 'length'),
        'L_le_main': Param(0.05, 'length'),
        'r_le_main': Param(0.6),
        'phi_le_main': Param(np.deg2rad(12.0)),
        'psi1_le_main': Param(np.deg2rad(10.0)),
        'psi2_le_main': Param(np.deg2rad(15.0)),
        'L1_te_main': Param(0.25, 'length'),
        'L2_te_main': Param(0.3, 'length'),
        'theta1_te_main': Param(np.deg2rad(2.0)),
        'theta2_te_main': Param(np.deg2rad(2.0)),
        't_te_main': Param(0.0, 'length', active=False),
        'r_te_main': Param(0.5, active=False),
        'phi_te_main': Param(np.deg2rad(0.0)),
        'dx_main': Param(0.0, active=False),
        'dy_main': Param(0.0, active=False),
    }

def _generate_airfoils():
    base_airfoil_params_main = BaseAirfoilParams(c=Param(10.0),
                                                 alf=Param(np.deg2rad(5.0)),
                                                 R_le=Param(0.03, 'length'),
                                                 L_le=Param(0.05, 'length'),
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
                                                 dx=Param(0.0, active=False),
                                                 dy=Param(0.0, active=False),
                                                 non_dim_by_chord=True
                                                 )

    anchor_point1_main = AnchorPoint(x=Param(0.55, units='length'),
                                     y=Param(0.03, units='length'),
                                     name='anchor-top',
                                     previous_anchor_point='te_1',
                                     L=Param(0.1, units='length'),
                                     R=Param(0.15, units='length'),
                                     r=Param(0.55),
                                     phi=Param(np.deg2rad(-5.0)),
                                     psi1=Param(np.deg2rad(80.0)),
                                     psi2=Param(np.deg2rad(40.0)),
                                     length_scale_dimension=base_airfoil_params_main.c.value
                                     )

    anchor_point2_main = AnchorPoint(x=Param(0.2, units='length'),
                                     y=Param(0.05, units='length'),
                                     name='anchor-top2',
                                     previous_anchor_point='anchor-top',
                                     L=Param(0.04, units='length'),
                                     R=Param(0.1, units='length'),
                                     r=Param(0.5),
                                     phi=Param(np.deg2rad(0.0)),
                                     psi1=Param(np.deg2rad(60.0)),
                                     psi2=Param(np.deg2rad(60.0)),
                                     length_scale_dimension=base_airfoil_params_main.c.value
                                     )

    anchor_point3_main = AnchorPoint(x=Param(0.35, units='length'),
                                     y=Param(-0.02, units='length'),
                                     name='anchor-bottom',
                                     previous_anchor_point='le',
                                     L=Param(0.13, units='length'),
                                     R=Param(0.2, units='length'),
                                     r=Param(0.7),
                                     phi=Param(np.deg2rad(8.0)),
                                     psi1=Param(np.deg2rad(100.0)),
                                     psi2=Param(np.deg2rad(100.0)),
                                     length_scale_dimension=base_airfoil_params_main.c.value
                                     )

    anchor_point_tuple_main = (anchor_point1_main, anchor_point2_main, anchor_point3_main)

    free_point1 = FreePoint(x=Param(0.15, units='length'),
                            y=Param(0.015, units='length'),
                            previous_anchor_point='le',
                            length_scale_dimension=base_airfoil_params_main.c.value)

    free_point2 = FreePoint(x=Param(0.58, units='length'),
                            y=Param(0.0, units='length'),
                            previous_anchor_point='anchor-bottom',
                            length_scale_dimension=base_airfoil_params_main.c.value)

    free_point3 = FreePoint(x=Param(0.3, units='length'),
                            y=Param(0.07, units='length'),
                            previous_anchor_point='anchor-top',
                            length_scale_dimension=base_airfoil_params_main.c.value)

    free_point4 = FreePoint(x=Param(0.28, units='length'),
                            y=Param(0.065, units='length'),
                            previous_anchor_point='anchor-top',
                            length_scale_dimension=base_airfoil_params_main.c.value)

    free_point_tuple = (free_point1, free_point2, free_point3, free_point4)

    airfoil_main = Airfoil(number_coordinates=100,
                           base_airfoil_params=base_airfoil_params_main,
                           anchor_point_tuple=anchor_point_tuple_main,
                           free_point_tuple=free_point_tuple)

    base_airfoil_params_nacelle = BaseAirfoilParams(c=Param(7.0),
                                                    alf=Param(np.deg2rad(1.0)),
                                                    R_le=Param(0.04, 'length'),
                                                    L_le=Param(0.05, 'length'),
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
                                                    dx=Param(0.2, 'length'),
                                                    dy=Param(0.2, 'length'),
                                                    non_dim_by_chord=True
                                                    )

    airfoil_nacelle = Airfoil(number_coordinates=100,
                              base_airfoil_params=base_airfoil_params_nacelle,
                              anchor_point_tuple=(),
                              free_point_tuple=())

    airfoil_tuple = (airfoil_main, airfoil_nacelle)
    return airfoil_tuple


def run():
    param_setup = ParamSetup(_generate_unlinked_param_dict, _generate_unlinked_param_dict)
    parametrization = AirfoilParametrization(param_setup, _generate_airfoils)
    m = Param(-0.05)
    b = Param(0.08, 'length', scale_value=param_setup.param_dict['c_main'])
    parametrization.mirror(axis=(m.value, b.value), fixed_airfoil_idx=0, linked_airfoil_idx=1,
                           fixed_anchor_point_range=('anchor-top', 'anchor-top2'),
                           starting_prev_anchor_point_str_linked='le')

    parametrization.extract_parameters()

    print(f"fixed_airfoil_N = {airfoil_main.N}, linked_airfoil_N = {airfoil_nacelle.N}")

    fig, axs = parametrization.airfoil_tuple[0].plot(
        ('airfoil', 'control-point-skeleton'),
        show_plot=False, show_legend=False)

    axs.plot([0, 10], [0.8, 0.3], color='orange', label='axis')

    parametrization.airfoil_tuple[1].plot(('airfoil', 'control-point-skeleton'),
                                          fig=fig, axs=axs)


if __name__ == '__main__':
    run()
