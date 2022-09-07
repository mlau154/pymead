import numpy as np

from pymead.core.free_point import FreePoint
from pymead.core.anchor_point import AnchorPoint
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.airfoil import Airfoil
from pymead.core.param import Param
from pymead.core.param_setup import ParamSetup
from pymead.core.parametrization import AirfoilParametrization


def generate_unlinked_param_dict():
    """
    ### Description:

    Generates the parameters in the `param_dict` which are to have `linked=False`. It is possible for `active` to be set
    to `False` in some `pymead.core.param.Param`s. These parameters will be ignored in the parameter extraction
    method.

    ### Returns:

    The dictionary filled with unlinked parameters.
    """
    param_dict = {
        # Base Airfoil Params:
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
        'c_nacelle': Param(7.0),
        'alf_nacelle': Param(np.deg2rad(1.0)),
        'R_le_nacelle': Param(0.04, 'length'),
        'L_le_nacelle': Param(0.05, 'length'),
        'r_le_nacelle': Param(0.6),
        'phi_le_nacelle': Param(np.deg2rad(12.0)),
        'psi1_le_nacelle': Param(np.deg2rad(10.0)),
        'psi2_le_nacelle': Param(np.deg2rad(15.0)),
        'L1_te_nacelle': Param(0.25, 'length'),
        'L2_te_nacelle': Param(0.3, 'length'),
        'theta1_te_nacelle': Param(np.deg2rad(2.0)),
        'theta2_te_nacelle': Param(np.deg2rad(2.0)),
        't_te_nacelle': Param(0.0, 'length'),
        'r_te_nacelle': Param(0.5),
        'phi_te_nacelle': Param(np.deg2rad(0.0)),
        'dx_nacelle': Param(0.2, 'length'),
        'dy_nacelle': Param(0.2, 'length'),
        # Anchor points:

    }
    return param_dict


def generate_linked_param_dict(param_dict):
    """
    ### Description:

    Generates more parameters for the parameter dictionary which are functions of other parameters in the `param_dict`
    and should not be included in the method which overrides the parameters (using `linked=True`). If this method there
    are no linked parameters required, this method can simply be empty and return the input parameter dictionary. E.g.,

    ```python
    def generate_linked_param_dict(param_dict):
        return param_dict
    ```

    ### Args:

    `param_dict`: The parameter dictionary which contains unlinked parameters.

    ### Returns:

    The dictionary of parameters
    """
    return param_dict


def generate_airfoils(param_dict):
    """
    ### Description:

    Converts the parameter dictionary into a tuple of `pymead.core.airfoil.Airfoil`s. Written as a method to increase
    flexibility in the implementation.
    """
    base_airfoil_params_main = BaseAirfoilParams(c=param_dict['c_main'],
                                                 alf=param_dict['alf_main'],
                                                 R_le=param_dict['R_le_main'],
                                                 L_le=param_dict['L_le_main'],
                                                 r_le=param_dict['r_le_main'],
                                                 phi_le=param_dict['phi_le_main'],
                                                 psi1_le=param_dict['psi1_le_main'],
                                                 psi2_le=param_dict['psi2_le_main'],
                                                 L1_te=param_dict['L1_te_main'],
                                                 L2_te=param_dict['L2_te_main'],
                                                 theta1_te=param_dict['theta1_te_main'],
                                                 theta2_te=param_dict['theta2_te_main'],
                                                 t_te=param_dict['t_te_main'],
                                                 r_te=param_dict['r_te_main'],
                                                 phi_te=param_dict['phi_te_main'],
                                                 dx=param_dict['dx_main'],
                                                 dy=param_dict['dy_main'],
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

    base_airfoil_params_nacelle = BaseAirfoilParams(c=param_dict['c_nacelle'],
                                                    alf=param_dict['alf_nacelle'],
                                                    R_le=param_dict['R_le_nacelle'],
                                                    L_le=param_dict['L_le_nacelle'],
                                                    r_le=param_dict['r_le_nacelle'],
                                                    phi_le=param_dict['phi_le_nacelle'],
                                                    psi1_le=param_dict['psi1_le_nacelle'],
                                                    psi2_le=param_dict['psi2_le_nacelle'],
                                                    L1_te=param_dict['L1_te_nacelle'],
                                                    L2_te=param_dict['L2_te_nacelle'],
                                                    theta1_te=param_dict['theta1_te_nacelle'],
                                                    theta2_te=param_dict['theta2_te_nacelle'],
                                                    t_te=param_dict['t_te_nacelle'],
                                                    r_te=param_dict['r_te_nacelle'],
                                                    phi_te=param_dict['phi_te_nacelle'],
                                                    dx=param_dict['dx_nacelle'],
                                                    dy=param_dict['dy_nacelle'],
                                                    non_dim_by_chord=True
                                                    )

    airfoil_nacelle = Airfoil(number_coordinates=100,
                              base_airfoil_params=base_airfoil_params_nacelle,
                              anchor_point_tuple=(),
                              free_point_tuple=())

    airfoil_tuple = (airfoil_main, airfoil_nacelle)
    return airfoil_tuple


def update(parametrization: AirfoilParametrization, param_setup: ParamSetup, parameter_list: list = None):
    """
    ### Description:

    Overrides parameter list using the input parameter list, generates the airfoil coordinates, and mirrors part of the
    main airfoil element across a specified axis.
    """
    if parameter_list is not None:
        parametrization.override_parameters(parameter_list, normalized=True)
    else:
        parametrization.generate_airfoils_()

    m = Param(-0.05)
    b = Param(0.08, 'length', scale_value=param_setup.param_dict['c_main'].value)
    parametrization.mirror(axis=(m.value, b.value), fixed_airfoil_idx=0, linked_airfoil_idx=1,
                           fixed_anchor_point_range=('anchor-top', 'anchor-top2'),
                           starting_prev_anchor_point_str_linked='le')


def run():
    """
    ### Description:

    In this example, a portion of one airfoil is mirrored across an axis onto another. Doing this decreases the number
    of dimensions in the parametrization and reduces the amount of parametrization setup required.
    """
    param_setup = ParamSetup(generate_unlinked_param_dict, generate_linked_param_dict)
    parametrization = AirfoilParametrization(param_setup, generate_airfoils)

    update(parametrization, param_setup, None)

    fig, axs = parametrization.airfoil_tuple[0].plot(
        ('airfoil', 'control-point-skeleton'),
        show_plot=False, show_legend=False)

    axs.plot([0, 10], [0.8, 0.3], color='orange', label='axis')

    parametrization.airfoil_tuple[1].plot(('airfoil', 'control-point-skeleton'),
                                          fig=fig, axs=axs)


if __name__ == '__main__':
    run()
