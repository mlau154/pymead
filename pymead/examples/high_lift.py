import numpy as np
from matplotlib.lines import Line2D
from matplotlib.pyplot import show
import os

from pymead.core.free_point import FreePoint
from pymead.core.anchor_point import AnchorPoint
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.airfoil import Airfoil
from pymead.core.param import Param
from pymead.core.param_setup import ParamSetup
from pymead.core.parametrization import AirfoilParametrization
from pymead.core.parametrization import rotate


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
        # Main chord length:
        'c_main': Param(1.0, active=False),
        # Angles of attack
        'alf_main': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
        'alf_flap': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
    }
    param_dict = {**param_dict, **{
        # Flap chord length:
        'c_flap': Param(0.3, 'length', bounds=[0.25, 0.35], scale_value=param_dict['c_main'].value),
        # Leading edge radius:
        'R_le_main': Param(0.03, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'R_le_flap': Param(0.01, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        # Leading edge length:
        'L_le_main': Param(0.081, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'L_le_flap': Param(0.025, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        # Leading edge length ratio:
        'r_le_main': Param(0.6, bounds=[0.01, 0.99]),
        'r_le_flap': Param(0.5, bounds=[0.01, 0.99]),
        # Leading edge tilt:
        'phi_le_flap': Param(np.deg2rad(0.0), active=False),
        # Upper curvature control arm angle:
        'psi1_le_main': Param(np.deg2rad(20.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi1_le_flap': Param(np.deg2rad(25.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Lower curvature control arm angle (upper and lower angles on hub are equal due to symmetry):
        'psi2_le_main': Param(np.deg2rad(10.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi2_le_flap': Param(np.deg2rad(15.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Upper trailing edge length:
        'L1_te_main': Param(0.0519, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        # Lower trailing edge length:
        'L2_te_main': Param(0.0997, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        'L2_te_flap': Param(0.15, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        # Upper trailing edge angle:
        'theta1_te_main': Param(np.deg2rad(6.0), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        # Lower trailing edge angle:
        'theta2_te_main': Param(np.deg2rad(-3.0), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        'theta2_te_flap': Param(np.deg2rad(1.0), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        # Anchor point lengths:
        'L_ap_flap_end_flap': Param(0.019, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        # Anchor point curvature:
        'kappa_ap_flap_end_flap': Param(1 / 1.0, 'inverse-length', bounds=[-1e2, 1e2],
                                        scale_value=param_dict['c_main'].value),
        # Anchor point length ratios:
        'r_ap_flap_end_flap': Param(0.5, bounds=[0.01, 0.99]),
        # Anchor point neighboring point line angles:
        'phi_ap_flap_end_flap': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Anchor point aft curvature control arm angles:
        'psi1_ap_flap_end_flap': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        # Anchor point fore curvature control arm angles:
        'psi2_ap_flap_end_flap': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        # Anchor point position for flap end anchor point:
        # Free point locations:
        'x_fp1_upper_main': Param(0.52, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_upper_main': Param(0.11, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        'x_fp1_lower_main': Param(0.44, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_lower_main': Param(-0.07, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        # 'x_fp1_upper_flap': Param(0.1, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        # 'y_fp1_upper_flap': Param(0.02, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        'x_fp1_lower_flap': Param(0.1, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_lower_flap': Param(-0.02, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        'parametrization_dictionary_name': 'v00'
    }}
    param_dict = {**param_dict, **{
        'x_ap_flap_le': Param
    }}
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
    param_dict = {**param_dict, **{
        'L1_te_flap': Param(param_dict['L2_te_main'].value, linked=True),
        'theta1_te_flap': Param(-param_dict['theta2_te_main'].value, linked=True),
        'L_ap_flap_le_main': Param(param_dict['L_le_flap'].value + 0.005 / param_dict['c_main'].value, linked=True),
        'L_ap_flap_end_main': Param(param_dict['L_ap_flap_end_flap'].value, linked=True),
        'kappa_ap_flap_le_main': Param(-1 / param_dict['R_le_flap'].value, linked=True),
        'kappa_ap_flap_end_main': Param(-param_dict['kappa_ap_flap_end_flap'].value, linked=True),
        'r_ap_flap_le_main': Param(param_dict['r_le_flap'].value, linked=True),
        'r_ap_flap_end_main': Param(param_dict['r_ap_flap_end_flap'].value, linked=True),
        'phi_ap_flap_le_main': Param(param_dict['phi_le_flap'].value - np.pi/2, linked=True),
        'phi_ap_flap_end_main': Param(param_dict['phi_ap_flap_end_flap'].value + np.pi, linked=True),
        'psi1_ap_flap_le_main': Param(param_dict['psi1_le_flap'].value + np.pi/2, linked=True),
        'psi1_ap_flap_end_main': Param(param_dict['psi1_ap_flap_end_flap'].value, linked=True),
        'psi2_ap_flap_le_main': Param(param_dict['psi2_le_flap'].value + np.pi/2, linked=True),
        'psi2_ap_flap_end_main': Param(param_dict['psi2_ap_flap_end_flap'].value, linked=True),
        'x_ap_flap_end_flap': Param(0.75, 'length', scale_value=param_dict['c_main'].value, linked=True),
        'y_ap_flap_end_flap': Param(-0.015, 'length', scale_value=param_dict['c_main'].value, linked=True),
        'x_ap_flap_end_main': Param(0.75, 'length', scale_value=param_dict['c_main'].value, linked=True),
        'y_ap_flap_end_main': Param(-0.02, 'length', scale_value=param_dict['c_main'].value, linked=True),
        'dx_flap': Param(param_dict['c_main'].value - param_dict['c_flap'].value,
                         'length', scale_value=param_dict['c_main'].value, linked=True),
        'dy_flap': Param(0.0, 'length', scale_value=param_dict['c_main'].value, active=False),
    }}
    param_dict = {**param_dict, **{
        'R_ap_flap_le_main': Param(np.divide(1, param_dict['kappa_ap_flap_le_main'].value), linked=True),
        'R_ap_flap_end_main': Param(np.divide(1, param_dict['kappa_ap_flap_end_main'].value), linked=True),
        'R_ap_flap_end_flap': Param(np.divide(1, param_dict['kappa_ap_flap_end_flap'].value), linked=True),
        'x_ap_flap_le_main': Param(param_dict['dx_flap'].value - 0.005 / param_dict['c_main'].value, linked=True),
        'y_ap_flap_le_main': Param(param_dict['dy_flap'].value, linked=True),
    }}
    return param_dict


def generate_param_dict():
    """
    ### Description:

    Method that executes the methods that generate the unlinked and linked parameter dictionaries.

    ### Returns:

    The combined dictionary of parameters
    """
    param_dict = generate_unlinked_param_dict()
    param_dict = generate_linked_param_dict(param_dict)
    return param_dict


def generate_airfoils(param_dict):
    """
    ### Description:

    Converts the parameter dictionary into a tuple of `pymead.core.airfoil.Airfoil`s. Written as a method to increase
    flexibility in the implementation.
    """
    base_airfoil_params_main = \
        BaseAirfoilParams(c=param_dict['c_main'], alf=param_dict['alf_main'], R_le=param_dict['R_le_main'],
                          L_le=param_dict['L_le_main'], r_le=param_dict['r_le_main'],
                          psi1_le=param_dict['psi1_le_main'], psi2_le=param_dict['psi2_le_main'],
                          L1_te=param_dict['L1_te_main'], L2_te=param_dict['L2_te_main'],
                          theta1_te=param_dict['theta1_te_main'], theta2_te=param_dict['theta2_te_main'],
                          non_dim_by_chord=True)

    base_airfoil_params_flap = \
        BaseAirfoilParams(c=param_dict['c_flap'], alf=param_dict['alf_flap'], R_le=param_dict['R_le_flap'],
                          L_le=param_dict['L_le_flap'], r_le=param_dict['r_le_flap'],
                          psi1_le=param_dict['psi1_le_flap'],
                          psi2_le=param_dict['psi2_le_flap'], L1_te=param_dict['L1_te_flap'],
                          L2_te=param_dict['L2_te_flap'], theta1_te=param_dict['theta1_te_flap'],
                          theta2_te=param_dict['theta2_te_flap'], dx=param_dict['dx_flap'],
                          dy=param_dict['dy_flap'], non_dim_by_chord=False)

    x = param_dict['x_ap_flap_le_main'].value
    y = param_dict['y_ap_flap_le_main'].value
    x, y = rotate(x, y, param_dict['alf_main'].value)

    ap_flap_le_main = AnchorPoint(x=Param(x, linked=True),
                                    y=Param(y, linked=True),
                                    name='ap_flap_le_main',
                                    previous_anchor_point='ap_flap_end_main',
                                    L=param_dict['L_ap_flap_le_main'],
                                    R=param_dict['R_ap_flap_le_main'],
                                    r=param_dict['r_ap_flap_le_main'],
                                    phi=param_dict['phi_ap_flap_le_main'],
                                    psi1=param_dict['psi1_ap_flap_le_main'],
                                    psi2=param_dict['psi2_ap_flap_le_main'])

    x = param_dict['x_ap_flap_end_main'].value
    y = param_dict['y_ap_flap_end_main'].value
    x, y = rotate(x, y, param_dict['alf_main'].value)

    ap_flap_end_main = AnchorPoint(x=Param(x, linked=True),
                                     y=Param(y, linked=True),
                                     name='ap_flap_end_main',
                                     previous_anchor_point='le',
                                     L=param_dict['L_ap_flap_end_main'],
                                     R=param_dict['R_ap_flap_end_main'],
                                     r=param_dict['r_ap_flap_end_main'],
                                     phi=param_dict['phi_ap_flap_end_main'],
                                     psi1=param_dict['psi1_ap_flap_end_main'],
                                     psi2=param_dict['psi2_ap_flap_end_main'],
                                    )

    anchor_point_tuple_main = (ap_flap_end_main, ap_flap_le_main)

    fp1_upper_main = FreePoint(x=param_dict['x_fp1_upper_main'],
                               y=param_dict['y_fp1_upper_main'],
                               previous_anchor_point='te_1')

    fp1_lower_main = FreePoint(x=param_dict['x_fp1_lower_main'],
                               y=param_dict['y_fp1_lower_main'],
                               previous_anchor_point='le')

    free_point_tuple_main = (fp1_upper_main, fp1_lower_main)

    fp1_lower_flap = FreePoint(x=param_dict['x_fp1_lower_flap'],
                               y=param_dict['y_fp1_lower_flap'],
                               previous_anchor_point='ap_flap_end_flap')

    free_point_tuple_flap = (fp1_lower_flap,)

    x = param_dict['x_ap_flap_end_flap'].value
    y = param_dict['y_ap_flap_end_flap'].value
    x -= param_dict['dx_flap'].value
    y -= param_dict['dy_flap'].value
    x, y = rotate(x, y, param_dict['alf_flap'].value)

    ap_flap_end_flap = AnchorPoint(x=Param(x, linked=True),
                                    y=Param(y, linked=True),
                                    name='ap_flap_end_flap',
                                    previous_anchor_point='le',
                                    L=param_dict['L_ap_flap_end_flap'],
                                    R=param_dict['R_ap_flap_end_flap'],
                                    r=param_dict['r_ap_flap_end_flap'],
                                    phi=param_dict['phi_ap_flap_end_flap'],
                                    psi1=param_dict['psi1_ap_flap_end_flap'],
                                    psi2=param_dict['psi2_ap_flap_end_flap'],
                                    )

    anchor_point_tuple_nacelle = (ap_flap_end_flap,)

    airfoil_main = Airfoil(number_coordinates=100,
                           base_airfoil_params=base_airfoil_params_main,
                           anchor_point_tuple=anchor_point_tuple_main,
                           free_point_tuple=free_point_tuple_main)

    airfoil_flap = Airfoil(number_coordinates=100,
                              base_airfoil_params=base_airfoil_params_flap,
                              anchor_point_tuple=anchor_point_tuple_nacelle,
                              free_point_tuple=free_point_tuple_flap)

    airfoil_tuple = (airfoil_main, airfoil_flap)
    return airfoil_tuple


def update(parametrization: AirfoilParametrization, parameter_list: list = None):
    """
    ### Description:

    Overrides parameter list using the input parameter list and generates the airfoil coordinates.
    """
    if parameter_list is not None:
        parametrization.override_parameters(parameter_list, normalized=True)
    else:
        parametrization.generate_airfoils_()


def run():
    """
    ### Description:

    An example implementation of a multi-element airfoil parametrization. Modeled in this example is a high-lift
    configuration consisting of a main airfoil element and a deployable Fowler flap.
    """
    param_setup = ParamSetup(generate_unlinked_param_dict, generate_linked_param_dict)
    parametrization = AirfoilParametrization(param_setup=param_setup,
                                             generate_airfoils=generate_airfoils)
    update(parametrization, None)

    fig, axs = parametrization.airfoil_tuple[0].plot(
        ('airfoil',), show_plot=False, show_legend=False)

    parametrization.airfoil_tuple[1].plot(('airfoil',),
                                          fig=fig, axs=axs, show_plot=False, show_legend=False,
                                          plot_kwargs=[{'color': 'indianred'}] * 4)

    parametrization.airfoil_tuple[1].translate(-0.7, 0.0)
    parametrization.airfoil_tuple[1].rotate(-np.deg2rad(30.0))
    parametrization.airfoil_tuple[1].translate(1.02, -0.02)
    parametrization.airfoil_tuple[1].update_anchor_point_array()
    parametrization.airfoil_tuple[1].generate_non_transformed_airfoil_coordinates()
    parametrization.airfoil_tuple[1].generate_coords()
    parametrization.airfoil_tuple[1].needs_update = False

    parametrization.airfoil_tuple[1].plot(('airfoil',),
                                          fig=fig, axs=axs, show_plot=False, show_legend=False,
                                          plot_kwargs=[{'color': 'indianred', 'ls': '--'}] * 4)

    parametrization.airfoil_tuple[1].plot(('control-point-skeleton',),
                                          fig=fig, axs=axs, show_plot=False, show_legend=False)

    airfoil_main_line_proxy = Line2D([], [], color='cornflowerblue')
    airfoil_flap_line_proxy = Line2D([], [], color='indianred')
    airfoil_flap_line_proxy2 = Line2D([], [], color='indianred', ls='--')
    control_point_skeleton_proxy = Line2D([], [], color='grey', ls='--', marker='*')

    fig.legend([airfoil_main_line_proxy, airfoil_flap_line_proxy, airfoil_flap_line_proxy2,
                control_point_skeleton_proxy],
               ['main element', 'flap (stowed)', 'flap (deployed)', 'control polygon'], fontsize=12)

    fig.suptitle('')
    axs.set_xlabel(r'$x/c$', fontsize=14)
    axs.set_ylabel(r'$y/c$', fontsize=14)
    fig.set_figheight(3)
    fig.set_figwidth(11)
    fig.tight_layout()

    show_flag = True
    save_flag = False

    if save_flag:
        save_name = os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(os.getcwd()))), 'docs', 'images', 'high_lift.png')
        fig.savefig(save_name, dpi=600)
        save_name_pdf = os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(os.getcwd()))), 'docs', 'images', 'high_lift.pdf')
        fig.savefig(save_name_pdf)
    if show_flag:
        show()


if __name__ == '__main__':
    run()
