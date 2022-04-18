import numpy as np
from matplotlib.lines import Line2D
from matplotlib.pyplot import show
import os

from pyairpar.core.free_point import FreePoint
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
from pyairpar.core.airfoil import Airfoil
from pyairpar.symmetric.symmetric_airfoil import SymmetricAirfoil
from pyairpar.core.param import Param
from pyairpar.core.param_setup import ParamSetup
from pyairpar.core.parametrization import AirfoilParametrization
from pyairpar.core.parametrization import rotate


def _generate_unlinked_param_dict():
    param_dict = {
        # Main chord length:
        'c_main': Param(1.0, active=False),
        # Angles of attack
        'alf_main': Param(np.deg2rad(2.0), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
        'alf_flap': Param(np.deg2rad(30.0), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
    }
    param_dict = {**param_dict, **{
        # Flap chord length:
        'c_flap': Param(0.3, 'length', bounds=[0.25, 0.35], scale_value=param_dict['c_main'].value),
        # Leading edge radius:
        'R_le_main': Param(0.03, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'R_le_flap': Param(0.01, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        # Leading edge length:
        'L_le_main': Param(0.081, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'L_le_flap': Param(0.0172, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        # Leading edge length ratio:
        'r_le_main': Param(0.6, bounds=[0.01, 0.99]),
        'r_le_flap': Param(0.5, bounds=[0.01, 0.99]),
        # Upper curvature control arm angle:
        'psi1_le_main': Param(np.deg2rad(20.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi1_le_flap': Param(np.deg2rad(10.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Lower curvature control arm angle (upper and lower angles on hub are equal due to symmetry):
        'psi2_le_main': Param(np.deg2rad(10.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi2_le_flap': Param(np.deg2rad(10.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Upper trailing edge length:
        'L1_te_main': Param(0.0519, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        'L1_te_flap': Param(0.138, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        # Lower trailing edge length:
        'L2_te_main': Param(0.0997, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        'L2_te_flap': Param(0.15, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        # Upper trailing edge angle:
        'theta1_te_main': Param(np.deg2rad(0.8), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        'theta1_te_flap': Param(np.deg2rad(1.0), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        # Lower trailing edge angle:
        'theta2_te_main': Param(np.deg2rad(2.4), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        'theta2_te_flap': Param(np.deg2rad(1.0), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        # Anchor point lengths:
        'L_ap_flap_le_main': Param(0.012, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        'L_ap_flap_end_main': Param(0.02, 'length', bounds=[0.001, 0.1], scale_value=param_dict['c_main'].value),
        'L_ap_flap_end_flap': Param(0.019, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        # Anchor point curvature:
        'kappa_ap_flap_end_flap': Param(1 / 1.0, 'inverse-length', bounds=[-1e2, 1e2],
                                  scale_value=param_dict['c_main'].value),
        # Anchor point length ratios:
        'r_ap_fan_upper_main': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_inlet_upper_main': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_mt_main': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_fan_upper_hub': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_ne_upper_hub': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_mt_nacelle': Param(0.5, bounds=[0.01, 0.99]),
        # Anchor point neighboring point line angles:
        'phi_ap_fan_upper_main': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_inlet_upper_main': Param(np.deg2rad(-10.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_mt_main': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_fan_upper_hub': Param(np.deg2rad(15.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_ne_upper_hub': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_mt_nacelle': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Anchor point aft curvature control arm angles:
        'psi1_ap_fan_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_inlet_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_mt_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_fan_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_ne_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_mt_nacelle': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        # Anchor point fore curvature control arm angles:
        'psi2_ap_fan_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_inlet_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_mt_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_fan_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_ne_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_mt_nacelle': Param(np.deg2rad(90.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        # Free point locations:
        'x_fp1_upper_main': Param(0.575, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_upper_main': Param(0.19, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        'x_fp1_lower_main': Param(0.49, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_lower_main': Param(-0.085, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        'parametrization_dictionary_name': 'v00'
    }}
    return param_dict


def _generate_linked_param_dict(param_dict):
    param_dict = {**param_dict, **{
        'kappa_ap_flap_le_main': Param(-1 / param_dict['R_le_flap'].value, linked=True),
        'kappa_ap_flap_end_main': Param(-param_dict['kappa_ap_flap_end_flap'].value, linked=True),
    }}
    param_dict = {**param_dict, **{

    }}
    param_dict['r_hne'].value = param_dict['r_hne'].value * param_dict['alpha_r_hne'].value
    return param_dict


def _generate_param_dict():
    param_dict = _generate_unlinked_param_dict()
    param_dict = _generate_linked_param_dict(param_dict)
    return param_dict


def _generate_airfoils(param_dict):
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

    x = param_dict['x_ap_fan_upper_main'].value
    y = param_dict['y_ap_fan_upper_main'].value
    x -= param_dict['dx_main'].value
    y -= param_dict['dy_main'].value
    x, y = rotate(x, y, param_dict['alf_main'].value)

    ap_fan_upper_main = AnchorPoint(x=Param(x, linked=True),
                                    y=Param(y, linked=True),
                                    name='ap_fan_upper_main',
                                    previous_anchor_point='te_1',
                                    L=param_dict['L_ap_fan_upper_main'],
                                    R=param_dict['R_ap_fan_upper_main'],
                                    r=param_dict['r_ap_fan_upper_main'],
                                    phi=param_dict['phi_ap_fan_upper_main'],
                                    psi1=param_dict['psi1_ap_fan_upper_main'],
                                    psi2=param_dict['psi2_ap_fan_upper_main'])

    x = param_dict['x_ap_inlet_upper_main'].value
    y = param_dict['y_ap_inlet_upper_main'].value
    x -= param_dict['dx_main'].value
    y -= param_dict['dy_main'].value
    x, y = rotate(x, y, param_dict['alf_main'].value)

    ap_inlet_upper_main = AnchorPoint(x=Param(x, linked=True),
                                      y=Param(y, linked=True),
                                      name='ap_inlet_upper_main',
                                      previous_anchor_point='ap_fan_upper_main',
                                      L=param_dict['L_ap_inlet_upper_main'],
                                      R=param_dict['R_ap_inlet_upper_main'],
                                      r=param_dict['r_ap_inlet_upper_main'],
                                      phi=param_dict['phi_ap_inlet_upper_main'],
                                      psi1=param_dict['psi1_ap_inlet_upper_main'],
                                      psi2=param_dict['psi2_ap_inlet_upper_main'],
                                      )

    x = param_dict['x_ap_mt_main'].value
    y = param_dict['y_ap_mt_main'].value
    x -= param_dict['dx_main'].value
    y -= param_dict['dy_main'].value
    x, y = rotate(x, y, param_dict['alf_main'].value)

    ap_mt_main = AnchorPoint(x=Param(x, linked=True),
                             y=Param(y, linked=True),
                             name='ap_mt_main',
                             previous_anchor_point='le',
                             L=param_dict['L_ap_mt_main'],
                             R=param_dict['R_ap_mt_main'],
                             r=param_dict['r_ap_mt_main'],
                             phi=param_dict['phi_ap_mt_main'],
                             psi1=param_dict['psi1_ap_mt_main'],
                             psi2=param_dict['psi2_ap_mt_main'],
                             )

    anchor_point_tuple_main = (ap_fan_upper_main, ap_inlet_upper_main, ap_mt_main)

    fp1_upper_main = FreePoint(x=param_dict['x_fp1_upper_main'],
                               y=param_dict['y_fp1_upper_main'],
                               previous_anchor_point='ap_inlet_upper_main')

    fp1_lower_main = FreePoint(x=param_dict['x_fp1_lower_main'],
                               y=param_dict['y_fp1_lower_main'],
                               previous_anchor_point='le')

    free_point_tuple_main = (fp1_upper_main, fp1_lower_main)

    x = param_dict['x_ap_mt_nacelle'].value
    y = param_dict['y_ap_mt_nacelle'].value
    x -= param_dict['dx_nacelle'].value
    y -= param_dict['dy_nacelle'].value
    x, y = rotate(x, y, param_dict['alf_nacelle'].value)

    ap_mt_nacelle = AnchorPoint(x=Param(x, linked=True),
                                y=Param(y, linked=True),
                                name='ap_mt_nacelle',
                                previous_anchor_point='te_1',
                                L=param_dict['L_ap_mt_nacelle'],
                                R=param_dict['R_ap_mt_nacelle'],
                                r=param_dict['r_ap_mt_nacelle'],
                                phi=param_dict['phi_ap_mt_nacelle'],
                                psi1=param_dict['psi1_ap_mt_nacelle'],
                                psi2=param_dict['psi2_ap_mt_nacelle'],
                                )

    anchor_point_tuple_nacelle = (ap_mt_nacelle,)

    airfoil_main = Airfoil(number_coordinates=100,
                           base_airfoil_params=base_airfoil_params_main,
                           anchor_point_tuple=anchor_point_tuple_main,
                           free_point_tuple=free_point_tuple_main)

    airfoil_flap = Airfoil(number_coordinates=100,
                              base_airfoil_params=base_airfoil_params_flap,
                              anchor_point_tuple=anchor_point_tuple_nacelle,
                              free_point_tuple=())

    airfoil_tuple = (airfoil_main, airfoil_flap)
    return airfoil_tuple


def update(parametrization: AirfoilParametrization, parameter_list: list = None):
    if parameter_list is not None:
        parametrization.override_parameters(parameter_list, normalized=True)
    else:
        parametrization.generate_airfoils()

    theta = -parametrization.param_setup.param_dict['alf_hub'].value
    xy_axis = np.array([parametrization.param_setup.param_dict['fan_center_x_over_c_main'].value,
                        parametrization.param_setup.param_dict['fan_center_y_over_c_main'].value])
    parametrization.mirror(axis=(theta, xy_axis), fixed_airfoil_idx=0, linked_airfoil_idx=2,
                           fixed_anchor_point_range=('ap_fan_upper_main', 'ap_inlet_upper_main'),
                           starting_prev_anchor_point_str_linked='le')


def run():
    param_setup = ParamSetup(_generate_unlinked_param_dict, _generate_linked_param_dict)
    parametrization = AirfoilParametrization(param_setup=param_setup,
                                             _generate_airfoils=_generate_airfoils)
    update(parametrization, None)

    fig, axs = parametrization.airfoil_tuple[0].plot(
        ('airfoil', 'control-point-skeleton'),
        show_plot=False, show_legend=False)

    parametrization.airfoil_tuple[1].plot(('airfoil', 'control-point-skeleton'),
                                          fig=fig, axs=axs, show_plot=False, show_legend=False)

    parametrization.airfoil_tuple[2].plot(('airfoil', 'control-point-skeleton'),
                                          fig=fig, axs=axs, show_plot=False, show_legend=False)

    airfoil_line_proxy = Line2D([], [], color='cornflowerblue')
    control_point_skeleton_proxy = Line2D([], [], color='grey', ls='--', marker='*')
    fig.legend([airfoil_line_proxy, control_point_skeleton_proxy],
               ['airfoil', 'control polygon'], fontsize=12)

    fig.suptitle('')
    axs.set_xlabel(r'$x/c$', fontsize=14)
    axs.set_ylabel(r'$y/c$', fontsize=14)
    fig.tight_layout()

    show_flag = True
    save_flag = False

    if save_flag:
        save_name = os.path.join(os.path.dirname(
            os.path.dirname(os.path.join(os.getcwd()))), 'docs', 'images', 'high_lift.png')
        fig.savefig(save_name, dpi=600)
    if show_flag:
        show()


if __name__ == '__main__':
    run()
