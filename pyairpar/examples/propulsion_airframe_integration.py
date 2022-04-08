import numpy as np

from pyairpar.core.free_point import FreePoint
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
from pyairpar.core.airfoil import Airfoil
from pyairpar.symmetric.symmetric_airfoil import SymmetricAirfoil
from pyairpar.core.param import Param
from pyairpar.core.parametrization import AirfoilParametrization
from pyairpar.core.parametrization import rotate


def run():
    param_dict = {
        # Main chord length:
        'c_main': Param(1.0, active=False),
        # Angles of attack
        'alf_main': Param(np.deg2rad(3.663), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
        'alf_hub': Param(np.deg2rad(2.0), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
        'alf_nacelle': Param(np.deg2rad(-3.67), bounds=[np.deg2rad(-15), np.deg2rad(15)]),
        # Hub-to-tip ratio of the fan:
        'r_htf': Param(0.47, active=False),
        # Hub-to-nozzle-exit diameter ratio (set by desired fan-face Mach number)
        'r_hne': Param(0.608, linked=True),
        # Hub-to-nozzle-exit diameter ratio multiplier (accounts for lack of accuracy in quasi-1D assumption)
        'alpha_r_hne': Param(0.9738, bounds=[0.9, 1.1]),
    }
    param_dict = {**param_dict, **{
        # Nacelle chord length:
        'c_nacelle': Param(0.2866, 'length', bounds=[0.25, 0.35], scale_value=param_dict['c_main'].value),
        # Distance from fan center to tip of nose along hub chordline:
        'l_nose': Param(0.048, 'length', bounds=[0.01, 0.08], scale_value=param_dict['c_main'].value),
        # Distance from fan center to nozzle exit center along hub chordline:
        'l_fne': Param(0.1473, 'length', bounds=[0.1, 0.2], scale_value=param_dict['c_main'].value),
        # Distance from nozzle exit center to hub trailing edge along hub chordline (pressure recovery distance):
        'l_pr': Param(0.352, 'length', bounds=[0.15, 0.5], scale_value=param_dict['c_main'].value),
        # Distance from tip of nose to line perpendicular to hub chordline and passing through inlet anchor points:
        'l_n2inlet': Param(0.03, 'length', bounds=[0.01, 0.07], scale_value=param_dict['c_main'].value),
        # Fan diameter:
        'd_f': Param(0.2, 'length', bounds=[0.17, 0.23], scale_value=param_dict['c_main'].value),
        # Nozzle exit diameter:
        'd_ne': Param(0.202, 'length', bounds=[0.17, 0.23], scale_value=param_dict['c_main'].value),
        # Inlet diameter:
        'd_inlet': Param(0.165, 'length', bounds=[0.13, 0.20], scale_value=param_dict['c_main'].value),
        # Minimum thickness:
        't_mt_main': Param(0.006, 'length', bounds=[0.001, 0.1], scale_value=param_dict['c_main'].value),
        't_mt_nacelle': Param(0.003, 'length', bounds=[0.001, 0.1], scale_value=param_dict['c_main'].value),
    }}
    param_dict = {**param_dict, **{
        'fan_center_x_over_c_main': Param(param_dict['c_main'].value * np.cos(-param_dict['alf_main'].value) +
                                          param_dict['d_ne'].value / 2 * np.cos(np.pi / 2 -
                                                                                param_dict['alf_main'].value) -
                                          param_dict['l_fne'].value * np.cos(-param_dict['alf_hub'].value),
                                          linked=True),
        'fan_center_y_over_c_main': Param(param_dict['c_main'].value * np.sin(-param_dict['alf_main'].value) +
                                          param_dict['d_ne'].value / 2 * np.sin(np.pi / 2 -
                                                                                param_dict['alf_main'].value) -
                                          param_dict['l_fne'].value * np.sin(-param_dict['alf_hub'].value),
                                          linked=True),
    }}
    param_dict = {**param_dict, **{
        # Leading edge radius:
        'R_le_main': Param(0.03, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'R_le_hub': Param(0.099, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'R_le_nacelle': Param(0.008, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        # Leading edge length:
        'L_le_main': Param(0.081, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'L_le_hub': Param(0.038, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        'L_le_nacelle': Param(0.0172, 'length', bounds=[0.001, 0.2], scale_value=param_dict['c_main'].value),
        # Leading edge length ratio (0.5 for hub due to symmetry):
        'r_le_main': Param(0.595, bounds=[0.01, 0.99]),
        'r_le_nacelle': Param(0.8, bounds=[0.01, 0.99]),
        # Leading edge tilt (no tilt on hub due to symmetry):
        'phi_le_main': Param(np.deg2rad(0.02), bounds=[np.deg2rad(-20), np.deg2rad(20)]),
        'phi_le_nacelle': Param(np.deg2rad(5.32), bounds=[np.deg2rad(-20), np.deg2rad(20)]),
        # Upper curvature control arm angle:
        'psi1_le_main': Param(np.deg2rad(0.974), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi1_le_hub': Param(np.deg2rad(22.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi1_le_nacelle': Param(np.deg2rad(21.03), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Lower curvature control arm angle (upper and lower angles on hub are equal due to symmetry):
        'psi2_le_main': Param(np.deg2rad(5.328), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'psi2_le_nacelle': Param(np.deg2rad(21.03), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Upper trailing edge length:
        'L1_te_main': Param(0.0519, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        'L1_te_hub': Param(0.104, 'length', scale_value=param_dict['c_main'].value),
        'L1_te_nacelle': Param(0.138, 'length', scale_value=param_dict['c_main'].value),
        # Lower trailing edge length (equal to upper trailing edge length on hub due to symmetry, and nacelle trailing
        # edge length is equal to the main upper trailing edge length due to the axisymmetric condition):
        'L2_te_main': Param(0.0997, 'length', bounds=[0.001, 0.5], scale_value=param_dict['c_main'].value),
        # Upper trailing edge angle:
        'theta1_te_main': Param(np.deg2rad(0.8), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        'theta1_te_hub': Param(np.deg2rad(27.33), bounds=[np.deg2rad(0.0), np.deg2rad(45)]),
        'theta1_te_nacelle': Param(np.deg2rad(8.59), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        # Lower trailing edge angle (equal to upper trailing edge angle on hub due to symmetry, and nacelle trailing
        # edge angle is missing here because it is linked to the main upper trailing edge angle by the axisymmetric
        # condition):
        'theta2_te_main': Param(np.deg2rad(2.4), bounds=[np.deg2rad(0.0), np.deg2rad(15)]),
        # Trailing edge thickness:
        't_te_main': Param(0.0, 'length', active=False),
        't_te_hub': Param(0.0, 'length', active=False),
        't_te_nacelle': Param(0.0, 'length', active=False),
        # Trailing edge thickness ratio (0.5 on hub due to symmetry):
        'r_te_main': Param(0.5, active=False),
        'r_te_nacelle': Param(0.5, active=False),
        # Trailing edge tilt (no tilt on hub due to symmetry):
        'phi_te_main': Param(np.deg2rad(0.0), active=False),
        'phi_te_nacelle': Param(np.deg2rad(0.0), active=False),
        # Translations:
        'dx_main': Param(0.0, active=False),
        'dx_hub': Param(param_dict['fan_center_x_over_c_main'].value - param_dict['l_nose'].value *
                        np.cos(-param_dict['alf_hub'].value), linked=True),
        'dx_nacelle': Param(param_dict['c_main'].value *
                            np.cos(-param_dict['alf_main'].value) + param_dict['d_ne'].value *
                            np.cos(np.pi / 2 - param_dict['alf_hub'].value) - param_dict['c_nacelle'].value *
                            np.cos(-param_dict['alf_nacelle'].value), linked=True),
        'dy_main': Param(0.0, active=False),
        'dy_hub': Param(param_dict['fan_center_y_over_c_main'].value - param_dict['l_nose'].value *
                        np.sin(-param_dict['alf_hub'].value), linked=True),
        'dy_nacelle': Param(param_dict['c_main'].value *
                            np.sin(-param_dict['alf_main'].value) + param_dict['d_ne'].value *
                            np.sin(np.pi / 2 - param_dict['alf_hub'].value) - param_dict['c_nacelle'].value *
                            np.sin(-param_dict['alf_nacelle'].value), linked=True),
        # Anchor point lengths:
        'L_ap_fan_upper_main': Param(0.012, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        'L_ap_inlet_upper_main': Param(0.02, 'length', bounds=[0.001, 0.1], scale_value=param_dict['c_main'].value),
        'L_ap_mt_main': Param(0.019, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        'L_ap_fan_upper_hub': Param(0.025, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        'L_ap_ne_upper_hub': Param(0.025, 'length', bounds=[0.001, 0.05], scale_value=param_dict['c_main'].value),
        # Anchor point radii of curvature:
        'R_ap_fan_upper_main': Param(-86.6, 'length', bounds=[0.001, 1e3], scale_value=param_dict['c_main'].value),
        'R_ap_inlet_upper_main': Param(50, 'length', bounds=[0.001, 1e3], scale_value=param_dict['c_main'].value),
        'R_ap_mt_main': Param(1.0, 'length', bounds=[0.001, 1e3], scale_value=param_dict['c_main'].value),
        'R_ap_fan_upper_hub': Param(100, 'length', bounds=[0.001, 1e3], scale_value=param_dict['c_main'].value),
        'R_ap_ne_upper_hub': Param(100, 'length', bounds=[0.001, 1e3], scale_value=param_dict['c_main'].value),
        # Anchor point length ratios:
        'r_ap_fan_upper_main': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_inlet_upper_main': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_mt_main': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_fan_upper_hub': Param(0.5, bounds=[0.01, 0.99]),
        'r_ap_ne_upper_hub': Param(0.5, bounds=[0.01, 0.99]),
        # Anchor point neighboring point line angles:
        'phi_ap_fan_upper_main': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_inlet_upper_main': Param(np.deg2rad(-10.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_mt_main': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_fan_upper_hub': Param(np.deg2rad(15.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        'phi_ap_ne_upper_hub': Param(np.deg2rad(0.0), bounds=[np.deg2rad(-45), np.deg2rad(45)]),
        # Anchor point aft curvature control arm angles:
        'psi1_ap_fan_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_inlet_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_mt_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_fan_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi1_ap_ne_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        # Anchor point fore curvature control arm angles:
        'psi2_ap_fan_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_inlet_upper_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_mt_main': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_fan_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        'psi2_ap_ne_upper_hub': Param(np.deg2rad(95.0), bounds=[np.deg2rad(10), np.deg2rad(170)]),
        # Anchor point locations:
        'x_ap_fan_upper_main': Param(param_dict['fan_center_x_over_c_main'].value - param_dict['d_f'].value / 2 *
                                     np.cos(np.pi / 2 - param_dict['alf_hub'].value), linked=True),
        'y_ap_fan_upper_main': Param(param_dict['fan_center_y_over_c_main'].value - param_dict['d_f'].value / 2 *
                                     np.sin(np.pi / 2 - param_dict['alf_hub'].value), linked=True),
        'x_ap_inlet_upper_main': Param(param_dict['fan_center_x_over_c_main'].value - (param_dict['l_n2inlet'].value +
                                                                                       param_dict['l_nose'].value) *
                                       np.cos(-param_dict['alf_hub'].value) - param_dict['d_inlet'].value / 2 *
                                       np.cos(np.pi / 2 - param_dict['alf_hub'].value)),
        'y_ap_inlet_upper_main': Param(param_dict['fan_center_y_over_c_main'].value - (param_dict['l_n2inlet'].value +
                                                                                       param_dict['l_nose'].value) *
                                       np.sin(-param_dict['alf_hub'].value) - param_dict['d_inlet'].value / 2 *
                                       np.sin(np.pi / 2 - param_dict['alf_hub'].value)),
        'x_ap_mt_main': Param(param_dict['fan_center_x_over_c_main'].value - param_dict['d_f'].value / 2 *
                              np.cos(np.pi / 2 - param_dict['alf_hub'].value) - param_dict['t_mt_main'].value *
                              np.cos(np.pi / 2 - param_dict['alf_hub'].value), linked=True),
        'y_ap_mt_main': Param(param_dict['fan_center_y_over_c_main'].value - param_dict['d_f'].value / 2 *
                              np.sin(np.pi / 2 - param_dict['alf_hub'].value) - param_dict['t_mt_main'].value *
                              np.sin(np.pi / 2 - param_dict['alf_hub'].value), linked=True),
        'x_ap_fan_upper_hub': Param(param_dict['fan_center_x_over_c_main'].value + param_dict['r_htf'].value *
                                    param_dict['d_f'].value / 2 * np.cos(np.pi / 2 - param_dict['alf_hub'].value)),
        'y_ap_fan_upper_hub': Param(param_dict['fan_center_y_over_c_main'].value + param_dict['r_htf'].value *
                                    param_dict['d_f'].value / 2 * np.sin(np.pi / 2 - param_dict['alf_hub'].value)),
        'x_ap_ne_upper_hub': Param(param_dict['fan_center_x_over_c_main'].value + param_dict['l_fne'].value *
                                   np.cos(-param_dict['alf_hub'].value) + param_dict['d_ne'].value *
                                   param_dict['r_hne'].value / 2 * np.cos(np.pi / 2 - param_dict['alf_hub'].value)),
        'y_ap_ne_upper_hub': Param(param_dict['fan_center_y_over_c_main'].value + param_dict['l_fne'].value *
                                   np.sin(-param_dict['alf_hub'].value) + param_dict['d_ne'].value *
                                   param_dict['r_hne'].value / 2 * np.sin(np.pi / 2 - param_dict['alf_hub'].value)),
        # Free point locations:
        'x_fp1_upper_main': Param(0.575, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_upper_main': Param(0.07, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
        'x_fp1_lower_main': Param(0.49, 'length', bounds=[0.1, 0.9], scale_value=param_dict['c_main'].value),
        'y_fp1_lower_main': Param(-0.045, 'length', bounds=[-0.2, 0.2], scale_value=param_dict['c_main'].value),
    }}
    param_dict = {**param_dict, **{
        'L2_te_nacelle': Param(param_dict['L1_te_main'].value, linked=True),
        'theta2_te_nacelle': Param(param_dict['theta1_te_main'].value - param_dict['alf_hub'].value, linked=True),
        'c_hub': Param(param_dict['l_nose'].value + param_dict['l_fne'].value + param_dict['l_pr'].value, linked=True),
    }}
    param_dict = {**param_dict, **{
        'parametrization_dictionary_name': 'v00'
    }}
    param_dict['r_hne'].value = param_dict['r_hne'].value * param_dict['alpha_r_hne'].value

    base_airfoil_params_main = \
        BaseAirfoilParams(c=param_dict['c_main'], alf=param_dict['alf_main'], R_le=param_dict['R_le_main'],
                          L_le=param_dict['L_le_main'], r_le=param_dict['r_le_main'], phi_le=param_dict['phi_le_main'],
                          psi1_le=param_dict['psi1_le_main'], psi2_le=param_dict['psi2_le_main'],
                          L1_te=param_dict['L1_te_main'], L2_te=param_dict['L2_te_main'],
                          theta1_te=param_dict['theta1_te_main'], theta2_te=param_dict['theta2_te_main'],
                          t_te=param_dict['t_te_main'], r_te=param_dict['r_te_main'],
                          phi_te=param_dict['phi_te_main'], dx=param_dict['dx_main'], dy=param_dict['dy_main'],
                          non_dim_by_chord=True)

    base_airfoil_params_hub = \
        SymmetricBaseAirfoilParams(c=param_dict['c_hub'], alf=param_dict['alf_hub'], R_le=param_dict['R_le_hub'],
                                   L_le=param_dict['L_le_hub'], psi1_le=param_dict['psi1_le_hub'],
                                   L1_te=param_dict['L1_te_hub'], theta1_te=param_dict['theta1_te_hub'],
                                   t_te=param_dict['t_te_hub'], dx=param_dict['dx_hub'], dy=param_dict['dy_hub'],
                                   non_dim_by_chord=False)

    base_airfoil_params_nacelle = \
        BaseAirfoilParams(c=param_dict['c_nacelle'], alf=param_dict['alf_nacelle'], R_le=param_dict['R_le_nacelle'],
                          L_le=param_dict['L_le_nacelle'], r_le=param_dict['r_le_nacelle'],
                          phi_le=param_dict['phi_le_nacelle'], psi1_le=param_dict['psi1_le_nacelle'],
                          psi2_le=param_dict['psi2_le_nacelle'], L1_te=param_dict['L1_te_nacelle'],
                          L2_te=param_dict['L2_te_nacelle'], theta1_te=param_dict['theta1_te_nacelle'],
                          theta2_te=param_dict['theta2_te_nacelle'], t_te=param_dict['t_te_nacelle'],
                          r_te=param_dict['r_te_nacelle'], phi_te=param_dict['phi_te_nacelle'],
                          dx=param_dict['dx_nacelle'], dy=param_dict['dy_nacelle'], non_dim_by_chord=False)

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

    x = param_dict['x_ap_ne_upper_hub'].value
    y = param_dict['y_ap_ne_upper_hub'].value
    x -= param_dict['dx_hub'].value
    y -= param_dict['dy_hub'].value
    x, y = rotate(x, y, param_dict['alf_hub'].value)

    ap_ne_upper_hub = AnchorPoint(x=Param(x, linked=True),
                                   y=Param(y, linked=True),
                                   name='ap_ne_upper_hub',
                                   previous_anchor_point='te_1',
                                   L=param_dict['L_ap_ne_upper_hub'],
                                   R=param_dict['R_ap_ne_upper_hub'],
                                   r=param_dict['r_ap_ne_upper_hub'],
                                   phi=param_dict['phi_ap_ne_upper_hub'],
                                   psi1=param_dict['psi1_ap_ne_upper_hub'],
                                   psi2=param_dict['psi2_ap_ne_upper_hub'])

    x = param_dict['x_ap_fan_upper_hub'].value
    y = param_dict['y_ap_fan_upper_hub'].value
    x -= param_dict['dx_hub'].value
    y -= param_dict['dy_hub'].value
    x, y = rotate(x, y, param_dict['alf_hub'].value)

    ap_fan_upper_hub = AnchorPoint(x=Param(x, linked=True),
                                   y=Param(y, linked=True),
                                   name='ap_fan_upper_hub',
                                   previous_anchor_point='ap_ne_upper_hub',
                                   L=param_dict['L_ap_fan_upper_hub'],
                                   R=param_dict['R_ap_fan_upper_hub'],
                                   r=param_dict['r_ap_fan_upper_hub'],
                                   phi=param_dict['phi_ap_fan_upper_hub'],
                                   psi1=param_dict['psi1_ap_fan_upper_hub'],
                                   psi2=param_dict['psi2_ap_fan_upper_hub'])

    anchor_point_tuple_hub = (ap_ne_upper_hub, ap_fan_upper_hub)
    free_point_tuple_hub = ()

    airfoil_main = Airfoil(number_coordinates=100,
                           base_airfoil_params=base_airfoil_params_main,
                           anchor_point_tuple=anchor_point_tuple_main,
                           free_point_tuple=free_point_tuple_main)

    airfoil_hub = SymmetricAirfoil(number_coordinates=100,
                                   base_airfoil_params=base_airfoil_params_hub,
                                   anchor_point_tuple=anchor_point_tuple_hub,
                                   free_point_tuple=free_point_tuple_hub)

    airfoil_nacelle = Airfoil(number_coordinates=100,
                              base_airfoil_params=base_airfoil_params_nacelle,
                              anchor_point_tuple=(),
                              free_point_tuple=())

    parametrization = AirfoilParametrization((airfoil_main, airfoil_hub, airfoil_nacelle))
    m = Param(np.tan(param_dict['alf_hub'].value), linked=True)
    b = Param(0.02, 'length', scale_value=param_dict['c_main'].value)
    parametrization.mirror(axis=(m.value, b.value), fixed_airfoil_idx=0, linked_airfoil_idx=2,
                           fixed_anchor_point_range=('ap_fan_upper_main', 'ap_inlet_upper_main'),
                           starting_prev_anchor_point_str_linked='le')

    parametrization.extract_parameters()
    print(parametrization.params)
    print(parametrization.bounds)
    print(len(parametrization.params))
    print(len(parametrization.bounds))

    print(f"fixed_airfoil_N = {airfoil_main.N}, linked_airfoil_N = {airfoil_nacelle.N}")

    fig, axs = parametrization.airfoil_tuple[0].plot(
        ('airfoil', 'control-point-skeleton'),
        show_plot=False, show_legend=False)

    # axs.plot([0, 1.4], [0.11, 0.11 - np.tan(hub_alf.value) * 1.4], color='orange', label='axis')

    parametrization.airfoil_tuple[1].plot(('airfoil', 'control-point-skeleton'),
                                          fig=fig, axs=axs, show_plot=False, show_legend=False)

    parametrization.airfoil_tuple[2].plot(('airfoil', 'control-point-skeleton'),
                                          fig=fig, axs=axs)


if __name__ == '__main__':
    run()
