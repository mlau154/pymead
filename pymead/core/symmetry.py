from pymead.core.line import InfiniteLine
from pymead.utils.transformations import transform_matrix
import numpy as np


def symmetry(param_name: str, x=None, y=None, alf_target=None, alf_tool=None, c_target=None, c_tool=None,
             dx_target=None, dx_tool=None, dy_target=None, dy_tool=None, upper_target=None, upper_tool=None,
             phi=None, psi1=None, psi2=None, r=None, L=None, R=None, x1=None, y1=None, x2=None, y2=None, m=None,
             theta_rad=None, theta_deg=None):
    new_x, new_y, new_xp, new_yp, rel_phi_target = None, None, None, None, None
    if param_name in ['x', 'y', 'phi']:
        inf_line = InfiniteLine(x1=x1, y1=y1, x2=x2, y2=y2, m=m, theta_rad=theta_rad, theta_deg=theta_deg)
        if param_name in ['x', 'y']:
            new_xpyp = transform_matrix(np.array([[x, y]]), dx_tool, dy_tool, -alf_tool, c_tool,
                                      ['scale', 'rotate', 'translate'])
            xp = new_xpyp[0][0]
            yp = new_xpyp[0][1]
            std_coeffs = inf_line.get_standard_form_coeffs()
            distance = (std_coeffs['A'] * xp + std_coeffs['B'] * yp + std_coeffs['C']
                        ) / np.hypot(std_coeffs['A'], std_coeffs['B'])
            over_under_value = y1 - yp - inf_line.m * (x1 - xp)
            if over_under_value > 0:
                angle = inf_line.theta_rad + np.pi / 2
            elif over_under_value < 0:
                angle = inf_line.theta_rad - np.pi / 2
            else:
                angle = 0.0
                distance = 0.0
            distance = abs(distance)
            new_xp = xp + 2 * distance * np.cos(angle)
            new_yp = yp + 2 * distance * np.sin(angle)
            new_xy = transform_matrix(np.array([[new_xp, new_yp]]), -dx_target, -dy_target, alf_tool, 1 / c_target,
                                      ['translate', 'rotate', 'scale'])
            new_x = new_xy[0][0]
            new_y = new_xy[0][1]
        else:
            if upper_tool:
                abs_phi_tool = phi + (-alf_tool)
            else:
                abs_phi_tool = -phi + (-alf_tool)
            delta_angle = abs_phi_tool - inf_line.theta_rad
            abs_phi_target = inf_line.theta_rad - delta_angle
            if upper_target:
                rel_phi_target = abs_phi_target + (-alf_target)
            else:
                rel_phi_target = -abs_phi_target + (-alf_target)

    output_dict = {
        'x': new_x,
        'y': new_y,
        'phi': rel_phi_target,
        'psi1': psi1,
        'psi2': psi2,
        'r': r,
        'L': L,
        'R': R,
    }
    return output_dict[param_name]


def _test():
    from pymead.core.param import Param
    from pymead.core.airfoil import Airfoil
    from pymead.core.mea import MEA
    from pymead.core.base_airfoil_params import BaseAirfoilParams
    from pymead.core.free_point import FreePoint
    import matplotlib.pyplot as plt
    mea = MEA()
    mea.add_airfoil(Airfoil(base_airfoil_params=BaseAirfoilParams(dx=Param(0.3), dy=Param(0.5))), 1, None)
    fp1 = FreePoint(x=Param(0.3), y=Param(0.1), previous_anchor_point='te_1', airfoil_tag='A0')
    fp2 = FreePoint(x=Param(0.5), y=Param(-0.1), previous_anchor_point='le', airfoil_tag='A1')
    mea.add_custom_parameters({'x1': {'value': 0.1}, 'y1': {'value': 0.1}, 'alf': {'value': -0.08}})
    mea.airfoils['A0'].insert_free_point(fp1)
    mea.airfoils['A1'].insert_free_point(fp2)
    fig, axs = plt.subplots()
    axs.plot(fp1.xp.value, fp1.yp.value, color='indianred', label='fp1', marker='o', ls='none')
    axs.plot(fp2.xp.value, fp2.yp.value, color='blue', label='fp2_original', marker='x', ls='none')
    fp2.xp.mea = mea
    fp2.xp.function_dict['symmetry'] = symmetry
    fp2.xp.function_dict['param_name'] = 'xp'
    fp2.xp.set_func_str('symmetry($A0.FreePoints.te_1.FP0.xp, $A0.FreePoints.te_1.FP0.yp, param_name, '
                        'x1=$Custom.x1, y1=$Custom.y1, theta_rad=$Custom.alf)')
    fp2.xp.update(show_q_error_messages=False)
    fp2.yp.mea = mea
    fp2.yp.function_dict['symmetry'] = symmetry
    fp2.yp.function_dict['param_name'] = 'yp'
    fp2.yp.set_func_str('symmetry($A0.FreePoints.te_1.FP0.xp, $A0.FreePoints.te_1.FP0.yp, param_name, '
                        'x1=$Custom.x1, y1=$Custom.y1, theta_rad=$Custom.alf)')
    fp2.yp.update(show_q_error_messages=False)
    axs.plot(fp2.xp.value, fp2.yp.value, color='green', label='fp2_final', marker='d', ls='none')
    axs.plot([0.1, 0.1 + np.cos(-0.08)], [0.1, 0.1 + np.sin(-0.08)], color='gold', label='symmetry line')
    axs.set_aspect('equal')
    axs.legend()
    plt.show()
    pass


if __name__ == '__main__':
    _test()
