from pymead.core.line import InfiniteLine
import numpy as np


def symmetry(xp, yp, param_name: str, x1=None, y1=None, x2=None, y2=None, m=None, theta_rad=None, theta_deg=None):
    inf_line = InfiniteLine(x1=x1, y1=y1, x2=x2, y2=y2, m=m, theta_rad=theta_rad, theta_deg=theta_deg)
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

    if param_name == 'xp':
        target_param_value = xp + 2 * distance * np.cos(angle)
    elif param_name == 'yp':
        target_param_value = yp + 2 * distance * np.sin(angle)
    else:
        raise ValueError('Invalid value for param_name')
    return target_param_value


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
