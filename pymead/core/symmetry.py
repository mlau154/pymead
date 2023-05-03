from pymead.core.line import InfiniteLine
from pymead.utils.transformations import transform_matrix
import numpy as np


def symmetry(name: str, xy=None, alf_target=None, alf_tool=None, c_target=None, c_tool=None,
             dx_target=None, dx_tool=None, dy_target=None, dy_tool=None, upper_target=None, upper_tool=None,
             phi=None, psi1=None, psi2=None, r=None, L=None, R=None, x1=None, y1=None, x2=None, y2=None, m=None,
             theta_rad=None, theta_deg=None):
    new_x, new_y, rel_phi_target = None, None, None
    if name in ['xy', 'phi']:
        inf_line = InfiniteLine(x1=x1, y1=y1, x2=x2, y2=y2, m=m, theta_rad=theta_rad, theta_deg=theta_deg)
        if name == 'xy':
            std_coeffs = inf_line.get_standard_form_coeffs()
            distance = (std_coeffs['A'] * xy[0] + std_coeffs['B'] * xy[1] + std_coeffs['C']
                        ) / np.hypot(std_coeffs['A'], std_coeffs['B'])
            over_under_value = y1 - xy[1] - inf_line.m * (x1 - xy[0])
            if over_under_value > 0:
                angle = inf_line.theta_rad + np.pi / 2
            elif over_under_value < 0:
                angle = inf_line.theta_rad - np.pi / 2
            else:
                angle = 0.0
                distance = 0.0
            distance = abs(distance)
            new_x = xy[0] + 2 * distance * np.cos(angle)
            new_y = xy[1] + 2 * distance * np.sin(angle)
        else:
            if upper_tool:
                abs_phi_tool = phi + (-alf_tool)
            else:
                abs_phi_tool = -phi + (-alf_tool)
            delta_angle = abs_phi_tool - inf_line.theta_rad
            abs_phi_target = inf_line.theta_rad - delta_angle
            if upper_target:
                rel_phi_target = abs_phi_target + alf_target
            else:
                rel_phi_target = -abs_phi_target + (-alf_target)
    if L is not None:
        L *= c_tool / c_target
    if R is not None:
        R *= c_tool / c_target
    output_dict = {
        'xy': [new_x, new_y],
        'phi': rel_phi_target,
        'psi1': psi1,
        'psi2': psi2,
        'r': r,
        'L': L,
        'R': R,
    }
    return output_dict[name]
