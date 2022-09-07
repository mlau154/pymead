from pymead.core.airfoil import Airfoil
from mplcanvas import MplCanvas
import numpy as np


def recalculate_airfoil_parameters(ind: int, x: np.ndarray, y: np.ndarray, cp_skeleton, dx, dy, airfoil: Airfoil, canvas: MplCanvas, lines):
    anchor_point = airfoil.anchor_points[airfoil.anchor_point_order.index(airfoil.control_points[ind].anchor_point_tag)]

    if airfoil.control_points[ind].cp_type == 'g2_minus':
        new_Lc = np.sqrt((x[ind] - x[ind + 1])**2 + (y[ind] - y[ind + 1])**2)
        # print(f"new_Lc = {new_Lc}")
        # phi_abs_angle = np.arctan2(y[ind + 1] - y[ind + 2], x[ind + 1] - x[ind + 2])
        new_psi1_abs_angle = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1])
        anchor_point.recalculate_ap_branch_props_from_g2_pt('minus', new_psi1_abs_angle, new_Lc)

    elif airfoil.control_points[ind].cp_type == 'g2_plus':
        new_Lc = np.sqrt((x[ind] - x[ind - 1])**2 + (y[ind] - y[ind - 1])**2)
        print(f"new_Lc = {new_Lc}")
        new_psi2_abs_angle = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1])
        anchor_point.recalculate_ap_branch_props_from_g2_pt('plus', new_psi2_abs_angle, new_Lc)

    elif airfoil.control_points[ind].cp_type == 'g1_minus':
        # x[ind - 1] += dx
        # y[ind - 1] += dy
        # cp_skeleton.set_xdata(x)
        # cp_skeleton.set_ydata(y)
        new_Lt = np.sqrt((x[ind] - x[ind + 1])**2 + (y[ind] - y[ind + 1])**2)
        new_abs_phi1 = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1])
        anchor_point.recalculate_ap_branch_props_from_g1_pt('minus', new_abs_phi1, new_Lt)

    elif airfoil.control_points[ind].cp_type == 'g1_plus':
        new_Lt = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2)
        new_abs_phi2 = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1])
        anchor_point.recalculate_ap_branch_props_from_g1_pt('plus', new_abs_phi2, new_Lt)

    elif airfoil.control_points[ind].name == 'le':
        airfoil.dx.value = x[ind]
        airfoil.dy.value = y[ind]

    # print(f"anchor_point.R.value = {anchor_point.R.value}")
    # airfoil.base_airfoil_params.R_le.value = anchor_point.R.value
    airfoil.update()
    airfoil.update_curvature_comb_normals()
    airfoil.update_curvature_comb_curve()
    # print(f"airfoil_ap_LE = {airfoil.anchor_points[1].R.value}")

    # lines = airfoil.plot_airfoil(canvas.axes, color='cornflowerblue', lw=2, label='airfoil')

    # # Update the value of the transformed control point in the airfoil control point objects
    # airfoil.control_point_array[ind].xp = x[ind]
    # airfoil.control_point_array[ind].yp = y[ind]
    #
    # airfoil.control_point_array = np.column_stack((x, y))
    # airfoil.curve_list = []
    # cp_end_idx, cp_start_idx = 0, 1
    # for idx, ap_name in enumerate(airfoil.anchor_point_order[:-1]):
    #     cp_end_idx += airfoil.N[ap_name] + 1
    #     P = airfoil.control_point_array[cp_start_idx - 1:cp_end_idx]
    #     airfoil.curve_list.append(Bezier(P, 150))
    #     cp_start_idx = deepcopy(cp_end_idx)
    for idx, line in enumerate(lines):
        line.set_xdata(airfoil.curve_list[idx].x)
        line.set_ydata(airfoil.curve_list[idx].y)
    cp_skeleton.set_xdata(airfoil.control_point_array[:, 0])
    cp_skeleton.set_ydata(airfoil.control_point_array[:, 1])
    print(airfoil.control_points[2].xp)
    return lines
