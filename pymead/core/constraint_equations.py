from collections import namedtuple

import numpy as np


def measure_distance(x1: float, y1: float, x2: float, y2: float):
    return np.hypot(x1 - x2, y1 - y2)


def measure_abs_angle(x1: float, y1: float, x2: float, y2: float):
    return (np.arctan2(y2 - y1, x2 - x1)) % (2 * np.pi)


def measure_rel_angle3(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return (np.arctan2(y1 - y2, x1 - x2) - np.arctan2(y3 - y2, x3 - x2)) % (2 * np.pi)


def measure_rel_angle4(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return (np.arctan2(y4 - y3, x4 - x3) - np.arctan2(y2 - y1, x2 - x1)) % (2 * np.pi)


def measure_point_line_distance_signed(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    """
    Measures the signed distance from a point :math:`(x_3, y_3)` to a line defined by
    two points: :math:`(x_1, y_1)` and :math:`(x_2, y_2)`. This signed version is used in the point-line
    symmetry constraint because a perpendicular constraint does not necessarily prevent the GCS from moving
    the symmetric point to the same side of the line as the target point.

    Parameters
    ----------
    x1: float
        x-coordinate of the line's start point
    y1: float
        y-coordinate of the line's start point
    x2: float
        x-coordinate of the line's end point
    y2: float
        y-coordinate of the line's end point
    x3: float
        x-coordinate of the target point
    y3: float
        y-coordinate of the target point

    Returns
    -------
    float
        Distance from the target point to the line
    """
    return ((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1)) / measure_distance(x1, y1, x2, y2)


def measure_point_line_distance_unsigned(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return np.abs(measure_point_line_distance_signed(x1, y1, x2, y2, x3, y3))


def measure_radius_of_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return np.abs(np.true_divide(Lt ** 2, Lc * (1 - 1 / n) * np.sin(psi)))


def measure_curvature_length_bezier(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, R: float, n: int):
    Lt = measure_distance(x1, y1, x2, y2)
    psi = measure_rel_angle3(x1, y1, x2, y2, x3, y3)
    Lc = np.abs(np.true_divide(Lt ** 2, R * (1 - 1 / n) * np.sin(psi)))
    return Lc


def measure_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return np.abs(np.true_divide(Lc * (1 - 1 / n) * np.sin(psi), Lt ** 2))


def measure_data_bezier_curve_joint(xy: np.ndarray, n: np.ndarray):
    phi1 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[1, 0], xy[1, 1])
    phi2 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    theta1 = measure_abs_angle(xy[1, 0], xy[1, 1], xy[0, 0], xy[0, 1])
    theta2 = measure_abs_angle(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    psi1 = theta1 - phi1
    psi2 = theta2 - phi2
    phi_rel = (phi1 - phi2) % (2 * np.pi)
    Lt1 = measure_distance(xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1])
    Lt2 = measure_distance(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    Lc1 = measure_distance(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])
    Lc2 = measure_distance(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    kappa1 = measure_curvature_bezier(Lt1, Lc1, n[0], psi1)
    kappa2 = measure_curvature_bezier(Lt2, Lc2, n[1], psi2)
    R1 = np.true_divide(1, kappa1)
    R2 = np.true_divide(1, kappa2)
    n1 = n[0]
    n2 = n[1]
    field_names = ["phi1", "phi2", "theta1", "theta2", "psi1", "psi2", "phi_rel", "Lt1", "Lt2", "Lc1", "Lc2",
                   "kappa1", "kappa2", "R1", "R2", "n1", "n2"]
    BezierCurveJointData = namedtuple("BezierCurveJointData", field_names=field_names)
    data = BezierCurveJointData(phi1=phi1, phi2=phi2, theta1=theta1, theta2=theta2, psi1=psi1, psi2=psi2,
                                phi_rel=phi_rel, Lt1=Lt1, Lt2=Lt2, Lc1=Lc1, Lc2=Lc2, kappa1=kappa1, kappa2=kappa2,
                                R1=R1, R2=R2, n1=n1, n2=n2)
    return data


def radius_of_curvature_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, R: float, n: int):
    Lt = measure_distance(x1, y1, x2, y2)
    psi = measure_rel_angle3(x1, y1, x2, y2, x3, y3)
    Lc = np.abs(np.true_divide(Lt ** 2, R * (1 - 1 / n) * np.sin(psi)))
    return distance_constraint(x2, y2, x3, y3, Lc)


def empty_constraint_weak():
    return 0.0


def fixed_param_constraint(p_val: float, val: float):
    return p_val - val


def fixed_param_constraint_weak(new_val: float, old_val: float):
    return new_val - old_val


def fixed_x_constraint(x: float, val: float):
    return x - val


def fixed_x_constraint_weak(x_new: float, x_old: float):
    return x_new - x_old


def fixed_y_constraint(y: float, val: float):
    return y - val


def fixed_y_constraint_weak(y_new: float, y_old: float):
    return y_new - y_old


def distance_constraint(x1: float, y1: float, x2: float, y2: float, dist: float):
    return measure_distance(x1, y1, x2, y2) - dist


def distance_constraint_weak(x1_new: float, y1_new: float, x2_new: float, y2_new: float,
                             x1_old: float, y1_old: float, x2_old: float, y2_old: float):
    return (measure_distance(x1_new, y1_new, x2_new, y2_new) -
            measure_distance(x1_old, y1_old, x2_old, y2_old))


def abs_angle_constraint(x1: float, y1: float, x2: float, y2: float, angle: float):
    return measure_abs_angle(x1, y1, x2, y2) - angle


def abs_angle_constraint_weak(x1_new: float, y1_new: float, x2_new: float, y2_new: float,
                              x1_old: float, y1_old: float, x2_old: float, y2_old: float):
    return (measure_abs_angle(x1_new, y1_new, x2_new, y2_new) -
            measure_abs_angle(x1_old, y1_old, x2_old, y2_old))


def rel_angle3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, angle: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - angle


def rel_angle3_constraint_weak(x1_new: float, y1_new: float, x2_new: float, y2_new: float, x3_new: float, y3_new: float,
                               x1_old: float, y1_old: float, x2_old: float, y2_old: float, x3_old: float,
                               y3_old: float):
    return (measure_rel_angle3(x1_new, y1_new, x2_new, y2_new, x3_new, y3_new) -
            measure_rel_angle3(x1_old, y1_old, x2_old, y2_old, x3_old, y3_old))


def rel_angle4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float,
                          angle: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - angle


def perp3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - (np.pi / 2)


def perp4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - (np.pi / 2)


def antiparallel3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - np.pi


def antiparallel4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - np.pi


def parallel3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3)


def parallel4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4)


def point_on_line_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return (y1 - y2) * (x3 - x1) - (y3 - y1) * (x1 - x2)


def points_equidistant_from_line_constraint_unsigned(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
                                                     x4: float, y4: float):
    return (measure_point_line_distance_unsigned(x1, y1, x2, y2, x3, y3) -
            measure_point_line_distance_unsigned(x1, y1, x2, y2, x4, y4))


def points_equidistant_from_line_constraint_signed(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float,
                                                   x4: float, y4: float):
    return (measure_point_line_distance_signed(x1, y1, x2, y2, x3, y3) +
            measure_point_line_distance_signed(x1, y1, x2, y2, x4, y4))
