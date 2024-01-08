from collections import namedtuple

import numpy as np
from jax import jit
from jax import numpy as jnp


@jit
def measure_distance(x1: float, y1: float, x2: float, y2: float):
    return jnp.hypot(x1 - x2, y1 - y2)


@jit
def measure_abs_angle(x1: float, y1: float, x2: float, y2: float):
    return (jnp.arctan2(y2 - y1, x2 - x1)) % (2 * jnp.pi)


@jit
def measure_rel_angle3(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return (jnp.arctan2(y1 - y2, x1 - x2) - jnp.arctan2(y3 - y2, x3 - x2)) % (2 * jnp.pi)


@jit
def measure_rel_angle4(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return (jnp.arctan2(y4 - y3, x4 - x3) - jnp.arctan2(y2 - y1, x2 - x1)) % (2 * jnp.pi)


@jit
def measure_radius_of_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return jnp.abs(jnp.true_divide(Lt ** 2, Lc * (1 - 1 / n) * jnp.sin(psi)))


@jit
def measure_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return jnp.abs(jnp.true_divide(Lc * (1 - 1 / n) * jnp.sin(psi), Lt ** 2))


@jit
def measure_data_bezier_curve_joint(xy: np.ndarray, n: np.ndarray):
    phi1 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[1, 0], xy[1, 1])
    phi2 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    theta1 = measure_abs_angle(xy[1, 0], xy[1, 1], xy[0, 0], xy[0, 1])
    theta2 = measure_abs_angle(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    psi1 = theta1 - phi1
    psi2 = theta2 - phi2
    phi_rel = (phi1 - phi2) % (2 * jnp.pi)
    Lt1 = measure_distance(xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1])
    Lt2 = measure_distance(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    Lc1 = measure_distance(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])
    Lc2 = measure_distance(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    kappa1 = measure_curvature_bezier(Lt1, Lc1, n[0], psi1)
    kappa2 = measure_curvature_bezier(Lt2, Lc2, n[1], psi2)
    R1 = jnp.true_divide(1, kappa1)
    R2 = jnp.true_divide(1, kappa2)
    n1 = n[0]
    n2 = n[1]
    field_names = ["phi1", "phi2", "theta1", "theta2", "psi1", "psi2", "phi_rel", "Lt1", "Lt2", "Lc1", "Lc2",
                   "kappa1", "kappa2", "R1", "R2", "n1", "n2"]
    BezierCurveJointData = namedtuple("BezierCurveJointData", field_names=field_names)
    data = BezierCurveJointData(phi1=phi1, phi2=phi2, theta1=theta1, theta2=theta2, psi1=psi1, psi2=psi2,
                                phi_rel=phi_rel, Lt1=Lt1, Lt2=Lt2, Lc1=Lc1, Lc2=Lc2, kappa1=kappa1, kappa2=kappa2,
                                R1=R1, R2=R2, n1=n1, n2=n2)
    return data


@jit
def empty_constraint_weak():
    return 0.0


@jit
def fixed_param_constraint(p_val: float, val: float):
    return p_val - val


@jit
def fixed_param_constraint_weak(new_val: float, old_val: float):
    return new_val - old_val


@jit
def fixed_x_constraint(x: float, val: float):
    return x - val


@jit
def fixed_x_constraint_weak(x_new: float, x_old: float):
    return x_new - x_old


@jit
def fixed_y_constraint(y: float, val: float):
    return y - val


@jit
def fixed_y_constraint_weak(y_new: float, y_old: float):
    return y_new - y_old


@jit
def distance_constraint(x1: float, y1: float, x2: float, y2: float, dist: float):
    return measure_distance(x1, y1, x2, y2) - dist


@jit
def distance_constraint_weak(x1_new: float, y1_new: float, x2_new: float, y2_new: float,
                             x1_old: float, y1_old: float, x2_old: float, y2_old: float):
    return (measure_distance(x1_new, y1_new, x2_new, y2_new) -
            measure_distance(x1_old, y1_old, x2_old, y2_old))


@jit
def abs_angle_constraint(x1: float, y1: float, x2: float, y2: float, angle: float):
    return measure_abs_angle(x1, y1, x2, y2) - angle


@jit
def abs_angle_constraint_weak(x1_new: float, y1_new: float, x2_new: float, y2_new: float,
                              x1_old: float, y1_old: float, x2_old: float, y2_old: float):
    return (measure_abs_angle(x1_new, y1_new, x2_new, y2_new) -
            measure_abs_angle(x1_old, y1_old, x2_old, y2_old))


@jit
def rel_angle3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, angle: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - angle


@jit
def rel_angle4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float,
                          angle: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - angle


@jit
def perp3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - (jnp.pi / 2)


@jit
def perp4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - (jnp.pi / 2)


@jit
def parallel3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - jnp.pi


@jit
def parallel4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - jnp.pi
