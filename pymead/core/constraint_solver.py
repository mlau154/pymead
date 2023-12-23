import time

from jax import jacfwd, jit, debug
import jax.numpy as jnp
from math import floor
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

from pymead.core.param2 import Param
from pymead.core.point import Point, PointSequence
from pymead.core.bezier2 import Bezier


@jit
def measure_distance(x1: float, y1: float, x2: float, y2: float):
    return jnp.hypot(x1 - x2, y1 - y2)


@jit
def measure_abs_angle(x1: float, y1: float, x2: float, y2: float):
    return jnp.arctan2(y2 - y1, x2 - x1)


@jit
def measure_rel_angle3(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return (jnp.arctan2(y1 - y2, x1 - x2) - jnp.arctan2(y3 - y2, x3 - x2)) % (2 * jnp.pi)


@jit
def measure_radius_of_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return jnp.abs(jnp.true_divide(Lt ** 2, Lc * (1 - 1 / n) * jnp.sin(psi)))


@jit
def measure_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return jnp.abs(jnp.true_divide(Lc * (1 - 1 / n) * jnp.sin(psi), Lt ** 2))


@jit
def f(x: np.ndarray):
    f1 = (1.0 - x[0]) ** 2 + (1.0 - x[1]) ** 2 - 3.0 ** 2
    f2 = (x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2 - 4.0 ** 2
    f3 = (x[2] - 1.0) ** 2 + (x[3] - 1.0) ** 2 - 5.0 ** 2
    f4 = jnp.arctan2(x[3] - x[1], x[2] - x[0]) - np.pi / 4
    return jnp.array([f1, f2, f3, f4])


@jit
def f_curvature(x: np.ndarray, x_pos: np.ndarray, xy: np.ndarray, n: np.ndarray):

    for idx, pos in enumerate(x_pos):
        xy = xy.at[pos[0], pos[1]].set(x[idx])

    debug.print(" {xy} ", xy=xy)

    phi1 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[1, 0], xy[1, 1])
    phi2 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    theta1 = measure_abs_angle(xy[1, 0], xy[1, 1], xy[0, 0], xy[0, 1])
    theta2 = measure_abs_angle(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    psi1 = theta1 - phi1
    psi2 = theta2 - phi2
    # phi_rel = measure_rel_angle3(xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    phi_rel = (phi1 - phi2) % (2 * jnp.pi)
    # psi1 = measure_rel_angle3(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1])
    # psi2 = measure_rel_angle3(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    Lt1 = measure_distance(xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1])
    Lt2 = measure_distance(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    Lc1 = measure_distance(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])
    Lc2 = measure_distance(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    kappa1 = measure_curvature_bezier(Lt1, Lc1, n[0], psi1)
    kappa2 = measure_curvature_bezier(Lt2, Lc2, n[1], psi2)
    f1 = kappa1 - kappa2
    f2 = phi_rel - jnp.pi

    debug.print(" {f1} ", f1=f1)
    f_arr = jnp.array([f1, f2])
    return f_arr


@jit
def fp(x: np.ndarray):
    jac = jacfwd(f)(x)
    return jac


@jit
def fp_curvature(x: np.ndarray, pos: np.ndarray, xy: np.ndarray, n: np.ndarray):
    return jacfwd(f_curvature)(x, pos, xy, n)


def main():
    x0 = np.array([4.87, -0.1, 16.221, 0.756])
    t1 = time.perf_counter()
    for _ in range(3500):
        res = root(f, x0=x0, jac=fp)
    t2 = time.perf_counter()
    print(f"Root found 3500 times in {t2 - t1} seconds. Estimated operations per second: {floor(3500 / (t2 - t1))}")
    fig, ax = plt.subplots()
    ax.plot([1.0, res.x[0], res.x[2]], [1.0, res.x[1], res.x[3]], ls="--", marker="o")
    ax.set_aspect("equal")
    plt.show()


def main2():
    x0 = np.array([0.25, 0.8])
    res = root(f_curvature, x0=x0, jac=fp_curvature)
    print(f"{res = }")


def main3():
    xy_original = np.array([
        [0.0, 0.0],
        [0.3, 0.5],
        [0.45, 0.55],
        [0.8, 0.3],
        [1.1, -0.1]
    ])
    original_point_seq = PointSequence.generate_from_array(xy_original)
    bez1 = Bezier(point_sequence=original_point_seq[:3])
    bez2 = Bezier(point_sequence=original_point_seq[2:][::-1])

    fig, ax = plt.subplots()
    ax.plot(xy_original[:, 0], xy_original[:, 1], ls="--", marker="o", mfc="grey", mec="grey",
            color="grey")
    bez1_original_data = bez1.evaluate()
    bez2_original_data = bez2.evaluate()
    ax.plot(bez1_original_data.xy[:, 0], bez1_original_data.xy[:, 1], color="steelblue")
    ax.plot(bez2_original_data.xy[:, 0], bez2_original_data.xy[:, 1], color="indianred")

    pos = np.array([[3, 1], [4, 1]])
    res = root(f_curvature, x0=np.array([xy_original[p[0], p[1]] for p in pos]), jac=fp_curvature,
               args=(pos, xy_original, np.array([bez1.degree, bez2.degree])))
    print(f"{res = }")

    for idx, x in enumerate(res.x):
        xy_original[pos[idx, 0], pos[idx, 1]] = x

    print(f"{xy_original = }")

    new_point_seq = PointSequence.generate_from_array(xy_original)
    bez1 = Bezier(point_sequence=new_point_seq[:3])
    bez2 = Bezier(point_sequence=new_point_seq[2:])
    bez1_new_data = bez1.evaluate()
    bez2_new_data = bez2.evaluate()

    ax.plot(xy_original[:, 0], xy_original[:, 1], ls="--", marker="o", mfc="magenta", mec="magenta",
            color="magenta")
    ax.plot(bez1_new_data.xy[:, 0], bez1_new_data.xy[:, 1], color="cornflowerblue")
    ax.plot(bez2_new_data.xy[:, 0], bez2_new_data.xy[:, 1], color="hotpink")

    print(f"{bez1_new_data.R[-1] = }, {bez2_new_data.R[0] = }")

    plt.show()


def main4():

    from pymead.core.constraints import GCS
    p1 = Point(0.0, 0.0)
    p2 = Point(0.5, 0.2)
    dist = Param(0.5, "dist")
    gcs = GCS(parent=p1)
    gcs.add_distance_constraint(p1, p2, dist)
    res = gcs.solve()
    print(f"{res = }")


if __name__ == "__main__":
    main4()
