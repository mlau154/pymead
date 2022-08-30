from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.param import Param

import numpy as np


def main():
    anchor_point = AnchorPoint(Param(0.0), Param(0.0), 'le', 'te_1', Param(0.1), Param(0.08), Param(0.6),
                               Param(np.deg2rad(15)), Param(np.deg2rad(20)), Param(np.deg2rad(30)))
    anchor_point.set_minus_plus_bezier_curve_orders(2, 2)
    anchor_point.generate_anchor_point_branch(['te_1', 'le', 'te_2'])
    import matplotlib.pyplot as plt
    from pyairpar.core.bezier import Bezier
    C1 = Bezier(anchor_point.ctrlpt_branch_array[:3, :], 100)
    C2 = Bezier(anchor_point.ctrlpt_branch_array[2:, :], 100)
    print(f"Assigned radius of curvature: {anchor_point.R.value}")
    print(f"C1_endpoint_curvature = {C1.k[-1]} | C1_endpoint_radius_of_curvature = {C1.R[-1]}")
    print(f"C2_startpoint_curvature = {C2.k[0]} | C2_startpoint_radius_of_curvature = {C2.R[0]}")
    print(f"curvature_difference = {np.abs(C2.k[0] - C1.k[-1])}, radius_difference = "
          f"{np.abs(C2.R[0] - C1.R[-1])}")
    scale_factor = 0.02
    max_k1 = np.max(np.abs(C1.k))
    max_k2 = np.max(np.abs(C2.k))
    max_k = np.max(np.array([max_k1, max_k2]))
    k_normalized_scale_factor = scale_factor / max_k

    fig, axs = plt.subplots(1, 1)
    axs.plot(anchor_point.ctrlpt_branch_array[:, 0], anchor_point.ctrlpt_branch_array[:, 1], 'ro')
    axs.plot(C1.x, C1.y, color='cornflowerblue')
    axs.plot(C2.x, C2.y, color='green')
    for idx, t in enumerate(C1.t):
        axs.plot([C1.x[idx], C1.x[idx] - C1.py[idx] / np.sqrt(C1.px[idx]**2 + C1.py[idx]**2)
                  * C1.k[idx] * k_normalized_scale_factor],
                 [C1.y[idx], C1.y[idx] + C1.px[idx] / np.sqrt(C1.px[idx]**2 + C1.py[idx]**2) *
                  C1.k[idx] * k_normalized_scale_factor], color='magenta', lw=1)
    for idx, t in enumerate(C2.t):
        axs.plot([C2.x[idx], C2.x[idx] - C2.py[idx] / np.sqrt(C2.px[idx]**2 + C2.py[idx]**2)
                  * C2.k[idx] * k_normalized_scale_factor],
                 [C2.y[idx], C2.y[idx] + C2.px[idx] / np.sqrt(C2.px[idx]**2 + C2.py[idx]**2) *
                  C2.k[idx] * k_normalized_scale_factor], color='orange', lw=1)
    axs.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()
