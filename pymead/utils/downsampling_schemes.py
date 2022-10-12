import numpy as np
import numpy.linalg as ln


def fractal_downsampler2(pos, ratio_thresh=None, abs_thresh=None):
    """
    Source: https://kaushikghose.wordpress.com/2017/11/25/adaptively-downsampling-a-curve/
    """
    if ratio_thresh is None:
        ratio_thresh = 1.001
    if abs_thresh is None:
        abs_thresh = 0.1
    d = np.diff(pos, axis=0)
    adaptive_pos = [pos[0, :]]
    last_n = 0
    for n in range(1, pos.shape[0]):
        if n == last_n:
            continue
        line_d = ln.norm(pos[n, :] - pos[last_n, :])
        curve_d = ln.norm(d[last_n:n, :], axis=1).sum()
        if curve_d / line_d > ratio_thresh or abs(curve_d - line_d) > abs_thresh:
            adaptive_pos.append(pos[n - 1, :])
            last_n = n - 1
    adaptive_pos.append(pos[-1, :])
    return np.vstack(adaptive_pos)


if __name__ == '__main__':
    from pymead.core.airfoil import Airfoil
    import matplotlib.pyplot as plt
    airfoil = Airfoil()
    airfoil.get_coords(body_fixed_csys=True)
    ds = fractal_downsampler2(airfoil.coords, ratio_thresh=1.0000001, abs_thresh=0.1)
    print(f"Number of down-sampled points: {len(ds)}")
    fig, axs = plt.subplots()
    axs.plot(ds[:, 0], ds[:, 1], marker='o')
    # axs.set_aspect('equal')
    plt.show()
