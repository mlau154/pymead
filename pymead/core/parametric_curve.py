import matplotlib.pyplot as plt
from abc import abstractmethod


class ParametricCurve:

    def __init__(self, t, x, y, px, py, ppx, ppy, k, R):
        self.t = t
        self.x = x
        self.y = y
        self.px = px
        self.py = py
        self.ppx = ppx
        self.ppy = ppy
        self.k = k
        self.R = R
        self.comb_heads = None
        self.comb_tails = None
        self.plt_handle_curve = None
        self.plt_handles_normals = None
        self.plt_handle_comb_curves = None
        self.pg_curve_handle = None
        self.scale_factor = None
        self.comb_interval = None

    def plot_curve(self, axs: plt.axes, **plt_kwargs):
        self.plt_handle_curve, = axs.plot(self.x, self.y, **plt_kwargs)
        return self.plt_handle_curve

    def init_curve_pg(self, v, pen):
        self.pg_curve_handle = v.plot(pen=pen)

    def update_curve(self):
        self.plt_handle_curve.set_xdata(self.x)
        self.plt_handle_curve.set_ydata(self.y)

    def update_curve_pg(self):
        self.pg_curve_handle.setData(self.x, self.y)

    @abstractmethod
    def get_curvature_comb(self, max_k_normalized_scale_factor, interval: int = 1):
        pass

    def plot_curvature_comb_normals(self, axs: plt.axes, scale_factor, interval: int = 1, **plt_kwargs):
        self.scale_factor = scale_factor
        self.comb_interval = interval
        if self.comb_heads is None or self.comb_tails is None:
            self.get_curvature_comb(scale_factor, interval)
        self.plt_handles_normals = []
        for idx, head in enumerate(self.comb_heads):
            h, = axs.plot([head[0], self.comb_tails[idx, 0]], [head[1], self.comb_tails[idx, 1]], **plt_kwargs)
            self.plt_handles_normals.append(h)

    def update_curvature_comb_normals(self):
        self.get_curvature_comb(self.scale_factor, self.comb_interval)
        for idx, head in enumerate(self.comb_heads):
            self.plt_handles_normals[idx].set_xdata([head[0], self.comb_tails[idx, 0]])
            self.plt_handles_normals[idx].set_ydata([head[1], self.comb_tails[idx, 1]])

    def plot_curvature_comb_curve(self, axs: plt.axes, scale_factor, interval: int = 1, **plt_kwargs):
        self.scale_factor = scale_factor
        self.comb_interval = interval
        if self.comb_heads is None or self.comb_tails is None:
            self.get_curvature_comb(scale_factor, interval)
        self.plt_handle_comb_curves, = axs.plot(self.comb_heads[:, 0], self.comb_heads[:, 1], **plt_kwargs)
        # print(self.plt_handle_comb_curves)

    def update_curvature_comb_curve(self):
        if self.comb_heads is None or self.comb_tails is None:
            self.get_curvature_comb(self.scale_factor, self.comb_interval)
        self.plt_handle_comb_curves.set_xdata(self.comb_heads[:, 0])
        self.plt_handle_comb_curves.set_ydata(self.comb_heads[:, 1])
