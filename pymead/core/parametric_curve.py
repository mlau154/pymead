from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from pymead.core.pymead_obj import PymeadObj


LOW_RES_NT = 100
INTERMEDIATE_NT = 150
HIGH_RES_NT = 200


class PCurveData:
    def __init__(self, t: np.ndarray, xy: np.ndarray, xpyp: np.ndarray, xppypp: np.ndarray, k: np.ndarray,
                 R: np.ndarray):
        self.t = t
        self.xy = xy
        self.xpyp = xpyp
        self.xppypp = xppypp
        self.k = k
        self.R = R
        self.R_abs_min = np.min(np.abs(self.R))

    def plot(self, ax: plt.Axes, **kwargs):
        ax.plot(self.xy[:, 0], self.xy[:, 1], **kwargs)

    def get_curvature_comb(self, max_k_normalized_scale_factor, interval: int = 1):
        first_deriv_mag = np.hypot(self.xpyp[:, 0], self.xpyp[:, 1])
        comb_heads_x = self.xy[:, 0] - self.xpyp[:, 1] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        comb_heads_y = self.xy[:, 1] + self.xpyp[:, 0] / first_deriv_mag * self.k * max_k_normalized_scale_factor
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        comb_tails = np.column_stack((self.xy[:, 0], self.xy[:, 1]))[:-1:interval, :]
        comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        comb_tails = np.row_stack((comb_tails, np.array([self.xy[-1, 0], self.xy[-1, 1]])))
        comb_heads = np.row_stack((comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1]])))
        return comb_tails, comb_heads

    def approximate_arc_length(self):
        return np.sum(np.hypot(self.xy[1:, 0] - self.xy[:-1, 0], self.xy[1:, 1] - self.xy[:-1, 1]))


class ParametricCurve(PymeadObj, ABC):
    def __init__(self, sub_container: str, reference: bool = False):
        super().__init__(sub_container=sub_container)
        self.curve_type = self.__class__.__name__
        self.reference = reference
        self.airfoil = None

    def update(self):
        p_curve_data = self.evaluate()
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(curve_data=p_curve_data)

    @staticmethod
    def generate_t_vec(nt: int = INTERMEDIATE_NT, spacing: str = "linear", start: int = 0.0, end: int = 1.0):
        if spacing == "linear":
            return np.linspace(start, end, nt)

    @abstractmethod
    def point_removal_deletes_curve(self) -> bool:
        pass

    @abstractmethod
    def evaluate(self, t: np.ndarray or None = None, **kwargs):
        pass
