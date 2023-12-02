from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from pymead.core.pymead_obj import PymeadObj


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


class ParametricCurve(PymeadObj, ABC):
    def __init__(self, sub_container: str, reference: bool = False):
        super().__init__(sub_container=sub_container)
        self.reference = reference

    def update(self):
        p_curve_data = self.evaluate()
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(curve_data=p_curve_data)

    def name(self) -> str:
        return self._name

    def set_name(self, name: str):
        # Rename the reference in the geometry collection
        if self.geo_col is not None and self.name() in self.geo_col.container()[self.sub_container].keys():
            sub_container = self.geo_col.container()[self.sub_container]
            sub_container[name] = sub_container[self.name()]
            sub_container.pop(self.name())

        self._name = name

    @staticmethod
    def generate_t_vec(nt: int = 100, spacing: str = "linear", start: int = 0.0, end: int = 1.0):
        if spacing == "linear":
            return np.linspace(start, end, nt)

    @abstractmethod
    def evaluate(self, t: np.ndarray or None = None, **kwargs):
        pass
