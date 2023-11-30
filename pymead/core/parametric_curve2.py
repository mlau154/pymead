from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from pymead.core.dual_rep import DualRep


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


class ParametricCurve(DualRep):
    def __init__(self, name: str, reference: bool = False):
        self._name = None
        self.gui_obj = None
        self.set_name(name)
        self.reference = reference

    def name(self):
        return self._name

    def set_name(self, name: str):
        self._name = name

    def update(self):
        p_curve_data = self.evaluate()
        if self.gui_obj is not None:
            self.gui_obj.updateGUIObj(curve_data=p_curve_data)

    @staticmethod
    def generate_t_vec(nt: int = 100, spacing: str = "linear", start: int = 0.0, end: int = 1.0):
        if spacing == "linear":
            return np.linspace(start, end, nt)

    @abstractmethod
    def evaluate(self, t: np.ndarray or None = None, **kwargs):
        pass
