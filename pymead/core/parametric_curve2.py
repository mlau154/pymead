from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from pymead.core.param2 import ParamCollection
from pymead.core.point import SurfPointSequence


class PCurveData:
    def __init__(self, t: ParamCollection, xy: SurfPointSequence):
        self.t = t
        self.xy = xy

    def plot(self, ax: plt.Axes, **kwargs):
        xy_arr = self.xy.as_array()
        ax.plot(xy_arr[:, 0], xy_arr[:, 1], **kwargs)


class PCurveDerivData:
    def __init__(self, t: ParamCollection, xpyp: SurfPointSequence, order: int):
        self.t = t
        self.xpyp = xpyp
        self.order = order


class ParametricCurve:
    def __init__(self, reference: bool = False):
        self.reference = reference

    @staticmethod
    def generate_t_collection(nt: int = 100, spacing: str = "linear", start: int = 0.0, end: int = 1.0):
        if spacing == "linear":
            return ParamCollection.generate_from_array(np.linspace(start, end, nt))

    @abstractmethod
    def evaluate(self, t: ParamCollection or None = None, **kwargs):
        pass
