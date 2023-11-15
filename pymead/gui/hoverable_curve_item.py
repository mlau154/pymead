import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal

from pymead.core.bezier2 import Bezier
from pymead.core.parametric_curve2 import PCurveData
from pymead.core.point import PointSequence


class HoverableCurveItem(pg.PlotCurveItem):
    """
    From https://stackoverflow.com/a/68857695
    """
    sigCurveHovered = pyqtSignal(object, object)
    sigCurveNotHovered = pyqtSignal(object, object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hoverable = True
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, ev):
        if self.hoverable:
            if hasattr(ev, "_scenePos") and self.mouseShape().contains(ev.pos()):
                self.sigCurveHovered.emit(self, ev)
            else:
                self.sigCurveNotHovered.emit(self, ev)


class BezierCurveItem(HoverableCurveItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_items = []

    @staticmethod
    def updateBezierCurveData(data: np.ndarray):
        point_sequence = PointSequence.generate_from_array(data)
        bezier_curve = Bezier(point_sequence)
        curve_data = bezier_curve.evaluate()
        return curve_data

    def updateBezierCurveItem(self, curve_data: PCurveData or np.ndarray or None = None):
        if curve_data is None:
            if len(self.point_items) == 0:
                raise ValueError("No curve data specified and no point items set in BezierCurveItem")
            curve_data = np.array([item.data["pos"][0, :] for item in self.point_items])
        if isinstance(curve_data, np.ndarray):
            curve_data = self.updateBezierCurveData(curve_data)
        self.setData(x=curve_data.xy.as_array()[:, 0], y=curve_data.xy.as_array()[:, 1])
