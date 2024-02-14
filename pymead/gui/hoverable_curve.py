import os
import typing

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal

from pymead import q_settings, GUI_SETTINGS_DIR
from pymead.core.bezier import Bezier
from pymead.core.line import LineSegment
from pymead.core.parametric_curve import PCurveData, ParametricCurve
from pymead.core.point import PointSequence, Point
from pymead.gui.draggable_point import DraggablePoint
from pymead.utils.misc import convert_str_to_Qt_dash_pattern
from pymead.utils.read_write_files import load_data

q_settings_descriptions = load_data(os.path.join(GUI_SETTINGS_DIR, "q_settings_descriptions.json"))


class HoverableCurve(pg.PlotCurveItem):
    """
    Hoverable curve item to be drawn on an AirfoilCanvas. Inspired in part by https://stackoverflow.com/a/68857695.
    """

    sigCurveHovered = pyqtSignal(object, object)
    sigCurveNotHovered = pyqtSignal(object, object)
    sigRemove = pyqtSignal(object)
    sigBezierAdded = pyqtSignal(object)
    sigLineAdded = pyqtSignal(object)
    sigLineItemAdded = pyqtSignal(object)

    def __init__(self, curve_type: str, *args, **kwargs):
        """
        Parameters
        ==========
        curve_type: str
            Type of curve to be drawn

        args
            Positional arguments which are passed to ``pyqtgraph.PlotCurveItem``

        kwargs
            Keyword arguments which are passed to ``pyqtgraph.PlotCurveItem``
        """
        implemented_curve_types = ["Bezier", "LineSegment"]
        if curve_type not in implemented_curve_types:
            raise NotImplementedError(f"Curve type {curve_type} either incorrectly labeled or not yet implemented."
                                      f"Currently implemented curve types: {implemented_curve_types}.")
        super().__init__(*args, **kwargs)
        self.hoverable = True
        self.setAcceptHoverEvents(True)
        self.curve_type = curve_type
        self.point_items = []
        self.control_point_line_items = None
        self.parametric_curve = None

    def hoverEvent(self, ev):
        """
        Trigger custom signals when a hover event is detected. Only active when ``hoverable==True``.
        """
        if self.hoverable:
            if hasattr(ev, "_scenePos") and self.mouseShape().contains(ev.pos()):
                self.sigCurveHovered.emit(self, ev)
            else:
                self.sigCurveNotHovered.emit(self, ev)

    def updateCurveData(self, data: np.ndarray):
        """
        Updates the data in the curve from a ``numpy`` array.

        Parameters
        ==========
        data: np.ndarray
            Two-column array, where the two columns represent the :math:`x` and :math:`y` values of a sequence of points
            that describes the curve. In the case of splines, these points are the control points.

        Returns
        =======
        PCurveData
            Parameter vector and "xy" ``PointSequence`` describing the parametric curve
        """
        point_sequence = PointSequence.generate_from_array(data)
        if self.curve_type == "Bezier":
            if self.parametric_curve is None:
                self.parametric_curve = Bezier(point_sequence)
                self.sigBezierAdded.emit(self.parametric_curve)
            else:
                self.parametric_curve.set_point_sequence(point_sequence)
        elif self.curve_type == "LineSegment":
            if self.parametric_curve is None:
                self.parametric_curve = LineSegment(point_sequence)
                self.sigLineAdded.emit(self.parametric_curve)
            else:
                self.parametric_curve.set_point_sequence(point_sequence)
        curve_data = self.parametric_curve.evaluate()
        return curve_data

    def getControlPointArray(self):
        control_point_array = np.array([item.data["pos"][0, :] for item in self.point_items])
        return control_point_array

    def updateCurveItem(self, curve_data: PCurveData or np.ndarray or None = None):
        """
        Function that wraps around ``updateCurveData`` and implements other input data types.

        Parameters
        ==========
        curve_data: PCurveData or np.ndarray or None
            If ``PCurveData``, directly update the GUI implementation of the curve. If an array, first update the curve
            data and then update the GUI. If ``None``, first need to extract an array of points from the ``point_items``
            attribute.
        """
        if curve_data is None:
            if len(self.point_items) == 0:
                raise ValueError("No curve data specified and no point items set in Curve Item")
            curve_data = self.getControlPointArray()
        if isinstance(curve_data, np.ndarray):
            curve_data = self.updateCurveData(curve_data)
        self.setData(x=curve_data.xy[:, 0], y=curve_data.xy[:, 1])
        # self.updateControlPointNetItem()

    def updateCanvasItem(self, curve_data: PCurveData):
        arr = curve_data.xy
        self.setData(x=arr[:, 0], y=arr[:, 1])

    @staticmethod
    def generateControlPointNetLine(parametric_curve: ParametricCurve, point_items: typing.List[DraggablePoint]):
        line_item = HoverableCurve(curve_type="LineSegment")
        line_item.parametric_curve = parametric_curve
        line_item.point_items = point_items
        line_item.updateCurveItem()
        line_item.hoverable = True
        # line_item.setZValue(-100)
        return line_item

    def generateControlPointNetItems(self):
        if self.curve_type != "Bezier":
            return
        self.control_point_line_items = []
        for line_idx, line in enumerate(self.parametric_curve.control_point_lines):
            point_items = [self.point_items[line_idx], self.point_items[line_idx + 1]]
            line_item = self.generateControlPointNetLine(line, point_items)
            self.control_point_line_items.append(line_item)
            self.sigLineItemAdded.emit(line_item)
        # Make the control point net behind everything else
        # self.control_point_net.setZValue(-100)

    def removeControlPointFromNet(self, pt: Point):
        if self.curve_type != "Bezier":
            return

        idx = self.point_items.index(pt)

        n_ctrl_pts = len(self.parametric_curve.point_sequence())

        # Case: first point
        if idx == 0:
            self.control_point_line_items.pop(0)

        # Case: last point
        elif idx == n_ctrl_pts - 1:
            self.control_point_line_items.pop(-1)

        # General case:
        else:
            # First, generate a line spanning the points on either side of the point we are deleting
            point_sequence = self.parametric_curve.point_sequence().extract_subsequence(indices=[idx - 1, idx + 1])
            line = LineSegment(point_sequence=point_sequence, reference=True)
            new_line_item = self.generateControlPointNetLine(
                parametric_curve=line, point_items=[self.point_items[idx - 1], self.point_items[idx + 1]]
            )

            # Then, delete the lines connecting to the points we are deleting
            self.control_point_line_items.pop(idx)
            self.control_point_line_items.pop(idx - 1)

            self.control_point_line_items.insert(idx, new_line_item)
            self.sigLineItemAdded.emit(new_line_item)

        # TODO: add a new line if necessary! only true for the general case

    # def updateControlPointNetItem(self):
    #     if self.control_point_line_items is None:
    #         return
    #     for line in self.control_point_line_items:
    #         line.setData(x=line.parametric_curve)

    def remove(self):
        for point in self.point_items:
            point.curves.remove(self)

    def setCurveStyle(self, mode: str = "default"):
        implemented_style_modes = ["default", "hovered"]
        if mode not in implemented_style_modes:
            raise NotImplementedError(f"Selected mode ({mode}) not implemented. "
                                      f"Currently implemented modes: {implemented_style_modes}.")
        self.setPen(pg.mkPen(color=q_settings.value(f"curve_{mode}_pen_color",
                                                    q_settings_descriptions[f"curve_{mode}_pen_color"][1]),
                             width=q_settings.value(f"curve_{mode}_pen_width",
                                                    q_settings_descriptions[f"curve_{mode}_pen_width"][1])))

        if self.parametric_curve is not None and self.parametric_curve.reference:
            dash_str = q_settings.value(f"cpnet_default_pen_dash", q_settings_descriptions[f"cpnet_default_pen_dash"][1])
            dash = convert_str_to_Qt_dash_pattern(dash_str)
            self.setPen(style=dash)

    def setControlPointNetStyle(self, mode: str = "default"):
        # if self.control_point_net is None:
        #     return
        # implemented_style_modes = ["default"]
        # if mode not in implemented_style_modes:
        #     raise NotImplementedError(f"Selected mode ({mode}) not implemented. "
        #                               f"Currently implemented modes: {implemented_style_modes}.")
        #
        # dash_str = q_settings.value(f"cpnet_{mode}_pen_dash", q_settings_descriptions[f"cpnet_{mode}_pen_dash"][1])
        # dash = convert_str_to_Qt_dash_pattern(dash_str)
        #
        # self.control_point_net.setPen(pg.mkPen(
        #     color=q_settings.value(f"cpnet_{mode}_pen_color",
        #                            q_settings_descriptions[f"cpnet_{mode}_pen_color"][1]),
        #     width=q_settings.value(f"cpnet_{mode}_pen_width",
        #                            q_settings_descriptions[f"cpnet_{mode}_pen_width"][1]),
        #     style=dash))
        pass
