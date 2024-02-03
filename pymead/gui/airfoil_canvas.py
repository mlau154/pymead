import os
import typing
import sys
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QEventLoop, Qt
from PyQt5.QtGui import QFont, QBrush, QColor
from PyQt5.QtWidgets import QApplication

from pymead.gui.input_dialog import PlotExportDialog
from pymead.gui.polygon_item import PolygonItem

from pymead.core.airfoil import Airfoil

from pymead.core.constraints import *
from pymead.core.constraint_equations import *
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.param import LengthParam
from pymead.core.parametric_curve import ParametricCurve
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj
from pymead.gui.constraint_items import *
from pymead.gui.hoverable_curve import HoverableCurve
from pymead.gui.draggable_point import DraggablePoint
from pymead.utils.read_write_files import load_data
from pymead.utils.misc import get_setting
from pymead.core import UNITS
from pymead import q_settings, GUI_SETTINGS_DIR

q_settings_descriptions = load_data(os.path.join(GUI_SETTINGS_DIR, "q_settings_descriptions.json"))


class AirfoilCanvas(pg.PlotWidget):
    sigEnterPressed = pyqtSignal()
    sigEscapePressed = pyqtSignal()
    sigStatusBarUpdate = pyqtSignal(str, int)

    def __init__(self, parent, geo_col: GeometryCollection, gui_obj):
        super().__init__(parent)
        self.setMenuEnabled(False)
        self.setAspectLocked(True)
        self.disableAutoRange()
        self.points = []
        self.airfoil_text_items = {}
        self.drawing_object = None
        self.creating_collinear_constraint = None
        self.adding_point_to_curve = None
        self.curve_hovered_item = None
        self.point_hovered_item = None
        self.constraint_hovered_item = None
        self.point_text_item = None
        self.geo_col = geo_col
        self.geo_col.canvas = self
        self.gui_obj = gui_obj
        self.plot = self.getPlotItem()
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)

    def showPymeadObjs(self, sub_container: str):
        for pymead_obj in self.geo_col.container()[sub_container].values():
            if pymead_obj.canvas_item is None:
                continue
            pymead_obj.canvas_item.show()

    def hidePymeadObjs(self, sub_container: str):
        for pymead_obj in self.geo_col.container()[sub_container].values():
            if pymead_obj.canvas_item is None:
                continue
            pymead_obj.canvas_item.hide()

    def showAllPymeadObjs(self):
        sub_containers = ("points", "bezier", "lines", "airfoils", "geocon")
        for sub_container in sub_containers:
            self.showPymeadObjs(sub_container)
        return {sub_container: True for sub_container in sub_containers}

    def hideAllPymeadObjs(self):
        sub_containers = ("points", "bezier", "lines", "airfoils", "geocon")
        for sub_container in sub_containers:
            self.hidePymeadObjs(sub_container)
        return {sub_container: False for sub_container in sub_containers}

    def setAxisLabels(self, theme: dict):
        label_font = f"{get_setting('axis-label-point-size')}pt {get_setting('axis-label-font-family')}"
        self.plot.setLabel(axis="bottom", text=f"x [{UNITS.current_length_unit()}]", font=label_font,
                           color=theme["main-color"])
        self.plot.setLabel(axis="left", text=f"y [{UNITS.current_length_unit()}]", font=label_font,
                           color=theme["main-color"])
        tick_font = QFont(get_setting("axis-tick-font-family"), get_setting("axis-tick-point-size"))
        self.plot.getAxis("bottom").setTickFont(tick_font)
        self.plot.getAxis("left").setTickFont(tick_font)
        self.plot.getAxis("bottom").setTextPen(theme["main-color"])
        self.plot.getAxis("left").setTextPen(theme["main-color"])

    def toggleGrid(self):
        x_state = self.plot.ctrl.xGridCheck.checkState()
        y_state = self.plot.ctrl.yGridCheck.checkState()
        if x_state or y_state:
            self.plot.showGrid(x=False, y=False)
        else:
            self.plot.showGrid(x=True, y=True)

    def getPointRange(self):
        """
        Gets the minimum and maximum :math:`x` and :math:`y` values of all the points in the GeometryCollection.

        Returns
        typing.Tuple[list]
            x-range and y-range of the points (two-element tuple of two-element lists)
        """
        point_seq = PointSequence(points=[pt for pt in self.geo_col.container()["points"].values()])
        pseq_arr = point_seq.as_array()
        min_x = pseq_arr[:, 0].min()
        max_x = pseq_arr[:, 0].max()
        min_y = pseq_arr[:, 1].min()
        max_y = pseq_arr[:, 1].max()
        return [min_x, max_x], [min_y, max_y]

    def drawPoint(self, x, y):
        """
        Draws a point on the airfoil canvas at a specified :math:`x`- and :math:`y`-location.

        Parameters
        ==========
        x: list
            One-element list where the element represents the :math:`x`-location

        y: list
            One-element list where the element represents the :math:`y`-location
        """
        self.geo_col.add_point(x[0], y[0])

    def addPymeadCanvasItem(self, pymead_obj: PymeadObj):
        # Type-specific actions
        if isinstance(pymead_obj, Point):
            # Create the canvas item
            point_gui = DraggablePoint()
            point_gui.setData(pos=np.array([[pymead_obj.x().value(), pymead_obj.y().value()]]), adj=None,
                              pen=pg.mkPen(color=q_settings.value("scatter_default_pen_color",
                                                                  q_settings_descriptions["scatter_default_pen_color"][
                                                                      1])),
                              pxMode=True, hoverable=True, tip=None)

            # Establish a two-way connection between the point data structure and the GUI representation
            pymead_obj.canvas_item = point_gui
            point_gui.point = pymead_obj

            # Set the style
            point_gui.setScatterStyle("default")

            # Connect signals
            point_gui.sigPointClicked.connect(self.pointClicked)
            point_gui.sigPointHovered.connect(self.pointHovered)
            point_gui.sigPointLeaveHovered.connect(self.pointLeaveHovered)
            point_gui.sigPointMoved.connect(self.pointMoved)

            # Add the point to the plot
            self.addItem(point_gui)

        elif isinstance(pymead_obj, ParametricCurve):
            # Create the canvas item
            curve_item = HoverableCurve(curve_type=pymead_obj.curve_type)

            # Establish a two-way connection between the curve data structure and the GUI representation
            pymead_obj.canvas_item = curve_item
            curve_item.parametric_curve = pymead_obj

            # Set the curve style
            curve_item.setCurveStyle("default")

            # Set the curve to be clickable within a specified radius
            curve_item.setClickable(True, width=4)

            # Connect hover/not hover signals
            curve_item.sigCurveHovered.connect(self.curveHovered)
            curve_item.sigCurveNotHovered.connect(self.curveLeaveHovered)
            curve_item.sigLineItemAdded.connect(self.onLineItemAdded)
            curve_item.sigRemove.connect(self.removeCurve)

            # Update the curve data based on the selected control points
            pymead_obj.update()

            # Add the curve to the plot
            self.addItem(curve_item)

        elif isinstance(pymead_obj, GeoCon):

            constraint_item = getattr(sys.modules[__name__], f"{type(pymead_obj).__name__}Item")(pymead_obj)
            # raise NotImplementedError(f"Constraint {pymead_obj.__class__.__name__} does not yet have a canvas item")

            constraint_item.addItems(self)

            # Connect hover/not hover signals
            # TODO: need to connect these

        elif isinstance(pymead_obj, Airfoil):

            pymead_obj.canvas_item = PolygonItem(data=pymead_obj.coords, airfoil=pymead_obj)
            self.addItem(pymead_obj.canvas_item)

            # Connect signals
            pymead_obj.canvas_item.sigPolyEnter.connect(self.airfoil_hovered)
            pymead_obj.canvas_item.sigPolyExit.connect(self.airfoil_exited)

    @staticmethod
    def runSelectionEventLoop(drawing_object: str, starting_message: str):
        drawing_object = drawing_object
        starting_message = starting_message

        def decorator(action: typing.Callable):
            def wrapped(self, *args, **kwargs):
                self.drawing_object = drawing_object
                self.sigStatusBarUpdate.emit(starting_message, 0)
                loop = QEventLoop()
                self.sigEnterPressed.connect(loop.quit)
                self.sigEscapePressed.connect(loop.quit)
                loop.exec()
                # if len(self.geo_col.selected_objects["points"]) > 0:
                action(self, *args, **kwargs)
                self.clearSelectedObjects()
                # elif len(self.geo_col.selected_objects["airfoils"]) > 0:
                #     action(self, *args, **kwargs)
                #     self.clearSelectedObjects()
                self.drawing_object = None
                self.sigStatusBarUpdate.emit("", 0)
            return wrapped
        return decorator

    @runSelectionEventLoop(drawing_object="Points", starting_message="Left click on the canvas to draw a point. "
                                                                     "Press Escape to stop drawing points.")
    def drawPoints(self):
        pass

    @runSelectionEventLoop(drawing_object="Bezier", starting_message="Select the first Bezier control point")
    def drawBezier(self):
        if len(self.geo_col.selected_objects["points"]) < 2:
            msg = f"Choose at least 2 points to define a curve"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return

        point_sequence = PointSequence([pt for pt in self.geo_col.selected_objects["points"]])
        self.geo_col.add_bezier(point_sequence=point_sequence)

    @runSelectionEventLoop(drawing_object="LineSegment", starting_message="Select the first line endpoint")
    def drawLineSegment(self):
        if len(self.geo_col.selected_objects["points"]) < 2:
            msg = f"Choose at least 2 points to define a curve"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return

        point_sequence = PointSequence([pt for pt in self.geo_col.selected_objects["points"]])
        self.geo_col.add_line(point_sequence=point_sequence)

    @runSelectionEventLoop(drawing_object="Airfoil", starting_message="Select the leading edge point")
    def generateAirfoil(self):
        if len(self.geo_col.selected_objects["points"]) not in [2, 4]:
            self.sigStatusBarUpdate.emit(
                "Choose either 2 points (sharp trailing edge) or 4 points (blunt trailing edge)"
                " to generate an airfoil", 4000)

        le = self.geo_col.selected_objects["points"][0]
        te = self.geo_col.selected_objects["points"][1]
        if len(self.geo_col.selected_objects["points"]) > 2:
            upper_surf_end = self.geo_col.selected_objects["points"][2]
            lower_surf_end = self.geo_col.selected_objects["points"][3]
        else:
            upper_surf_end = te
            lower_surf_end = te

        self.geo_col.add_airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=upper_surf_end,
                                 lower_surf_end=lower_surf_end)

    @runSelectionEventLoop(drawing_object="MEA", starting_message="Select the first airfoil")
    def generateMEA(self):
        if len(self.geo_col.selected_objects["airfoils"]) == 0:
            self.sigStatusBarUpdate.emit("Must choose at least 1 airfoil for a multi-element airfoil (MEA) object",
                                         4000)
            return

        self.geo_col.add_mea(airfoils=self.geo_col.selected_objects["airfoils"].copy())

    @runSelectionEventLoop(drawing_object="LengthDimension", starting_message="Select the tool point")
    def addLengthDimension(self):
        if len(self.geo_col.selected_objects["points"]) not in [2, 3]:
            self.sigStatusBarUpdate.emit("Choose either 2 points (no length parameter) or 3 points "
                                         "(specified length parameter)"
                                         " to add a length dimension", 4000)

        tool_point = self.geo_col.selected_objects["points"][0]
        target_point = self.geo_col.selected_objects["points"][1]
        length_param = None if len(self.geo_col.selected_objects["points"]) <= 2 else self.geo_col.selected_objects["points"][2]

        self.geo_col.add_length_dimension(tool_point=tool_point, target_point=target_point, length_param=length_param)

    @runSelectionEventLoop(drawing_object="DistanceConstraint", starting_message="Select the first point")
    def addDistanceConstraint(self):
        if len(self.geo_col.selected_objects["points"]) != 2:
            self.sigStatusBarUpdate.emit("Choose exactly two points to define a distance constraint", 4000)

        # p1 = self.geo_col.selected_objects["points"][0]
        # p2 = self.geo_col.selected_objects["points"][1]
        #
        # cnstr = self.geo_col.add_distance_constraint(p1, p2)
        # cnstr.add_constraint_to_gcs()

        distance = self.geo_col.selected_objects["points"][0].measure_distance(
            self.geo_col.selected_objects["points"][1])
        par = self.geo_col.add_param(value=distance, unit_type="length", assign_unique_name=True)
        constraint = DistanceConstraint(*self.geo_col.selected_objects["points"], value=par)
        self.geo_col.add_constraint(constraint)

    @runSelectionEventLoop(drawing_object="AngleDimension", starting_message="Select the tool point")
    def addAngleDimension(self):
        if len(self.geo_col.selected_objects["points"]) not in [2, 3]:
            self.sigStatusBarUpdate.emit("Choose either 2 points (no angle parameter) or 3 points "
                                         "(specified angle parameter)"
                                         " to add an angle dimension", 4000)
        tool_point = self.geo_col.selected_objects["points"][0]
        target_point = self.geo_col.selected_objects["points"][1]
        angle_param = None if len(self.geo_col.selected_objects["points"]) <= 2 else self.geo_col.selected_objects["points"][2]

        self.geo_col.add_angle_dimension(tool_point=tool_point, target_point=target_point, angle_param=angle_param)

    @runSelectionEventLoop(drawing_object="RelAngle3Constraint", starting_message="Select any point other than "
                                                                                  "the vertex")
    def addRelAngle3Constraint(self):
        if len(self.geo_col.selected_objects["points"]) != 3:
            msg = (f"Choose exactly three points (start, vertex, and end) for a "
                   f"relative angle 3 constraint")
            self.sigStatusBarUpdate.emit(msg, 4000)
            return
        args = []
        for point in self.geo_col.selected_objects["points"]:
            args.extend([point.x().value(), point.y().value()])
        value = measure_rel_angle3(*args)
        par = self.geo_col.add_param(value, unit_type="angle")
        constraint = RelAngle3Constraint(*self.geo_col.selected_objects["points"], value=par)
        self.geo_col.add_constraint(constraint)

    @runSelectionEventLoop(drawing_object="AntiParallel3Constraint", starting_message="Select any point other than "
                                                                                      "the vertex")
    def addAntiParallel3Constraint(self):
        if len(self.geo_col.selected_objects["points"]) != 3:
            msg = (f"Choose exactly three points (start, vertex, and end) for an "
                   f"anti-parallel 3 constraint")
            self.sigStatusBarUpdate.emit(msg, 4000)
            return
        constraint = AntiParallel3Constraint(*self.geo_col.selected_objects["points"])
        self.geo_col.add_constraint(constraint)

    @runSelectionEventLoop(drawing_object="SymmetryConstraint", starting_message="Select the start point of the "
                                                                                 "mirror axis")
    def addSymmetryConstraint(self):
        if len(self.geo_col.selected_objects["points"]) != 4:
            msg = (f"Choose exactly four points (mirror axis start, mirror axis end, tool point, and target point) "
                   f"for a symmetry constraint")
            self.sigStatusBarUpdate.emit(msg, 4000)
            return

        constraint = SymmetryConstraint(*self.geo_col.selected_objects["points"])
        self.geo_col.add_constraint(constraint)

    @runSelectionEventLoop(drawing_object="Perp3Constraint", starting_message="Select the first point (not the vertex)")
    def addPerp3Constraint(self):
        if len(self.geo_col.selected_objects["points"]) != 3:
            msg = f"Choose exactly three points (start, vertex, and end) for a Perp3Constraint"
            self.sigStatusBarUpdate.emit(msg, 4000)
            return
        constraint = Perp3Constraint(*self.geo_col.selected_objects["points"])
        self.geo_col.add_constraint(constraint)

    @runSelectionEventLoop(drawing_object="ROCurvatureConstraint", starting_message="Select the curve joint")
    def addROCurvatureConstraint(self):
        if len(self.geo_col.selected_objects["points"]) != 1:
            msg = (f"Choose exactly one point (the curve joint) for a "
                   f"radius of curvature constraint")
            self.sigStatusBarUpdate.emit(msg, 4000)
            return
        curvature_data = ROCurvatureConstraint.calculate_curvature_data(self.geo_col.selected_objects["points"][0])
        R = 0.5 * (curvature_data.R1 + curvature_data.R2)
        # R = curvature_data.R1
        R_param = self.geo_col.add_param(R, name="ROC-1", unit_type="length")
        constraint = ROCurvatureConstraint(*self.geo_col.selected_objects["points"], value=R_param)
        self.geo_col.add_constraint(constraint)

    def addPointToCurve(self, curve_item: HoverableCurve):
        self.adding_point_to_curve = curve_item
        self.sigStatusBarUpdate.emit("First, click the point to add to the curve", 0)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        loop.exec()
        point_to_add = self.geo_col.selected_objects["points"][0]
        preceding_point = self.geo_col.selected_objects["points"][1]
        prev_item_index = curve_item.point_items.index(preceding_point)
        curve_item.point_items.insert(prev_item_index + 1, point_to_add)
        point_to_add.curveOwners.append(curve_item)
        curve_item.updateCurveItem()
        self.clearSelectedObjects()
        self.adding_point_to_curve = None

    def appendSelectedPoint(self, plot_data_item: pg.PlotDataItem):
        self.geo_col.selected_objects["points"].append(plot_data_item.point)

    def pointClicked(self, scatter_item, spot, ev, point_item):
        if point_item in self.geo_col.selected_objects["points"]:
            return

        if self.drawing_object == "Points":
            return

        if self.point_text_item is not None:
            self.removeItem(self.point_text_item)
            self.point_text_item = None
        # point_item.hoverable = False
        # point_item.setScatterStyle("selected")
        # point_item.point.tree_item.setSelected(True)
        if self.drawing_object == "Bezier":
            self.geo_col.select_object(point_item.point)
            n_ctrl_pts = len(self.geo_col.selected_objects["points"])
            degree = n_ctrl_pts - 1
            msg = (f"Added control point to curve. Number of control points: {len(self.geo_col.selected_objects['points'])} "
                   f"(degree: {degree}). Press 'Enter' to generate the curve.")
            self.sigStatusBarUpdate.emit(msg, 0)
        elif self.drawing_object == "LineSegment":
            if len(self.geo_col.selected_objects["points"]) < 2:
                self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 2:
                self.sigEnterPressed.emit()  # Complete the line after selecting the second point
        elif self.adding_point_to_curve is not None:
            if len(self.geo_col.selected_objects["points"]) < 2:
                self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the preceding point in the sequence", 0)
            if len(self.geo_col.selected_objects["points"]) == 2:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "Airfoil":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, select the trailing edge point. For a blunt trailing edge, the "
                                             "point must have two associated lines (connecting to the upper and lower"
                                             " surface end points).", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                self.sigStatusBarUpdate.emit("Now, select the upper surface endpoint. For a sharp trailing edge, "
                                             "press the enter key to finish generating the airfoil.", 0)
            elif len(self.geo_col.selected_objects["points"]) == 3:
                self.sigStatusBarUpdate.emit("Now, select the lower surface endpoint.", 0)
            elif len(self.geo_col.selected_objects["points"]) == 4:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "LengthDimension":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the target point.", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                # TODO: implement the ability to select a parameter from the tree here
                self.sigEnterPressed.emit()
            elif len(self.geo_col.selected_objects["points"]) == 3:
                # TODO: this currently will not be called until the above TODO is implemented
                self.sigEnterPressed.emit()
        elif self.drawing_object == "AngleDimension":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the target point.", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                # TODO: implement the ability to select a parameter from the tree here
                self.sigEnterPressed.emit()
            elif len(self.geo_col.selected_objects["points"]) == 3:
                # TODO: this currently will not be called until the above TODO is implemented
                self.sigEnterPressed.emit()
        elif self.drawing_object == "DistanceConstraint":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the last point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "CollinearConstraint":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the middle point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                self.sigStatusBarUpdate.emit("Finally, choose the end point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 3:
                self.sigEnterPressed.emit()
        elif self.drawing_object in ["RelAngle3Constraint", "Perp3Constraint", "AntiParallel3Constraint"]:
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the vertex", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                self.sigStatusBarUpdate.emit("Finally, choose the end point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 3:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "SymmetryConstraint":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the mirror axis end point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                self.sigStatusBarUpdate.emit("Now, choose the tool point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 3:
                self.sigStatusBarUpdate.emit("Finally, choose the target point", 0)
            elif len(self.geo_col.selected_objects["points"]) == 4:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "ROCurvatureConstraint":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigEnterPressed.emit()
        elif self.drawing_object in ["RelAngle4Constraint", "ParallelConstraint", "PerpendicularConstraint"]:
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the end point of the tool line", 0)
            elif len(self.geo_col.selected_objects["points"]) == 2:
                self.sigStatusBarUpdate.emit("Choose the start point of the target line", 0)
            elif len(self.geo_col.selected_objects["points"]) == 3:
                self.sigStatusBarUpdate.emit("Now, choose the end point of the target line", 0)
            elif len(self.geo_col.selected_objects["points"]) == 4:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "CurvatureConstraint":
            self.geo_col.select_object(point_item.point)
            if len(self.geo_col.selected_objects["points"]) == 1:
                self.sigEnterPressed.emit()
        else:
            self.geo_col.select_object(point_item.point)

    def pointMoved(self, point: DraggablePoint):
        if self.point_text_item is not None:
            self.removeItem(self.point_text_item)
            self.point_text_item = None
        for curve in point.curveOwners:
            curve.updateCurveItem()

    def setItemStyle(self, item, style: str):
        valid_styles = ["default", "hovered", "selected"]
        if style not in valid_styles:
            raise ValueError(f"Style found ({style}) is not a valid style. Must be one of {valid_styles}.")

        if style == "hovered":
            # Point
            if isinstance(item, DraggablePoint):
                self.point_hovered_item = item
                point = self.point_hovered_item
                if self.point_text_item is None:
                    self.point_text_item = pg.TextItem(
                        f"{point.point.name()}\nx: {point.point.x().value():.6f}\ny: {point.point.y().value():.6f}",
                        anchor=(0, 1))
                    self.point_text_item.setFont(QFont("DejaVu Sans", 8))
                    self.addItem(self.point_text_item)
                    self.point_text_item.setPos(point.point.x().value(), point.point.y().value())
                item.setScatterStyle(mode="hovered")
            # Curve
            elif isinstance(item, HoverableCurve):
                self.curve_hovered_item = item
                item.setCurveStyle("hovered")
            elif isinstance(item, ConstraintItem):
                self.constraint_hovered_item = item
                item.setStyle(mode="hovered")

        elif style == "default":
            # Point
            if isinstance(item, DraggablePoint):
                self.point_hovered_item = None
                self.removeItem(self.point_text_item)
                self.point_text_item = None
                item.setScatterStyle(mode="default")
            # Curve
            elif isinstance(item, HoverableCurve):
                self.curve_hovered_item = None
                item.setCurveStyle("default")
            elif isinstance(item, ConstraintItem):
                self.constraint_hovered_item = None
                item.setStyle(theme=self.gui_obj.themes[self.gui_obj.current_theme])

        elif style == "selected":
            # Point
            if isinstance(item, DraggablePoint):
                self.point_hovered_item = None
                self.removeItem(self.point_text_item)
                self.point_text_item = None
                item.setScatterStyle(mode="selected")
            elif isinstance(item, ConstraintItem):
                self.constraint_hovered_item = None
                item.setStyle(mode="selected")
            # Curve
            # elif isinstance(item, HoverableCurve):
            #     self.curve_hovered_item = None
            #     item.setCurveStyle("default")

    def pointHovered(self, scatter_item, spot, ev, point_item):
        if point_item.dragPoint is not None:
            return
        self.geo_col.hover_enter_obj(point_item.point)

    def pointLeaveHovered(self, scatter_item, spot, ev, point_item):
        self.geo_col.hover_leave_obj(point_item.point)

    def curveHovered(self, item):
        self.geo_col.hover_enter_obj(item.parametric_curve)

    def curveLeaveHovered(self, item):
        self.geo_col.hover_leave_obj(item.parametric_curve)

    def constraintHovered(self, item):
        self.geo_col.hover_enter_obj(item.canvas_item)

    def constraintLeaveHovered(self, item):
        self.geo_col.hover_leave_obj(item.canvas_item)

    def airfoil_hovered(self, airfoil: Airfoil, x_centroid: float, y_centroid: float):
        """
        Adds the name of the airfoil as a label to the airfoil's centroid when the airfoil shape is
        hovered with the mouse

        Parameters
        ==========
        airfoil_name: str
            Name of the airfoil (A0, A1, etc.)

        x_centroid: float
            x-location of the airfoil's centroid

        y_centroid: float
            y-location of the airfoil's centroid
        """

        # Get the color for the text
        # main_color = None if self.gui_obj is None else self.gui_obj.themes[self.gui_obj.current_theme]["main-color"]
        main_color = self.gui_obj.themes[self.gui_obj.current_theme]["main-color"]

        # Add the name of the airfoil as a pg.TextItem if not dragging
        text_item = pg.TextItem(airfoil.name(), anchor=(0.5, 0.5), color=main_color)
        text_item.setFont(QFont("DejaVu Sans", 10))
        self.airfoil_text_items[airfoil] = text_item
        self.airfoil_text_items[airfoil].setPos(x_centroid, y_centroid)
        self.addItem(self.airfoil_text_items[airfoil])

    def airfoil_exited(self, airfoil: Airfoil):
        """
        Remove the label from the airfoil centroid on mouse hover exit
        """
        if airfoil in self.airfoil_text_items.keys():
            self.removeItem(self.airfoil_text_items[airfoil])

    def removeCurve(self, item):
        self.geo_col.remove_pymead_obj(item.parametric_curve)

    def selectPointsToDeepcopy(self):
        self.sigStatusBarUpdate.emit("Click the point to deepcopy", 0)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        loop.exec()
        self.deepcopy_point()
        self.clearSelectedObjects()

    def exportPlot(self):
        dialog = PlotExportDialog(self, gui_obj=self.gui_obj, theme=self.gui_obj.themes[self.gui_obj.current_theme])
        if dialog.exec_():
            # Get the inputs from the dialog
            inputs = dialog.valuesFromWidgets()

            # Create the pyqtgraph ImageExporter object from the airfoil canvas
            exporter = pg.exporters.ImageExporter(self.plot)

            # Set the image width parameter
            exporter.parameters()["width"] = inputs["width"]

            # Prevent saves to the local GUI directory
            if inputs["save_dir"] == "":
                self.gui_obj.disp_message_box(f"Please select a save directory")
                return

            # Prevent saves to an invalid file path
            if not os.path.exists(inputs["save_dir"]):
                self.gui_obj.disp_message_box(f"The save directory {inputs['save_dir']} does not exist",
                                              message_mode="error")
                return

            # If there is no .png extension, add one
            file_path = os.path.join(inputs["save_dir"], inputs["save_name"])
            if not os.path.splitext(file_path)[-1] == ".png":
                file_path += ".png"

            # Export the image
            exporter.export(file_path)

            # Display success message
            self.gui_obj.disp_message_box(f"Plot saved to {file_path}", message_mode="info")

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(f"""
        QMenu::item:selected {{ background-color: 
            {self.gui_obj.themes[self.gui_obj.current_theme]['menu-item-selected-color']} }}
        """)
        create_geometry_menu = menu.addMenu("Create Geometry")
        modify_geometry_menu = menu.addMenu("Modify Geometry")
        add_constraint_menu = menu.addMenu("Add Constraint")

        removeCurveAction, curve_item, insertCurvePointAction = None, None, None
        if self.curve_hovered_item is not None:
            # Curve removal action
            removeCurveAction = modify_geometry_menu.addAction("Remove Curve")
            curve_item = self.curve_hovered_item

            # Curve point insertion action
            insertCurvePointAction = modify_geometry_menu.addAction(
                "Insert Curve Point") if self.curve_hovered_item is not None else None

        drawPointAction = create_geometry_menu.addAction("Insert Point")
        drawBezierCurveThroughPointsAction = create_geometry_menu.addAction("Bezier Curve Through Points")
        drawLineSegmentThroughPointsAction = create_geometry_menu.addAction("Line Segment Through Points")
        generateAirfoilAction = create_geometry_menu.addAction("Generate Airfoil")
        generateMEAAction = create_geometry_menu.addAction("Generate MEA")
        addRelAngle3ConstraintAction = add_constraint_menu.addAction("Add Relative Angle 3 Constraint")
        addPerp3ConstraintAction = add_constraint_menu.addAction("Add Perpendicular 3 constraint")
        addAntiParallel3ConstraintAction = add_constraint_menu.addAction("Add Anti-Parallel 3 Constraint")
        addSymmetryConstraintAction = add_constraint_menu.addAction("Add Symmetry Constraint")
        addROCurvatureConstraintAction = add_constraint_menu.addAction("Add Radius of Curvature Constraint")
        addLengthDimensionAction = add_constraint_menu.addAction("Add Length Dimension")
        addAngleDimensionAction = add_constraint_menu.addAction("Add Angle Dimension")
        addDistanceConstraintAction = add_constraint_menu.addAction("Add Distance Constraint")
        exportPlotAction = menu.addAction("Export Plot")
        view_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
        res = menu.exec_(event.globalPos())
        if res == drawPointAction:
            self.drawPoint(x=[view_pos.x()], y=[view_pos.y()])
        elif res == drawBezierCurveThroughPointsAction:
            self.drawBezier()
        elif res == drawLineSegmentThroughPointsAction:
            self.drawLineSegment()
        elif res == generateAirfoilAction:
            self.generateAirfoil()
        elif res == generateMEAAction:
            self.generateMEA()
        elif res == addRelAngle3ConstraintAction:
            self.addRelAngle3Constraint()
        elif res == addAntiParallel3ConstraintAction:
            self.addAntiParallel3Constraint()
        elif res == addPerp3ConstraintAction:
            self.addPerp3Constraint()
        elif res == addSymmetryConstraintAction:
            self.addSymmetryConstraint()
        elif res == addROCurvatureConstraintAction:
            self.addROCurvatureConstraint()
        elif res == addLengthDimensionAction:
            self.addLengthDimension()
        elif res == addAngleDimensionAction:
            self.addAngleDimension()
        elif res == addDistanceConstraintAction:
            self.addDistanceConstraint()
        elif res == removeCurveAction and curve_item is not None:
            self.removeCurve(curve_item)
        elif res == insertCurvePointAction and curve_item is not None:
            self.addPointToCurve(curve_item)
        elif res == exportPlotAction:
            self.exportPlot()

    def removeSelectedPoints(self):
        self.geo_col.remove_selected_objects()

    def clearSelectedObjects(self):
        self.geo_col.clear_selected_objects()

    def onLineItemAdded(self, line_item):
        line_item.sigCurveHovered.connect(self.curveHovered)
        line_item.sigCurveNotHovered.connect(self.curveLeaveHovered)

    def arrowKeyPointMove(self, key, mods):
        step = 10 * self.geo_col.single_step if mods == Qt.ShiftModifier else self.geo_col.single_step
        if key == Qt.Key_Left:
            for point in self.geo_col.selected_objects["points"]:
                point.request_move(point.x().value() - step, point.y().value())
        elif key == Qt.Key_Right:
            for point in self.geo_col.selected_objects["points"]:
                point.request_move(point.x().value() + step, point.y().value())
        elif key == Qt.Key_Up:
            for point in self.geo_col.selected_objects["points"]:
                point.request_move(point.x().value(), point.y().value() + step)
        elif key == Qt.Key_Down:
            for point in self.geo_col.selected_objects["points"]:
                point.request_move(point.x().value(), point.y().value() - step)

    def mousePressEvent(self, ev):

        super().mousePressEvent(ev)

        if not self.drawing_object == "Points":
            return

        view_pos = self.getPlotItem().getViewBox().mapSceneToView(ev.pos())
        self.geo_col.add_point(view_pos.x(), view_pos.y())

    def keyPressEvent(self, ev):
        mods = QApplication.keyboardModifiers()
        if ev.key() == Qt.Key_Return:
            self.sigEnterPressed.emit()
        elif ev.key() == Qt.Key_Escape:
            self.sigEscapePressed.emit()
            self.geo_col.clear_selected_objects()
            self.sigStatusBarUpdate.emit("", 0)
        elif ev.key() == Qt.Key_Delete:
            self.geo_col.remove_selected_objects()
            self.sigStatusBarUpdate.emit("", 0)
        elif ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Down, Qt.Key_Up) and len(self.geo_col.selected_objects["points"]) > 0:
            self.arrowKeyPointMove(ev.key(), mods)
        elif ev.key() == Qt.Key_P:
            self.drawPoints()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    _geo_col = GeometryCollection()
    plot = AirfoilCanvas(geo_col=_geo_col)
    plot.show()
    app.exec_()
