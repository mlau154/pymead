import os
import typing
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QEventLoop, Qt
from PyQt5.QtWidgets import QApplication

from pymead.core.bezier2 import Bezier
from pymead.core.line2 import LineSegment
from pymead.core.constraints import CollinearConstraint
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.parametric_curve2 import ParametricCurve
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj
from pymead.gui.hoverable_curve import HoverableCurve
from pymead.gui.draggable_point import DraggablePoint
from pymead.utils.read_write_files import load_data
from pymead import q_settings, GUI_SETTINGS_DIR

q_settings_descriptions = load_data(os.path.join(GUI_SETTINGS_DIR, "q_settings_descriptions.json"))


class AirfoilCanvas(pg.PlotWidget):
    sigEnterPressed = pyqtSignal()
    sigEscapePressed = pyqtSignal()
    sigStatusBarUpdate = pyqtSignal(str, int)

    def __init__(self, geo_col: GeometryCollection):
        super().__init__()
        self.setMenuEnabled(False)
        self.setAspectLocked(True)
        self.disableAutoRange()
        self.points = []
        self.drawing_object = None
        self.creating_collinear_constraint = None
        self.adding_point_to_curve = None
        self.curve_hovered_item = None
        self.point_hovered_item = None
        self.point_text_item = None
        self.geo_col = geo_col
        self.geo_col.canvas = self
        self.plot = self.getPlotItem()

    def toggleGrid(self):
        x_state = self.plot.ctrl.xGridCheck.checkState()
        y_state = self.plot.ctrl.yGridCheck.checkState()
        if x_state or y_state:
            self.plot.showGrid(x=False, y=False)
        else:
            self.plot.showGrid(x=True, y=True)

    def drawPoint(self, x, y):
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
                if len(self.geo_col.selected_points) > 0:
                    action(self, *args, **kwargs)
                    self.clearSelectedPoints()
                self.drawing_object = None
                self.sigStatusBarUpdate.emit("", 0)
            return wrapped
        return decorator

    @runSelectionEventLoop(drawing_object="Bezier", starting_message="Select the first Bezier control point")
    def drawBezier(self):
        if len(self.geo_col.selected_points) < 2:
            msg = f"Choose at least 2 points to define a curve"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return

        point_sequence = PointSequence([pt for pt in self.geo_col.selected_points])
        self.geo_col.add_bezier(point_sequence=point_sequence)

    @runSelectionEventLoop(drawing_object="LineSegment", starting_message="Select the first line endpoint")
    def drawLineSegment(self):
        if len(self.geo_col.selected_points) < 2:
            msg = f"Choose at least 2 points to define a curve"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return

        point_sequence = PointSequence([pt for pt in self.geo_col.selected_points])
        self.geo_col.add_line(point_sequence=point_sequence)

    @runSelectionEventLoop(drawing_object="Airfoil", starting_message="Select the leading edge point")
    def generateAirfoil(self):
        if len(self.geo_col.selected_points) not in [2, 4]:
            self.sigStatusBarUpdate.emit(
                "Choose either 2 points (sharp trailing edge) or 4 points (blunt trailing edge)"
                " to generate an airfoil", 4000)

        le = self.geo_col.selected_points[0]
        te = self.geo_col.selected_points[1]
        if len(self.geo_col.selected_points) > 2:
            upper_surf_end = self.geo_col.selected_points[2]
            lower_surf_end = self.geo_col.selected_points[3]
        else:
            upper_surf_end = te
            lower_surf_end = te

        self.geo_col.add_airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=upper_surf_end,
                                 lower_surf_end=lower_surf_end)

    @runSelectionEventLoop(drawing_object="LengthDimension", starting_message="Select the tool point")
    def addLengthDimension(self):
        if len(self.geo_col.selected_points) not in [2, 3]:
            self.sigStatusBarUpdate.emit("Choose either 2 points (no length parameter) or 3 points "
                                         "(specified length parameter)"
                                         " to add a length dimension", 4000)

        tool_point = self.geo_col.selected_points[0]
        target_point = self.geo_col.selected_points[1]
        length_param = None if len(self.geo_col.selected_points) <= 2 else self.geo_col.selected_points[2]

        self.geo_col.add_length_dimension(tool_point=tool_point, target_point=target_point, length_param=length_param)

    @runSelectionEventLoop(drawing_object="AngleDimension", starting_message="Select the tool point")
    def addAngleDimension(self):
        if len(self.geo_col.selected_points) not in [2, 3]:
            self.sigStatusBarUpdate.emit("Choose either 2 points (no angle parameter) or 3 points "
                                         "(specified angle parameter)"
                                         " to add an angle dimension", 4000)
        tool_point = self.geo_col.selected_points[0]
        target_point = self.geo_col.selected_points[1]
        angle_param = None if len(self.geo_col.selected_points) <= 2 else self.geo_col.selected_points[2]

        self.geo_col.add_angle_dimension(tool_point=tool_point, target_point=target_point, angle_param=angle_param)

    @runSelectionEventLoop(drawing_object="CollinearConstraint", starting_message="Select the start point")
    def addCollinearConstraint(self):
        if len(self.geo_col.selected_points) != 3:
            msg = f"Choose exactly 3 points for a collinear constraint"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return
        constraint = self.geo_col.add_collinear_constraint(start_point=self.geo_col.selected_points[0],
                                                           middle_point=self.geo_col.selected_points[1],
                                                           end_point=self.geo_col.selected_points[2])
        constraint.enforce("start")

    @runSelectionEventLoop(drawing_object="CurvatureConstraint", starting_message="Select the curve joint")
    def addCurvatureConstraint(self):
        if len(self.geo_col.selected_points) != 1:
            msg = f"Choose only one point (the curve joint) for a curvature constraint"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return
        constraint = self.geo_col.add_curvature_constraint(curve_joint=self.geo_col.selected_points[0])
        constraint.enforce(constraint.target().points()[1])

    def addPointToCurve(self, curve_item: HoverableCurve):
        self.adding_point_to_curve = curve_item
        self.sigStatusBarUpdate.emit("First, click the point to add to the curve", 0)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        loop.exec()
        point_to_add = self.geo_col.selected_points[0]
        preceding_point = self.geo_col.selected_points[1]
        prev_item_index = curve_item.point_items.index(preceding_point)
        curve_item.point_items.insert(prev_item_index + 1, point_to_add)
        point_to_add.curveOwners.append(curve_item)
        curve_item.updateCurveItem()
        self.clearSelectedPoints()
        self.adding_point_to_curve = None

    def appendSelectedPoint(self, plot_data_item: pg.PlotDataItem):
        self.geo_col.selected_points.append(plot_data_item.point)

    def pointClicked(self, scatter_item, spot, ev, point_item):
        if point_item in self.geo_col.selected_points:
            return
        if self.point_text_item is not None:
            self.removeItem(self.point_text_item)
            self.point_text_item = None
        # point_item.hoverable = False
        # point_item.setScatterStyle("selected")
        # point_item.point.tree_item.setSelected(True)

        if self.drawing_object == "Bezier":
            self.geo_col.select_point(point_item.point)
            n_ctrl_pts = len(self.geo_col.selected_points)
            degree = n_ctrl_pts - 1
            msg = (f"Added control point to curve. Number of control points: {len(self.geo_col.selected_points)} "
                   f"(degree: {degree}). Press 'Enter' to generate the curve.")
            self.sigStatusBarUpdate.emit(msg, 0)
        elif self.drawing_object == "LineSegment":
            if len(self.geo_col.selected_points) < 2:
                self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 2:
                self.sigEnterPressed.emit()  # Complete the line after selecting the second point
        elif self.adding_point_to_curve is not None:
            if len(self.geo_col.selected_points) < 2:
                self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the preceding point in the sequence", 0)
            if len(self.geo_col.selected_points) == 2:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "Airfoil":
            self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 1:
                self.sigStatusBarUpdate.emit("Now, select the trailing edge point. For a blunt trailing edge, the "
                                             "point must have two associated lines (connecting to the upper and lower"
                                             " surface end points).", 0)
            elif len(self.geo_col.selected_points) == 2:
                self.sigStatusBarUpdate.emit("Now, select the upper surface endpoint. For a sharp trailing edge, "
                                             "press the enter key to finish generating the airfoil.", 0)
            elif len(self.geo_col.selected_points) == 3:
                self.sigStatusBarUpdate.emit("Now, select the lower surface endpoint.", 0)
            elif len(self.geo_col.selected_points) == 4:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "LengthDimension":
            self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the target point.", 0)
            elif len(self.geo_col.selected_points) == 2:
                # TODO: implement the ability to select a parameter from the tree here
                self.sigEnterPressed.emit()
            elif len(self.geo_col.selected_points) == 3:
                # TODO: this currently will not be called until the above TODO is implemented
                self.sigEnterPressed.emit()
        elif self.drawing_object == "AngleDimension":
            self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the target point.", 0)
            elif len(self.geo_col.selected_points) == 2:
                # TODO: implement the ability to select a parameter from the tree here
                self.sigEnterPressed.emit()
            elif len(self.geo_col.selected_points) == 3:
                # TODO: this currently will not be called until the above TODO is implemented
                self.sigEnterPressed.emit()
        elif self.drawing_object == "CollinearConstraint":
            self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the middle point", 0)
            elif len(self.geo_col.selected_points) == 2:
                self.sigStatusBarUpdate.emit("Finally, choose the end point", 0)
            elif len(self.geo_col.selected_points) == 3:
                self.sigEnterPressed.emit()
        elif self.drawing_object == "CurvatureConstraint":
            self.geo_col.select_point(point_item.point)
            if len(self.geo_col.selected_points) == 1:
                self.sigEnterPressed.emit()
        else:
            self.geo_col.select_point(point_item.point)

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
                    self.addItem(self.point_text_item)
                    self.point_text_item.setPos(point.point.x().value(), point.point.y().value())
                item.setScatterStyle(mode="hovered")
            # Curve
            elif isinstance(item, HoverableCurve):
                self.curve_hovered_item = item
                item.setCurveStyle("hovered")

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

        elif style == "selected":
            # Point
            if isinstance(item, DraggablePoint):
                self.point_hovered_item = None
                self.removeItem(self.point_text_item)
                self.point_text_item = None
                item.setScatterStyle(mode="selected")
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

    def removeCurve(self, item):
        self.geo_col.remove_pymead_obj(item.parametric_curve)

    def selectPointsToDeepcopy(self):
        self.sigStatusBarUpdate.emit("Click the point to deepcopy", 0)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        loop.exec()
        self.deepcopy_point()
        self.clearSelectedPoints()

    def deepcopy_point(self):
        point = self.geo_col.selected_points[0]
        new_point = deepcopy(point)
        print(f"{point.tree_item = }, {new_point.tree_item = }, {point.canvas_item = }, {new_point.canvas_item = }")

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)

        removeCurveAction, curve_item, insertCurvePointAction = None, None, None
        if self.curve_hovered_item is not None:
            # Curve removal action
            removeCurveAction = menu.addAction("Remove Curve")
            curve_item = self.curve_hovered_item

            # Curve point insertion action
            insertCurvePointAction = menu.addAction(
                "Insert Curve Point") if self.curve_hovered_item is not None else None

        drawPointAction = menu.addAction("Insert Point")
        drawBezierCurveThroughPointsAction = menu.addAction("Bezier Curve Through Points")
        drawLineSegmentThroughPointsAction = menu.addAction("Line Segment Through Points")
        generateAirfoilAction = menu.addAction("Generate Airfoil")
        makePointsCollinearAction = menu.addAction("Add Collinear Constraint")
        addCurvatureConstraintAction = menu.addAction("Add Curvature Constraint")
        addLengthDimensionAction = menu.addAction("Add Length Dimension")
        addAngleDimensionAction = menu.addAction("Add Angle Dimension")
        deepcopyPointsAction = menu.addAction("Deepcopy point")
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
        elif res == makePointsCollinearAction:
            self.addCollinearConstraint()
        elif res == addCurvatureConstraintAction:
            self.addCurvatureConstraint()
        elif res == addLengthDimensionAction:
            self.addLengthDimension()
        elif res == addAngleDimensionAction:
            self.addAngleDimension()
        elif res == removeCurveAction and curve_item is not None:
            self.removeCurve(curve_item)
        elif res == insertCurvePointAction and curve_item is not None:
            self.addPointToCurve(curve_item)
        elif res == deepcopyPointsAction:
            self.selectPointsToDeepcopy()

    def removeSelectedPoints(self):
        self.geo_col.remove_selected_points()

    def clearSelectedPoints(self):
        self.geo_col.clear_selected_points()

    def onLineItemAdded(self, line_item):
        line_item.sigCurveHovered.connect(self.curveHovered)
        line_item.sigCurveNotHovered.connect(self.curveLeaveHovered)

    def arrowKeyPointMove(self, key, mods):
        step = 10 * self.geo_col.single_step if mods == Qt.ShiftModifier else self.geo_col.single_step
        if key == Qt.Key_Left:
            for point in self.geo_col.selected_points:
                point.request_move(point.x().value() - step, point.y().value())
        elif key == Qt.Key_Right:
            for point in self.geo_col.selected_points:
                point.request_move(point.x().value() + step, point.y().value())
        elif key == Qt.Key_Up:
            for point in self.geo_col.selected_points:
                point.request_move(point.x().value(), point.y().value() + step)
        elif key == Qt.Key_Down:
            for point in self.geo_col.selected_points:
                point.request_move(point.x().value(), point.y().value() - step)

    def keyPressEvent(self, ev):
        mods = QApplication.keyboardModifiers()
        if ev.key() == Qt.Key_Return:
            self.sigEnterPressed.emit()
        elif ev.key() == Qt.Key_Delete:
            self.geo_col.remove_selected_points()
        elif ev.key() == Qt.Key_Escape:
            self.geo_col.clear_selected_points()
            self.sigStatusBarUpdate.emit("", 0)
            self.sigEscapePressed.emit()
        elif ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Down, Qt.Key_Up) and len(self.geo_col.selected_points) > 0:
            self.arrowKeyPointMove(ev.key(), mods)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    _geo_col = GeometryCollection()
    plot = AirfoilCanvas(geo_col=_geo_col)
    plot.show()
    app.exec_()
