import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QEventLoop, Qt
from PyQt5.QtWidgets import QApplication

from pymead.core.bezier2 import Bezier
from pymead.core.line2 import LineSegment
from pymead.core.constraints import CollinearConstraint
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.point import PointSequence
from pymead.gui.hoverable_curve_item import HoverableCurveItem
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
        self.selected_points = None
        self.drawing_curve = None
        self.creating_collinear_constraint = None
        self.adding_point_to_curve = None
        self.curve_hovered_item = None
        self.point_hovered_item = None
        self.point_text_item = None
        self.geo_col = geo_col
        self.geo_col.geo_canvas = self

    def drawPoint(self, x, y):
        point_gui = DraggablePoint()
        point_gui.setData(pos=np.array([[x[0], y[0]]]), adj=None,
                      pen=pg.mkPen(color=q_settings.value("scatter_default_pen_color",
                                                          q_settings_descriptions["scatter_default_pen_color"][1])),
                      pxMode=True, hoverable=True, tip=None)
        point = self.geo_col.add_point(x[0], y[0])

        # Establish a two-way connection between the point data structure and the GUI representation
        point.gui_obj = point_gui
        point_gui.point = point

        # Set the style
        point_gui.setScatterStyle("default")

        # Connect signals
        point_gui.sigPointClicked.connect(self.pointClicked)
        point_gui.sigPointHovered.connect(self.pointHovered)
        point_gui.sigPointLeaveHovered.connect(self.pointLeaveHovered)
        point_gui.sigPointMoved.connect(self.pointMoved)
        self.addItem(point_gui)

    def generateCurve(self, curve_type: str):
        if len(self.selected_points) < 2:
            msg = f"Choose at least 2 points to define a curve"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return
        # selected_data = np.zeros(shape=(len(self.selected_points), 2))
        # for idx, pt in enumerate(self.selected_points):
        #     selected_data[idx, :] = pt.data["pos"][0, :]

        # bezier = Bezier(point_sequence=PointSequence(points=[pt.point for pt in self.selected_points]))

        if curve_type == "Bezier":
            curve = self.geo_col.add_bezier(
                point_sequence=PointSequence(points=[pt.point for pt in self.selected_points])
            )
        elif curve_type == "LineSegment":
            curve = self.geo_col.add_line(
                point_sequence=PointSequence(points=[pt.point for pt in self.selected_points])
            )
        else:
            raise NotImplementedError(f"Curve type {curve_type} not implemented")

        # Generate the curve item
        curve_item = HoverableCurveItem(curve_type=curve_type)
        curve.gui_obj = curve_item
        curve_item.parametric_curve = curve

        curve_item.setCurveStyle("default")
        # curve_item.point_items = self.selected_points

        # Set the curve to be clickable within a specified radius
        curve_item.setClickable(True, width=4)

        # Connect hover/not hover signals
        curve_item.sigCurveHovered.connect(self.curveHovered)
        curve_item.sigCurveNotHovered.connect(self.curveLeaveHovered)
        curve_item.sigLineItemAdded.connect(self.onLineItemAdded)
        curve_item.sigRemove.connect(self.removeCurve)

        # Update the curve data based on the selected control points
        # curve_item.updateCurveItem(curve_data=bezier.evaluate())
        curve.update()

        # Update the control point net
        # curve_item.generateControlPointNetItems()
        # curve_item.setControlPointNetStyle("default")

        # Add the curve as an owner to each of the curve's point items
        # for pt in curve_item.point_items:
        #     pt.curveOwners.append(curve_item)
        # for line_item in curve_item.control_point_line_items:
        #     for pt in line_item.point_items:
        #         pt.curveOwners.append(line_item)

        # Add the curve to the plot
        self.addItem(curve_item)

        # Add the control point net to the plot
        # if curve_item.control_point_line_items is not None:
        #     for item in curve_item.control_point_line_items:
        #         self.addItem(item)

        # Reset the status bar
        self.sigStatusBarUpdate.emit("", 0)

    def drawCurveThroughPoints(self, curve_type: str):
        self.drawing_curve = curve_type
        self.setWindowModality(Qt.ApplicationModal)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        self.sigEscapePressed.connect(loop.quit)
        loop.exec()
        if self.selected_points is not None:
            self.generateCurve(curve_type=curve_type)
            self.clearSelectedPoints()
        self.drawing_curve = None

    def makeCollinearConstraint(self):
        if len(self.selected_points) != 3:
            msg = f"Choose exactly 3 points for a collinear constraint"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return
        constraint = CollinearConstraint(start_point=self.selected_points[0].point,
                                         middle_point=self.selected_points[1].point,
                                         end_point=self.selected_points[2].point)
        constraint.enforce("start")

    def makePointsCollinear(self):
        self.creating_collinear_constraint = True
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        self.sigEscapePressed.connect(loop.quit)
        loop.exec()
        if self.selected_points is not None:
            self.makeCollinearConstraint()
            self.clearSelectedPoints()
        self.creating_collinear_constraint = None

    def addPointToCurve(self, curve_item: HoverableCurveItem):
        self.adding_point_to_curve = curve_item
        self.sigStatusBarUpdate.emit("First, click the point to add to the curve", 0)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        loop.exec()
        point_to_add = self.selected_points[0]
        preceding_point = self.selected_points[1]
        prev_item_index = curve_item.point_items.index(preceding_point)
        curve_item.point_items.insert(prev_item_index + 1, point_to_add)
        point_to_add.curveOwners.append(curve_item)
        curve_item.updateCurveItem()
        self.clearSelectedPoints()
        self.adding_point_to_curve = None

    def appendSelectedPoint(self, plot_data_item: pg.PlotDataItem):
        self.selected_points.append(plot_data_item)

    def pointClicked(self, scatter_item, spot, ev, point_item):
        if self.selected_points is None:
            self.selected_points = []
        if point_item in self.selected_points:
            return
        if self.point_text_item is not None:
            self.removeItem(self.point_text_item)
            self.point_text_item = None
        point_item.hoverable = False
        point_item.setScatterStyle("selected")
        if self.drawing_curve is None and self.adding_point_to_curve is None:
            self.appendSelectedPoint(point_item)
        elif self.drawing_curve == "Bezier":
            self.appendSelectedPoint(point_item)
            n_ctrl_pts = len(self.selected_points)
            degree = n_ctrl_pts - 1
            msg = (f"Added control point to curve. Number of control points: {len(self.selected_points)} "
                   f"(degree: {degree}). Press 'Enter' to generate the curve.")
            self.sigStatusBarUpdate.emit(msg, 0)
        elif self.drawing_curve == "LineSegment":
            if len(self.selected_points) < 2:
                self.appendSelectedPoint(point_item)
            if len(self.selected_points) == 2:
                self.sigEnterPressed.emit()  # Complete the line after selecting the second point
        elif self.adding_point_to_curve is not None:
            if len(self.selected_points) < 2:
                self.appendSelectedPoint(point_item)
            if len(self.selected_points) == 1:
                self.sigStatusBarUpdate.emit("Now, choose the preceding point in the sequence", 0)
            if len(self.selected_points) == 2:
                self.sigEnterPressed.emit()
                self.sigStatusBarUpdate.emit("", 0)

    def pointMoved(self, point: DraggablePoint):
        if self.point_text_item is not None:
            self.removeItem(self.point_text_item)
            self.point_text_item = None
        for curve in point.curveOwners:
            curve.updateCurveItem()

    def pointHovered(self, scatter_item, spot, ev, point_item):
        if point_item.dragPoint is not None:
            return
        self.point_hovered_item = point_item
        point = self.point_hovered_item.point
        if self.point_text_item is None:
            self.point_text_item = pg.TextItem(
                f"{point.name()}\nx: {point.x().value():.6f}\ny: {point.y().value():.6f}", anchor=(0, 1))
            self.addItem(self.point_text_item)
            self.point_text_item.setPos(point.x().value(), point.y().value())
        point_item.setScatterStyle(mode="hovered")

    def pointLeaveHovered(self, scatter_item, spot, ev, point_item):
        self.point_hovered_item = None
        self.removeItem(self.point_text_item)
        self.point_text_item = None
        point_item.setScatterStyle(mode="default")

    def curveHovered(self, item):
        self.curve_hovered_item = item
        item.setCurveStyle("hovered")

    def curveLeaveHovered(self, item):
        self.curve_hovered_item = None
        item.setCurveStyle("default")

    def removeCurve(self, item):
        # if item.control_point_line_items is not None:
        #     for sub_item in item.control_point_line_items:
        #         self.removeItem(sub_item)
        # item.remove()
        if isinstance(item.parametric_curve, Bezier):
            self.geo_col.remove_bezier(item.parametric_curve)
        elif isinstance(item.parametric_curve, LineSegment):
            self.geo_col.remove_line(item.parametric_curve)
        self.removeItem(item)

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)

        removeCurveAction, curve_item, insertCurvePointAction = None, None, None
        if self.curve_hovered_item is not None:
            print(f"{self.curve_hovered_item.parametric_curve.parent_curve = }")
        if self.curve_hovered_item is not None and self.curve_hovered_item.parametric_curve.parent_curve is None:
            # Curve removal action
            removeCurveAction = menu.addAction("Remove Curve")
            curve_item = self.curve_hovered_item

            # Curve point insertion action
            insertCurvePointAction = menu.addAction(
                "Insert Curve Point") if self.curve_hovered_item is not None else None

        drawPointAction = menu.addAction("Insert Point")
        drawBezierCurveThroughPointsAction = menu.addAction("Bezier Curve Through Points")
        drawLineSegmentThroughPointsAction = menu.addAction("Line Segment Through Points")
        makePointsCollinearAction = menu.addAction("Make 3 Points Collinear")
        view_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
        res = menu.exec_(event.globalPos())
        if res == drawPointAction:
            self.drawPoint(x=[view_pos.x()], y=[view_pos.y()])
        elif res == drawBezierCurveThroughPointsAction:
            self.drawCurveThroughPoints(curve_type="Bezier")
        elif res == drawLineSegmentThroughPointsAction:
            self.drawCurveThroughPoints(curve_type="LineSegment")
        elif res == makePointsCollinearAction:
            self.makePointsCollinear()
        elif res == removeCurveAction and curve_item is not None:
            curve_item.remove()
            self.removeCurve(curve_item)
        elif res == insertCurvePointAction and curve_item is not None:
            self.addPointToCurve(curve_item)

    def removeSelectedPoints(self):
        curves_to_delete = []
        curves_to_update = []
        for pt in self.selected_points:
            self.geo_col.remove_point(pt.point)
            delete_curve = False
            for curve in pt.point.curves:
                delete_curve = curve.remove_point(point=pt.point)
                if delete_curve and curve not in curves_to_delete:
                    curves_to_delete.append(curve)
                    if curve in curves_to_update:
                        curves_to_update.remove(curve)
                if not delete_curve and curve not in curves_to_update:
                    curves_to_update.append(curve)

            # for curve in pt.curveOwners:
            #     curve.removeControlPointFromNet(pt)  # Action only occurs if curve_type == "Bezier"
            #     curve.point_items.remove(pt)
            #     if len(curve.point_items) < 2:  # No curve is valid with only one point, so just remove the curve
            #         self.removeCurve(curve)
            #     else:  # Otherwise, update the curve
            #         curve.updateCurveItem()
            self.removeItem(pt)
        self.selected_points.clear()

        for curve in curves_to_delete:
            curve.remove()
        for curve in curves_to_update:
            curve.update()

    def clearSelectedPoints(self):
        if self.selected_points is None:
            return
        for pt in self.selected_points:
            pt.setScatterStyle("default")
            pt.hoverable = True
        self.selected_points = None

    def onLineItemAdded(self, line_item):
        line_item.sigCurveHovered.connect(self.curveHovered)
        line_item.sigCurveNotHovered.connect(self.curveLeaveHovered)

    def arrowKeyPointMove(self, key, mods):
        step = 10 * self.geo_col.single_step if mods == Qt.ShiftModifier else self.geo_col.single_step
        if key == Qt.Key_Left:
            for point in self.selected_points:
                point.point.request_move(point.point.x().value() - step, point.point.y().value())
        elif key == Qt.Key_Right:
            for point in self.selected_points:
                point.point.request_move(point.point.x().value() + step, point.point.y().value())
        elif key == Qt.Key_Up:
            for point in self.selected_points:
                point.point.request_move(point.point.x().value(), point.point.y().value() + step)
        elif key == Qt.Key_Down:
            for point in self.selected_points:
                point.point.request_move(point.point.x().value(), point.point.y().value() - step)

    def keyPressEvent(self, ev):
        mods = QApplication.keyboardModifiers()
        if ev.key() == Qt.Key_Return:
            self.sigEnterPressed.emit()
        elif ev.key() == Qt.Key_Delete:
            self.removeSelectedPoints()
        elif ev.key() == Qt.Key_Escape:
            self.clearSelectedPoints()
            self.sigEscapePressed.emit()
        elif ev.key() in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Down, Qt.Key_Up) and self.selected_points is not None:
            self.arrowKeyPointMove(ev.key(), mods)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    _geo_col = GeometryCollection()
    plot = AirfoilCanvas(geo_col=_geo_col)
    plot.show()
    app.exec_()
