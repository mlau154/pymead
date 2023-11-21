import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QEventLoop, Qt

from pymead.core.geometry_collection import GeometryCollection
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
        self.adding_point_to_curve = None
        self.curve_hovered_item = None
        self.point_hovered_item = None
        self.geo_col = geo_col

    def drawPoint(self, x, y):
        point = DraggablePoint()
        point.setData(pos=np.array([[x[0], y[0]]]), adj=None,
                      pen=pg.mkPen(color=q_settings.value("scatter_default_pen_color",
                                                          q_settings_descriptions["scatter_default_pen_color"][1])),
                      pxMode=True, hoverable=True, tip=point.hover_tip)
        point.setScatterStyle("default")
        point.sigPointClicked.connect(self.pointClicked)
        point.sigPointHovered.connect(self.pointHovered)
        point.sigPointLeaveHovered.connect(self.pointLeaveHovered)
        point.sigPointMoved.connect(self.pointMoved)
        self.addItem(point)

    def generateCurve(self, curve_type: str):
        if len(self.selected_points) < 2:
            msg = f"Choose at least 2 points to define a curve"
            self.sigStatusBarUpdate.emit(msg, 2000)
            return
        # selected_data = np.zeros(shape=(len(self.selected_points), 2))
        # for idx, pt in enumerate(self.selected_points):
        #     selected_data[idx, :] = pt.data["pos"][0, :]

        # Generate the curve item
        curve_item = HoverableCurveItem(curve_type=curve_type)
        curve_item.setCurveStyle("default")
        curve_item.point_items = self.selected_points

        # Set the curve to be clickable within a specified radius
        curve_item.setClickable(True, width=4)

        # Connect hover/not hover signals
        curve_item.sigCurveHovered.connect(self.curveHovered)
        curve_item.sigCurveNotHovered.connect(self.curveLeaveHovered)
        curve_item.sigLineItemAdded.connect(self.onLineItemAdded)

        # Update the curve data based on the selected control points
        curve_item.updateCurveItem()

        # Update the control point net
        curve_item.generateControlPointNetItems()
        # curve_item.setControlPointNetStyle("default")

        # Add the curve as an owner to each of the curve's point items
        for pt in curve_item.point_items:
            pt.curveOwners.append(curve_item)
        for line_item in curve_item.control_point_line_items:
            for pt in line_item.point_items:
                pt.curveOwners.append(line_item)

        # Add the curve to the plot
        self.addItem(curve_item)

        # Add the control point net to the plot
        if curve_item.control_point_line_items is not None:
            for item in curve_item.control_point_line_items:
                self.addItem(item)

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

    @staticmethod
    def pointMoved(point: DraggablePoint):
        for curve in point.curveOwners:
            curve.updateCurveItem()

    def pointHovered(self, scatter_item, spot, ev, point_item):
        self.point_hovered_item = point_item
        point_item.setScatterStyle(mode="hovered")

    def pointLeaveHovered(self, scatter_item, spot, ev, point_item):
        self.point_hovered_item = None
        point_item.setScatterStyle(mode="default")

    def curveHovered(self, item):
        self.curve_hovered_item = item
        item.setCurveStyle("hovered")

    def curveLeaveHovered(self, item):
        self.curve_hovered_item = None
        item.setCurveStyle("default")

    def removeCurve(self, item):
        if item.control_point_line_items is not None:
            for sub_item in item.control_point_line_items:
                self.removeItem(sub_item)
        item.remove()
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
        view_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
        res = menu.exec_(event.globalPos())
        if res == drawPointAction:
            self.drawPoint(x=[view_pos.x()], y=[view_pos.y()])
        elif res == drawBezierCurveThroughPointsAction:
            self.drawCurveThroughPoints(curve_type="Bezier")
        elif res == drawLineSegmentThroughPointsAction:
            self.drawCurveThroughPoints(curve_type="LineSegment")
        elif res == removeCurveAction and curve_item is not None:
            curve_item.remove()
            self.removeCurve(curve_item)
        elif res == insertCurvePointAction and curve_item is not None:
            self.addPointToCurve(curve_item)

    def removeSelectedPoints(self):
        for pt in self.selected_points:
            for curve in pt.curveOwners:
                curve.removeControlPointFromNet(pt)  # Action only occurs if curve_type == "Bezier"
                curve.point_items.remove(pt)
                if len(curve.point_items) < 2:  # No curve is valid with only one point, so just remove the curve
                    self.removeCurve(curve)
                else:  # Otherwise, update the curve
                    curve.updateCurveItem()
            self.removeItem(pt)
        self.selected_points.clear()

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

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Return:
            self.sigEnterPressed.emit()
        elif ev.key() == Qt.Key_Delete:
            self.removeSelectedPoints()
        elif ev.key() == Qt.Key_Escape:
            self.clearSelectedPoints()
            self.sigEscapePressed.emit()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    geo_col = GeometryCollection()
    plot = AirfoilCanvas(geo_col=geo_col)
    plot.show()
    app.exec_()
