import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QEventLoop, Qt

from pymead.gui.hoverable_curve_item import BezierCurveItem
from pymead.gui.draggable_point import DraggablePoint


class AirfoilCanvas(pg.PlotWidget):
    sigEnterPressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMenuEnabled(False)
        self.setAspectLocked(True)
        self.disableAutoRange()
        self.points = []
        self.selected_points = []
        self.drawing_curve = False
        self.wait_for_clicked_points_thread = None

    def drawPoint(self, x, y):
        point = DraggablePoint()
        point.setData(pos=np.array([[x[0], y[0]]]), adj=None, pen=pg.mkPen(color="indianred"), symbol="x", size=10,
                      pxMode=True, hoverable=True, hoverBrush=pg.mkBrush(color='gold'), tip=point.hover_tip)
        # point.sigPointsHovered.connect(self.pointHovered)
        # point.scatter.setData(hoverable=True, hoverSymbol="+", hoverBrush=pg.mkBrush(color="green"),
        #                       hoverPen=pg.mkPen(color="limegreen"), hoverSize=15, tip=None)
        # point.sigPointsClicked.connect(self.pointClicked)
        point.sigPointClicked.connect(self.pointClicked)
        point.sigPointMoved.connect(self.pointMoved)
        self.addItem(point)

    def generateBezierCurve(self):
        selected_data = np.zeros(shape=(len(self.selected_points), 2))
        for idx, pt in enumerate(self.selected_points):
            selected_data[idx, :] = pt.data["pos"][0, :]
        bez_item = BezierCurveItem(pen=pg.mkPen(color="steelblue", width=2))
        bez_item.point_items = self.selected_points
        bez_item.setClickable(True, width=4)
        bez_item.sigCurveHovered.connect(self.curveHovered)
        bez_item.sigCurveNotHovered.connect(self.curveLeaveHovered)
        bez_item.updateBezierCurveItem(selected_data)
        for pt in bez_item.point_items:
            pt.curveOwners.append(bez_item)
        self.addItem(bez_item)

    def drawBezierCurveThroughPoints(self):
        self.drawing_curve = True
        self.setWindowModality(Qt.ApplicationModal)
        loop = QEventLoop()
        self.sigEnterPressed.connect(loop.quit)
        loop.exec()
        self.generateBezierCurve()
        self.selected_points = []
        self.drawing_curve = False

    def appendSelectedPoint(self, plot_data_item: pg.PlotDataItem):
        print("Appending point!")
        self.selected_points.append(plot_data_item)

    def pointClicked(self, scatter_item, spot, ev, point_item):
        if self.drawing_curve:
            self.appendSelectedPoint(point_item)

    @staticmethod
    def pointMoved(point: DraggablePoint):
        for curve in point.curveOwners:
            curve.updateBezierCurveItem()

    @staticmethod
    def curveHovered(item):
        item.setPen(pg.mkPen(color="white", width=3))

    @staticmethod
    def curveLeaveHovered(item):
        item.setPen(pg.mkPen(color="cornflowerblue", width=2))

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        drawPointAction = menu.addAction("Insert Point")
        drawBezierCurveThroughPointsAction = menu.addAction("Bezier Curve Through Points")
        view_pos = self.getPlotItem().getViewBox().mapSceneToView(event.pos())
        res = menu.exec_(event.globalPos())
        if res == drawPointAction:
            self.drawPoint(x=[view_pos.x()], y=[view_pos.y()])
        elif res == drawBezierCurveThroughPointsAction:
            self.drawBezierCurveThroughPoints()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Return:
            self.sigEnterPressed.emit()


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    plot = AirfoilCanvas()
    plot.show()
    app.exec_()
