import os

import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np

from pymead import q_settings, GUI_SETTINGS_DIR
from pymead.utils.read_write_files import load_data

q_settings_descriptions = load_data(os.path.join(GUI_SETTINGS_DIR, "q_settings_descriptions.json"))


class DraggablePoint(pg.GraphItem):
    sigPointClicked = pyqtSignal(object, object, object, object)
    sigPointHovered = pyqtSignal(object, object, object, object)
    sigPointLeaveHovered = pyqtSignal(object, object, object, object)
    sigPointStartedMoving = pyqtSignal(object)
    sigPointMoving = pyqtSignal(object)
    sigPointFinishedMoving = pyqtSignal(object)

    def __init__(self):
        self.point = None
        self.dragPoint = None
        self.dragOffset = None
        self.curveOwners = []
        self.textItems = []
        super().__init__()
        self.scatter.sigClicked.connect(self.clicked)
        self.scatter.sigHovered.connect(self.hovered)
        self.hoverable = True

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def updateCanvasItem(self, x: float, y: float):
        self.data["pos"][0] = np.array([x, y])
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        self.sigPointMoving.emit(self)

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def mouseDragEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            self.sigPointStartedMoving.emit(self)
            # While dragging, disable hovering for curves
            self.hoverable = False
            for curve in self.curveOwners:
                curve.hoverable = False

            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
        elif ev.isFinish():
            self.sigPointFinishedMoving.emit(self)
            # Re-enable hovering for curves
            self.hoverable = True
            for curve in self.curveOwners:
                curve.hoverable = True

            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        data = ev.pos() + self.dragOffset
        x = data.x()
        y = data.y()

        # Make a request to the API to move the point. The GUI representation of the point will move if successful
        self.point.request_move(x, y)

        ev.accept()

    # @staticmethod
    # def hover_tip(x: float, y: float, data):
    #     """
    #     Shows data about a point when it is hovered
    #
    #     Parameters
    #     ==========
    #     x: float
    #         x-location of the control point
    #
    #     y: float
    #         y-location of the control point
    #
    #     data
    #         Signaled by the hover
    #     """
    #     return f"x: {x:.8f}\ny: {y:.8f}\n{data}"

    def clicked(self, item, spot, ev):
        self.sigPointClicked.emit(item, spot, ev, self)

    def hovered(self, item, spot, ev):
        if self.hoverable:
            if ev.exit:
                self.sigPointLeaveHovered.emit(item, spot, ev, self)
            else:
                self.sigPointHovered.emit(item, spot, ev, self)

    def setScatterStyle(self, mode: str = "default"):
        implemented_style_modes = ["default", "hovered", "selected"]
        if mode not in implemented_style_modes:
            raise NotImplementedError(f"Selected mode ({mode}) not implemented. "
                                      f"Currently implemented modes: {implemented_style_modes}.")
        self.scatter.setPen(pg.mkPen(color=q_settings.value(f"scatter_{mode}_pen_color",
                                                            q_settings_descriptions[f"scatter_{mode}_pen_color"][1])))
        self.scatter.setBrush(
            pg.mkBrush(color=q_settings.value(f"scatter_{mode}_brush_color",
                                              q_settings_descriptions[f"scatter_{mode}_brush_color"][1]))
        )
        self.scatter.setSize(q_settings.value(f"scatter_{mode}_size",
                                              q_settings_descriptions[f"scatter_{mode}_size"][1]))
        self.scatter.setSymbol(q_settings.value(f"scatter_{mode}_symbol",
                                                q_settings_descriptions[f"scatter_{mode}_symbol"][1]))
