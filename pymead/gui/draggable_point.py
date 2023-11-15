import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np


class DraggablePoint(pg.GraphItem):
    sigPointClicked = pyqtSignal(object, object, object, object)
    sigPointMoved = pyqtSignal(object)

    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.curveOwners = []
        self.textItems = []
        super().__init__()
        self.scatter.sigClicked.connect(self.clicked)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        self.sigPointMoved.emit(self)

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def mouseDragEvent(self, ev):
        # if self.last_time is not None:
        #     print(f"Time since last update: {t1 - self.last_time} seconds")

        if ev.button() != Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            # While dragging, disable hovering for curves
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
            # Re-enable hovering for curves
            for curve in self.curveOwners:
                curve.hoverable = True

            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        x = self.data['pos'][:, 0]
        y = self.data['pos'][:, 1]
        self.updateGraph()

        ev.accept()

    def hover_tip(self, x: float, y: float, data):
        """
        Shows data about a point when it is hovered

        Parameters
        ==========
        x: float
            x-location of the control point

        y: float
            y-location of the control point

        data
            Signaled by the hover
        """
        idx = data[0]
        return f"x: {x:.8f}\ny: {y:.8f}\nindex: {idx}"

    def clicked(self, item, spot, ev):
        self.sigPointClicked.emit(item, spot, ev, self)
