"""
Simple example of subclassing GraphItem.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from PyQt5.QtGui import QPen
from PyQt5 import QtCore

from pyairpar.core.airfoil import Airfoil


class AirfoilGraph(pg.GraphItem):
    def __init__(self, airfoil: Airfoil = Airfoil(), pen=pg.mkPen(color='cornflowerblue', width=2),
                 size: tuple = (1000, 300), background_color: str = 'w', w=None, v=None, airfoil2: Airfoil = Airfoil()):
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        if w is None:
            self.w = pg.GraphicsLayoutWidget(show=True, size=size)
            # self.w.setWindowTitle('Airfoil')
            self.w.setBackground(background_color)
        else:
            self.w = w

        if v is None:
            self.v = self.w.addPlot()
            self.v.setAspectLocked()
        else:
            self.v = v

        self.airfoil = airfoil
        self.airfoil2 = airfoil2
        self.airfoil.init_airfoil_curve_pg(self.v, pen)
        self.dragPoint = None
        self.dragOffset = None
        self.te_thickness_edit_mode = False
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.v.addItem(self)

        # Define positions of nodes
        pos = self.airfoil.control_point_array

        # Define the set of connections in the graph
        adj = np.zeros(shape=pos.shape, dtype=int)
        for idx, _ in enumerate(pos):
            adj[idx, 0] = idx
            if idx == len(pos) - 1:
                adj[idx, 1] = 0
            else:
                adj[idx, 1] = idx + 1

        # Define the symbol to use for each node (this is optional)
        # symbols = ['o', 'o', 'o', 'o', 't', '+']
        symbols = ['+' if cp.cp_type == 'anchor_point' else 'o' for cp in self.airfoil.control_points]

        # Define the line style for each connection (this is optional)
        # lines = np.array([
        #     (255, 0, 0, 255, 1),
        #     (255, 0, 255, 255, 2),
        #     (255, 0, 255, 255, 3),
        #     (255, 255, 0, 255, 2),
        #     (255, 0, 0, 255, 1),
        #     (255, 255, 255, 255, 4),
        # ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])

        # Define text to show next to each symbol
        # texts = ["Point %d" % i for i in range(6)]

        # Update the graph
        self.setData(pos=pos, adj=adj, size=8, pxMode=True, symbol=symbols)
        self.v.disableAutoRange()
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

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        # print(self.data)
        self.airfoil.update_airfoil_curve_pg()
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
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

        anchor_point = self.airfoil.anchor_points[
            self.airfoil.anchor_point_order.index(self.airfoil.control_points[ind].anchor_point_tag)]

        if self.airfoil.control_points[ind].cp_type == 'g2_minus':
            new_Lc = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            # print(f"new_Lc = {new_Lc}")
            # phi_abs_angle = np.arctan2(y[ind + 1] - y[ind + 2], x[ind + 1] - x[ind + 2])
            new_psi1_abs_angle = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('minus', new_psi1_abs_angle, new_Lc)

        elif self.airfoil.control_points[ind].cp_type == 'g2_plus':
            new_Lc = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
            # print(f"new_Lc = {new_Lc}")
            new_psi2_abs_angle = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('plus', new_psi2_abs_angle, new_Lc)

        elif self.airfoil.control_points[ind].cp_type == 'g1_minus':
            new_Lt = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            new_abs_phi1 = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g1_pt('minus', new_abs_phi1, new_Lt)

        elif self.airfoil.control_points[ind].cp_type == 'g1_plus':
            new_Lt = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
            new_abs_phi2 = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g1_pt('plus', new_abs_phi2, new_Lt)

        elif self.airfoil.control_points[ind].name == 'le':
            self.airfoil.dx.value = x[ind]
            self.airfoil.dy.value = y[ind]
            # print(f"dx, dy for airfoil {repr(self.airfoil)} is {self.airfoil.dx.value}, {self.airfoil.dy.value}")

        elif self.airfoil.control_points[ind].name in ['te_1', 'te_2']:
            if self.te_thickness_edit_mode:
                pass
            else:
                chord = np.sqrt((x[ind] - self.airfoil.dx.value)**2 + (y[ind] - self.airfoil.dy.value)**2)
                angle_of_attack = -np.arctan2(y[ind] - self.airfoil.dy.value, x[ind] - self.airfoil.dx.value)
                print(f"Before update, airfoil 1 chord is {self.airfoil.c.value}")
                print(f"Before update, airfoil 2 chord is {self.airfoil2.c.value}")
                print(self.airfoil.base_airfoil_params.c)
                print(self.airfoil2.base_airfoil_params.c)
                self.airfoil.c.value = chord
                print(f"After update, airfoil 1 chord is {self.airfoil.c.value}")
                print(f"After update, airfoil 2 chord is {self.airfoil2.c.value}")
                self.airfoil.alf.value = angle_of_attack

        self.airfoil.update()
        self.data['pos'] = self.airfoil.control_point_array

        self.updateGraph()
        ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)


if __name__ == '__main__':
    g = AirfoilGraph()
    pg.exec()
