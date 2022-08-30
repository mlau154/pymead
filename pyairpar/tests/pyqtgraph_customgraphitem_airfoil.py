"""
Simple example of subclassing GraphItem.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from pyairpar.core.airfoil import Airfoil

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

w = pg.GraphicsLayoutWidget(show=True)
w.setWindowTitle('pyqtgraph example: CustomGraphItem')
w.setBackground('w')
v = w.addViewBox()
v.setAspectLocked()


class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)
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

        anchor_point = airfoil.anchor_points[
            airfoil.anchor_point_order.index(airfoil.control_points[ind].anchor_point_tag)]

        if airfoil.control_points[ind].cp_type == 'g2_minus':
            new_Lc = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2)
            # print(f"new_Lc = {new_Lc}")
            # phi_abs_angle = np.arctan2(y[ind + 1] - y[ind + 2], x[ind + 1] - x[ind + 2])
            new_psi1_abs_angle = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1])
            anchor_point.recalculate_ap_branch_props_from_g2_pt('minus', new_psi1_abs_angle, new_Lc)

        elif airfoil.control_points[ind].cp_type == 'g2_plus':
            new_Lc = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2)
            print(f"new_Lc = {new_Lc}")
            new_psi2_abs_angle = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1])
            anchor_point.recalculate_ap_branch_props_from_g2_pt('plus', new_psi2_abs_angle, new_Lc)

        elif airfoil.control_points[ind].cp_type == 'g1_minus':
            # x[ind - 1] += dx
            # y[ind - 1] += dy
            # cp_skeleton.set_xdata(x)
            # cp_skeleton.set_ydata(y)
            new_Lt = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2)
            new_abs_phi1 = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1])
            anchor_point.recalculate_ap_branch_props_from_g1_pt('minus', new_abs_phi1, new_Lt)

        elif airfoil.control_points[ind].cp_type == 'g1_plus':
            new_Lt = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2)
            new_abs_phi2 = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1])
            anchor_point.recalculate_ap_branch_props_from_g1_pt('plus', new_abs_phi2, new_Lt)

        # print(f"anchor_point.R.value = {anchor_point.R.value}")
        # airfoil.base_airfoil_params.R_le.value = anchor_point.R.value
        airfoil.update()
        # airfoil.update_curvature_comb_normals()
        # airfoil.update_curvature_comb_curve()
        # print(f"airfoil_ap_LE = {airfoil.anchor_points[1].R.value}")

        # lines = airfoil.plot_airfoil(canvas.axes, color='cornflowerblue', lw=2, label='airfoil')

        # # Update the value of the transformed control point in the airfoil control point objects
        # airfoil.control_point_array[ind].xp = x[ind]
        # airfoil.control_point_array[ind].yp = y[ind]
        #
        # airfoil.control_point_array = np.column_stack((x, y))
        # airfoil.curve_list = []
        # cp_end_idx, cp_start_idx = 0, 1
        # for idx, ap_name in enumerate(airfoil.anchor_point_order[:-1]):
        #     cp_end_idx += airfoil.N[ap_name] + 1
        #     P = airfoil.control_point_array[cp_start_idx - 1:cp_end_idx]
        #     airfoil.curve_list.append(Bezier(P, 150))
        #     cp_start_idx = deepcopy(cp_end_idx)

        # for idx, line in enumerate(lines):
        #     line.set_xdata(airfoil.curve_list[idx].x)
        #     line.set_ydata(airfoil.curve_list[idx].y)

        # cp_skeleton.set_xdata(airfoil.control_point_array[:, 0])
        # cp_skeleton.set_ydata(airfoil.control_point_array[:, 1])
        self.data['pos'] = airfoil.control_point_array
        print(airfoil.control_points[2].xp)


        self.updateGraph()
        ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)


g = Graph()
v.addItem(g)

## Define positions of nodes
# pos = np.array([
#     [0, 0],
#     [10, 0],
#     [0, 10],
#     [10, 10],
#     [5, 5],
#     [15, 5]
# ], dtype=float)
airfoil = Airfoil()
pos = airfoil.control_point_array

## Define the set of connections in the graph
# adj = np.array([
#     [0, 1],
#     [1, 3],
#     [3, 2],
#     [2, 0],
#     [1, 5],
#     [3, 5],
# ])
adj = np.zeros(shape=pos.shape, dtype=int)
for idx, _ in enumerate(pos):
    adj[idx, 0] = idx
    if idx == len(pos) - 1:
        adj[idx, 1] = 0
    else:
        adj[idx, 1] = idx + 1
print(adj)

## Define the symbol to use for each node (this is optional)
# symbols = ['o', 'o', 'o', 'o', 't', '+']

## Define the line style for each connection (this is optional)
# lines = np.array([
#     (255, 0, 0, 255, 1),
#     (255, 0, 255, 255, 2),
#     (255, 0, 255, 255, 3),
#     (255, 255, 0, 255, 2),
#     (255, 0, 0, 255, 1),
#     (255, 255, 255, 255, 4),
# ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])

## Define text to show next to each symbol
# texts = ["Point %d" % i for i in range(6)]

## Update the graph
g.setData(pos=pos, adj=adj, size=8, pxMode=True)

if __name__ == '__main__':
    pg.exec()