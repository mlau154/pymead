"""
Interactive airfoil graph object
"""
from time import time

import numpy as np

import pyqtgraph as pg
from PyQt5.QtCore import Qt

from pymead.gui.polygon_item import PolygonItem
from pymead.core.pos_param import PosParam

from pymead.core.airfoil import Airfoil
from pymead.analysis.single_element_inviscid import single_element_inviscid

from time import time


class AirfoilGraph(pg.GraphItem):

    def __init__(self, airfoil: Airfoil = None, pen=None,
                 size: tuple = (1000, 300), background_color: str = 'w', w=None, v=None):
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color='cornflowerblue', width=2)

        if w is None:
            self.w = pg.GraphicsLayoutWidget(show=True, size=size)
            # self.w.setWindowTitle('Airfoil')
            self.w.setBackground(background_color)
            # self.w.setFont(QFont("Arial"))
        else:
            self.w = w

        if v is None:
            self.v = self.w.addPlot()
            self.v.setAspectLocked()
            self.v.hideButtons()
            # self.v.getViewBox().setFont(QFont("Arial"))
        else:
            self.v = v

        # self.point_label = pg.LabelItem(size="12pt")
        # self.point_label.setParentItem(self.v)
        # self.point_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

        self.airfoil = airfoil
        self.last_time = None
        if not self.airfoil:
            self.airfoil = Airfoil()
        self.airfoil.init_airfoil_curve_pg(self.v, pen)
        self.polygon_item = PolygonItem(self.airfoil.get_coords())
        self.poly = self.v.addItem(self.polygon_item)
        self.dragPoint = None
        self.dragOffset = None
        self.te_thickness_edit_mode = False
        self.param_tree = None
        self.airfoil_parameters = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.v.addItem(self)

        pos, adj, symbols = self.update_airfoil_data()

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
        # texts = ["%d" % i for i in range(9)]

        # Update the graph
        self.setData(pos=pos, adj=adj, size=8, pxMode=True, symbol=symbols, hoverable=True,
                     hoverBrush=pg.mkBrush(color='gold'), tip=self.hover_tip)
        # self.v.disableAutoRange()
        # self.scatter.sigClicked.connect(self.clicked)
        # self.scatter.sigHovered.connect(self.hovered)
        # self.scatter.sigPlotChanged.connect(self.clicked)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def update_airfoil_data(self):
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
        symbols = ['x' if cp.cp_type == 'anchor_point' else 'o' for cp in self.airfoil.control_points]
        return pos, adj, symbols

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
        self.airfoil.update_airfoil_curve_pg()
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

        # Update the polygonal patch
        self.polygon_item.data = self.airfoil.get_coords()
        self.polygon_item.generatePicture()

    def update_ap_fp(self):
        for ap_tag in self.airfoil.free_points.keys():
            for fp in self.airfoil.free_points[ap_tag].values():
                fp.set_ctrlpt_value()
        for ap in self.airfoil.anchor_points:
            if ap.tag not in ['te_1', 'le', 'te_2']:
                ap.set_ctrlpt_value()

    def mouseDragEvent(self, ev):
        t1 = time()
        # if self.last_time is not None:
        #     print(f"Time since last update: {t1 - self.last_time} seconds")
        if ev.button() != Qt.MouseButton.LeftButton:
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

        if self.param_tree is not None:
            if self.airfoil_parameters is None:
                self.airfoil_parameters = self.param_tree.p.param('Airfoil Parameters')

        if self.airfoil.control_points[ind].cp_type == 'g2_minus':
            new_Lc = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            new_psi_abs_angle = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('minus', new_psi_abs_angle, new_Lc)

        elif self.airfoil.control_points[ind].cp_type == 'g2_plus':
            new_Lc = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
            new_psi_abs_angle = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('plus', new_psi_abs_angle, new_Lc)

        elif self.airfoil.control_points[ind].cp_type == 'g1_minus':
            new_Lt = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            new_abs_phi1 = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g1_pt('minus', new_abs_phi1, new_Lt)

        elif self.airfoil.control_points[ind].cp_type == 'g1_plus':
            new_Lt = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
            new_abs_phi2 = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g1_pt('plus', new_abs_phi2, new_Lt)

        elif self.airfoil.control_points[ind].tag == 'le':
            if self.airfoil.dx.active and not self.airfoil.dx.linked:
                self.airfoil.dx.value = x[ind]
            if self.airfoil.dy.active and not self.airfoil.dy.linked:
                self.airfoil.dy.value = y[ind]
            self.update_ap_fp()

        elif self.airfoil.control_points[ind].tag in ['te_1', 'te_2']:
            if self.te_thickness_edit_mode:
                x_te1_old = self.airfoil.control_points[0].xp
                y_te1_old = self.airfoil.control_points[0].yp
                x_te2_old = self.airfoil.control_points[-1].xp
                y_te2_old = self.airfoil.control_points[-1].yp
                x_te = self.airfoil.dx.value + self.airfoil.c.value * np.cos(-self.airfoil.alf.value)
                y_te = self.airfoil.dy.value + self.airfoil.c.value * np.sin(-self.airfoil.alf.value)
                if self.airfoil.control_points[ind].tag == 'te_1':
                    x_te1_new = x[ind]
                    y_te1_new = y[ind]
                    if self.airfoil.t_te.active and not self.airfoil.t_te.linked:
                        self.airfoil.t_te.value = np.sqrt((x_te1_new - x_te2_old) ** 2 +
                                                          (y_te1_new - y_te2_old) ** 2) / self.airfoil.c.value
                    if self.airfoil.r_te.active and not self.airfoil.r_te.linked:
                        self.airfoil.r_te.value = np.sqrt((x_te1_new - x_te) ** 2 +
                                                          (y_te1_new - y_te) ** 2) / np.sqrt(
                            (x_te1_new - x_te2_old) ** 2 + (y_te1_new - y_te2_old) ** 2)
                    if self.airfoil.phi_te.active and not self.airfoil.phi_te.linked:
                        self.airfoil.phi_te.value = np.arctan2(y_te1_new - y_te2_old, x_te1_new - x_te2_old) - np.pi / 2 + self.airfoil.alf.value
                else:
                    x_te2_new = x[ind]
                    y_te2_new = y[ind]
                    if self.airfoil.t_te.active and not self.airfoil.t_te.linked:
                        self.airfoil.t_te.value = np.sqrt((x_te2_new - x_te1_old) ** 2 +
                                                          (y_te2_new - y_te1_old) ** 2) / self.airfoil.c.value
                    if self.airfoil.r_te.active and not self.airfoil.r_te.linked:
                        self.airfoil.r_te.value = np.sqrt((x_te1_old - x_te) ** 2 +
                                                          (y_te1_old - y_te) ** 2) / np.sqrt(
                            (x_te1_old - x_te2_new) ** 2 + (y_te1_old - y_te2_new) ** 2)
                    if self.airfoil.phi_te.active and not self.airfoil.phi_te.linked:
                        self.airfoil.phi_te.value = np.arctan2(y_te1_old - y_te2_new, x_te1_old - x_te2_new) - np.pi / 2 + self.airfoil.alf.value
            else:
                chord = np.sqrt((x[ind] - self.airfoil.dx.value)**2 + (y[ind] - self.airfoil.dy.value)**2)
                angle_of_attack = -np.arctan2(y[ind] - self.airfoil.dy.value, x[ind] - self.airfoil.dx.value)
                if self.airfoil.c.active and not self.airfoil.c.linked:
                    self.airfoil.c.value = chord
                if self.airfoil.alf.active and not self.airfoil.alf.linked:
                    self.airfoil.alf.value = angle_of_attack
                self.update_ap_fp()

        elif self.airfoil.control_points[ind].cp_type == 'free_point':
            ap_tag = self.airfoil.control_points[ind].anchor_point_tag
            fp_tag = self.airfoil.control_points[ind].tag
            self.airfoil.free_points[ap_tag][fp_tag].set_xp_yp_value(x[ind], y[ind])

        elif self.airfoil.control_points[ind].cp_type == 'anchor_point':
            selected_anchor_point = self.airfoil.anchor_points[
                self.airfoil.anchor_point_order.index(self.airfoil.control_points[ind].tag)]
            selected_anchor_point.set_xp_yp_value(x[ind], y[ind])

        self.plot_change_recursive(self.airfoil_parameters.child('Custom').children())

        for a_tag, airfoil in self.airfoil.mea.airfoils.items():
            airfoil.update()
            airfoil.airfoil_graph.data['pos'] = airfoil.control_point_array
            airfoil.airfoil_graph.updateGraph()
            airfoil.airfoil_graph.plot_change_recursive(
                airfoil.airfoil_graph.airfoil_parameters.child(a_tag).children())

        ev.accept()
        t2 = time()
        self.last_time = t2
    #
    # @staticmethod
    # def update_affected_parameters(obj, param_name_list: list, affected_airfoil_list: list):
    #     for param_name in param_name_list:
    #         if hasattr(obj, param_name):
    #             for affected_param in getattr(obj, param_name).affects:
    #                 print(f"{affected_param.name = }")
    #                 affected_param.update()
    #                 if affected_param.airfoil_tag is not None:
    #                     if affected_param.airfoil_tag not in affected_airfoil_list:
    #                         affected_airfoil_list.append(affected_param.airfoil_tag)
    #     return affected_airfoil_list

    def plot_change_recursive(self, child_list: list):

        def block_changes(pg_param):
            pg_param.blockTreeChangeSignal()

        def flush_changes(pg_param):
            pg_param.treeStateChanges = []
            pg_param.blockTreeChangeEmit = 1
            pg_param.unblockTreeChangeSignal()

        for idx, child in enumerate(child_list):
            if hasattr(child, "airfoil_param"):
                if child.hasChildren():
                    if child.children()[0].name() == 'Equation Definition':
                        block_changes(child)
                        if isinstance(child.airfoil_param, PosParam):
                            child.setValue([-999.0, -999.0])  # hack to force PosParam to update
                        child.setValue(child.airfoil_param.value)
                        flush_changes(child)
                    else:
                        self.plot_change_recursive(child.children())
                else:
                    block_changes(child)
                    child.setValue(child.airfoil_param.value)
                    flush_changes(child)
            else:
                if child.hasChildren():
                    self.plot_change_recursive(child.children())

    def hover_tip(self, x, y, data):
        idx = data[0]
        return f"{self.airfoil.control_points[idx]}\nx: {x:.8f}\ny: {y:.8f}\nindex: {idx}"


if __name__ == '__main__':
    g = AirfoilGraph()
    pg.exec()
