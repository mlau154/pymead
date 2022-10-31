"""
Simple example of subclassing GraphItem.
"""

import numpy as np

import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import Qt

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from pymead.gui.polygon_item import PolygonItem

from pymead.core.airfoil import Airfoil


class AirfoilGraph(pg.GraphItem):
    my_signal = pyqtSignal(str)
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

        self.airfoil = airfoil
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
        self.setData(pos=pos, adj=adj, size=8, pxMode=True, symbol=symbols)
        self.v.disableAutoRange()
        self.scatter.sigClicked.connect(self.clicked)
        # self.scatter.sigPlotChanged.connect(self.clicked)
        self.my_signal.connect(self.slot)

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
        # print(f"Updating graph!")
        # print(f"self.data for airfoil {self.airfoil} is {self.data}")
        # t1 = time()
        pg.GraphItem.setData(self, **self.data)
        # t2 = time()
        # print(f"graph item setting data time = {t2 - t1:.3e} seconds")
        # print(self.data)
        # t3 = time()
        self.airfoil.update_airfoil_curve_pg()
        # t4 = time()
        # print(f"updating airfoil curve pg time = {t4 - t3:.3e} seconds")
        # t5 = time()
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

        # Update the polygonal patch
        self.polygon_item.data = self.airfoil.get_coords()
        self.polygon_item.generatePicture()


        # t6 = time()
        # print(f"Setting pos time = {t6 - t5:.3e} seconds")

    def mouseDragEvent(self, ev):
        # print(f"event = {ev}")
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

        # self.param_tree.p.param('Airfoil Parameters').param('A0').param('Base').param('A0.Base.psi1_le').blockTreeChangeSignal()
        # self.param_tree.p.param('Airfoil Parameters').param('A0').param('Base').param('A0.Base.R_le').blockTreeChangeSignal()

        # other_airfoils_affected = []

        if self.airfoil.control_points[ind].cp_type == 'g2_minus':
            new_Lc = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            new_psi1_abs_angle = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('minus', new_psi1_abs_angle, new_Lc)
            # other_airfoils_affected = self.update_affected_parameters(anchor_point, ['R', 'psi1'], other_airfoils_affected)

        elif self.airfoil.control_points[ind].cp_type == 'g2_plus':
            new_Lc = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
            new_psi2_abs_angle = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('plus', new_psi2_abs_angle, new_Lc)
            # other_airfoils_affected = self.update_affected_parameters(anchor_point, ['R', 'psi2'], other_airfoils_affected)

        elif self.airfoil.control_points[ind].cp_type == 'g1_minus':
            new_Lt = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            new_abs_phi1 = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g1_pt('minus', new_abs_phi1, new_Lt)
            # other_airfoils_affected = self.update_affected_parameters(anchor_point, ['r', 'L', 'phi', 't', 'theta'], other_airfoils_affected)

        elif self.airfoil.control_points[ind].cp_type == 'g1_plus':
            new_Lt = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
            new_abs_phi2 = np.arctan2(y[ind] - y[ind - 1], x[ind] - x[ind - 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g1_pt('plus', new_abs_phi2, new_Lt)
            # other_airfoils_affected = self.update_affected_parameters(anchor_point, ['r', 'L', 'phi', 't', 'theta'], other_airfoils_affected)

        elif self.airfoil.control_points[ind].tag == 'le':
            if self.airfoil.dx.active and not self.airfoil.dx.linked:
                self.airfoil.dx.value = x[ind]
            if self.airfoil.dy.active and not self.airfoil.dy.linked:
                self.airfoil.dy.value = y[ind]
            # self.airfoil.update(generate_curves=False)
            for ap_key, ap_val in self.airfoil.free_points.items():
                for fp_key, fp_val in ap_val.items():
                    fp_val.set_xy(x=fp_val.x.value, y=fp_val.y.value)
            for ap in self.airfoil.anchor_points:
                if ap.tag not in ['te_1', 'le', 'te_2']:
                    ap.set_xy(x=ap.x.value, y=ap.y.value)
            # self.airfoil.update_control_point_array()
            # self.airfoil.generate_curves()
            # self.airfoil.update()

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
                        # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['t_te'], other_airfoils_affected)
                    if self.airfoil.r_te.active and not self.airfoil.r_te.linked:
                        self.airfoil.r_te.value = np.sqrt((x_te1_new - x_te) ** 2 +
                                                          (y_te1_new - y_te) ** 2) / np.sqrt(
                            (x_te1_new - x_te2_old) ** 2 + (y_te1_new - y_te2_old) ** 2)
                        # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['r_te'], other_airfoils_affected)
                    if self.airfoil.phi_te.active and not self.airfoil.phi_te.linked:
                        self.airfoil.phi_te.value = np.arctan2(y_te1_new - y_te2_old, x_te1_new - x_te2_old) - np.pi / 2 + self.airfoil.alf.value
                        # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['phi_te'], other_airfoils_affected)
                else:
                    x_te2_new = x[ind]
                    y_te2_new = y[ind]
                    if self.airfoil.t_te.active and not self.airfoil.t_te.linked:
                        self.airfoil.t_te.value = np.sqrt((x_te2_new - x_te1_old) ** 2 +
                                                          (y_te2_new - y_te1_old) ** 2) / self.airfoil.c.value
                        # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['t_te'], other_airfoils_affected)
                    if self.airfoil.r_te.active and not self.airfoil.r_te.linked:
                        self.airfoil.r_te.value = np.sqrt((x_te1_old - x_te) ** 2 +
                                                          (y_te1_old - y_te) ** 2) / np.sqrt(
                            (x_te1_old - x_te2_new) ** 2 + (y_te1_old - y_te2_new) ** 2)
                        # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['r_te'], other_airfoils_affected)
                    if self.airfoil.phi_te.active and not self.airfoil.phi_te.linked:
                        self.airfoil.phi_te.value = np.arctan2(y_te1_old - y_te2_new, x_te1_old - x_te2_new) - np.pi / 2 + self.airfoil.alf.value
                        # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['phi_te'], other_airfoils_affected)
            else:
                chord = np.sqrt((x[ind] - self.airfoil.dx.value)**2 + (y[ind] - self.airfoil.dy.value)**2)
                angle_of_attack = -np.arctan2(y[ind] - self.airfoil.dy.value, x[ind] - self.airfoil.dx.value)
                if self.airfoil.c.active and not self.airfoil.c.linked:
                    self.airfoil.c.value = chord
                    # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['c'], other_airfoils_affected)
                if self.airfoil.alf.active and not self.airfoil.alf.linked:
                    self.airfoil.alf.value = angle_of_attack
                    # other_airfoils_affected = self.update_affected_parameters(self.airfoil, ['alf'], other_airfoils_affected)
                # self.airfoil.update(generate_curves=False)

            for ap_key, ap_val in self.airfoil.free_points.items():
                for fp_key, fp_val in ap_val.items():
                    fp_val.set_xy(x=fp_val.x.value, y=fp_val.y.value)
                    # self.airfoil.update_control_point_array()
                    # self.airfoil.generate_curves()
            for ap in self.airfoil.anchor_points:
                if ap.tag not in ['te_1', 'le', 'te_2']:
                    ap.set_xy(x=ap.x.value, y=ap.y.value)
                    # self.airfoil.update_control_point_array()
                    # self.airfoil.generate_curves()

        elif self.airfoil.control_points[ind].cp_type == 'free_point':
            ap_tag = self.airfoil.control_points[ind].anchor_point_tag
            fp_tag = self.airfoil.control_points[ind].tag
            self.airfoil.free_points[ap_tag][fp_tag].set_xy(xp=x[ind], yp=y[ind])

        elif self.airfoil.control_points[ind].cp_type == 'anchor_point':
            selected_anchor_point = self.airfoil.anchor_points[
                self.airfoil.anchor_point_order.index(self.airfoil.control_points[ind].tag)]
            selected_anchor_point.set_xy(xp=x[ind], yp=y[ind])

        # self.airfoil.update()
        #
        # self.data['pos'] = self.airfoil.control_point_array
        #
        # self.updateGraph()
        # self.plot_change_recursive(self.airfoil_parameters.child(self.airfoil.tag).children())

        self.plot_change_recursive(self.airfoil_parameters.child('Custom').children())

        for ap in self.airfoil.anchor_points:
            if ap.tag not in ['te_1', 'le', 'te_2']:
                # print(f"ap xp is {ap.xp.value}")
                # print(f"ap yp is {ap.yp.value}")
                # ap.ctrlpt.xp = ap.xp.value
                # ap.ctrlpt.yp = ap.yp.value
                pass

        # for a_tag in set(other_airfoils_affected):  # Use set to ignore duplicate values
        for a_tag, airfoil in self.airfoil.mea.airfoils.items():
            # airfoil = self.airfoil.mea.airfoils[a_tag].airfoil_graph
            airfoil.update()
            airfoil.airfoil_graph.data['pos'] = airfoil.control_point_array
            airfoil.airfoil_graph.updateGraph()
            airfoil.airfoil_graph.plot_change_recursive(airfoil.airfoil_graph.airfoil_parameters.child(a_tag).children())

        # print(f"alf = {self.airfoil.alf.value}")

        ev.accept()

    @staticmethod
    def update_affected_parameters(obj, param_name_list: list, affected_airfoil_list: list):
        for param_name in param_name_list:
            if hasattr(obj, param_name):
                for affected_param in getattr(obj, param_name).affects:
                    affected_param.update()
                    if affected_param.airfoil_tag is not None:
                        if affected_param.airfoil_tag not in affected_airfoil_list:
                            affected_airfoil_list.append(affected_param.airfoil_tag)
        return affected_airfoil_list

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

    @pyqtSlot(str)
    def slot(self, string):
        print(f"Signal emitted! Emitted signal was {string}")

    def clicked(self, pts):
        print("clicked: %s" % pts)


if __name__ == '__main__':
    g = AirfoilGraph()
    pg.exec()
