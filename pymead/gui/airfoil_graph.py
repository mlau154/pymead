"""
Simple example of subclassing GraphItem.
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from PyQt5.QtGui import QPen, QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot

from pymead.core.airfoil import Airfoil
from time import time

from pymead.utils.transformations import rotate, translate, scale, transform


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
        t1 = time()
        pg.GraphItem.setData(self, **self.data)
        t2 = time()
        # print(f"graph item setting data time = {t2 - t1:.3e} seconds")
        # print(self.data)
        t3 = time()
        self.airfoil.update_airfoil_curve_pg()
        t4 = time()
        # print(f"updating airfoil curve pg time = {t4 - t3:.3e} seconds")
        t5 = time()
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        t6 = time()
        # print(f"Setting pos time = {t6 - t5:.3e} seconds")

    def mouseDragEvent(self, ev):
        # print(f"event = {ev}")
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

        if self.param_tree is not None:
            if self.airfoil_parameters is None:
                self.airfoil_parameters = self.param_tree.p.param('Airfoil Parameters')

        # self.param_tree.p.param('Airfoil Parameters').param('A0').param('Base').param('A0.Base.psi1_le').blockTreeChangeSignal()
        # self.param_tree.p.param('Airfoil Parameters').param('A0').param('Base').param('A0.Base.R_le').blockTreeChangeSignal()

        other_airfoils_affected = []

        if self.airfoil.control_points[ind].cp_type == 'g2_minus':
            new_Lc = np.sqrt((x[ind] - x[ind + 1]) ** 2 + (y[ind] - y[ind + 1]) ** 2) / self.airfoil.c.value
            new_psi1_abs_angle = np.arctan2(y[ind] - y[ind + 1], x[ind] - x[ind + 1]) + self.airfoil.alf.value
            anchor_point.recalculate_ap_branch_props_from_g2_pt('minus', new_psi1_abs_angle, new_Lc)
            # self.airfoil_parameters.param(self.airfoil.tag).param('Base').param(f"{self.airfoil.tag}.Base.psi1_le").setValue(self.airfoil.psi1_le.value)

        elif self.airfoil.control_points[ind].cp_type == 'g2_plus':
            new_Lc = np.sqrt((x[ind] - x[ind - 1]) ** 2 + (y[ind] - y[ind - 1]) ** 2) / self.airfoil.c.value
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

        elif self.airfoil.control_points[ind].tag == 'le':
            old_dx = self.airfoil.dx.value
            old_dy = self.airfoil.dy.value
            if self.airfoil.dx.active and not self.airfoil.dx.linked:
                self.airfoil.dx.value = x[ind]
                new_dx = x[ind]
            else:
                new_dx = old_dx
            if self.airfoil.dy.active and not self.airfoil.dy.linked:
                self.airfoil.dy.value = y[ind]
                new_dy = y[ind]
            else:
                new_dy = old_dx
            for ap_key, ap_val in self.airfoil.free_points.items():
                for fp_key, fp_val in ap_val.items():
                    fp_val.set_x_value(None)
                    fp_val.set_y_value(None)
            for ap in self.airfoil.anchor_points:
                if ap.tag not in ['te_1', 'le', 'te_2']:
                    ap.set_x_value(None)
                    ap.set_y_value(None)
            # for ap_key, ap_val in self.airfoil.free_points.items():
            #     for fp_key, fp_val in ap_val.items():
            #         fp_val.xp.value, fp_val.yp.value = translate(fp_val.x.value, fp_val.y.value, new_dx - old_dx,
            #                                                      new_dy - old_dy)
                    # print(f"xp val = {fp_val.xp.value}")
            # for ap_key, ap_val in self.airfoil.free_points.items():
            #     for fp_key, fp_val in ap_val.items():
            #         fp_val.set_y_value(None)
            # print(f"value = {self.airfoil.free_points['ap0']['FP0'].xp.value}")
            # print(f"transformation fp = {self.airfoil.free_points['ap0']['FP0'].airfoil_transformation}")
            # print(f"airfoil dx = {self.airfoil.dx}")

        elif self.airfoil.control_points[ind].tag in ['te_1', 'te_2']:
            if self.te_thickness_edit_mode:
                pass
            else:
                chord = np.sqrt((x[ind] - self.airfoil.dx.value)**2 + (y[ind] - self.airfoil.dy.value)**2)
                angle_of_attack = -np.arctan2(y[ind] - self.airfoil.dy.value, x[ind] - self.airfoil.dx.value)
                if self.airfoil.c.active and not self.airfoil.c.linked:
                    self.airfoil.c.value = chord
                if self.airfoil.alf.active and not self.airfoil.alf.linked:
                    self.airfoil.alf.value = angle_of_attack
                self.airfoil.update(generate_curves=False)

            for ap_key, ap_val in self.airfoil.free_points.items():
                for fp_key, fp_val in ap_val.items():
                    # other_airfoils_affected.extend(fp_val.set_xy(x=fp_val.x.value, y=fp_val.y.value))
                    other_airfoils_affected.extend(fp_val.set_xy(x=fp_val.x.value, y=fp_val.y.value))
                    # print(f"xp now is {fp_val.xp.value}")
                    self.airfoil.update_control_point_array()
                    self.airfoil.generate_curves()
                    # other_airfoils_affected.extend(fp_val.set_xy(xp=fp_val.xp.value, yp=fp_val.yp.value))
            for ap in self.airfoil.anchor_points:
                if ap.tag not in ['te_1', 'le', 'te_2']:
                    ap.set_x_value(None)
                    ap.set_y_value(None)

            # for ap_key, ap_val in self.airfoil.free_points.items():
            #     for fp_key, fp_val in ap_val.items():
            #         fp_val.set_y_value(None)

        elif self.airfoil.control_points[ind].cp_type == 'free_point':
            ap_tag = self.airfoil.control_points[ind].anchor_point_tag
            fp_tag = self.airfoil.control_points[ind].tag
            if self.airfoil.free_points[ap_tag][fp_tag].airfoil_transformation is None:
                self.airfoil.free_points[ap_tag][fp_tag].airfoil_transformation = {'dx': self.airfoil.dx,
                                                                                   'dy': self.airfoil.dy,
                                                                                   'alf': self.airfoil.alf,
                                                                                   'c': self.airfoil.c}
            other_airfoils_affected.extend(self.airfoil.free_points[ap_tag][fp_tag].set_xy(xp=x[ind], yp=y[ind]))
            # self.airfoil.update_free_point_only()

        elif self.airfoil.control_points[ind].cp_type == 'anchor_point':
            selected_anchor_point = self.airfoil.anchor_points[
                self.airfoil.anchor_point_order.index(self.airfoil.control_points[ind].tag)]
            # print(f"ap = {self.airfoil.anchor_points}")
            # ap_tag = self.airfoil.control_points[ind].tag
            # ap_idx = next((idx for idx, ap in enumerate(self.airfoil.anchor_points) if ap.tag == ap_tag))
            if selected_anchor_point.airfoil_transformation is None:
                selected_anchor_point.airfoil_transformation = {'dx': self.airfoil.dx, 'dy': self.airfoil.dy,
                                                                'alf': self.airfoil.alf, 'c': self.airfoil.c}
            if selected_anchor_point.xp.active and not selected_anchor_point.xp.linked:
                xp_input = x[ind]
            else:
                xp_input = None
            if selected_anchor_point.yp.active and not selected_anchor_point.yp.linked:
                yp_input = y[ind]
            else:
                yp_input = None
            selected_anchor_point.set_xp_yp_value(xp_input, yp_input)

        # if not self.airfoil.control_points[ind].cp_type == 'free_point':  # try dealing with this case separately
        if not self.airfoil.control_points[ind].tag in ['te_1', 'te_2']:
            self.airfoil.update()

        self.data['pos'] = self.airfoil.control_point_array

        # self.my_signal.emit("Hi!")

        self.updateGraph()
        self.plot_change_recursive(self.airfoil_parameters.child(self.airfoil.tag).children())

        for a_tag in set(other_airfoils_affected):  # Use set to ignore duplicate values
            other_graph = self.airfoil.mea.airfoils[a_tag].airfoil_graph
            other_graph.airfoil.update()
            other_graph.data['pos'] = other_graph.airfoil.control_point_array
            other_graph.updateGraph()
            other_graph.plot_change_recursive(other_graph.airfoil_parameters.child(a_tag).children())

        # print(f"Made it before accept")
        ev.accept()
        # print(f"Made it after accept")
        # self.airfoil.update()

    # def update_dependencies(self):
    #     list_of_vals = []
    #
    #     def get_list_of_vals_from_dict(d):
    #         for k, v in d.items():
    #             # print(f"k = {k}, v = {v}")
    #             if isinstance(v, dict):
    #                 get_list_of_vals_from_dict(v)
    #             else:
    #                 # print("{0} : {1}".format(k, v))
    #                 list_of_vals.append(v)
    #                 # return list_of_vals
    #
    #     # IMPORTANT
    #     if mea.param_dict is not None:
    #         get_list_of_vals_from_dict(mea.param_dict)
    #         for val in list_of_vals:
    #             for v in val.depends_on.values():
    #                 # print(f"v = {v}")
    #                 if param.airfoil_param is v:
    #                     val.update()

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
