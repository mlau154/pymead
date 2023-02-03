"""
Interactive airfoil graph object
"""

import numpy as np

import pyqtgraph as pg
from PyQt5.QtCore import Qt

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from pymead.gui.polygon_item import PolygonItem
from pymead.core.pos_param import PosParam
from pymead.utils.transformations import transform_matrix

from pymead.core.airfoil import Airfoil, AirfoilTransformation

from time import time


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
        self.setData(pos=pos, adj=adj, size=8, pxMode=True, symbol=symbols)
        # self.v.disableAutoRange()
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
        pg.GraphItem.setData(self, **self.data)
        self.airfoil.update_airfoil_curve_pg()
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

        # Update the polygonal patch
        self.polygon_item.data = self.airfoil.get_coords()
        self.polygon_item.generatePicture()

    def update_ap_fp(self, old_transformation: AirfoilTransformation = None,
                     new_transformation: AirfoilTransformation = None):
        for ap_tag in self.airfoil.free_points.keys():
            for fp in self.airfoil.free_points[ap_tag].values():
                if old_transformation is not None and new_transformation is not None:
                    old_coords = np.array([fp.xy.value])
                    print(f"{old_transformation.transform_rel(old_coords)}")
                    new_coords = new_transformation.transform_abs(old_transformation.transform_rel(old_coords))
                    fp.xy.value = new_coords[0].tolist()
                fp.set_ctrlpt_value()
        for ap in self.airfoil.anchor_points:
            if ap.tag not in ['te_1', 'le', 'te_2']:
                if old_transformation is not None and new_transformation is not None:
                    old_coords = np.array([ap.xy.value])
                    new_coords = new_transformation.transform_abs(old_transformation.transform_rel(old_coords))
                    ap.xy.value = new_coords[0].tolist()
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
            old_transformation = AirfoilTransformation(self.airfoil)
            if self.airfoil.dx.active and not self.airfoil.dx.linked:
                self.airfoil.dx.value = x[ind]
            if self.airfoil.dy.active and not self.airfoil.dy.linked:
                self.airfoil.dy.value = y[ind]
            new_transformation = AirfoilTransformation(self.airfoil)
            self.update_ap_fp(old_transformation=old_transformation, new_transformation=new_transformation)

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
                old_transformation = AirfoilTransformation(self.airfoil)
                chord = np.sqrt((x[ind] - self.airfoil.dx.value)**2 + (y[ind] - self.airfoil.dy.value)**2)
                angle_of_attack = -np.arctan2(y[ind] - self.airfoil.dy.value, x[ind] - self.airfoil.dx.value)
                if self.airfoil.c.active and not self.airfoil.c.linked:
                    self.airfoil.c.value = chord
                if self.airfoil.alf.active and not self.airfoil.alf.linked:
                    self.airfoil.alf.value = angle_of_attack
                new_transformation = AirfoilTransformation(self.airfoil)
                self.update_ap_fp(old_transformation=old_transformation, new_transformation=new_transformation)

        elif self.airfoil.control_points[ind].cp_type == 'free_point':
            ap_tag = self.airfoil.control_points[ind].anchor_point_tag
            fp_tag = self.airfoil.control_points[ind].tag
            self.airfoil.free_points[ap_tag][fp_tag].set_xp_yp_value(x[ind], y[ind])

        elif self.airfoil.control_points[ind].cp_type == 'anchor_point':
            selected_anchor_point = self.airfoil.anchor_points[
                self.airfoil.anchor_point_order.index(self.airfoil.control_points[ind].tag)]
            selected_anchor_point.set_xp_yp_value(x[ind], y[ind])

        self.plot_change_recursive(self.airfoil_parameters.child('Custom').children())

        # try:
        # print(f"Before airfoil update, {self.param_tree.p.child('Airfoil Parameters').child('A0').child('FreePoints').child('te_1').child('FP0').child('A0.FreePoints.te_1.FP0.xy').value() = }")
        # except:
        #     print('ERror!!#J')
        #     pass

        for a_tag, airfoil in self.airfoil.mea.airfoils.items():
            airfoil.update()
            airfoil.airfoil_graph.data['pos'] = airfoil.control_point_array
            airfoil.airfoil_graph.updateGraph()
            airfoil.airfoil_graph.plot_change_recursive(
                airfoil.airfoil_graph.airfoil_parameters.child(a_tag).children())


        # print(
        #     f"After airfoil update, {self.param_tree.p.child('Airfoil Parameters').child('A0').child('FreePoints').child('te_1').child('FP0').child('A0.FreePoints.te_1.FP0.xy').value() = }")

        ev.accept()
        t2 = time()
        self.last_time = t2
        # print(f"Time to update airfoil graph for {self.airfoil.tag}: {t2 - t1} seconds")
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
                        # print(f"Setting airfoil_param {child.airfoil_param.name = } to {child.airfoil_param.value = }")
                        block_changes(child)
                        child.setValue(child.airfoil_param.value)
                        # print(f"Now, {child.value() = }")
                        flush_changes(child)
                    else:
                        self.plot_change_recursive(child.children())
                else:
                #     if child.airfoil_param.name.split('.')[-1] == 'xy':
                #         print(f"Setting airfoil_param {child.airfoil_param.name = } to {child.airfoil_param.value = }")
                    # block_changes(child)
                    # child.setValue(child.airfoil_param.value)
                    # if isinstance(child.airfoil_param, PosParam):
                    #     print("Setting value!")
                    #     child.opts['value'][0] = child.airfoil_param.value[0]
                    #     child.opts['value'][1] = child.airfoil_param.value[1]
                    #     child.opts['default'][0] = child.airfoil_param.value[0]
                    #     child.opts['default'][1] = child.airfoil_param.value[1]
                    child.setValue(child.airfoil_param.value)
                    # if child.airfoil_param.name.split('.')[-1] == 'xy':
                    #     print(f"Now, {child.value() = }")
                    # if child.airfoil_param.name.split('.')[-1] == 'xy':
                    #     print(f"{vars(child) = }")
                    # flush_changes(child)
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
