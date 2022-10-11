"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types
"""
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, registerParameterItemType
from pymead.core.mea import MEA
import pyqtgraph as pg
from pymead.core.param import Param
from pymead.core.free_point import FreePoint
from pymead.core.anchor_point import AnchorPoint
from pymead.gui.autocomplete import AutoStrParameterItem
from pymead.gui.selectable_header import SelectableHeaderParameterItem
from pymead.gui.input_dialog import FreePointInputDialog, AnchorPointInputDialog
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.core.airfoil import Airfoil
from PyQt5.QtWidgets import QCompleter, QWidget, QGridLayout, QLabel, QInputDialog, QHeaderView
from pymead.utils.downsampling_schemes import fractal_downsampler2
from PyQt5.QtCore import Qt
from functools import partial
import numpy as np
from time import time

import ctypes

app = pg.mkQApp("Parameter Tree Example")


# test subclassing parameters
# This parameter automatically generates two child parameters which are always reciprocals of each other
class MEAParameters(pTypes.GroupParameter):
    def __init__(self, mea: MEA, status_bar, **opts):
        registerParameterItemType('selectable_header', SelectableHeaderParameterItem, override=True)
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.mea = mea
        self.status_bar = status_bar
        self.airfoil_headers = []
        self.custom_header = self.addChild(HeaderParameter(name='Custom', type='bool', value=True))
        for idx, a in enumerate(self.mea.airfoils.values()):
            self.add_airfoil(a, idx)

    def add_airfoil(self, airfoil: Airfoil, idx: int):
        self.airfoil_headers.append(self.addChild(dict(name=airfoil.tag, type='selectable_header', value=True,
                                                       context={"add_fp": "Add FreePoint", "remove_fp": "Remove FreePoint",
                                                                "add_ap": "Add AnchorPoint", "remove_ap": "Remove AnchorPoint"})))
        header_params = ['Base', 'AnchorPoints', 'FreePoints', 'Custom']
        for hp in header_params:
            # print(f"children = {self.airfoil_headers[idx].children()}")
            self.airfoil_headers[idx].addChild(HeaderParameter(name=hp, type='bool', value=True))
        for p_key, p_val in self.mea.param_dict[airfoil.tag]['Base'].items():
            self.airfoil_headers[idx].children()[0].addChild(AirfoilParameter(self.mea.param_dict[airfoil.tag]['Base'][p_key],
                                                                              name=f"{airfoil.tag}.Base.{p_key}",
                                                                              type='float',
                                                                              value=self.mea.param_dict[airfoil.tag]['Base'][
                                                                                  p_key].value,
                                                                              context={'add_eq': 'Define by equation'}))
        # print(f"param_dict = {self.mea.param_dict}")
        for ap_key, ap_val in self.mea.param_dict[airfoil.tag]['AnchorPoints'].items():
            self.child(airfoil.tag).child('AnchorPoints').addChild(
                HeaderParameter(name=ap_key, type='bool', value='true'))
            for p_key, p_val in self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key].items():
                self.child(airfoil.tag).child('AnchorPoints').child(ap_key).addChild(AirfoilParameter(
                    self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key][p_key],
                    name=f"{airfoil.tag}.AnchorPoints.{ap_key}.{p_key}", type='float',
                    value=self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key][
                        p_key].value,
                    context={'add_eq': 'Define by equation'}))
        for ap_key, ap_val in self.mea.param_dict[airfoil.tag]['FreePoints'].items():
            self.child(airfoil.tag).child('FreePoints').addChild(
                HeaderParameter(name=ap_key, type='bool', value='true'))
            for fp_key, fp_val in ap_val.items():
                self.child(airfoil.tag).child('FreePoints').child(ap_key).addChild(
                    HeaderParameter(name=fp_key, type='bool', value='true'))
                for p_key, p_val in fp_val.items():
                    self.child(airfoil.tag).child('FreePoints').child(ap_key).child(fp_key).addChild(
                        AirfoilParameter(self.mea.param_dict[airfoil.tag]['FreePoints'][ap_key][fp_key][p_key],
                                         name=f"{airfoil.tag}.FreePoints.{ap_key}.{fp_key}.{p_key}", type='float',
                                         value=self.mea.param_dict[airfoil.tag]['FreePoints'][ap_key][fp_key][p_key].value,
                                         context={'add_eq': 'Define by equation'}))


class AirfoilParameter(pTypes.SimpleParameter):
    def __init__(self, airfoil_param: Param, **opts):
        self.airfoil_param = airfoil_param
        pTypes.SimpleParameter.__init__(self, **opts)


class HeaderParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        pTypes.GroupParameter.__init__(self, **opts)


# this group includes a menu allowing the user to add new parameters into its child list
class ScalableGroup(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'group'
        opts['addText'] = "Add"
        opts['addList'] = ['str', 'float', 'int']
        pTypes.GroupParameter.__init__(self, **opts)

    def addNew(self, typ):
        val = {
            'str': '',
            'float': 0.0,
            'int': 0
        }[typ]
        self.addChild(
            dict(name="ScalableParam %d" % (len(self.childs) + 1), type=typ, value=val, removable=True, renamable=True))


# Create two ParameterTree widgets, both accessing the same data
class MEAParamTree:
    def __init__(self, mea: MEA, status_bar):
        self.params = [
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action', 'children': [
                    {'name': 'Add missing items', 'type': 'bool', 'value': True},
                    {'name': 'Remove extra items', 'type': 'bool', 'value': True},
                ]},
            ]},
            {'name': 'Analysis', 'type': 'group', 'children': [{'name': 'Inviscid Cp Calc', 'type': 'list', 'limits': [a.tag for a in mea.airfoils.values()]}]},
            MEAParameters(mea, status_bar, name='Airfoil Parameters'),
            # ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
            #     {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
            #     {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            # ]),
        ]

        # Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=self.params)

        self.mea = mea
        self.cl_label = pg.LabelItem(size="18pt", color="#000000")
        self.cl_label.setParentItem(self.mea.v)
        self.cl_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

        self.cl_airfoil_tag = 'A0'

        self.line_edit_refs = {}

        registerParameterItemType('auto_str', AutoStrParameterItem, override=True)

        for a_name, a in mea.airfoils.items():
            # print(f"airfoil_name = {a_name}")
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.plot_changed, a_name))  # Needs to change with airfoil added or removed

        ## If anything changes in the tree, print a message
        def change(param, changes):
            # single_element_inviscid(np.array([[1, 0], [0, 0], [1, 0]]))
            # print(f"params = {vars(self.params[-1])}")
            # print("tree changes:")
            # print(f"change = {change}")

            for param, change, data in changes:
                # print(f"change = {change}")
                # print(f"param = {param}")
                # print(f"data = {data}")
                # if param.children():
                #     print(f"param name = {param.children()[0].name()}")
                # print(f"has children = {param.hasChildren()}")
                if change == 'childRemoved' and data.opts['name'] == 'Equation Definition':
                    # print('Removing equation...')
                    param.airfoil_param.remove_func()
                    param.setReadonly(False)
                    self.line_edit_refs.pop(param.name())
                path = self.p.childPath(param)
                # print(f"path = {path}")
                if path is not None:
                    if path[-1] == 'Equation Definition':
                        if change == 'value':
                            # self.t.resizeColumnToContents(0)
                            # self.t.resizeColumnToContents(1)
                            # print("Equation definition value changed")
                            # print(f"asdf vars={param.makeTreeItem(0)}")

                            param.parent().setReadonly()
                            param.parent().airfoil_param.mea = mea
                            # print(f"param_dict rn is {param.parent().airfoil_param.param_dict}")
                            param.parent().airfoil_param.set_func_str(str(data))
                            param.parent().airfoil_param.update()
                            # print(f"depends on = {param.parent().airfoil_param.depends_on}")
                            if len(param.parent().airfoil_param.depends_on) > 0:
                                for air_par in param.parent().airfoil_param.depends_on.values():
                                    air_par.update()
                                if param.parent().airfoil_param.linked:
                                    param.parent().setValue(param.parent().airfoil_param.value)

                                # print(f"parent name, success is {param.parent().name()}")
                                self.line_edit_refs[param.parent().name()].widget.setStyleSheet(
                                    'border: 0px; color: green;')
                                # print(f"great grandparent = {param.parent().parent().parent()}")
                                if param.parent().parent().parent().parent().name() == 'FreePoints':
                                    # print(f"Setting freepoint!")
                                    fp_name = param.parent().parent().name()
                                    ap_name = param.parent().parent().parent().name()
                                    a_name = param.parent().parent().parent().parent().parent().name()
                                    param_name = param.parent().name().split('.')[-1]
                                    fp = self.mea.airfoils[a_name].free_points[ap_name][fp_name]
                                    if param_name in ['x', 'y']:
                                        fp.set_x_value(None)
                                        fp.set_y_value(None)
                                    fp.set_ctrlpt_value()
                                elif param.parent().parent().parent().name() == 'AnchorPoints':
                                    # print(f"Setting AnchorPoint!")
                                    ap_name = param.parent().parent().name()
                                    a_name = param.parent().parent().parent().parent().name()
                                    param_name = param.parent().name().split('.')[-1]
                                    # print(f"param_name = {param_name}")
                                    ap = self.mea.airfoils[a_name].anchor_points[self.mea.airfoils[a_name].anchor_point_order.index(ap_name)]
                                    if param_name in ['x', 'y']:
                                        ap.set_x_value(None)
                                        ap.set_y_value(None)
                                    else:
                                        ap.set_ctrlpt_value()
                            else:
                                # print(f"parent name is {param.parent().name()}")
                                param.parent().setReadonly(False)
                                self.line_edit_refs[param.parent().name()].widget.setStyleSheet('border: 0px; color: red;')

                if hasattr(param, 'airfoil_param') and change == 'value':
                    param_name = param.name().split('.')[-1]
                    # print(f"name = {param_name}")
                    if param.airfoil_param.active and not param.airfoil_param.linked:
                        if param_name not in ['x', 'y', 'xp', 'yp']:
                            param.airfoil_param.value = data

                    def block_changes(pg_param):
                        pg_param.blockTreeChangeSignal()

                    def flush_changes(pg_param):
                        pg_param.treeStateChanges = []
                        pg_param.blockTreeChangeEmit = 1
                        pg_param.unblockTreeChangeSignal()

                    # Treat the x, y, xp, and yp locations of the FreePoints and AnchorPoints separately
                    if param_name in ['x', 'y', 'xp', 'yp']:
                        if param.parent().parent().parent().name() == 'FreePoints':
                            fp_name = param.parent().name()
                            ap_name = param.parent().parent().name()
                            a_name = param.parent().parent().parent().parent().name()
                            fp_or_ap = self.mea.airfoils[a_name].airfoil_graph.airfoil.free_points[ap_name][fp_name]
                        elif param.parent().parent().name() == 'AnchorPoints':
                            ap_name = param.parent().name()
                            a_name = param.parent().parent().parent().name()
                            airfoil = self.mea.airfoils[a_name].airfoil_graph.airfoil
                            aps = airfoil.anchor_points
                            ap_order = airfoil.anchor_point_order
                            fp_or_ap = aps[ap_order.index(ap_name)]
                        else:
                            raise ValueError('Parameter names \'x\', \'y\', \'xp\', and \'yp\' are reserved for'
                                             'FreePoints and AnchorPoints. Please choose a different parameter name.')

                        if not self.mea.airfoils[a_name].airfoil_graph.dragPoint:
                            if param_name == 'x':
                                fp_or_ap.set_x_value(data)
                            elif param_name == 'y':
                                fp_or_ap.set_y_value(data)
                            elif param_name == 'xp':
                                fp_or_ap.set_xp_value(data)
                            elif param_name == 'yp':
                                fp_or_ap.set_yp_value(data)

                    param.airfoil_param.update()

                    if param.airfoil_param.linked:
                        param.setValue(param.airfoil_param.value)

                    list_of_vals = []

                    def get_list_of_vals_from_dict(d):
                        for k, v in d.items():
                            # print(f"k = {k}, v = {v}")
                            if isinstance(v, dict):
                                get_list_of_vals_from_dict(v)
                            else:
                                # print("{0} : {1}".format(k, v))
                                list_of_vals.append(v)
                                # return list_of_vals

                    # IMPORTANT
                    if mea.param_dict is not None:
                        get_list_of_vals_from_dict(mea.param_dict)
                        for val in list_of_vals:
                            for v in val.depends_on.values():
                                if param.airfoil_param is v:
                                    print(f"val before update is {val.value}")
                                    val.update()
                                    print(f"val after update is {val.value}")

                    for a in mea.airfoils.values():
                        for fp_dict in a.free_points.values():
                            if len(fp_dict) > 0:
                                for fp in fp_dict.values():
                                    for x_or_y in ['x', 'y', 'xp', 'yp']:
                                        self.params[-1].child(a.tag).child("FreePoints").child(
                                            fp.anchor_point_tag).child(fp.tag).child(
                                            f"{a.tag}.FreePoints.{fp.anchor_point_tag}.{fp.tag}.{x_or_y}").blockTreeChangeSignal()
                                    if fp.airfoil_transformation is None:
                                        fp.airfoil_transformation = {'dx': a.dx, 'dy': a.dy, 'alf': a.alf, 'c': a.c}
                                    fp.set_x_value(None)
                                    fp.set_y_value(None)
                                    # fp.set_xp_value(None)
                                    # fp.set_yp_value(None)
                                    fp.set_ctrlpt_value()
                        for ap in a.anchor_points:
                            if ap.tag not in ['te_1', 'le', 'te_2']:
                                for x_or_y in ['x', 'y']:
                                    self.params[-1].child(a.tag).child(
                                        "AnchorPoints").child(ap.tag).child(
                                        f"{a.tag}.AnchorPoints.{ap.tag}.{x_or_y}").blockTreeChangeSignal()
                                if ap.airfoil_transformation is None:
                                    ap.airfoil_transformation = {'dx': a.dx, 'dy': a.dy, 'alf': a.alf, 'c': a.c}
                                    ap.set_x_value(None)
                                    ap.set_y_value(None)
                                ap.set_xp_value(None)
                                ap.set_yp_value(None)

                        a.update()
                        # if 'ap0' in a.free_points.keys():
                        #     print(f"x, y, xp, yp = {a.free_points['ap0']['FP0'].x.value}, {a.free_points['ap0']['FP0'].y.value}, {a.free_points['ap0']['FP0'].xp.value}, {a.free_points['ap0']['FP0'].yp.value}")
                        a.airfoil_graph.data['pos'] = a.control_point_array
                        # print(f"cp = {a.control_point_array[7, :]}")
                        a.airfoil_graph.updateGraph()

                        # Run inviscid CL calculation after any geometry change
                        if a.tag == self.cl_airfoil_tag:
                            a.get_coords(body_fixed_csys=True)
                            ds = fractal_downsampler2(a.coords, ratio_thresh=1.000005, abs_thresh=0.1)
                            _, _, CL = single_element_inviscid(ds, a.alf.value * 180 / np.pi)
                            self.cl_label.setText(f"{self.cl_airfoil_tag} Inviscid CL = {CL:.3f}")

                        # Flush AnchorPoint x and y changes and unblock AnchorPoint x and y parameters
                        for fp_dict in a.free_points.values():
                            if len(fp_dict) > 0:
                                for fp in fp_dict.values():
                                    for x_or_y in ['x', 'y', 'xp', 'yp']:
                                        param_val = self.params[-1].child(a.tag).child("FreePoints").child(
                                            fp.anchor_point_tag).child(fp.tag).child(
                                            f"{a.tag}.FreePoints.{fp.anchor_point_tag}.{fp.tag}.{x_or_y}")
                                        flush_changes(param_val)
                            for ap in a.anchor_points:
                                if ap.tag not in ['te_1', 'le', 'te_2']:
                                    for x_or_y in ['x', 'y', 'xp', 'yp']:
                                        param_val = self.params[-1].child(a.tag).child(
                                            "AnchorPoints").child(ap.tag).child(
                                            f"{a.tag}.AnchorPoints.{ap.tag}.{x_or_y}")
                                        flush_changes(param_val)

                # Add equation child parameter if the change is the selection of the "equation" button in the
                # contextMenu
                if change == 'contextMenu' and data == 'add_eq':
                    if not param.hasChildren():
                        param.addChild(dict(name='Equation Definition', type='auto_str', value='', removable=True))
                        for k in param.children()[0].items.data.keys():
                            id_val = ''
                            for ch in str(k)[-19:-1]:
                                id_val += ch
                            self.line_edit_refs[param.name()] = ctypes.cast(int(id_val, 0), ctypes.py_object).value
                            self.update_auto_complete()
                        # print(f"self.line_edit_refs = {self.line_edit_refs}")
                    # for id_val in self.id_list:
                    #     value = ctypes.cast(id_val, ctypes.py_object).value
                    #     print(f"value = {value}")

                if change == 'contextMenu' and data == 'add_fp':
                    a_tag = param.name()
                    self.dialog = FreePointInputDialog(items=[("x", "double"), ("y", "double"),
                                                              ("Previous Anchor Point", "combo"),
                                                              ("Previous Free Point", "combo")],
                                                       fp=self.mea.airfoils[a_tag].free_points)
                    if self.dialog.exec():
                        inputs = self.dialog.getInputs()
                    else:
                        inputs = None
                    if inputs:
                        if inputs[3] == 'None':
                            pfp = None
                        else:
                            pfp = inputs[3]
                        fp = FreePoint(Param(inputs[0]), Param(inputs[1]),
                                       previous_anchor_point=inputs[2], previous_free_point=pfp)
                        self.mea.airfoils[a_tag].insert_free_point(fp)
                        self.mea.airfoils[a_tag].update()
                        self.dialog.update_fp_ap_tags()
                        pos, adj, symbols = self.mea.airfoils[a_tag].airfoil_graph.update_airfoil_data()
                        self.mea.airfoils[a_tag].airfoil_graph.setData(pos=pos, adj=adj, size=8, pxMode=True, symbol=symbols)
                        if not fp.anchor_point_tag in [p.name() for p in self.params[-1].child(a_tag).child('FreePoints')]:
                            self.params[-1].child(a_tag).child('FreePoints').addChild(HeaderParameter(name=fp.anchor_point_tag, type='bool', value='true'))
                        self.params[-1].child(a_tag).child('FreePoints').child(fp.anchor_point_tag).addChild(
                            HeaderParameter(name=fp.tag, type='bool', value='true'))
                        # print(self.mea.param_dict['A0'])
                        for p_key, p_val in self.mea.param_dict[a_tag]['FreePoints'][fp.anchor_point_tag][fp.tag].items():
                            self.params[-1].child(a_tag).child('FreePoints').child(fp.anchor_point_tag).child(fp.tag).addChild(AirfoilParameter(self.mea.param_dict[a_tag]['FreePoints'][fp.anchor_point_tag][fp.tag][p_key],
                                                 name=f"{a_tag}.FreePoints.{fp.anchor_point_tag}.{fp.tag}.{p_key}", type='float',
                                                 value=self.mea.param_dict[a_tag]['FreePoints'][fp.anchor_point_tag][fp.tag][p_key].value,
                                                 context={'add_eq': 'Define by equation'}))

                if change == 'contextMenu' and data == 'add_ap':
                    a_tag = param.name()
                    self.dialog = AnchorPointInputDialog(items=[("x", "double"), ("y", "double"),
                                                                ("L", "double"), ("R", "double"),
                                                                ("r", "double"), ("phi", "double"),
                                                                ("psi1", "double"), ("psi2", "double"),
                                                                ("Previous Anchor Point", "combo"),
                                                                ("Anchor Point Name", "string")],
                                                         ap=self.mea.airfoils[a_tag].anchor_points)
                    if self.dialog.exec():
                        inputs = self.dialog.getInputs()
                    else:
                        inputs = None
                    if inputs:
                        # id_list = []
                        for curve in self.mea.airfoils[a_tag].curve_list:
                            # print(f'curve_handle = {curve.pg_curve_handle}')
                            # id_list.append(id(curve.pg_curve_handle))
                            curve.pg_curve_handle.clear()
                        ap = AnchorPoint(x=Param(inputs[0]), y=Param(inputs[1]), L=Param(inputs[2]), R=Param(inputs[3]),
                                         r=Param(inputs[4]), phi=Param(inputs[5]), psi1=Param(inputs[6]),
                                         psi2=Param(inputs[7]), previous_anchor_point=inputs[8], tag=inputs[9])
                        self.mea.airfoils[a_tag].insert_anchor_point(ap)
                        self.mea.airfoils[a_tag].update()
                        self.mea.airfoils[a_tag].init_airfoil_curve_pg(self.mea.airfoils[a_tag].airfoil_graph.v,
                                                                       pen=pg.mkPen(color='cornflowerblue', width=2))

                        # for curve in self.mea.airfoils[a_tag].curve_list:
                        #     print(f'curve_handle = {curve.pg_curve_handle}')
                        self.dialog.update_ap_tags()
                        pos, adj, symbols = self.mea.airfoils[a_tag].airfoil_graph.update_airfoil_data()
                        self.mea.airfoils[a_tag].airfoil_graph.setData(pos=pos, adj=adj, size=8, pxMode=True,
                                                                       symbol=symbols)
                        self.params[-1].child(a_tag).child('AnchorPoints').addChild(
                            HeaderParameter(name=ap.tag, type='bool', value='true'))
                        # print(self.mea.param_dict['A0'])
                        for p_key, p_val in self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag].items():
                            self.params[-1].child(a_tag).child('AnchorPoints').child(ap.tag).addChild(AirfoilParameter(
                                self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag][p_key],
                                name=f"{a_tag}.AnchorPoints.{ap.tag}.{p_key}", type='float',
                                value=self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag][
                                    p_key].value,
                                context={'add_eq': 'Define by equation'}))

                if change == 'value' and param.name() == 'Inviscid Cp Calc':
                    self.cl_airfoil_tag = data

        self.p.sigTreeStateChanged.connect(change)

        def save():
            global state
            state = self.p.saveState()

        def restore():
            global state
            add = self.p['Save/Restore functionality', 'Restore State', 'Add missing items']
            rem = self.p['Save/Restore functionality', 'Restore State', 'Remove extra items']
            self.p.restoreState(state, addChildren=add, removeChildren=rem)

        self.p.param('Save/Restore functionality', 'Save State').sigActivated.connect(save)
        self.p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(restore)
        self.t = ParameterTree()
        # self.t.header().setResizeMode(QHeaderView.ResizeToContents)
        # self.t.header().setStretchLastSection(False)
        # for idx in [0, 1]:
        #     self.t.resizeColumnToContents(idx)
        self.t.setParameters(self.p, showTop=False)
        self.t.setWindowTitle('pymead ParameterTree')
        # header_view = QHeaderView(Qt.Horizontal, self.t)
        # print(header_view)
        self.t.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.t.header().setSectionResizeMode(1, QHeaderView.Interactive)

        self.win = QWidget()
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)
        self.layout.addWidget(QLabel("These are two views of the same data. They should always display the same values."),
                         0, 0, 1, 2)
        self.layout.addWidget(self.t, 1, 0, 1, 1)

    def plot_changed(self, airfoil_name: str):
        # print(f"plot changed for airfoil {airfoil_name}!")
        def plot_change_recursive(child_list: list):
            for idx, child in enumerate(child_list):
                if hasattr(child, "airfoil_param"):
                    if child.hasChildren():
                        if child.children()[0].name() == 'Equation Definition':
                            child.setValue(child.airfoil_param.value)
                        else:
                            plot_change_recursive(child.children())
                    else:
                        child.setValue(child.airfoil_param.value)
                else:
                    if child.hasChildren():
                        plot_change_recursive(child.children())

        plot_change_recursive(self.params[-1].child(airfoil_name).children())

    def update_auto_complete(self):

        for v in self.line_edit_refs.values():
            v.widget.setCompleter(QCompleter(self.mea.get_keys()))
# win.show()

# ## test save/restore
# state = p.saveState()
# p.restoreState(state)
# compareState = p.saveState()
# assert pg.eq(compareState, state)


if __name__ == '__main__':
    pg.exec()