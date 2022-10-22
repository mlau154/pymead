import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, registerParameterItemType
from pymead.core.mea import MEA
import pyqtgraph as pg
from pymead.core.param import Param
from pymead.core.free_point import FreePoint
from pymead.core.anchor_point import AnchorPoint
from pymead.gui.autocomplete import AutoStrParameterItem
from pymead.gui.selectable_header import SelectableHeaderParameterItem
from pymead.gui.input_dialog import FreePointInputDialog, AnchorPointInputDialog, BoundsDialog
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.core.airfoil import Airfoil
from PyQt5.QtWidgets import QCompleter, QWidget, QGridLayout, QLabel, QInputDialog, QHeaderView
from pymead.utils.downsampling_schemes import fractal_downsampler2
from pymead.gui.autocomplete import Completer
from PyQt5.QtCore import Qt
from functools import partial
import numpy as np
from time import time

import ctypes

app = pg.mkQApp("Parameter Tree")


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
        self.custom_header = self.addChild(CustomGroup(mea, name='Custom'))
        for k, v in self.mea.param_dict['Custom'].items():
            pg_param = self.custom_header.addChild(dict(name=k, type='float', value=v.value, removable=True,
                                                        renamable=True, context={'add_eq': 'Define by equation',
                                                                                 'deactivate': 'Deactivate parameter',
                                                                                 'activate': 'Activate parameter',
                                                                                 'setbounds': 'Set parameter bounds'}))
            pg_param.airfoil_param = v
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
                                                                              context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                         'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}))
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
                    context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                         'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}))
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
                                         context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                         'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}))


class AirfoilParameter(pTypes.SimpleParameter):
    def __init__(self, airfoil_param: Param, **opts):
        self.airfoil_param = airfoil_param
        pTypes.SimpleParameter.__init__(self, **opts)


class HeaderParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        pTypes.GroupParameter.__init__(self, **opts)


# this group includes a menu allowing the user to add new parameters into its child list
class CustomGroup(pTypes.GroupParameter):
    def __init__(self, mea: MEA, **opts):
        opts['type'] = 'group'
        opts['addText'] = 'Add'
        opts['addList'] = ['New']
        pTypes.GroupParameter.__init__(self, **opts)
        self.mea = mea

    def addNew(self, typ):
        default_value = 0.0
        default_name = f"CustomParam{(len(self.childs) + 1)}"
        airfoil_param = Param(default_value)
        pg_param = self.addChild(dict(name=default_name,
                                      type='float', value=default_value, removable=True, renamable=True,
                                      context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                               'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}
                                      ))
        pg_param.airfoil_param = airfoil_param
        self.mea.param_dict['Custom'][default_name] = airfoil_param
        # pg_param.sigNameChanged.connect(self.name_changed_action)

    # def name_changed_action(self, pg_param):
        # new_name = pg_param.name()
        # key_to_change = ''
        # for k, v in self.mea.param_dict['Custom'].items():
        #     if v is pg_param.airfoil_param:
        #         key_to_change = k
        #         break
        # if key_to_change == '':
        #     raise ValueError('This shouldn\'t be possible...')
        # self.mea.param_dict['Custom'][new_name] = self.mea.param_dict['Custom'].pop(key_to_change)
        # print(f"new param_dict = {self.mea.param_dict['Custom']}")


# Create two ParameterTree widgets, both accessing the same data
class MEAParamTree:
    def __init__(self, mea: MEA, status_bar, parent=None):
        self.params = [
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action', 'children': [
                    {'name': 'Add missing items', 'type': 'bool', 'value': True},
                    {'name': 'Remove extra items', 'type': 'bool', 'value': True},
                ]},
            ]},
            {'name': 'Analysis', 'type': 'group', 'children': [{'name': 'Inviscid Cl Calc', 'type': 'list', 'limits': [a.tag for a in mea.airfoils.values()]}]},
            MEAParameters(mea, status_bar, name='Airfoil Parameters'),
            # ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
            #     {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
            #     {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            # ]),
        ]

        # Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=self.params)

        self.equation_strings = {}

        def add_equation_boxes_recursively(child_list):
            for child in child_list:
                if hasattr(child, 'airfoil_param'):
                    if child.airfoil_param.func_str is not None:
                        self.add_equation_box(child, child.airfoil_param.func_str)
                else:
                    if child.hasChildren():
                        add_equation_boxes_recursively(child.children())

        def set_readonly_recursively(child_list):
            for child in child_list:
                if hasattr(child, 'airfoil_param'):
                    if child.airfoil_param.linked or not child.airfoil_param.active:
                        child.setReadonly(True)
                else:
                    if child.hasChildren():
                        set_readonly_recursively(child.children())

        self.mea = mea
        self.cl_label = pg.LabelItem(size="18pt")
        self.cl_label.setParentItem(self.mea.v)
        self.cl_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

        self.cl_airfoil_tag = 'A0'

        registerParameterItemType('auto_str', AutoStrParameterItem, override=True)

        for a_name, a in mea.airfoils.items():
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.plot_changed, a_name))  # Needs to change with airfoil added or removed

        # For any change in the tree:
        def change(_, changes):

            for param, change, data in changes:
                print(f"change = {change}")
                print(f"param = {param}")
                print(f"data = {data}")

                # Removing an equation:
                if change == 'childRemoved' and data.opts['name'] == 'Equation Definition':
                    # print('Removing equation...')
                    param.airfoil_param.remove_func()
                    param.setReadonly(False)
                    self.equation_strings.pop(param.name())

                path = self.p.childPath(param)
                # print(f"path = {path}")
                if path is not None:
                    # Defining an equation:
                    if path[-1] == 'Equation Definition':
                        if change == 'value':

                            param.parent().setReadonly()
                            param.parent().airfoil_param.mea = mea
                            param.parent().airfoil_param.set_func_str(str(data))
                            param.parent().airfoil_param.update()
                            if len(param.parent().airfoil_param.depends_on) > 0:
                                for air_par in param.parent().airfoil_param.depends_on.values():
                                    air_par.update()
                                if param.parent().airfoil_param.linked:
                                    param.parent().setValue(param.parent().airfoil_param.value)
                                self.equation_widget(param).setStyleSheet('border: 0px; color: green;')
                                if param.parent().parent().parent().parent().name() == 'FreePoints':
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
                                    ap_name = param.parent().parent().name()
                                    a_name = param.parent().parent().parent().parent().name()
                                    param_name = param.parent().name().split('.')[-1]
                                    ap = self.mea.airfoils[a_name].anchor_points[self.mea.airfoils[a_name].anchor_point_order.index(ap_name)]
                                    if param_name in ['x', 'y']:
                                        ap.set_x_value(None)
                                        ap.set_y_value(None)
                                    ap.set_ctrlpt_value()
                            else:
                                # print(f"parent name is {param.parent().name()}")
                                param.parent().setReadonly(False)
                                self.equation_widget(param).setStyleSheet('border: 0px; color: red;')

                def block_changes(pg_param):
                    pg_param.blockTreeChangeSignal()

                def flush_changes(pg_param):
                    pg_param.treeStateChanges = []
                    pg_param.blockTreeChangeEmit = 1
                    pg_param.unblockTreeChangeSignal()

                # Value change for any parameter:
                if hasattr(param, 'airfoil_param') and change == 'value':
                    param_name = param.name().split('.')[-1]
                    if param.airfoil_param.active and not param.airfoil_param.linked:
                        block_changes(param)
                        if param_name not in ['x', 'y', 'xp', 'yp']:
                            param.airfoil_param.value = data
                        else:
                            if 'FreePoints' in param.name():
                                fp_name = param.parent().name()
                                ap_name = param.parent().parent().name()
                                a_name = param.parent().parent().parent().parent().name()
                                fp_or_ap = self.mea.airfoils[a_name].airfoil_graph.airfoil.free_points[ap_name][fp_name]
                            else:
                                ap_name = param.parent().name()
                                a_name = param.parent().parent().parent().name()
                                airfoil = self.mea.airfoils[a_name].airfoil_graph.airfoil
                                aps = airfoil.anchor_points
                                ap_order = airfoil.anchor_point_order
                                fp_or_ap = aps[ap_order.index(ap_name)]
                            if param_name == 'x':
                                fp_or_ap.set_xy(x=data, y=fp_or_ap.y.value)
                            elif param_name == 'y':
                                fp_or_ap.set_xy(y=data, x=fp_or_ap.x.value)
                            elif param_name == 'xp':
                                fp_or_ap.set_xy(xp=data, yp=fp_or_ap.yp.value)
                            elif param_name == 'yp':
                                fp_or_ap.set_xy(yp=data, xp=fp_or_ap.xp.value)

                    param.airfoil_param.update()

                    if param.airfoil_param.linked:
                        param.setValue(param.airfoil_param.value)

                    list_of_vals = []

                    def get_list_of_vals_from_dict(d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                get_list_of_vals_from_dict(v)
                            else:
                                list_of_vals.append(v)

                    # IMPORTANT
                    if mea.param_dict is not None:
                        get_list_of_vals_from_dict(mea.param_dict)
                        for val in list_of_vals:
                            for v in val.depends_on.values():
                                # print(f"v = {v}")
                                if param.airfoil_param is v:
                                    val.update()

                    self.plot_change_recursive(self.p.param('Airfoil Parameters').child('Custom').children())

                    for a in mea.airfoils.values():
                        a.update()
                        a.airfoil_graph.data['pos'] = a.control_point_array
                        a.airfoil_graph.updateGraph()

                        # Run inviscid CL calculation after any geometry change
                        if a.tag == self.cl_airfoil_tag:
                            a.get_coords(body_fixed_csys=True)
                            ds = fractal_downsampler2(a.coords, ratio_thresh=1.000005, abs_thresh=0.1)
                            _, _, CL = single_element_inviscid(ds, a.alf.value * 180 / np.pi)
                            self.cl_label.setText(f"{self.cl_airfoil_tag} Inviscid CL = {CL:.3f}")

                        self.plot_change_recursive(self.p.param('Airfoil Parameters').child(a.tag).children())

                    flush_changes(param)

                # Add equation child parameter if the change is the selection of the "equation" button in the
                # contextMenu
                if change == 'contextMenu' and data == 'add_eq':
                    self.add_equation_box(param)

                if change == 'contextMenu' and data == 'deactivate':
                    param.airfoil_param.active = False
                    param.setReadonly(True)

                if change == 'contextMenu' and data == 'activate':
                    param.airfoil_param.active = True
                    param.setReadonly(False)

                if change == 'contextMenu' and data == 'setbounds':
                    self.dialog = BoundsDialog(param.airfoil_param.bounds, parent=self.t)
                    if self.dialog.exec():
                        inputs = self.dialog.getInputs()
                    else:
                        inputs = None
                    if inputs:
                        param.airfoil_param.bounds = np.array([inputs[0], inputs[1]])

                # Adding a FreePoint
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
                        fp = FreePoint(Param(inputs[0]), Param(inputs[1]), airfoil_tag=a_tag,
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
                                                 context={'add_eq': 'Define by equation',
                                                          'deactivate': 'Deactivate parameter',
                                                          'activate': 'Activate parameter',
                                                          'setbounds': 'Set parameter bounds'}))

                # Adding an AnchorPoint
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
                                         psi2=Param(inputs[7]), previous_anchor_point=inputs[8], tag=inputs[9], airfoil_tag=a_tag)
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
                                context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                         'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}))

                # Different value in QComboBox for the inviscid CL calculation is selected
                if change == 'value' and param.name() == 'Inviscid Cl Calc':
                    self.cl_airfoil_tag = data

                if change == 'name':
                    new_name = param.name()
                    key_to_change = ''
                    for k, v in self.mea.param_dict['Custom'].items():
                        if v is param.airfoil_param:
                            key_to_change = k
                            break
                    if key_to_change == '':
                        raise ValueError('This shouldn\'t be possible...')
                    self.mea.param_dict['Custom'][new_name] = self.mea.param_dict['Custom'].pop(key_to_change)
                    # print(f"new param_dict = {self.mea.param_dict['Custom']}")

                if change == 'childRemoved' and param.name() == 'Custom':
                    self.mea.param_dict['Custom'].pop(data.name())
                    print(f"new custom param_dict = {self.mea.param_dict['Custom']}")

                    def recursive_refactor(child_list):
                        for child in child_list:
                            if hasattr(child, 'airfoil_param') and child.airfoil_param.func_str is not None:
                                print(f"This one has a func_str!")
                                if key_to_change in child.airfoil_param.func_str:
                                    print(f"Replacing {key_to_change} with {new_name}...")
                                    child.airfoil_param.func_str = \
                                        child.airfoil_param.func_str.replace(key_to_change, new_name)
                                    print(f"func_str now is {child.airfoil_param.func_str}")
                                    child.child('Equation Definition').setValue(child.airfoil_param.func_str)
                            else:
                                recursive_refactor(child.children())

                    recursive_refactor(self.p.param('Airfoil Parameters').children())

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
        self.t = ParameterTree(parent=parent)
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
        self.t.setAlternatingRowColors(False)

        if self.t.parent().dark_mode:
            self.set_dark_mode()
        else:
            self.set_light_mode()

        add_equation_boxes_recursively(self.p.param('Airfoil Parameters').children())
        set_readonly_recursively(self.p.param('Airfoil Parameters').children())

        self.win = QWidget()
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)
        self.layout.addWidget(QLabel("These are two views of the same data. They should always display the same values."),
                         0, 0, 1, 2)
        self.layout.addWidget(self.t, 1, 0, 1, 1)

    def set_dark_mode(self):
        self.t.setStyleSheet('''QTreeWidget {color: #dce1e6; alternate-background-color: #dce1e6;
                            selection-background-color: #36bacfaa;} 
                            QTreeView::item:hover {background: #36bacfaa;} QTreeView::item {border: 0px solid gray; color: #dce1e6}''')

    def set_light_mode(self):
        self.t.setStyleSheet('''QTreeWidget {color: white; alternate-background-color: white; 
                    selection-background-color: #36bacfaa}
                    QTreeView::item::hover {background: #36bacfaa;} QTreeView::item {border: 0px solid gray; color: black}''')

    def add_equation_box(self, pg_param, equation: str = None):
        if equation is None:
            value = ''
        else:
            value = equation
        if not pg_param.hasChildren():
            pg_param.addChild(dict(name='Equation Definition', type='auto_str', value=value, removable=True))
            # for k in pg_param.children()[0].items.data.keys():
            #     id_val = ''
            #     for ch in str(k)[-19:-1]:
            #         id_val += ch
            #     self.equation_strings[pg_param.name()] = ctypes.cast(int(id_val, 0), ctypes.py_object).value
            #     # print(f"line edit refs in add equation box is {self.line_edit_refs}")
            #     self.update_auto_complete()
            self.equation_strings[pg_param.name()] = self.equation_widget(pg_param.child('Equation Definition'))
            self.update_auto_complete()

    def plot_changed(self, airfoil_name: str):

        def block_changes(pg_param):
            pg_param.blockTreeChangeSignal()

        def flush_changes(pg_param):
            pg_param.treeStateChanges = []
            pg_param.blockTreeChangeEmit = 1
            pg_param.unblockTreeChangeSignal()

        def plot_change_recursive(child_list: list):
            for idx, child in enumerate(child_list):
                if hasattr(child, "airfoil_param"):
                    if child.hasChildren():
                        if child.children()[0].name() == 'Equation Definition':
                            block_changes(child)
                            child.setValue(child.airfoil_param.value)
                            flush_changes(child)
                        else:
                            plot_change_recursive(child.children())
                    else:
                        block_changes(child)
                        child.setValue(child.airfoil_param.value)
                        flush_changes(child)
                else:
                    if child.hasChildren():
                        plot_change_recursive(child.children())

        # plot_change_recursive(self.params[-1].child(airfoil_name).children())
        pass

    def update_auto_complete(self):
        for v in self.equation_strings.values():
            v.setCompleter(Completer(self.mea.get_keys()))

    def equation_widget(self, pg_param):
        return next((p for p in self.t.listAllItems() if hasattr(p, 'param') and p.param is pg_param)).widget

    @staticmethod
    def block_changes(pg_param):
        pg_param.blockTreeChangeSignal()

    @staticmethod
    def flush_changes(pg_param):
        pg_param.treeStateChanges = []
        pg_param.blockTreeChangeEmit = 1
        pg_param.unblockTreeChangeSignal()

    def plot_change_recursive(self, child_list: list):
        for idx, child in enumerate(child_list):
            # print(f"child name is {child.name()}")
            if hasattr(child, "airfoil_param"):
                if child.hasChildren():
                    if child.children()[0].name() == 'Equation Definition':
                        # print(f"Setting value of {child.name()}")
                        child.setValue(child.airfoil_param.value)
                    else:
                        self.plot_change_recursive(child.children())
                else:
                    child.setValue(child.airfoil_param.value)
            else:
                if child.hasChildren():
                    self.plot_change_recursive(child.children())


if __name__ == '__main__':
    pg.exec()
