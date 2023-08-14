import os.path
import typing

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, registerParameterItemType, ParameterItem
from pymead.core.mea import MEA
import pyqtgraph as pg
from pymead.core.param import Param
from pymead.core.pos_param import PosParam
from pymead.core.free_point import FreePoint
from pymead.core.anchor_point import AnchorPoint
from pymead.gui.autocomplete import AutoStrParameterItem
from pymead.gui.selectable_header import SelectableHeaderParameterItem
from pymead.gui.airfoil_pos_parameter import AirfoilPositionParameterItem
from pymead.gui.input_dialog import FreePointInputDialog, AnchorPointInputDialog, BoundsDialog
from pymead.analysis.single_element_inviscid import single_element_inviscid
from pymead.core.airfoil import Airfoil
from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QHeaderView
from PyQt5.QtWidgets import QAbstractItemView
from PyQt5.QtCore import pyqtSignal
from pymead.utils.downsampling_schemes import fractal_downsampler2
import pymead.core.symmetry  # DO NOT REMOVE
from pymead.gui.autocomplete import Completer
from functools import partial
from pymead import INCLUDE_FILES
import importlib.util
# import importlib.machinery
import numpy as np
from time import time


progress_idx = 0
# pymead.core.symmetry.symmetry("r")


class MEAParameters(pTypes.GroupParameter):
    """Class for storage of all the Multi-Element Airfoil Parameters."""
    def __init__(self, mea: MEA, status_bar, **opts):
        registerParameterItemType('selectable_header', SelectableHeaderParameterItem, override=True)
        registerParameterItemType('pos_parameter', AirfoilPositionParameterItem, override=True)
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.mea = mea
        self.status_bar = status_bar
        self.airfoil_headers = []
        self.custom_header = self.addChild(CustomGroup(mea, name='Custom'))
        for k, v in self.mea.param_dict['Custom'].items():
            if isinstance(v.value, list):
                pg_param = Parameter.create(name=k, type='pos_parameter', value=v.value,
                                            removable=True,
                                            renamable=True, context={'add_eq': 'Define by equation',
                                                                     'deactivate': 'Deactivate parameter',
                                                                     'activate': 'Activate parameter',
                                                                     'setbounds': 'Set parameter bounds'})
            else:
                pg_param = Parameter.create(name=k, type='float', value=v.value, removable=True,
                                            renamable=True, context={'add_eq': 'Define by equation',
                                                                     'deactivate': 'Deactivate parameter',
                                                                     'activate': 'Activate parameter',
                                                                     'setbounds': 'Set parameter bounds'})

            self.custom_header.addChild(pg_param)
            pg_param.airfoil_param = v
            pg_param.airfoil_param.name = pg_param.name()

        for idx, a in enumerate(self.mea.airfoils.values()):
            self.add_airfoil(a, idx)

    def add_airfoil(self, airfoil: Airfoil, idx: int):
        """Adds an airfoil, along with its associated Parameters, into a Multi-Element Airfoil system using the GUI."""
        self.airfoil_headers.append(self.addChild(dict(name=airfoil.tag, type='selectable_header', value=True,
                                                       context={"add_fp": "Add FreePoint",
                                                                "add_ap": "Add AnchorPoint"})))
        header_params = ['Base', 'AnchorPoints', 'FreePoints']
        for hp in header_params:
            # print(f"children = {self.airfoil_headers[idx].children()}")
            self.airfoil_headers[idx].addChild(HeaderParameter(name=hp, type='bool', value=True))
        for p_key, p_val in self.mea.param_dict[airfoil.tag]['Base'].items():
            pg_param = AirfoilParameter(self.mea.param_dict[airfoil.tag]['Base'][p_key],
                                        name=f"{airfoil.tag}.Base.{p_key}",
                                        type='float',
                                        value=self.mea.param_dict[airfoil.tag]['Base'][
                                            p_key].value,
                                        context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                                 'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'})
            self.airfoil_headers[idx].children()[0].addChild(pg_param)
            pg_param.airfoil_param.name = pg_param.name()
        # print(f"param_dict = {self.mea.param_dict}")
        for ap_key, ap_val in self.mea.param_dict[airfoil.tag]['AnchorPoints'].items():
            self.child(airfoil.tag).child('AnchorPoints').addChild(
                HeaderParameter(name=ap_key, type='bool', value='true', context={'remove_ap': 'Remove AnchorPoint'}))
            for p_key, p_val in self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key].items():
                if p_key != 'xy':
                    self.child(airfoil.tag).child('AnchorPoints').child(ap_key).addChild(AirfoilParameter(
                        self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key][p_key],
                        name=f"{airfoil.tag}.AnchorPoints.{ap_key}.{p_key}", type='float',
                        value=self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key][
                            p_key].value,
                        context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                 'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}))
                else:
                    airfoil_param = self.mea.param_dict[airfoil.tag]['AnchorPoints'][ap_key][p_key]
                    pg_param = Parameter.create(name=f"{airfoil.tag}.AnchorPoints.{ap_key}.{p_key}",
                                                type='pos_parameter',
                                                value=[-999.0, -999.0],
                                                context={'add_eq': 'Define by equation',
                                                         'deactivate': 'Deactivate parameter',
                                                         'activate': 'Activate parameter',
                                                         'setbounds': 'Set parameter bounds'})
                    pg_param.airfoil_param = airfoil_param
                    airfoil_param.name = pg_param.name()
                    self.child(airfoil.tag).child('AnchorPoints').child(
                        ap_key).addChild(pg_param)
                    pg_param.setValue([pg_param.airfoil_param.value[0], pg_param.airfoil_param.value[1]])
        for ap_key, ap_val in self.mea.param_dict[airfoil.tag]['FreePoints'].items():
            self.child(airfoil.tag).child('FreePoints').addChild(
                HeaderParameter(name=ap_key, type='bool', value='true'))
            for fp_key, fp_val in ap_val.items():
                self.child(airfoil.tag).child('FreePoints').child(ap_key).addChild(
                    HeaderParameter(name=fp_key, type='bool', value='true', context={'remove_fp': 'Remove FreePoint'}))
                for p_key, p_val in fp_val.items():
                    airfoil_param = self.mea.param_dict[airfoil.tag]['FreePoints'][ap_key][fp_key][p_key]
                    pg_param = Parameter.create(name=f"{airfoil.tag}.FreePoints.{ap_key}.{fp_key}.{p_key}",
                                                type='pos_parameter',
                                                value=[-999.0, -999.0],
                                                context={'add_eq': 'Define by equation',
                                                         'deactivate': 'Deactivate parameter',
                                                         'activate': 'Activate parameter',
                                                         'setbounds': 'Set parameter bounds'})
                    pg_param.airfoil_param = airfoil_param
                    airfoil_param.name = pg_param.name()
                    self.child(airfoil.tag).child('FreePoints').child(ap_key).child(
                        fp_key).addChild(pg_param)
                    pg_param.setValue([pg_param.airfoil_param.value[0], pg_param.airfoil_param.value[1]])
                # print(f"{self.child(airfoil.tag).child('FreePoints').child(ap_key).child(fp_key).children() = }")


class AirfoilParameter(pTypes.SimpleParameter):
    """Subclass of SimpleParameter which adds the airfoil_param attribute (a `pymead.core.param.Param`)."""
    def __init__(self, airfoil_param: Param, **opts):
        self.airfoil_param = airfoil_param
        pTypes.SimpleParameter.__init__(self, **opts)


class HeaderParameter(pTypes.GroupParameter):
    """Simple class for containing Parameters with no value. HeaderParameter has a similar purpose to a key in a
    nested dictionary."""
    def __init__(self, **opts):
        pTypes.GroupParameter.__init__(self, **opts)


class CustomGroup(pTypes.GroupParameter):
    """Class for addition of Custom Parameters to the Multi-Element Airfoil system within the GUI"""
    def __init__(self, mea: MEA, **opts):
        opts['type'] = 'group'
        opts['addText'] = 'Add'
        opts['addList'] = ['Param', 'PosParam']
        pTypes.GroupParameter.__init__(self, **opts)
        self.mea = mea

    def addNew(self, typ):
        default_value = {'Param': 0.0, 'PosParam': [0.0, 0.0]}[typ]
        default_name = f"CustomParam{(len(self.childs) + 1)}"
        if typ == 'Param':
            airfoil_param = Param(default_value)
            pg_param = Parameter.create(name=default_name, type='float', value=default_value, removable=True,
                                        renamable=True, context={'add_eq': 'Define by equation',
                                                                 'deactivate': 'Deactivate parameter',
                                                                 'activate': 'Activate parameter',
                                                                 'setbounds': 'Set parameter bounds'})
        elif typ == 'PosParam':
            airfoil_param = PosParam(default_value)
            pg_param = Parameter.create(name=default_name, type='pos_parameter', value=default_value, removable=True,
                                        renamable=True, context={'add_eq': 'Define by equation',
                                                                 'deactivate': 'Deactivate parameter',
                                                                 'activate': 'Activate parameter',
                                                                 'setbounds': 'Set parameter bounds'})
        else:
            raise ValueError("Current supported types of Custom Parameters are \'Param\' and \'PosParam\'")
        self.addChild(pg_param)
        pg_param.airfoil_param = airfoil_param
        self.mea.param_dict['Custom'][default_name] = airfoil_param


class MEAParamTree:
    """Class for containment of all Multi-Element Airfoil Parameters in the GUI"""
    def __init__(self, mea: MEA, status_bar, parent, progress_info: typing.NamedTuple = None):
        global progress_idx
        progress_idx = 0
        self.user_mods = {}
        for f in INCLUDE_FILES:
            name = os.path.split(f)[-1]  # get the name of the file without the directory
            name_no_ext = os.path.splitext(name)[-2]  # get the name of the file without the .py extension
            spec = importlib.util.spec_from_file_location(name_no_ext, f)
            self.user_mods[name_no_ext] = importlib.util.module_from_spec(spec)  # generate the module from the name
            # print(f"{self.user_mods[name_no_ext] = }")
            spec.loader.exec_module(self.user_mods[name_no_ext])  # compile and execute the module
            # TODO: implement "add user module" functionality into GUI
        self.dialog = None
        self.parent = parent
        self.params = [
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action', 'children': [
                    {'name': 'Add missing items', 'type': 'bool', 'value': True},
                    {'name': 'Remove extra items', 'type': 'bool', 'value': True},
                ]},
            ]},
            {'name': 'Analysis', 'type': 'group', 'children': [
                {'name': 'Inviscid Cl Calc', 'type': 'list', 'limits': [a.tag for a in mea.airfoils.values()]}]},
            MEAParameters(mea, status_bar, name='Airfoil Parameters'),
            # ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
            #     {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
            #     {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            # ]),
        ]

        # Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=self.params)

        self.equation_strings = {}

        progress_values = None
        if progress_info is not None:
            progress_values = np.round(np.linspace(progress_info.start, progress_info.end, progress_info.n)).astype(int)

        def add_equation_boxes_recursively(child_list):
            """Adds equation boxes for each loaded Airfoil Param which already contains an equation."""
            global progress_idx
            for child in child_list:
                if hasattr(child, 'airfoil_param'):
                    func_str = child.airfoil_param.func_str
                    if func_str is not None:
                        # Here we set update_auto_completer to False because it is really slow. We update the auto
                        # completer once afterward.
                        self.add_equation_box(child, func_str, update_auto_completer=False)
                        self.update_equation(child.child('Equation Definition'), func_str)
                        if progress_values is not None:
                            truncated_func_str = func_str if len(func_str) <= 50 else func_str[:50] + "..."
                            self.parent.statusBar().showMessage(f"Adding equation {truncated_func_str}")
                            self.parent.progress_bar.setValue(progress_values[progress_idx])
                            progress_idx += 1
                else:
                    if child.hasChildren():
                        add_equation_boxes_recursively(child.children())

        def set_readonly_recursively(child_list):
            """Sets the state to ReadOnly for each Airfoil Parameter which has an equation."""
            for child in child_list:
                if hasattr(child, 'airfoil_param'):
                    if not isinstance(child.airfoil_param,
                                      PosParam) and child.airfoil_param.linked or not child.airfoil_param.active:
                        child.setReadonly(True)
                else:
                    if child.hasChildren():
                        set_readonly_recursively(child.children())

        self.mea = mea
        self.mea.param_tree = self
        # self.cl_label = pg.LabelItem(size="18pt")
        # self.cl_label.setParentItem(self.mea.v)
        # self.cl_label.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        self.cl_label = None
        #
        self.cl_airfoil_tag = 'A0'

        registerParameterItemType('auto_str', AutoStrParameterItem, override=True)

        for a_name, a in self.mea.airfoils.items():
            a.airfoil_graph.scatter.sigPlotChanged.connect(
                partial(self.plot_changed, a_name))  # Needs to change with airfoil added or removed

        # For any change in the tree:
        def change(_, changes):
            """This function gets called any time the state of the ParameterTree or any of its child Parameters is
            changed."""

            for param, change, data in changes:

                # Removing an equation:
                if change == 'childRemoved' and data.opts['name'] == 'Equation Definition':
                    param.airfoil_param.remove_func()
                    param.setReadonly(False)
                    self.equation_strings.pop(param.name())

                path = self.p.childPath(param)
                if path is not None:
                    # Defining an equation:
                    if path[-1] == 'Equation Definition':
                        if change == 'value':
                            self.update_equation(param, data)

                def block_changes(pg_param):
                    pg_param.blockTreeChangeSignal()

                def flush_changes(pg_param):
                    pg_param.treeStateChanges = []
                    pg_param.blockTreeChangeEmit = 1
                    pg_param.unblockTreeChangeSignal()

                # Value change for any parameter:
                if hasattr(param, 'airfoil_param') and change == 'value':
                    param_name = param.name().split('.')[-1]
                    block_changes(param)
                    param.airfoil_param.value = data
                    Param.update_ap_fp(param.airfoil_param)
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
                                if param.airfoil_param is v:
                                    val.update()

                    self.plot_change_recursive(self.p.param('Airfoil Parameters').child('Custom').children())

                    for a in mea.airfoils.values():
                        for ap in a.anchor_points:
                            if ap.tag not in ['te_1', 'le', 'te_2']:
                                ap.set_ctrlpt_value()
                        for fp_dict in a.free_points.values():
                            for fp in fp_dict.values():
                                fp.set_ctrlpt_value()
                        a.update()
                        a.airfoil_graph.data['pos'] = a.control_point_array
                        a.airfoil_graph.updateGraph()

                        # Run inviscid CL calculation after any geometry change
                        # if a.tag == self.cl_airfoil_tag:
                        #     a.get_coords(body_fixed_csys=True)
                        #     # ds = fractal_downsampler2(a.coords, ratio_thresh=1.000005, abs_thresh=0.1)
                        #     _, _, CL = single_element_inviscid(a, a.alf.value * 180 / np.pi)
                        #     print(f"{CL = }")
                        #     self.cl_label.setText(f"{self.cl_airfoil_tag} Inviscid CL = {CL:.3f}")

                        self.plot_change_recursive(self.p.param('Airfoil Parameters').child(a.tag).children())

                    flush_changes(param)

                # Add equation child parameter if the change is the selection of the "equation" button in the
                # contextMenu
                if change == 'contextMenu' and data == 'add_eq':
                    self.add_equation_box(param)

                if change == 'contextMenu' and data == 'deactivate':
                    for p in self.t.multi_select:
                        if isinstance(p.param.airfoil_param, PosParam):
                            p.param.airfoil_param.active = [False, False]
                        else:
                            p.param.airfoil_param.active = False
                        p.param.setReadonly(True)

                if change == 'contextMenu' and data == 'activate':
                    for p in self.t.multi_select:
                        if isinstance(p.param.airfoil_param, PosParam):
                            p.param.airfoil_param.active = [True, True]
                        else:
                            p.param.airfoil_param.active = True
                        p.param.setReadonly(False)

                if change == 'contextMenu' and data == 'setbounds':
                    pos_param = isinstance(param.airfoil_param, PosParam)
                    self.dialog = BoundsDialog(param.airfoil_param.bounds, parent=self.t, pos_param=pos_param)
                    if self.dialog.exec():
                        inputs = self.dialog.getInputs()
                    else:
                        inputs = None
                    if inputs:
                        # param.airfoil_param.bounds = np.array([inputs[0], inputs[1]])
                        param.airfoil_param.bounds = inputs

                # Adding a FreePoint
                if change == 'contextMenu' and data == 'add_fp':
                    self.add_free_point(param)

                # Adding an AnchorPoint
                if change == 'contextMenu' and data == 'add_ap':
                    self.add_anchor_point(param)

                # Removing a FreePoint
                if change == 'contextMenu' and data == 'remove_fp':
                    self.remove_free_point(param)

                if change == 'contextMenu' and data == 'remove_ap':
                    self.remove_anchor_point(param)

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
                        raise ValueError('This should not be possible...')
                    self.mea.param_dict['Custom'][new_name] = self.mea.param_dict['Custom'].pop(key_to_change)
                    self.mea.param_dict['Custom'][new_name].name = new_name
                    # print(f"new param_dict = {self.mea.param_dict['Custom']}")

                if change == 'childRemoved' and param.name() == 'Custom':
                    self.mea.param_dict['Custom'].pop(data.name())

                    def recursive_refactor(child_list):
                        for child in child_list:
                            if hasattr(child, 'airfoil_param') and child.airfoil_param.func_str is not None:
                                # print(f"This one has a func_str!")
                                if key_to_change in child.airfoil_param.func_str:
                                    # print(f"Replacing {key_to_change} with {new_name}...")
                                    child.airfoil_param.func_str = \
                                        child.airfoil_param.func_str.replace(key_to_change, new_name)
                                    # print(f"func_str now is {child.airfoil_param.func_str}")
                                    child.child('Equation Definition').setValue(child.airfoil_param.func_str)
                            else:
                                recursive_refactor(child.children())

                    recursive_refactor(self.p.param('Airfoil Parameters').children())

        self.p.sigTreeStateChanged.connect(change)
        self.t = CustomParameterTree(parent=parent)
        self.t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.t.setParameters(self.p, showTop=False)
        self.t.setWindowTitle('pymead ParameterTree')
        self.t.header().setSectionResizeMode(0, QHeaderView.Interactive)
        self.t.header().setSectionResizeMode(1, QHeaderView.Interactive)
        self.t.setAlternatingRowColors(False)

        if self.t.parent().dark_mode:
            self.set_dark_mode()
        else:
            self.set_light_mode()

        add_equation_boxes_recursively(self.p.param('Airfoil Parameters').children())
        self.update_auto_complete()
        set_readonly_recursively(self.p.param('Airfoil Parameters').children())

        self.win = QWidget()
        self.layout = QGridLayout()
        self.win.setLayout(self.layout)
        self.layout.addWidget(QLabel("These are two views of the same data. "
                                     "They should always display the same values."), 0, 0, 1, 2)
        self.layout.addWidget(self.t, 1, 0, 1, 1)

    def set_dark_mode(self):
        """Sets the theme of the ParameterTree to dark by modifying the CSS"""
        self.t.setStyleSheet('''QTreeWidget {color: #3e3f40; alternate-background-color: #3e3f40;
                            selection-background-color: #3e3f40;}
                            QTreeView::item:hover {background: #36bacfaa;} QTreeWidget::item {border: 0px solid gray; color: #dce1e6}
                            QTreeWidget::item:selected {background-color: #36bacfaa; alternate-background-color: #36bacfaa} 
                            QTreeView::branch:has-siblings:adjoins-item {border-image: url(../icons/branch-more.png) 0;}
                            QTreeView::branch:!has-children:!has-siblings:adjoins-item {border-image: url(../icons/branch-end.png) 0;}
                            QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings  {
                            border-image: none;
                            image: url(../icons/closed-arrow.png);} 
                            QTreeView::branch:has-siblings:!adjoins-item {border-image: url(../icons/vline.png) 0;} 
                            QTreeView::branch:has-children:!has-siblings:closed,
                            QTreeView::branch:closed:has-children:has-siblings {border-image: none; image: url(../icons/opened-arrow.png);}''')

    def set_light_mode(self):
        """Sets the theme of the ParameterTree to light by modifying the CSS"""
        self.t.setStyleSheet('''QTreeWidget {color: white; alternate-background-color: white; 
                    selection-background-color: white;}
                    QTreeView::item::hover {background: #36bacfaa;} QTreeView::item {border: 0px solid gray; color: black}
                    QTreeWidget::item:selected {background-color: #36bacfaa; alternate-background-color: #36bacfaa}
                    QTreeView::branch:has-siblings:adjoins-item {border-image: url(../icons/branch-more.png) 0;}
                    QTreeView::branch:!has-children:!has-siblings:adjoins-item {border-image: url(../icons/branch-end.png) 0;}
                    QTreeView::branch:open:has-children:!has-siblings, QTreeView::branch:open:has-children:has-siblings  {
                    border-image: none;
                    image: url(../icons/closed-arrow.png);} 
                    QTreeView::branch:has-siblings:!adjoins-item {border-image: url(../icons/vline.png) 0;} 
                    QTreeView::branch:has-children:!has-siblings:closed,
                    QTreeView::branch:closed:has-children:has-siblings {border-image: none; image: url(../icons/opened-arrow.png);}''')

    def add_equation_box(self, pg_param, equation: str = None, update_auto_completer: bool = True):
        """Adds a QLineEdit to the ParameterTreeItem for equation editing"""
        if equation is None:
            value = ''
        else:
            value = equation
        if not pg_param.hasChildren():
            pg_param.addChild(dict(name='Equation Definition', type='auto_str', value=value, removable=True))
            self.equation_strings[pg_param.name()] = self.equation_widget(pg_param.child('Equation Definition'))
            if update_auto_completer:
                self.update_auto_complete()

    @staticmethod
    def plot_changed(airfoil_name: str):
        """This function gets called any time an Airfoil is added or removed

        Parameters
        ==========
        airfoil_name: str
          Name of the airfoil being added or removed
        """

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

        pass

    def update_auto_complete(self):
        """Update the auto-completer for the equation text"""
        for v in self.equation_strings.values():
            v.setCompleter(Completer(self.mea.get_keys()))

    def update_equation(self, pg_param: Parameter, equation_str: str, **func_dict_kwargs):
        """Updates the parameter equation based on user input inside the GUI. The text is colored green if the equation
        compiles successfully and the text is colored red otherwise.

        Parameters
        ==========
        pg_param: Parameter
          The pyqtgraph parameter representing the equation box

        equation_str: str
          The string to apply as an equation to pg_param's parent

        **func_dict_kwargs
          Key-value pairs to merge into the airfoil parameter's function dictionary, useful for custom functions. For
          example, to add a custom function named "foo" with required input "bar" (a str with value "h"), simply add
          foo=foo and bar="h" as keyword arguments to update_equation.
        """
        pg_eq_parent = pg_param.parent()
        airfoil_param = pg_eq_parent.airfoil_param
        pg_eq_parent.setReadonly()
        airfoil_param.mea = self.mea
        temp_func_dict = {**func_dict_kwargs}
        # INCLUDE_FILES.append('test2.py')
        # for k, v in temp_func_dict.items():
        #     if isinstance(v, str) and v[:2] == '$$':

        # if 'name' not in airfoil_param.function_dict.keys():
        if airfoil_param.name is None:
            airfoil_param.name = pg_eq_parent.name()
        airfoil_param.function_dict['name'] = airfoil_param.name.split('.')[-1]
        airfoil_param.function_dict = {**airfoil_param.function_dict, **func_dict_kwargs}
        airfoil_param.set_func_str(equation_str)
        try:
            airfoil_param.update(func_str_changed=True)
        except (SyntaxError, NameError, AttributeError):
            pg_eq_parent.setReadonly(False)
            airfoil_param.remove_func()
            self.equation_widget(pg_param).setStyleSheet('border: 0px; color: #fa4b4b;')  # make the equation red
            self.parent.disp_message_box("Could not compile function")
            return

        # Update the parameter's dependencies
        if len(airfoil_param.depends_on) > 0:
            for air_par in airfoil_param.depends_on.values():
                air_par.update()
            if airfoil_param.linked:
                pg_eq_parent.setValue(airfoil_param.value)
            self.equation_widget(pg_param).setStyleSheet('border: 0px; color: #a1fa9d;')  # make the equation green
            Param.update_ap_fp(airfoil_param)
        else:
            pg_eq_parent.setReadonly(False)
            self.equation_widget(pg_param).setStyleSheet('border: 0px; color: #fa4b4b;')  # make the equation red
            self.parent.disp_message_box("Could not compile function")

        for a in self.mea.airfoils.values():
            a.update()
            a.airfoil_graph.data['pos'] = a.control_point_array
            a.airfoil_graph.updateGraph()

    def add_free_point(self, pg_param: Parameter):
        """Adds a FreePoint to the specified Airfoil and updates the graph.

        Parameters
        ==========
        pg_param: Parameter
          HeaderParameter within the ParameterTree named with the FreePoint's Airfoil tag
        """
        a_tag = pg_param.name()
        self.dialog = FreePointInputDialog(items=[("x", "double", 0.5), ("y", "double", 0.1),
                                                  ("Previous Anchor Point", "combo"),
                                                  ("Previous Free Point", "combo")],
                                           fp=self.mea.airfoils[a_tag].free_points, parent=self.parent)
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
        else:
            inputs = None
        if inputs:
            if inputs[3] == 'None':
                pfp = None
            else:
                pfp = inputs[3]
            fp = FreePoint(PosParam(value=(inputs[0], inputs[1])), airfoil_tag=a_tag,
                           previous_anchor_point=inputs[2], previous_free_point=pfp)
            self.mea.airfoils[a_tag].insert_free_point(fp)
            self.mea.airfoils[a_tag].update()
            self.mea.assign_names_to_params_in_param_dict()
            self.dialog.update_fp_ap_tags()
            pos, adj, symbols = self.mea.airfoils[a_tag].airfoil_graph.update_airfoil_data()
            self.mea.airfoils[a_tag].airfoil_graph.setData(pos=pos, adj=adj, size=8, pxMode=True, symbol=symbols)
            if fp.anchor_point_tag not in [p.name() for p in self.params[-1].child(a_tag).child('FreePoints')]:
                self.params[-1].child(a_tag).child('FreePoints').addChild(
                    HeaderParameter(name=fp.anchor_point_tag, type='bool', value='true'))
            self.params[-1].child(a_tag).child('FreePoints').child(fp.anchor_point_tag).addChild(
                HeaderParameter(name=fp.tag, type='bool', value='true', context={'remove_fp': 'Remove FreePoint'}))
            for p_key, p_val in self.mea.param_dict[a_tag]['FreePoints'][fp.anchor_point_tag][fp.tag].items():
                airfoil_param = self.mea.param_dict[a_tag]['FreePoints'][fp.anchor_point_tag][fp.tag][p_key]
                pg_param = Parameter.create(name=f"{a_tag}.FreePoints.{fp.anchor_point_tag}.{fp.tag}.{p_key}",
                                            type='pos_parameter',
                                            value=[-999.0, -999.0],
                                            context={'add_eq': 'Define by equation',
                                                     'deactivate': 'Deactivate parameter',
                                                     'activate': 'Activate parameter',
                                                     'setbounds': 'Set parameter bounds'})
                pg_param.airfoil_param = airfoil_param
                self.params[-1].child(a_tag).child('FreePoints').child(fp.anchor_point_tag).child(
                    fp.tag).addChild(pg_param)
                pg_param.setValue([pg_param.airfoil_param.value[0], pg_param.airfoil_param.value[1]])

    def add_anchor_point(self, pg_param: Parameter):
        """Adds an AnchorPoint to the specified Airfoil and updates the graph.

        Parameters
        ==========
        pg_param: Parameter
          HeaderParameter within the ParameterTree named with the AnchorPoint's Airfoil tag
        """
        a_tag = pg_param.name()
        self.dialog = AnchorPointInputDialog(items=[("x", "double", 0.5), ("y", "double", 0.1),
                                                    ("L", "double", 0.1), ("R", "double", 2.0),
                                                    ("r", "double", 0.5), ("phi", "double", 0.0),
                                                    ("psi1", "double", 1.5), ("psi2", "double", 1.5),
                                                    ("Previous Anchor Point", "combo"),
                                                    ("Anchor Point Name", "string")],
                                             ap=self.mea.airfoils[a_tag].anchor_points, parent=self.parent)
        if self.dialog.exec():
            inputs = self.dialog.getInputs()
        else:
            inputs = None

        if inputs:  # Continue only if the dialog was accepted:
            for curve in self.mea.airfoils[a_tag].curve_list:
                curve.clear_curve_pg()
            ap = AnchorPoint(xy=PosParam((inputs[0], inputs[1])), L=Param(inputs[2]), R=Param(inputs[3]),
                             r=Param(inputs[4]), phi=Param(inputs[5]), psi1=Param(inputs[6]),
                             psi2=Param(inputs[7]), previous_anchor_point=inputs[8], tag=inputs[9], airfoil_tag=a_tag)

            # Stop execution if the AnchorPoint tag already exists for the airfoil
            if ap.tag in self.mea.airfoils[a_tag].anchor_point_order:
                self.parent.disp_message_box(f"AnchorPoint tag {ap.tag} already exists in Airfoil {a_tag}. "
                                             f"Please choose a different name.")
                return

            # Insert the AnchorPoint and update the Airfoil
            self.mea.airfoils[a_tag].insert_anchor_point(ap)
            self.mea.airfoils[a_tag].update()

            self.mea.assign_names_to_params_in_param_dict()
            self.mea.airfoils[a_tag].init_airfoil_curve_pg(self.mea.airfoils[a_tag].airfoil_graph.v,
                                                           pen=pg.mkPen(color='cornflowerblue', width=2))

            self.dialog.update_ap_tags()
            pos, adj, symbols = self.mea.airfoils[a_tag].airfoil_graph.update_airfoil_data()
            self.mea.airfoils[a_tag].airfoil_graph.setData(pos=pos, adj=adj, size=8, pxMode=True,
                                                           symbol=symbols)

            # Add the appropriate Headers to the ParameterTree:
            self.params[-1].child(a_tag).child('AnchorPoints').addChild(
                HeaderParameter(name=ap.tag, type='bool', value='true', context={'remove_ap': 'Remove AnchorPoint'}))
            self.params[-1].child(a_tag).child('FreePoints').addChild(
                HeaderParameter(name=ap.tag, type='bool', value='true'))

            # Add the appropriate Parameters to the ParameterTree:
            for p_key, p_val in self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag].items():
                if p_key != 'xy':
                    self.params[-1].child(a_tag).child('AnchorPoints').child(ap.tag).addChild(AirfoilParameter(
                        self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag][p_key],
                        name=f"{a_tag}.AnchorPoints.{ap.tag}.{p_key}", type='float',
                        value=self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag][
                            p_key].value,
                        context={'add_eq': 'Define by equation', 'deactivate': 'Deactivate parameter',
                                 'activate': 'Activate parameter', 'setbounds': 'Set parameter bounds'}))
                else:
                    airfoil_param = self.mea.param_dict[a_tag]['AnchorPoints'][ap.tag][p_key]
                    pg_param = Parameter.create(name=f"{a_tag}.AnchorPoints.{ap.tag}.{p_key}",
                                                type='pos_parameter',
                                                value=[-999.0, -999.0],
                                                context={'add_eq': 'Define by equation',
                                                         'deactivate': 'Deactivate parameter',
                                                         'activate': 'Activate parameter',
                                                         'setbounds': 'Set parameter bounds'})
                    pg_param.airfoil_param = airfoil_param
                    self.params[-1].child(a_tag).child('AnchorPoints').child(
                        ap.tag).addChild(pg_param)
                    pg_param.setValue([pg_param.airfoil_param.value[0], pg_param.airfoil_param.value[1]])

    def remove_free_point(self, pg_param: Parameter):
        """Removes a FreePoint from an Airfoil and updates the graph

        Parameters
        ==========
        pg_param: Parameter
          HeaderParameter within the ParameterTree named with the FreePoint's tag
        """
        # First, make sure that a FreePoint parameter was selected:
        if not pg_param.parent() or not pg_param.parent().parent() or not pg_param.parent().parent().name() == 'FreePoints':
            self.parent.disp_message_box('A FreePoint must be selected to remove; e.g., \'FP0\'')
            return

        # Get the Airfoil from which to remove the FreePoint:
        airfoil_name = pg_param.parent().parent().parent().name()
        ap_name = pg_param.parent().name()
        fp_name = pg_param.name()
        airfoil = self.mea.airfoils[airfoil_name]

        # Delete the FreePoint from the Airfoil:
        airfoil.delete_free_point(fp_name, ap_name)
        airfoil.update()

        # Delete the FreePoint from the parameter tree:
        pg_param.clearChildren()
        pg_param.remove()

        # Update the Graph:
        # self.dialog.update_fp_ap_tags()
        pos, adj, symbols = self.mea.airfoils[airfoil_name].airfoil_graph.update_airfoil_data()
        self.mea.airfoils[airfoil_name].airfoil_graph.setData(pos=pos, adj=adj, size=8, pxMode=True,
                                                              symbol=symbols)

    def remove_anchor_point(self, pg_param: Parameter):
        """Removes an AnchorPoint from an Airfoil and updates the graph

        Parameters
        ==========
        pg_param: Parameter
          HeaderParameter within the ParameterTree named with the AnchorPoint's tag
        """
        # First, make sure that a FreePoint parameter was selected:
        if not pg_param.parent() or not pg_param.parent().name() == 'AnchorPoints':
            self.parent.disp_message_box('An AnchorPoint must be selected to remove; e.g., \'AP0\'')
            return

        # Get the Airfoil from which to remove the AnchorPoint:
        airfoil_name = pg_param.parent().parent().name()
        ap_name = pg_param.name()
        airfoil = self.mea.airfoils[airfoil_name]

        # Delete the AnchorPoint from the Airfoil:
        airfoil.delete_anchor_point(ap_name)
        airfoil.update()

        # Delete the AnchorPoint and its children from the FreePoint header in the parameter tree:
        fp_ap_param = pg_param.parent().parent().child('FreePoints').child(ap_name)
        fp_ap_param.clearChildren()
        fp_ap_param.remove()

        # Delete the AnchorPoint from the parameter tree:
        pg_param.clearChildren()
        pg_param.remove()

        # Update the Graph:
        # self.dialog.update_ap_tags()
        pos, adj, symbols = self.mea.airfoils[airfoil_name].airfoil_graph.update_airfoil_data()
        self.mea.airfoils[airfoil_name].airfoil_graph.setData(pos=pos, adj=adj, size=8, pxMode=True,
                                                              symbol=symbols)

    def equation_widget(self, pg_param: Parameter):
        """Acquires the equation's container QWidget"""
        return next((p for p in self.t.listAllItems() if hasattr(p, 'param') and p.param is pg_param)).widget

    @staticmethod
    def block_changes(pg_param: Parameter):
        """Blocks all changes to the specified parameter

        Parameters
        ==========
        pg_param: Parameter
          Parameter for which to block state changes
        """
        pg_param.blockTreeChangeSignal()

    @staticmethod
    def flush_changes(pg_param):
        """Flushes all changes from the specified parameter

        Parameters
        ==========
        pg_param: Parameter
          Parameter from which to flush state changes
        """
        pg_param.treeStateChanges = []
        pg_param.blockTreeChangeEmit = 1
        pg_param.unblockTreeChangeSignal()

    def plot_change_recursive(self, child_list: list):
        """Moves through the pyqtgraph ParameterTree recursively, setting the value of each pyqtgraph Parameter to the
        value of the contained pymead parameter value."""
        for idx, child in enumerate(child_list):
            if hasattr(child, "airfoil_param"):
                if child.hasChildren():
                    if child.children()[0].name() == 'Equation Definition':
                        child.setValue(child.airfoil_param.value)
                    else:
                        self.plot_change_recursive(child.children())
                else:
                    child.setValue(child.airfoil_param.value)
            else:
                if child.hasChildren():
                    self.plot_change_recursive(child.children())


class CustomParameterTree(ParameterTree):
    """A custom version of pyqtgraph's ParameterTree (allows for multiple selection and custom signal emitting)"""
    sigSymmetry = pyqtSignal(str)
    sigPosConstraint = pyqtSignal(str)
    sigSelChanged = pyqtSignal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.multi_select = None

    def selectionChanged(self, *args):
        """Override method in pyqtgraph's ParameterTree and make some modifications"""
        sel = self.selectedItems()
        if sel and len(sel) > 0:
            # Emit signals required for symmetry enforcement
            param = sel[-1].param
            parent = sel[-1].param.parent()
            if (parent and isinstance(parent, CustomGroup)) or isinstance(param, HeaderParameter):
                self.sigSymmetry.emit(self.get_full_param_name_path(param))
                self.sigPosConstraint.emit(self.get_full_param_name_path(param))
                self.sigSelChanged.emit((self.get_full_param_name_path(param), param.value()))
            elif isinstance(param, AirfoilParameter):
                self.sigSymmetry.emit(f"${param.name()}")
                self.sigPosConstraint.emit(f"${param.name()}")
                self.sigSelChanged.emit((f"${param.name()}", param.value()))
        self.multi_select = sel
        if len(sel) != 1:
            sel = None
        if self.lastSel is not None and isinstance(self.lastSel, ParameterItem):
            self.lastSel.selected(False)
        if sel is None:
            self.lastSel = None
            return
        self.lastSel = sel[0]
        if hasattr(sel[0], 'selected'):
            sel[0].selected(True)
        for selection in self.multi_select:
            if hasattr(selection, 'widget'):
                selection.widget.setMinimumHeight(20)
        return super().selectionChanged(*args)

    @staticmethod
    def get_full_param_name_path(param: Parameter):
        """Get the full path of the parameter in the parameter tree (dot seperator, leading $)"""
        path_list = []
        for idx in range(5):
            if not param.name() or param.name() == 'Airfoil Parameters':
                break
            path_list.append(param.name())
            param = param.parent()
        return f"${'.'.join(path_list[::-1])}"


if __name__ == '__main__':
    pg.exec()
