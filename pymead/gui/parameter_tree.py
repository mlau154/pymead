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
from pymead.gui.autocomplete import AutoStrParameterItem
from pymead.core.airfoil import Airfoil
from PyQt5.QtWidgets import QCompleter, QWidget, QGridLayout, QLabel
from functools import partial

import ctypes

app = pg.mkQApp("Parameter Tree Example")


# test subclassing parameters
# This parameter automatically generates two child parameters which are always reciprocals of each other
class MEAParameters(pTypes.GroupParameter):
    def __init__(self, mea: MEA, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.mea = mea
        self.airfoil_headers = []
        self.custom_header = self.addChild(HeaderParameter(name='Custom', type='bool', value=True))
        for idx, a in enumerate(self.mea.airfoils.values()):
            self.add_airfoil(a, idx)

    def add_airfoil(self, airfoil: Airfoil, idx: int):
        self.airfoil_headers.append(self.addChild(HeaderParameter(name=airfoil.tag, type='bool', value=True)))
        header_params = ['Base', 'AnchorPoints', 'FreePoints', 'Custom']
        for hp in header_params:
            self.airfoil_headers[idx].addChild(HeaderParameter(name=hp, type='bool', value=True))
        for p_key, p_val in self.mea.param_dict[airfoil.tag]['Base'].items():
            self.airfoil_headers[idx].children()[0].addChild(AirfoilParameter(self.mea.param_dict[airfoil.tag]['Base'][p_key],
                                                                              name=f"{airfoil.tag}.Base.{p_key}",
                                                                              type='float',
                                                                              value=self.mea.param_dict[airfoil.tag]['Base'][
                                                                                  p_key].value,
                                                                              context={'add_eq': 'Define by equation'}))
        for fp_key, fp_val in self.mea.param_dict[airfoil.tag]['FreePoints'].items():
            for p_key, p_val in fp_val.items():
                self.airfoil_headers[idx].children()[2].addChild(
                    AirfoilParameter(self.mea.param_dict[airfoil.tag]['FreePoints'][fp_key][p_key],
                                     name=f"{airfoil.tag}.FreePoints.{fp_key}.{p_key}", type='float',
                                     value=self.mea.param_dict[airfoil.tag]['FreePoints'][fp_key][p_key].value,
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
    def __init__(self, mea: MEA):
        self.params = [
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action', 'children': [
                    {'name': 'Add missing items', 'type': 'bool', 'value': True},
                    {'name': 'Remove extra items', 'type': 'bool', 'value': True},
                ]},
            ]},
            MEAParameters(mea, name='Airfoil Parameters'),
            # ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
            #     {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
            #     {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            # ]),
        ]

        # Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=self.params)

        self.mea = mea

        self.line_edit_refs = {}

        registerParameterItemType('auto_str', AutoStrParameterItem)

        for a_name, a in mea.airfoils.items():
            # print(f"airfoil_name = {a_name}")
            a.airfoil_graph.scatter.sigPlotChanged.connect(partial(self.plot_changed, a_name))  # Needs to change with airfoil added or removed

        ## If anything changes in the tree, print a message
        def change(param, changes):
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
                            # print("Equation definition value changed")
                            # print(f"asdf vars={param.makeTreeItem(0)}")

                            param.parent().setReadonly()
                            param.parent().airfoil_param.param_dict = mea.param_dict
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
                                if param.parent().parent().name() == 'FreePoints':
                                    # print(f'Modifying freepoint!: Value of airfoil_param is {param.parent().airfoil_param.value}')
                                    # fp = mea.airfoils['A0'].free_points['te_1']['FP0']
                                    # c = mea.airfoils['A0'].c.value
                                    # alf = mea.airfoils['A0'].alf.value
                                    # dx = mea.airfoils['A0'].dx.value
                                    # dy = mea.airfoils['A0'].dy.value
                                    # fp_x, fp_y = scale(fp.x.value, fp.y.value, c)
                                    # fp_x, fp_y = rotate(fp_x, fp_y, -alf)
                                    # fp_x, fp_y = translate(fp_x, fp_y, dx, dy)
                                    #
                                    # mea.airfoils['A0'].free_points['te_1']['FP0'].ctrlpt.xp = fp_x
                                    # mea.airfoils['A0'].free_points['te_1']['FP0'].ctrlpt.yp = fp_y
                                    # mea.airfoils['A0'].control_point_array = np.array([[cp.xp, cp.yp] for cp in mea.airfoils['A0'].control_points])
                                    # for a in
                                    # self.params[-1].airfoil_graph.data['pos'] = mea.airfoils['A0'].control_point_array
                                    # self.params[-1].airfoil_graph.updateGraph()
                                    pass
                            else:
                                # print(f"parent name is {param.parent().name()}")
                                param.parent().setReadonly(False)
                                self.line_edit_refs[param.parent().name()].widget.setStyleSheet('border: 0px; color: red;')

                if hasattr(param, 'airfoil_param') and change == 'value':
                    # print(f"airfoil_param is in here? {hasattr(param, 'airfoil_param')}")
                    if not param.airfoil_param.linked:
                        param.airfoil_param.value = data
                    param.airfoil_param.update()
                    if param.airfoil_param.linked:
                        param.setValue(param.airfoil_param.value)

                    list_of_vals = []

                    def get_list_of_vals_from_dict(d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                get_list_of_vals_from_dict(v)
                            else:
                                # print("{0} : {1}".format(k, v))
                                list_of_vals.append(v)
                                # return list_of_vals

                    if mea.param_dict is not None:
                        # print(f"param_dict = {mea.param_dict}")
                        get_list_of_vals_from_dict(mea.param_dict)
                        # print(f"list_of_vals = {list_of_vals}")
                        for val in list_of_vals:
                            # print(f"repr = {repr(param.airfoil_param)}")
                            # print(f"val.depends_on = {val.depends_on}")
                            for v in val.depends_on.values():
                                repr_val = repr(v)
                                # print(f"repr1 = {repr(param.airfoil_param)}")
                                # print(f"repr2 = {repr_val}")
                                if repr(param.airfoil_param) == repr_val:
                                    # print(f"Made it here")
                                    val.update()
                                    # print(f"new val for {repr(val)} = {val.value}")
                                    # print(f"param children = {self.params[-1].children()}")
                                    # for c in self.params[-1].children():
                                    #     if 'airfoil_param' in vars(c):
                                    #         print(f"Setting value for {c}")
                                    #         c.setValue(c.airfoil_param.value)
                                    # plot_changed(None)
                    # print(path)
                    # print(vars(param))
                    # print(
                    #     f"new_airfoil_param_value = {param.airfoil_param.value}") #!!!!!!
                    for a in mea.airfoils.values():
                        # print(a)
                        # a.free_points['te_1']['FP0'].x = a.param_dicts['FreePoints']['FP0']['x']
                        # a.free_points['te_1']['FP0'].y = a.param_dicts['FreePoints']['FP0']['y']
                        # a.free_points['te_1']['FP0'].ctrlpt.xp = a.param_dicts['FreePoints']['FP0']['x'].value
                        # a.free_points['te_1']['FP0'].ctrlpt.yp = a.param_dicts['FreePoints']['FP0']['y'].value
                        # a.free_points['te_1']['FP0'].ctrlpt.yp = a.free_points['te_1']['FP0'].y.value
                        # a.control_points[2].xp = a.free_points['te_1']['FP0'].x.value
                        # a.control_points[2].yp = a.free_points['te_1']['FP0'].y.value
                        # print(f"Before update, {a.free_points['te_1']['FP0'].x.value}")
                        # print(f"Before cp_array_2, fpx = {a.control_point_array[2, 0]}")
                        # plot_changed(None)
                        a.update()
                        # print(f"After cp_array_2, fpx = {a.control_point_array[2, 0]}")
                        # print(f"After update, {a.free_points['te_1']['FP0'].x.value}")
                        # a.control_points[2].xp = a.free_points['te_1']['FP0'].x.value
                        # a.control_points[2].yp = a.free_points['te_1']['FP0'].y.value
                        # print(f"a freepoint x,y = {hex(id(a.param_dicts['FreePoints']['FP0']['x'].value))}, "
                        #       f"{hex(id(a.param_dicts['FreePoints']['FP0']['y'].value))}")
                        # print(f"param airfoil_param = {param.airfoil_param.value}")

                        # print(f"new control point array = {a.control_point_array}")
                        a.airfoil_graph.data['pos'] = a.control_point_array
                        # self.params[-1].airfoil_graph.data['pos'] = a.control_point_array
                        # print(f"control_point_array  = {a.control_point_array[2, :]}")
                        # print(f"free points = {hex(id(a.free_points['te_1']['FP0'].x))}, {hex(id(a.free_points['te_1']['FP0'].y))}")
                        a.airfoil_graph.updateGraph()
                        # self.params[-1].airfoil_graph.updateGraph()

                        # fp = mea.airfoils['A0'].free_points['te_1']['FP0']
                        # c = mea.airfoils['A0'].c.value
                        # alf = mea.airfoils['A0'].alf.value
                        # dx = mea.airfoils['A0'].dx.value
                        # dy = mea.airfoils['A0'].dy.value
                        # fp_x, fp_y = scale(fp.x.value, fp.y.value, c)
                        # fp_x, fp_y = rotate(fp_x, fp_y, -alf)
                        # fp_x, fp_y = translate(fp_x, fp_y, dx, dy)
                        #
                        # mea.airfoils['A0'].free_points['te_1']['FP0'].ctrlpt.xp = fp_x
                        # mea.airfoils['A0'].free_points['te_1']['FP0'].ctrlpt.yp = fp_y
                        # mea.airfoils['A0'].control_point_array = np.array(
                        #     [[cp.xp, cp.yp] for cp in mea.airfoils['A0'].control_points])

                        # self.params[-1].airfoil_graph.data['pos'] = mea.airfoils['A0'].control_point_array
                        # self.params[-1].airfoil_graph.updateGraph()

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

        self.p.sigTreeStateChanged.connect(change)

        # def valueChanging(param, value):
        #     print("Value changing (not finalized): %s %s" % (param, value))

        # Only listen for changes of the 'widget' child:
        # for child in p.child('Example Parameters'):
        #     if 'widget' in child.names:
        #         child.child('widget').sigValueChanging.connect(valueChanging)

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
        self.t.setParameters(self.p, showTop=False)
        self.t.setWindowTitle('pyqtgraph example: Parameter Tree')

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
