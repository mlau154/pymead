"""
This example demonstrates the use of pyqtgraph's parametertree system. This provides
a simple way to generate user interfaces that control sets of parameters. The example
demonstrates a variety of different parameter types (int, float, list, etc.)
as well as some customized parameter types
"""

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = pg.mkQApp("Parameter Tree Example")
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree
from pymead.gui.airfoil_graph import AirfoilGraph


## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class MultelemAirfoilParameter(pTypes.GroupParameter):
    def __init__(self, airfoil_graph: AirfoilGraph, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.airfoil_graph = airfoil_graph
        self.addChild({'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 'siPrefix': True})
        self.addChild({'name': 'B = 1/A', 'type': 'float', 'value': 1 / 7., 'suffix': 's', 'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)

    def aChanged(self):
        self.b.setValue(1.0 / self.a.value(), blockSignal=self.bChanged)

    def bChanged(self):
        self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)
        self.airfoil_graph.airfoil.alf.value = self.b.value()
        self.airfoil_graph.airfoil.update()
        print(self.airfoil_graph.airfoil.control_point_array[0, :])
        self.airfoil_graph.data['pos'] = self.airfoil_graph.airfoil.control_point_array

        self.airfoil_graph.updateGraph()


    ## test add/remove
## this group includes a menu allowing the user to add new parameters into its child list
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

## Create two ParameterTree widgets, both accessing the same data
class CustomParamTree:
    def __init__(self, airfoil_graph: AirfoilGraph):
        self.params = [
            {'name': 'Save/Restore functionality', 'type': 'group', 'children': [
                {'name': 'Save State', 'type': 'action'},
                {'name': 'Restore State', 'type': 'action', 'children': [
                    {'name': 'Add missing items', 'type': 'bool', 'value': True},
                    {'name': 'Remove extra items', 'type': 'bool', 'value': True},
                ]},
            ]},
            {'name': 'Custom context menu', 'type': 'group', 'children': [
                {'name': 'List contextMenu', 'type': 'float', 'value': 0, 'context': [
                    'menu1',
                    'menu2'
                ]},
                {'name': 'Dict contextMenu', 'type': 'float', 'value': 0, 'context': {
                    'changeName': 'Title',
                    'internal': 'What the user sees',
                }},
            ]},
            MultelemAirfoilParameter(airfoil_graph=airfoil_graph, name='Custom parameter group (reciprocal values)'),
            ScalableGroup(name="Expandable Parameter Group", tip='Click to add children', children=[
                {'name': 'ScalableParam 1', 'type': 'str', 'value': "default param 1"},
                {'name': 'ScalableParam 2', 'type': 'str', 'value': "default param 2"},
            ]),
        ]

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=self.params)

        ## If anything changes in the tree, print a message
        def change(param, changes):
            print("tree changes:")
            for param, change, data in changes:
                path = self.p.childPath(param)
                if path is not None:
                    childName = '.'.join(path)
                else:
                    childName = param.name()
                print('  parameter: %s' % childName)
                print('  change:    %s' % change)
                print('  data:      %s' % str(data))
                print('  ----------')

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

        self.win = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.win.setLayout(self.layout)
        self.layout.addWidget(QtWidgets.QLabel("These are two views of the same data. They should always display the same values."),
                         0, 0, 1, 2)
        self.layout.addWidget(self.t, 1, 0, 1, 1)
# win.show()

# ## test save/restore
# state = p.saveState()
# p.restoreState(state)
# compareState = p.saveState()
# assert pg.eq(compareState, state)

if __name__ == '__main__':
    pg.exec()
