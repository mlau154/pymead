from pyqtgraph.parametertree.parameterTypes import GroupParameterItem
from pyqtgraph import mkPen


class SelectableHeaderParameterItem(GroupParameterItem):

    def __init__(self, param, depth):
        super().__init__(param, depth)

    def selected(self, sel):
        if sel:
            self.param.parent().status_bar.showMessage(f"Airfoil {self.param.name()} selected", 3000)
            for curve in self.param.parent().mea.airfoils[self.param.name()].curve_list:
                curve.pg_curve_handle.setPen(mkPen(color='fuchsia', width=2))
        else:
            for curve in self.param.parent().mea.airfoils[self.param.name()].curve_list:
                curve.pg_curve_handle.setPen(mkPen(color='cornflowerblue', width=2))
