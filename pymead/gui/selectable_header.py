from pyqtgraph.parametertree.parameterTypes import GroupParameterItem
from pyqtgraph import mkPen
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMenu

translate = QtCore.QCoreApplication.translate

from pymead.gui.custom_context_menu_event import custom_context_menu_event


class SelectableHeaderParameterItem(GroupParameterItem):

    def __init__(self, param, depth):
        super().__init__(param, depth)

    def selected(self, sel):
        if self.param.parent() is None:
            return
        if sel:
            self.param.parent().status_bar.showMessage(f"Airfoil {self.param.name()} selected", 3000)
            for curve in self.param.parent().mea.airfoils[self.param.name()].curve_list:
                curve.pg_curve_handle.setPen(mkPen(color='fuchsia', width=2))
        else:
            for curve in self.param.parent().mea.airfoils[self.param.name()].curve_list:
                curve.pg_curve_handle.setPen(mkPen(color='cornflowerblue', width=2))

    def contextMenuEvent(self, ev):
        custom_context_menu_event(ev, self)
