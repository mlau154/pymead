from PyQt5.QtWidgets import QLineEdit, QCompleter
from pyqtgraph.parametertree.parameterTypes.basetypes import WidgetParameterItem, Parameter


class AutoStrParameterItem(WidgetParameterItem):
    """Registered parameter type which displays a QLineEdit"""

    def __init__(self, param, depth):
        self.widget = None
        super().__init__(param, depth)

    def makeWidget(self):
        w = QLineEdit()
        completer = QCompleter(['Apple', 'Banana'])
        w.setCompleter(completer)
        w.setStyleSheet('border: 0px')
        w.sigChanged = w.editingFinished
        w.value = w.text
        w.setValue = w.setText
        w.sigChanging = w.textChanged
        self.widget = w
        # print(f"widget = {self.widget}")
        return w


# class AutoStrParameter(Parameter):
#     def __init__(self, **opts):
#         self.q_line_edit = None
#         Parameter.__init__(self, **opts)


