from PyQt5.QtWidgets import QLineEdit, QCompleter
from PyQt5.QtCore import Qt
from pyqtgraph.parametertree.parameterTypes.basetypes import WidgetParameterItem

from pymead.gui.custom_context_menu_event import custom_context_menu_event


class AutoStrParameterItem(WidgetParameterItem):
    """Parameter type which displays a QLineEdit with an auto-completion mechanism built in"""

    def __init__(self, param, depth):
        self.widget = None
        super().__init__(param, depth)

    def makeWidget(self):
        w = QLineEdit()
        completer = Completer()
        w.setCompleter(completer)
        w.setStyleSheet('border: 0px')
        w.sigChanged = w.editingFinished
        w.value = w.text
        w.setValue = w.setText
        w.sigChanging = w.textChanged
        w.setPlaceholderText("$")
        self.widget = w
        return w

    def contextMenuEvent(self, ev):
        custom_context_menu_event(ev, self)


class Completer(QCompleter):
    """
    From https://gitter.im/baudren/NoteOrganiser?at=55afbefdcce129d570a3c188
    """

    def __init__(self, parent=None):
        super(Completer, self).__init__(parent)

        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompletionMode(QCompleter.PopupCompletion)
        self.setWrapAround(False)

    # Add texts instead of replace
    def pathFromIndex(self, index):
        path = QCompleter.pathFromIndex(self, index)

        lst = str(self.widget().text()).split('$')

        if len(lst) > 1:
            path = '%s%s' % ('$'.join(lst[:-1]), path)

        return path

    # Add operator to separate between texts
    def splitPath(self, path):
        for ch in [' ', '+', '-', '*', '/', '(']:
            path = str(path.split(ch)[-1]).lstrip(' ')
        return [path]
