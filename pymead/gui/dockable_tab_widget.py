import typing

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QDockWidget, QGridLayout, QWidget
from PyQt6.QtCore import pyqtSignal


class PymeadDockWidget(QDockWidget):
    """
    Simple re-implementation of QDockWidget with special handling of tab close with a custom signal.
    """
    tab_closed = pyqtSignal(str, QCloseEvent)

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Sends a signal that the tab has been closed along with the name of the tab and the QCloseEvent object.
        """
        self.tab_closed.emit(f"{self.windowTitle()}", event)


class DockableTabWidget(QMainWindow):
    tab_closed = pyqtSignal(str, QCloseEvent)

    def __init__(self, parent=None, cancel_if_tab_name_exists: bool = False):
        super().__init__(parent=parent)
        self.setDockNestingEnabled(True)
        self.w = QWidget()
        layout = QGridLayout()
        self.dock_widgets = []
        self.names = []
        self.w.setLayout(layout)
        self.setCentralWidget(self.w)
        self.first_dock_widget = None
        self.setWindowTitle('Geometry & Analysis')
        self.cancel_if_tab_name_exists = cancel_if_tab_name_exists
        self.tabifiedDockWidgetActivated.connect(self.activated)

    def add_new_tab_widget(self, widget, name, connect_func: typing.Callable):
        if not (self.cancel_if_tab_name_exists and name in self.names):
            dw = PymeadDockWidget(name, self)
            dw.setWidget(widget)
            dw.setFloating(False)
            dw.tab_closed.connect(self.on_tab_closed)
            self.dock_widgets.append(dw)
            self.names.append(name)
            if len(self.dock_widgets) == 2:
                self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock_widgets[-2])
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)
                self.setCentralWidget(QWidget())
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.AlignmentFlag.Horizontal)
            elif len(self.dock_widgets) == 3:
                self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.AlignmentFlag.Vertical)
            elif len(self.dock_widgets) > 3:
                self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])
            else:
                self.setCentralWidget(dw)
            self.tab_closed.connect(connect_func)

    def on_tab_closed(self, name: str, event: QCloseEvent):
        if name == "Geometry":
            event.ignore()
            return
        idx = self.names.index(name)
        self.names.pop(idx)
        self.dock_widgets.pop(idx)
        self.tab_closed.emit(name, event)

    def activated(self, dw: QDockWidget):
        self.current_dock_widget = dw
