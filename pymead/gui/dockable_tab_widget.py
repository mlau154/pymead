import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QGridLayout, QApplication, QWidget


class DockableTabWidget(QMainWindow):
    def __init__(self, parent=None, cancel_if_tab_name_exists: bool = False):
        super(DockableTabWidget, self).__init__(parent)
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

    def add_new_tab_widget(self, widget, name):
        if not (self.cancel_if_tab_name_exists and name in self.names):
            dw = QDockWidget(name, self)
            dw.setWidget(widget)
            dw.setFloating(False)
            self.dock_widgets.append(dw)
            self.names.append(name)
            if len(self.dock_widgets) == 2:
                self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widgets[-2])
                self.addDockWidget(Qt.RightDockWidgetArea, dw)
                self.setCentralWidget(QWidget())
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])
                self.splitDockWidget(self.dock_widgets[-2], self.dock_widgets[-1], Qt.Horizontal)
            elif len(self.dock_widgets) > 2:
                self.addDockWidget(Qt.RightDockWidgetArea, dw)
                self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])
            else:
                self.setCentralWidget(dw)

    def activated(self, dw: QDockWidget):
        self.current_dock_widget = dw


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = DockableTabWidget()
    demo.show()
    sys.exit(app.exec_())
