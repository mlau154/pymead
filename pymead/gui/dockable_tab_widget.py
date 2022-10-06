import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QDockWidget, QGridLayout, QApplication


class DockableTabWidget(QMainWindow):
    def __init__(self, parent=None):
        super(DockableTabWidget, self).__init__(parent)
        self.setDockNestingEnabled(True)
        layout = QGridLayout()
        self.dock_widgets = []
        self.setLayout(layout)
        self.setWindowTitle('Dock')

    def add_new_tab_widget(self, widget, name):
        dw = QDockWidget(name, self)
        dw.setWidget(widget)
        dw.setFloating(False)
        self.dock_widgets.append(dw)
        self.addDockWidget(Qt.LeftDockWidgetArea, dw)
        if len(self.dock_widgets) > 1:
            self.tabifyDockWidget(self.dock_widgets[-2], self.dock_widgets[-1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = DockableTabWidget()
    demo.show()
    sys.exit(app.exec_())
