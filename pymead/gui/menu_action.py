from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction


class MenuAction(QAction):

    menuActionClicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triggered.connect(self.onTriggered)

    def onTriggered(self):
        self.menuActionClicked.emit()
