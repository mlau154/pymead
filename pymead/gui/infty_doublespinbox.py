from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import Qt
import numpy as np


class InftyDoubleSpinBox(QDoubleSpinBox):
    """
    From StackOverflow answer https://stackoverflow.com/a/57411410
    """
    def __init__(self, lower: bool):
        super(QDoubleSpinBox, self).__init__()

        self.setMinimum(-np.inf)
        self.setMaximum(np.inf)

        self.lower = lower

    def keyPressEvent(self, e: QKeyEvent):

        if e.key() == Qt.Key.Key_Home:
            if self.lower:
                self.setValue(self.minimum())
            else:
                self.setValue(self.maximum())
        else:
            super(QDoubleSpinBox, self).keyPressEvent(e)
