from PyQt5.QtWidgets import QPushButton

from pymead.core.param2 import Param


class ParamButton(QPushButton):
    def __init__(self, text: str, parent):
        super().__init__(text, parent)


class ParamGUI(Param):
    def __init__(self, value: float, qt_parent):
        super().__init__(value=value)
        self.widget = ParamButton()
