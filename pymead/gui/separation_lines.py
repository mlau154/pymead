from PyQt6 import QtWidgets


class QHSeperationLine(QtWidgets.QFrame):
    """From LegendaryCodingNoob @ https://stackoverflow.com/a/61389578"""
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)


class QVSeperationLine(QtWidgets.QFrame):
    """From LegendaryCodingNoob @ https://stackoverflow.com/a/61389578"""
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setFixedWidth(20)
        self.setMinimumHeight(1)
        self.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
