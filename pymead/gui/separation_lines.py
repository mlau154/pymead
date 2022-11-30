from PyQt5 import QtWidgets


class QHSeperationLine(QtWidgets.QFrame):
    """From LegendaryCodingNoob @ https://stackoverflow.com/a/61389578"""
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)


class QVSeperationLine(QtWidgets.QFrame):
    """From LegendaryCodingNoob @ https://stackoverflow.com/a/61389578"""
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setFixedWidth(20)
        self.setMinimumHeight(1)
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
