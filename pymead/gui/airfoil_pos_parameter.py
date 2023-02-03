from pyqtgraph.parametertree.parameterTypes import WidgetParameterItem
from pyqtgraph import SpinBox
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QDoubleSpinBox
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject
import numpy as np


class SignalContainer(QObject):
    """Need to create a separate class to contain the composite signal because the WidgetParameterItem cannot be cast
    as a QObject"""
    signal = pyqtSignal()


class AirfoilPositionParameterItem(WidgetParameterItem):
    """Sub-classing from WidgetParameterItem to create a ParameterTree item with two spin boxes"""
    left_spin: QDoubleSpinBox
    right_spin: QDoubleSpinBox

    def __init__(self, param, depth):
        super().__init__(param, depth)

    def makeWidget(self):
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        SC = SignalContainer()
        self.left_spin = QDoubleSpinBox()
        self.left_spin.setMinimum(-np.inf)
        self.left_spin.setMaximum(np.inf)
        self.left_spin.setDecimals(12)
        self.left_spin.setSingleStep(0.001)
        self.left_spin.setKeyboardTracking(False)
        self.right_spin = QDoubleSpinBox()
        self.right_spin.setMinimum(-np.inf)
        self.right_spin.setMaximum(np.inf)
        self.right_spin.setDecimals(12)
        self.right_spin.setSingleStep(0.001)
        self.right_spin.setKeyboardTracking(False)
        layout.addWidget(self.left_spin)
        layout.addWidget(self.right_spin)

        def setValue(v):
            self.left_spin.setValue(v[0])
            self.right_spin.setValue(v[1])

        def value():
            return [self.left_spin.value(), self.right_spin.value()]

        @pyqtSlot()
        def either_changed():
            SC.signal.emit()

        self.left_spin.valueChanged.connect(either_changed)
        self.right_spin.valueChanged.connect(either_changed)

        widget.sigChanged = SC.signal
        widget.value = value
        widget.setValue = setValue
        return widget
