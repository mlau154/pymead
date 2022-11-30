from PyQt5.QtWidgets import QWidget, QGridLayout, QDoubleSpinBox, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from pymead.gui.separation_lines import QHSeperationLine
import numpy as np


class GridBounds(QWidget):
    boundsChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent=parent)
        layout = QGridLayout()
        self.setLayout(layout)
        label_names = ['Left', 'Right', 'Top', 'Bottom']
        self.labels = {k: QLabel(k, self) for k in label_names}
        label_positions = {
            'Left': [1, 0],
            'Right': [1, 2],
            'Top': [2, 0],
            'Bottom': [2, 2],
        }
        self.widgets = {k: QDoubleSpinBox() for k in label_positions}
        defaults = {
            'Left': [-5.0, 1, 1],
            'Right': [5.0, 1, 3],
            'Top': [5.0, 2, 1],
            'Bottom': [-5.0, 2, 3],
        }
        for k, v in defaults.items():
            self.widgets[k].setMinimum(-np.inf)
            self.widgets[k].setMaximum(np.inf)
            self.widgets[k].setValue(v[0])
            self.widgets[k].setMinimumWidth(75)
        layout.addWidget(QHSeperationLine(self), 0, 0, 1, 1)
        layout.addWidget(QLabel('Grid Bounds', self), 0, 1, 1, 2, Qt.AlignCenter)
        layout.addWidget(QHSeperationLine(self), 0, 3, 1, 1)
        for label_name in label_names:
            layout.addWidget(self.labels[label_name], label_positions[label_name][0], label_positions[label_name][1],
                             1, 1, Qt.AlignRight)
            layout.addWidget(self.widgets[label_name], defaults[label_name][1], defaults[label_name][2], 1, 1,
                             Qt.AlignLeft)
        layout.addWidget(QHSeperationLine(self), 3, 0, 1, 4)
        for w in self.widgets.values():
            w.valueChanged.connect(self.valueChanged)

    def setValues(self, value_list: list):
        if len(value_list) != 4:
            raise ValueError('Length of input value list must be 4')
        self.widgets['Left'].setValue(value_list[0])
        self.widgets['Right'].setValue(value_list[1])
        self.widgets['Top'].setValue(value_list[2])
        self.widgets['Bottom'].setValue(value_list[3])

    def values(self):
        return [self.widgets['Left'].value(), self.widgets['Right'].value(), self.widgets['Top'].value(),
                self.widgets['Bottom'].value()]

    def valueChanged(self, _):
        self.boundsChanged.emit()
