from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QCheckBox, QLabel, QWidget

from pymead.gui.input_dialog import PymeadDialog


class ShowHideDialog(PymeadDialog):
    def __init__(self, parent, state: dict, theme: dict):
        widget = QWidget()
        super().__init__(parent=parent, window_title="Show/Hide", widget=widget, theme=theme)
        self.lay = QGridLayout()
        widget.setLayout(self.lay)
        self.state = state
        self.addRows()
        self.lay.setColumnStretch(0, 0)
        self.lay.setColumnStretch(1, 1)
        self.setMinimumWidth(250)

    def addRows(self):

        items = {
            "points": "Points",
            "lines": "Lines",
            "bezier": "Bezier",
            "airfoils": "Airfoils",
            "geocon": "Constraints"
        }

        for row_idx, (sub_container, title) in enumerate(items.items()):
            checkbox = QCheckBox(self)
            checkbox.setChecked(self.state[sub_container])
            checkbox.clicked.connect(partial(self.parent().showHidePymeadObjs, sub_container))
            checkbox.clicked.connect(partial(self.onCheckBoxStateChanged, sub_container))

            label = QLabel(title, self)
            self.lay.addWidget(checkbox, row_idx, 0)
            self.lay.addWidget(label, row_idx, 1)

    def onCheckBoxStateChanged(self, sub_container: str, shown: bool):
        self.state[sub_container] = shown
