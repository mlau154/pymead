from functools import partial

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QGridLayout, QCheckBox, QLabel


class ShowHideDialog(QDialog):
    def __init__(self, parent, state: dict):
        super().__init__(parent=parent)
        self.setWindowTitle("Show/Hide")
        self.lay = QGridLayout()
        self.setLayout(self.lay)
        self.state = state
        self.addRows()
        self.lay.setColumnMinimumWidth(1, 100)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(150, 120)

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
