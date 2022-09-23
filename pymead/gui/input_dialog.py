import numpy as np
from typing import List
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QComboBox, QLineEdit


class FreePointInputDialog(QDialog):
    def __init__(self, items: List[tuple], fp: dict, parent=None):
        super().__init__(parent)
        self.fp = fp
        self.ap_list = [k for k in self.fp.keys()]
        self.fp_list = ['None']
        self.fp_list.extend([k for k in self.fp['te_1'].keys()])

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.inputs = []
        for item in items:
            if item[1] == 'double':
                self.inputs.append(QDoubleSpinBox(self))
                self.inputs[-1].setMinimum(-np.inf)
                self.inputs[-1].setMaximum(np.inf)
                self.inputs[-1].setSingleStep(0.01)
                self.inputs[-1].setDecimals(16)
            elif item[1] == 'combo':
                self.inputs.append(QComboBox(self))
                if item[0] == "Previous Anchor Point":
                    self.inputs[-1].addItems(self.ap_list)
                if item[0] == "Previous Free Point":
                    self.inputs[-1].addItems(self.fp_list)
            layout.addRow(item[0], self.inputs[-1])

        self.inputs[-2].currentTextChanged.connect(self.update_fp_items)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return tuple(val.value() if isinstance(val, QDoubleSpinBox) else val.currentText() for val in self.inputs)

    def update_fp_items(self, text):
        self.fp_list = ['None']
        self.fp_list.extend([k for k in self.fp[text].keys()])
        self.inputs[-1].clear()
        self.inputs[-1].addItems(self.fp_list)

    def update_fp_ap_tags(self):
        self.ap_list = [k for k in self.fp.keys()]


class AnchorPointInputDialog(QDialog):
    def __init__(self, items: List[tuple], ap: dict, parent=None):
        super().__init__(parent)
        self.ap = ap
        self.ap_list = [anchor_point.tag for anchor_point in self.ap]

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.inputs = []
        for item in items:
            if item[1] == 'double':
                self.inputs.append(QDoubleSpinBox(self))
                self.inputs[-1].setMinimum(-np.inf)
                self.inputs[-1].setMaximum(np.inf)
                self.inputs[-1].setSingleStep(0.01)
                self.inputs[-1].setDecimals(16)
            elif item[1] == 'combo':
                self.inputs.append(QComboBox(self))
                if item[0] == "Previous Anchor Point":
                    self.inputs[-1].addItems(self.ap_list)
            elif item[1] == 'string':
                self.inputs.append(QLineEdit(self))
            else:
                raise ValueError(f"AnchorPointInputDialog item types must be \'double\', \'combo\', or \'string\'")
            layout.addRow(item[0], self.inputs[-1])

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return_vals = []
        for val in self.inputs:
            if isinstance(val, QDoubleSpinBox):
                return_vals.append(val.value())
            elif isinstance(val, QLineEdit):
                return_vals.append(val.text())
            elif isinstance(val, QComboBox):
                return_vals.append(val.currentText())
            else:
                raise TypeError(f'QFormLayout widget must be of type {type(QComboBox)}, {type(QDoubleSpinBox)}, '
                                f'or {type(QLineEdit)}')
        return tuple(return_vals)

    def update_ap_tags(self):
        self.ap_list = [anchor_point.tag for anchor_point in self.ap]
