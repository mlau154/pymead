import numpy as np
from typing import List
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QComboBox, QLineEdit, QSpinBox, \
    QTabWidget, QLabel, QMessageBox, QCheckBox, QFileDialog
from PyQt5.QtCore import QEvent
from pymead.gui.infty_doublespinbox import InftyDoubleSpinBox


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


class SingleAirfoilInviscidDialog(QDialog):
    def __init__(self, items: List[tuple], a_list: list, parent=None):
        super().__init__(parent)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.inputs = []
        for item in items:
            if item[1] == 'double':
                self.inputs.append(QDoubleSpinBox(self))
                self.inputs[-1].setValue(item[2])
                self.inputs[-1].setMinimum(0.0)
                self.inputs[-1].setMaximum(np.inf)
                self.inputs[-1].setSingleStep(1.0)
                self.inputs[-1].setDecimals(5)
            elif item[1] == 'combo':
                self.inputs.append(QComboBox(self))
                if item[0] == "Airfoil":
                    self.inputs[-1].addItems(a_list)
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


class SingleAirfoilViscousDialog(QDialog):
    def __init__(self, items: List[tuple], a_list: list, parent=None):
        super().__init__(parent)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.inputs = []
        for item in items:
            if item[1] == 'double':
                self.inputs.append(QDoubleSpinBox(self))
                if item[0] == "Reynolds Number":
                    self.inputs[-1].setMinimum(0.0)
                else:
                    self.inputs[-1].setMinimum(-np.inf)
                self.inputs[-1].setMaximum(np.inf)
                self.inputs[-1].setValue(item[2])
                self.inputs[-1].setSingleStep(1.0)
                self.inputs[-1].setDecimals(5)
            elif item[1] == 'int':
                self.inputs.append(QSpinBox(self))
                self.inputs[-1].setMaximum(99999)
                self.inputs[-1].setValue(item[2])
            elif item[1] == 'combo':
                self.inputs.append(QComboBox(self))
                if item[0] == "Airfoil":
                    self.inputs[-1].addItems(a_list)
                    # self.inputs[-1].valueChanged.connect(self.set_airfoil_name)
            elif item[1] == 'string':
                self.inputs.append(QLineEdit(self))
                self.inputs[-1].setText(item[2])
            elif item[1] == 'checkbox':
                self.inputs.append(QCheckBox(self))
                self.inputs[-1].setCheckState(item[2])
            else:
                raise ValueError(f"AnchorPointInputDialog item types must be \'double\', \'combo\', or \'string\'")
            layout.addRow(item[0], self.inputs[-1])

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    # def set_airfoil_name(self, value):
    #     for item in self.inputs:
    #         if item.

    def getInputs(self):
        return_vals = []
        for val in self.inputs:
            if isinstance(val, QDoubleSpinBox):
                return_vals.append(val.value())
            elif isinstance(val, QLineEdit):
                return_vals.append(val.text())
            elif isinstance(val, QComboBox):
                return_vals.append(val.currentText())
            elif isinstance(val, QSpinBox):
                return_vals.append(val.value())
            elif isinstance(val, QCheckBox):
                return_vals.append(val.checkState())
            else:
                raise TypeError(f'QFormLayout widget must be of type {type(QComboBox)}, {type(QDoubleSpinBox)}, '
                                f'or {type(QLineEdit)}')
        return tuple(return_vals)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)


class InviscidCpCalcDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)


class BoundsDialog(QDialog):
    def __init__(self, bounds, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter Bounds Modification")
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.lower_bound = InftyDoubleSpinBox(lower=True)
        self.lower_bound.setValue(bounds[0])
        self.lower_bound.setDecimals(16)
        layout.addRow("Lower Bound", self.lower_bound)

        self.upper_bound = InftyDoubleSpinBox(lower=False)
        self.upper_bound.setValue(bounds[1])
        self.upper_bound.setDecimals(16)
        layout.addRow("Upper Bound", self.upper_bound)

        self.hint_label = QLabel("<Home> key: +/-inf", parent=self)
        self.hint_label.setWordWrap(True)

        layout.addWidget(self.hint_label)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return self.lower_bound.value(), self.upper_bound.value()

    def event(self, event):
        if event.type() == QEvent.EnterWhatsThisMode:
            mbox = QMessageBox()
            mbox.setText("Note: When focused on the Lower Bound field, press the <Home> "
                         "key to set the value to negative infinity. When focused on the Upper Bound "
                         "field, press the <Home> key to set the value to positive infinity.")
            mbox.exec()
        elif event.type() == QEvent.LeaveWhatsThisMode:
            pass
        return super().event(event)


class LoadDialog(QFileDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setFileMode(QFileDialog.ExistingFile)
        self.setNameFilter(self.tr("PyMEAD Files (*.mead)"))
        self.setViewMode(QFileDialog.Detail)


class SaveAsDialog(QFileDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setFileMode(QFileDialog.AnyFile)
        self.setNameFilter(self.tr("PyMEAD Files (*.mead)"))
        self.setViewMode(QFileDialog.Detail)
