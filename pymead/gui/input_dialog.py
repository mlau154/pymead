import numpy as np
from typing import List, Any
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QComboBox, QLineEdit, QSpinBox, \
    QTabWidget, QLabel, QMessageBox, QCheckBox, QFileDialog, QVBoxLayout, QWidget, QRadioButton, QHBoxLayout, \
    QButtonGroup, QGridLayout, QPushButton, QPlainTextEdit
from PyQt5.QtCore import QEvent
from pymead.gui.infty_doublespinbox import InftyDoubleSpinBox
from pymead.gui.scientificspinbox_master.ScientificDoubleSpinBox import ScientificDoubleSpinBox
from pymead.gui.pyqt_vertical_tab_widget.pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget
import sys
import os
from functools import partial
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.dict_recursion import recursive_get
from pymead.gui.opt_settings_default import opt_settings_default


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

        self.setFont(self.parent().font())
        self.setWindowTitle("Single Airfoil Viscous Analysis")

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
        self.setFont(self.parent().font())
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


class OptimizationSetupDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.setWindowTitle("Optimization Setup")
        self.setFont(self.parent().font())

        self.grid_widget = None
        self.grid_layout = None

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton("Run", QDialogButtonBox.AcceptRole)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        layout = QVBoxLayout(self)
        self.tab_widget = VerticalTabWidget(self)
        layout.addWidget(self.tab_widget)

        if parent.opt_settings is None:
            self.inputs = opt_settings_default()
        else:
            self.inputs = parent.opt_settings

        self.setInputs()

        parent.opt_settings = self.inputs

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def add_tab(self, name: str):
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self)
        self.grid_widget.setLayout(self.grid_layout)
        self.tab_widget.addTab(self.grid_widget, name)

    def select_directory(self, line_edit: QLineEdit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory", os.path.expanduser("~"),
                                                        QFileDialog.ShowDirsOnly)
        if selected_dir:
            line_edit.setText(selected_dir)

    @staticmethod
    def activate_deactivate_checkbox(widget: QLineEdit or QSpinBox or QDoubleSpinBox, checked: bool):
        if checked:
            widget.setReadOnly(False)
        else:
            widget.setReadOnly(True)

    def select_existing_mead_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(self.tr("MEAD Parametrization (*.mead)"))
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def select_existing_json_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(self.tr("JSON Settings Files (*.json)"))
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def select_any_json_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setNameFilter(self.tr("JSON Settings Files (*.json)"))
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def select_directory_for_json_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def select_multiple_coord_files(self, text_edit: QPlainTextEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(self.tr("Data Files (*.txt *.dat *.csv)"))
        if file_dialog.exec_():
            text_edit.insertPlainText('\n\n'.join(file_dialog.selectedFiles()))

    def select_multiple_json_files(self, text_edit: QPlainTextEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(self.tr("JSON Settings Files (*.json)"))
        if file_dialog.exec_():
            text_edit.insertPlainText('\n\n'.join(file_dialog.selectedFiles()))

    def save_opt_settings(self):
        input_filename = self.inputs['Save/Load Settings']['settings_save_dir']['text']
        save_data(self.inputs, input_filename)
        msg_box = QMessageBox()
        msg_box.setText(f"Settings saved as {input_filename}")
        msg_box.setWindowTitle('Save Notification')
        msg_box.setFont(self.parent().font())
        msg_box.exec()

    def load_opt_settings(self):
        self.inputs = load_data(self.inputs['Save/Load Settings']['settings_load_dir']['text'])
        self.setInputs()

    def saveas_opt_settings(self):
        input_filename = os.path.join(self.inputs['Save/Load Settings']['settings_saveas_dir']['text'],
                                      self.inputs['Save/Load Settings']['settings_saveas_filename']['text'])
        save_data(self.inputs, input_filename)
        msg_box = QMessageBox()
        msg_box.setText(f"Settings saved as {input_filename}")
        msg_box.setWindowTitle('Save Notification')
        msg_box.setFont(self.parent().font())
        msg_box.exec()

    def enable_disable_from_checkbox(self, key1: str, key2: str):
        widget = self.widget_dict[key1][key2]['widget']
        if 'widgets_to_enable' in self.inputs[key1][key2].keys() and widget.checkState() or (
            'widgets_to_disable' in self.inputs[key1][key2].keys() and not widget.checkState()
        ):
            enable_disable_key = 'widgets_to_enable' if 'widgets_to_enable' in self.inputs[
                key1][key2].keys() else 'widgets_to_disable'
            for enable_list in self.inputs[key1][key2][enable_disable_key]:
                dict_to_enable = recursive_get(self.widget_dict, *enable_list)
                if not isinstance(dict_to_enable['widget'], QPushButton):
                    dict_to_enable['widget'].setReadOnly(False)
                    if 'push_button' in dict_to_enable.keys():
                        dict_to_enable['push_button'].setEnabled(True)
                    if 'checkbox' in dict_to_enable.keys():
                        dict_to_enable['checkbox'].setReadOnly(False)
                else:
                    dict_to_enable['widget'].setEnabled(True)
        elif 'widgets_to_enable' in self.inputs[key1][key2].keys() and not widget.checkState() or (
            'widgets_to_disable' in self.inputs[key1][key2].keys() and widget.checkState()
        ):
            enable_disable_key = 'widgets_to_enable' if 'widgets_to_enable' in self.inputs[
                key1][key2].keys() else 'widgets_to_disable'
            for disable_list in self.inputs[key1][key2][enable_disable_key]:
                dict_to_enable = recursive_get(self.widget_dict, *disable_list)
                if not isinstance(dict_to_enable['widget'], QPushButton):
                    dict_to_enable['widget'].setReadOnly(True)
                    if 'push_button' in dict_to_enable.keys():
                        dict_to_enable['push_button'].setEnabled(False)
                    if 'checkbox' in dict_to_enable.keys():
                        dict_to_enable['checkbox'].setReadOnly(True)
                else:
                    dict_to_enable['widget'].setEnabled(False)

    def dict_connection(self, widget: QLineEdit | QSpinBox | QDoubleSpinBox | ScientificDoubleSpinBox | QComboBox | QCheckBox,
                        key1: str, key2: str):
        if isinstance(widget, QLineEdit):
            self.inputs[key1][key2]['text'] = widget.text()
        elif isinstance(widget, QPlainTextEdit):
            self.inputs[key1][key2]['texts'] = widget.toPlainText().split('\n\n')
        elif isinstance(widget, QComboBox):
            self.inputs[key1][key2]['current_text'] = widget.currentText()
        elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox) or isinstance(
                widget, ScientificDoubleSpinBox):
            self.inputs[key1][key2]['value'] = widget.value()
        elif isinstance(widget, QCheckBox):
            if 'active_checkbox' in self.inputs[key1][key2].keys():
                self.inputs[key1][key2]['active_checkbox'] = widget.checkState()
            else:
                self.inputs[key1][key2]['state'] = widget.checkState()

        self.enable_disable_from_checkbox(key1, key2)

        if key2 == 'pop_size':
            self.widget_dict[key1]['n_offspring']['widget'].setMinimum(widget.value())

    def change_prescribed_aero_parameter(self, current_text: str):
        w1 = self.widget_dict['XFOIL']['alfa']['widget']
        w2 = self.widget_dict['XFOIL']['Cl']['widget']
        w3 = self.widget_dict['XFOIL']['CLI']['widget']
        if current_text == 'Angle of Attack (deg)':
            bools = (False, True, True)
        elif current_text == 'Viscous Cl':
            bools = (True, False, True)
        elif current_text == 'Inviscid Cl':
            bools = (True, True, False)
        else:
            raise ValueError('Invalid value of currentText for QComboBox (alfa/Cl/CLI')
        w1.setReadOnly(bools[0])
        w2.setReadOnly(bools[1])
        w3.setReadOnly(bools[2])

    def setInputs(self):
        self.tab_widget.clear()
        self.widget_dict = {}
        for k, v in self.inputs.items():
            self.widget_dict[k] = {}
            grid_counter = 0
            self.add_tab(k)
            for k_, v_ in v.items():
                label = QLabel(v_['label'], self)
                widget = getattr(sys.modules[__name__], v_['widget_type'])(self)
                self.widget_dict[k][k_] = {'widget': widget}
                if 'text' in v_.keys():
                    widget.setText(v_['text'])
                    widget.textChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'texts' in v_.keys():
                    widget.insertPlainText('\n\n'.join(v_['texts']))
                    widget.textChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'items' in v_.keys():
                    widget.addItems(v_['items'])
                if 'current_text' in v_.keys():
                    widget.setCurrentText(v_['current_text'])
                    widget.currentTextChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'lower_bound' in v_.keys():
                    widget.setMinimum(v_['lower_bound'])
                if 'upper_bound' in v_.keys():
                    widget.setMaximum(v_['upper_bound'])
                if 'value' in v_.keys():
                    # print(f"Setting value of {v_['label']} to {v_['value']}")
                    widget.setValue(v_['value'])
                    widget.valueChanged.connect(partial(self.dict_connection, widget, k, k_))
                    # print(f"Actual returned value is {widget.value()}")
                if 'state' in v_.keys():
                    widget.setCheckState(v_['state'])
                    widget.setTristate(False)
                    widget.stateChanged.connect(partial(self.dict_connection, widget, k, k_))
                if isinstance(widget, QPushButton):
                    if 'button_title' in v_.keys():
                        widget.setText(v_['button_title'])
                    if 'click_connect' in v_.keys():
                        push_button_action = getattr(self, v_['click_connect'])
                        widget.clicked.connect(push_button_action)
                if 'decimals' in v_.keys():
                    widget.setDecimals(v_['decimals'])
                if 'tool_tip' in v_.keys():
                    label.setToolTip(v_['tool_tip'])
                    widget.setToolTip(v_['tool_tip'])
                push_button = None
                checkbox = None
                if 'push_button' in v_.keys():
                    push_button = QPushButton(v_['push_button'], self)
                    push_button_action = getattr(self, v_['push_button_action'])
                    push_button.clicked.connect(partial(push_button_action, widget))
                    self.widget_dict[k][k_]['push_button'] = push_button
                if 'active_checkbox' in v_.keys():
                    checkbox = QCheckBox('Active?', self)
                    checkbox.setCheckState(v_['active_checkbox'])
                    checkbox.setTristate(False)
                    checkbox_action = getattr(self, 'activate_deactivate_checkbox')
                    checkbox.stateChanged.connect(partial(checkbox_action, widget))
                    checkbox.stateChanged.connect(partial(self.dict_connection, checkbox, k, k_))
                    self.widget_dict[k][k_]['checkbox'] = checkbox_action
                if 'combo_callback' in v_.keys():
                    combo_callback_action = getattr(self, v_['combo_callback'])
                    widget.currentTextChanged.connect(combo_callback_action)
                if 'editable' in v_.keys():
                    widget.setReadOnly(not v_['editable'])
                if k_ == 'mea_dir' and self.widget_dict[k]['use_current_mea']['widget'].checkState():
                    widget.setReadOnly(True)
                    self.widget_dict[k][k_]['push_button'].setEnabled(False)
                self.grid_layout.addWidget(label, grid_counter, 0)
                if push_button is None:
                    if checkbox is None:
                        self.grid_layout.addWidget(widget, grid_counter, 1, 1, 3)
                    else:
                        self.grid_layout.addWidget(widget, grid_counter, 1, 1, 2)
                        self.grid_layout.addWidget(checkbox, grid_counter, 3)
                else:
                    self.grid_layout.addWidget(widget, grid_counter, 1, 1, 2)
                    self.grid_layout.addWidget(push_button, grid_counter, 3)
                grid_counter += 1

        for k, v in self.inputs.items():
            for k_ in v.keys():
                self.enable_disable_from_checkbox(k, k_)

    def getInputs(self):
        return self.inputs
