import numpy as np
from typing import List, Any
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QComboBox, QLineEdit, QSpinBox, \
    QTabWidget, QLabel, QMessageBox, QCheckBox, QFileDialog, QVBoxLayout, QWidget, QRadioButton, QHBoxLayout, \
    QButtonGroup, QGridLayout, QPushButton, QPlainTextEdit, QGraphicsScene, QGraphicsView, QListView
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QPixmap, QPainter, QStandardItem, QStandardItemModel
from PyQt5.QtSvg import QSvgWidget
from functools import partial
from PyQt5.QtCore import pyqtSlot
from pymead.gui.infty_doublespinbox import InftyDoubleSpinBox
from pymead.gui.scientificspinbox_master.ScientificDoubleSpinBox import ScientificDoubleSpinBox
from pymead.gui.pyqt_vertical_tab_widget.pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget
import sys
import os
from functools import partial
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.dict_recursion import recursive_get
from pymead.gui.default_settings import opt_settings_default, xfoil_settings_default, mset_settings_default, \
    mses_settings_default, mplot_settings_default
from pymead.optimization.objectives_and_constraints import Objective, Constraint, FunctionCompileError
from pymead.analysis import cfd_output_templates
from pymead.gui.grid_bounds_widget import GridBounds
from pymead.gui.mset_multigrid_widget import MSETMultiGridWidget, XTRSWidget, ADWidget
from pymead.analysis.utils import viscosity_calculator
from pymead.gui.custom_graphics_view import CustomGraphicsView
from pymead import RESOURCE_DIR, DATA_DIR
import pyqtgraph as pg


def convert_dialog_to_mset_settings(dialog_input: dict):
    mset_settings = {
        'airfoil_order': dialog_input['airfoil_order']['text'].split(','),
        'grid_bounds': dialog_input['grid_bounds']['values'],
        'verbose': dialog_input['verbose']['state'],
        'airfoil_analysis_dir': dialog_input['airfoil_analysis_dir']['text'],
        'airfoil_coord_file_name': dialog_input['airfoil_coord_file_name']['text'],
    }
    values_list = ['airfoil_side_points', 'exp_side_points', 'inlet_pts_left_stream', 'outlet_pts_right_stream',
                   'num_streams_top', 'num_streams_bot', 'max_streams_between', 'elliptic_param',
                   'stag_pt_aspect_ratio', 'x_spacing_param', 'alf0_stream_gen', 'timeout']
    for value in values_list:
        mset_settings[value] = dialog_input[value]['value']
    for idx, airfoil in enumerate(dialog_input['multi_airfoil_grid']['values'].values()):
        for k, v in airfoil.items():
            if idx == 0:
                mset_settings[k] = [v]
            else:
                mset_settings[k].append(v)
    mset_settings['n_airfoils'] = len(mset_settings['airfoil_order'])
    # for k, v in mset_settings.items():
    #     print(f"{k}: {v}")
    return mset_settings


def convert_dialog_to_mses_settings(dialog_input: dict):
    mses_settings = {
        'ISMOVE': 0,
        'ISPRES': 0,
        'NMODN': 0,
        'NPOSN': 0,
        'viscous_flag': dialog_input['viscous_flag']['state'],
        'inverse_flag': 0,
        'inverse_side': 1,
        'verbose': dialog_input['verbose']['state'],
    }

    for idx, item in enumerate(dialog_input['ISMOM']['items']):
        if dialog_input['ISMOM']['current_text'] == item:
            mses_settings['ISMOM'] = idx + 1
            break

    for idx, item in enumerate(dialog_input['IFFBC']['items']):
        if dialog_input['IFFBC']['current_text'] == item:
            mses_settings['IFFBC'] = idx + 1
            break

    if dialog_input['AD_active']['state']:
        mses_settings['AD_flags'] = [1 for _ in range(dialog_input['AD_number']['value'])]
    else:
        mses_settings['AD_flags'] = [0 for _ in range(dialog_input['AD_number']['value'])]

    values_list = ['REYNIN', 'MACHIN', 'ALFAIN', 'CLIFIN', 'ACRIT', 'MCRIT', 'MUCON',
                   'timeout', 'iter']
    for value in values_list:
        mses_settings[value] = dialog_input[value]['value']

    if dialog_input['spec_alfa_Cl']['current_text'] == 'Specify Angle of Attack':
        mses_settings['target'] = 'alfa'
    elif dialog_input['spec_alfa_Cl']['current_text'] == 'Specify Lift Coefficient':
        mses_settings['target'] = 'Cl'

    for idx, airfoil in enumerate(dialog_input['xtrs']['values'].values()):
        for k, v in airfoil.items():
            if idx == 0:
                mses_settings[k] = [v]
            else:
                mses_settings[k].append(v)

    # for k, v in mses_settings.items():
    #     print(f"{k}: {v}")
    return mses_settings


def convert_dialog_to_mplot_settings(dialog_input: dict):
    mplot_settings = {
        'timeout': dialog_input['timeout']['value'],
        'Mach': dialog_input['Mach']['state'],
        'Grid': dialog_input['Grid']['state'],
        'Grid_Zoom': dialog_input['Grid_Zoom']['state'],
    }
    # for k, v in mplot_settings.items():
    #     print(f"{k}: {v}")
    return mplot_settings


class FreePointInputDialog(QDialog):
    def __init__(self, items: List[tuple], fp: dict, parent):
        super().__init__(parent)
        self.setWindowTitle('Insert FreePoint')
        self.setFont(self.parent().font())
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
                self.inputs[-1].setValue(item[2])
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
    def __init__(self, items: List[tuple], ap: dict, parent):
        super().__init__(parent)
        self.setWindowTitle('Insert AnchorPoint')
        self.setFont(self.parent().font())
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
                self.inputs[-1].setValue(item[2])
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
                self.inputs[-1].setDecimals(5)
                self.inputs[-1].setValue(item[2])
                self.inputs[-1].setMinimum(0.0)
                self.inputs[-1].setMaximum(np.inf)
                self.inputs[-1].setSingleStep(1.0)
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


class ColorInputDialog(QDialog):
    def __init__(self, parent, default_color: tuple):
        super().__init__(parent=parent)

        self.setFont(self.parent().font())
        self.setWindowTitle("Color Selector")

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton("Apply", QDialogButtonBox.AcceptRole)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.layout = QGridLayout(self)
        self.widget = QWidget(self)
        self.layout.addWidget(self.widget)

        self.color_button_widget = pg.ColorButton(parent=self, color=default_color)
        self.layout.addWidget(self.color_button_widget)

        self.layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)


class SingleAirfoilViscousDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFont(self.parent().font())
        self.setWindowTitle("Single Airfoil Analysis")
        self.widget_dict = None

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton("Run", QDialogButtonBox.AcceptRole)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.layout = QGridLayout(self)
        self.widget = QWidget(self)
        self.layout.addWidget(self.widget)

        if parent.xfoil_settings is None:
            self.inputs = xfoil_settings_default(self.parent().airfoil_name_list)
        else:
            self.inputs = parent.xfoil_settings
            self.inputs['airfoil']['items'] = self.parent().airfoil_name_list

        self.setInputs()

        parent.xfoil_settings = self.inputs

        self.layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def select_directory_for_airfoil_analysis(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def enable_disable_from_checkbox(self, key: str):
        widget = self.widget_dict[key]['widget']
        if 'widgets_to_enable' in self.inputs[key].keys() and widget.checkState() or (
                'widgets_to_disable' in self.inputs[key].keys() and not widget.checkState()
        ):
            enable_disable_key = 'widgets_to_enable' if 'widgets_to_enable' in self.inputs[
                key].keys() else 'widgets_to_disable'
            for enable_list in self.inputs[key][enable_disable_key]:
                dict_to_enable = recursive_get(self.widget_dict, *enable_list)
                if not isinstance(dict_to_enable['widget'], QPushButton):
                    dict_to_enable['widget'].setReadOnly(False)
                    if 'push_button' in dict_to_enable.keys():
                        dict_to_enable['push_button'].setEnabled(True)
                    if 'checkbox' in dict_to_enable.keys():
                        dict_to_enable['checkbox'].setReadOnly(False)
                else:
                    dict_to_enable['widget'].setEnabled(True)
        elif 'widgets_to_enable' in self.inputs[key].keys() and not widget.checkState() or (
                'widgets_to_disable' in self.inputs[key].keys() and widget.checkState()
        ):
            enable_disable_key = 'widgets_to_enable' if 'widgets_to_enable' in self.inputs[
                key].keys() else 'widgets_to_disable'
            for disable_list in self.inputs[key][enable_disable_key]:
                dict_to_enable = recursive_get(self.widget_dict, *disable_list)
                if not isinstance(dict_to_enable['widget'], QPushButton):
                    dict_to_enable['widget'].setReadOnly(True)
                    if 'push_button' in dict_to_enable.keys():
                        dict_to_enable['push_button'].setEnabled(False)
                    if 'checkbox' in dict_to_enable.keys():
                        dict_to_enable['checkbox'].setReadOnly(True)
                else:
                    dict_to_enable['widget'].setEnabled(False)

    def dict_connection(self,
                        widget: QLineEdit | QSpinBox | QDoubleSpinBox | ScientificDoubleSpinBox | QComboBox | QCheckBox,
                        key: str):
        if isinstance(widget, QLineEdit):
            self.inputs[key]['text'] = widget.text()
        elif isinstance(widget, QPlainTextEdit):
            if key == 'additional_data':
                self.inputs[key]['texts'] = widget.toPlainText().split('\n')
            else:
                self.inputs[key]['texts'] = widget.toPlainText().split('\n\n')
        elif isinstance(widget, QComboBox):
            self.inputs[key]['current_text'] = widget.currentText()
        elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox) or isinstance(
                widget, ScientificDoubleSpinBox):
            self.inputs[key]['value'] = widget.value()
        elif isinstance(widget, QCheckBox):
            if 'active_checkbox' in self.inputs[key].keys():
                self.inputs[key]['active_checkbox'] = widget.checkState()
            else:
                self.inputs[key]['state'] = widget.checkState()

        self.enable_disable_from_checkbox(key)

    def change_prescribed_aero_parameter(self, current_text: str):
        w1 = self.widget_dict['alfa']['widget']
        w2 = self.widget_dict['Cl']['widget']
        w3 = self.widget_dict['CLI']['widget']
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
        # self.widget.clear()
        self.widget_dict = {}
        grid_counter = 0
        for k, v in self.inputs.items():
            label = QLabel(v['label'], self)
            widget = getattr(sys.modules[__name__], v['widget_type'])(self)
            self.widget_dict[k] = {'widget': widget}
            if 'decimals' in v.keys():
                widget.setDecimals(v['decimals'])
            if 'text' in v.keys():
                widget.setText(v['text'])
                widget.textChanged.connect(partial(self.dict_connection, widget, k))
            if 'texts' in v.keys():
                if k == 'additional_data':
                    widget.insertPlainText('\n'.join(v['texts']))
                else:
                    widget.insertPlainText('\n\n'.join(v['texts']))
                widget.textChanged.connect(partial(self.dict_connection, widget, k))
            if 'items' in v.keys():
                widget.addItems(v['items'])
            if 'current_text' in v.keys():
                widget.setCurrentText(v['current_text'])
                widget.currentTextChanged.connect(partial(self.dict_connection, widget, k))
            if 'lower_bound' in v.keys():
                widget.setMinimum(v['lower_bound'])
            if 'upper_bound' in v.keys():
                widget.setMaximum(v['upper_bound'])
            if 'value' in v.keys():
                # print(f"Setting value of {v_['label']} to {v_['value']}")
                widget.setValue(v['value'])
                widget.valueChanged.connect(partial(self.dict_connection, widget, k))
                # print(f"Actual returned value is {widget.value()}")
            if 'state' in v.keys():
                widget.setCheckState(v['state'])
                widget.setTristate(False)
                widget.stateChanged.connect(partial(self.dict_connection, widget, k))
            if isinstance(widget, QPushButton):
                if 'button_title' in v.keys():
                    widget.setText(v['button_title'])
                if 'click_connect' in v.keys():
                    push_button_action = getattr(self, v['click_connect'])
                    widget.clicked.connect(push_button_action)
            if 'tool_tip' in v.keys():
                label.setToolTip(v['tool_tip'])
                widget.setToolTip(v['tool_tip'])
            push_button = None
            checkbox = None
            if 'push_button' in v.keys():
                push_button = QPushButton(v['push_button'], self)
                push_button_action = getattr(self, v['push_button_action'])
                push_button.clicked.connect(partial(push_button_action, widget))
                self.widget_dict[k]['push_button'] = push_button
            if 'active_checkbox' in v.keys():
                checkbox = QCheckBox('Active?', self)
                checkbox.setCheckState(v['active_checkbox'])
                checkbox.setTristate(False)
                checkbox_action = getattr(self, 'activate_deactivate_checkbox')
                checkbox.stateChanged.connect(partial(checkbox_action, widget))
                checkbox.stateChanged.connect(partial(self.dict_connection, checkbox, k))
                self.widget_dict[k]['checkbox'] = checkbox_action
            if 'combo_callback' in v.keys():
                combo_callback_action = getattr(self, v['combo_callback'])
                widget.currentTextChanged.connect(combo_callback_action)
            if 'editable' in v.keys():
                widget.setReadOnly(not v['editable'])
            if 'text_changed_callback' in v.keys():
                action = getattr(self, v['text_changed_callback'])
                widget.textChanged.connect(partial(action, widget))
                action(widget, v['text'])
            if k == 'mea_dir' and self.widget_dict[k]['use_current_mea']['widget'].checkState():
                widget.setReadOnly(True)
                self.widget_dict[k]['push_button'].setEnabled(False)
            if k == 'additional_data':
                widget.setMaximumHeight(50)
            self.layout.addWidget(label, grid_counter, 0)
            if push_button is None:
                if checkbox is None:
                    self.layout.addWidget(widget, grid_counter, 1, 1, 3)
                else:
                    self.layout.addWidget(widget, grid_counter, 1, 1, 2)
                    self.layout.addWidget(checkbox, grid_counter, 3)
            else:
                self.layout.addWidget(widget, grid_counter, 1, 1, 2)
                self.layout.addWidget(push_button, grid_counter, 3)
            grid_counter += 1

        for k, v in self.inputs.items():
            self.enable_disable_from_checkbox(k)

        self.change_prescribed_aero_parameter(self.inputs['prescribe']['current_text'])

    def getInputs(self):
        return self.inputs


class MultiAirfoilDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.setWindowTitle("Multi-Element-Airfoil Analysis")
        self.setFont(self.parent().font())

        self.grid_widget = None
        self.grid_layout = None

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton("Run", QDialogButtonBox.AcceptRole)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        layout = QVBoxLayout(self)
        self.tab_widget = VerticalTabWidget(self)
        layout.addWidget(self.tab_widget)

        if parent.mset_settings is None:
            self.inputs = {'MSET': mset_settings_default([a for a in self.parent().mea.airfoils.keys()])}
        else:
            self.inputs = {'MSET': parent.mset_settings}

        if parent.mses_settings is None:
            self.inputs['MSES'] = mses_settings_default()
        else:
            self.inputs['MSES'] = parent.mses_settings

        if parent.mplot_settings is None:
            self.inputs['MPLOT'] = mplot_settings_default()
        else:
            self.inputs['MPLOT'] = parent.mplot_settings

        self.setInputs()

        parent.mset_settings = self.inputs['MSET']
        parent.mses_settings = self.inputs['MSES']
        parent.mplot_settings = self.inputs['MPLOT']

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

    def select_directory_for_airfoil_analysis(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    @staticmethod
    def activate_deactivate_checkbox(widget: QLineEdit or QSpinBox or QDoubleSpinBox, checked: bool):
        if checked:
            widget.setReadOnly(False)
        else:
            widget.setReadOnly(True)

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

    def select_coord_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(self.tr("Data Files (*.txt *.dat *.csv)"))
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
        msg_box.setIcon(QMessageBox.Information)
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
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setFont(self.parent().font())
        msg_box.exec()

    @staticmethod
    def convert_text_array_to_dict(text_array: list):
        data_dict = {}
        for text in text_array:
            text_split = text.split(': ')
            if len(text_split) > 1:
                k = text_split[0]
                v = float(text_split[1])
                data_dict[k] = v
        return data_dict

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

    def dict_connection(self,
                        widget: QLineEdit | QSpinBox | QDoubleSpinBox | ScientificDoubleSpinBox | QComboBox | QCheckBox,
                        key1: str, key2: str):
        if isinstance(widget, QLineEdit):
            self.inputs[key1][key2]['text'] = widget.text()
        elif isinstance(widget, QPlainTextEdit):
            if key2 == 'additional_data':
                self.inputs[key1][key2]['texts'] = widget.toPlainText().split('\n')
            else:
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
        elif isinstance(widget, GridBounds):
            self.inputs[key1][key2]['values'] = widget.values()
        elif isinstance(widget, MSETMultiGridWidget):
            self.inputs[key1][key2]['values'] = widget.values()
        elif isinstance(widget, XTRSWidget):
            self.inputs[key1][key2]['values'] = widget.values()
        elif isinstance(widget, ADWidget):
            self.inputs[key1][key2]['values'] = widget.values()

        self.enable_disable_from_checkbox(key1, key2)

        if key2 in ['P', 'T', 'rho', 'R', 'gam', 'L', 'MACHIN'] and not widget.isReadOnly():
            P = self.inputs['MSES']['P']['value']
            T = self.inputs['MSES']['T']['value']
            rho = self.inputs['MSES']['rho']['value']
            R = self.inputs['MSES']['R']['value']
            # gam = self.inputs['MSES']['gam']['value']
            # L = self.inputs['MSES']['L']['value']
            # M = self.inputs['MSES']['MACHIN']['value']
            P_widget = self.widget_dict['MSES']['P']['widget']
            T_widget = self.widget_dict['MSES']['T']['widget']
            rho_widget = self.widget_dict['MSES']['rho']['widget']
            # Re_widget = self.widget_dict['MSES']['REYNIN']['widget']
            if P_widget.isReadOnly():
                P_widget.setValue(rho * R * T)
            elif T_widget.isReadOnly():
                T_widget.setValue(P / R / rho)
            elif rho_widget.isReadOnly():
                rho_widget.setValue(P / R / T)
            self.calculate_and_set_Reynolds_number()

    def calculate_and_set_Reynolds_number(self):
        T = self.inputs['MSES']['T']['value']
        rho = self.inputs['MSES']['rho']['value']
        R = self.inputs['MSES']['R']['value']
        gam = self.inputs['MSES']['gam']['value']
        L = self.inputs['MSES']['L']['value']
        M = self.inputs['MSES']['MACHIN']['value']
        Re_widget = self.widget_dict['MSES']['REYNIN']['widget']
        nu = viscosity_calculator(T, rho=rho)
        a = np.sqrt(gam * R * T)
        V = M * a
        Re_widget.setValue(V * L / nu)

    def change_prescribed_aero_parameter(self, current_text: str):
        w1 = self.widget_dict['MSES']['ALFAIN']['widget']
        w2 = self.widget_dict['MSES']['CLIFIN']['widget']
        if current_text == 'Specify Angle of Attack':
            bools = (False, True)
        elif current_text == 'Specify Lift Coefficient':
            bools = (True, False)
        else:
            raise ValueError('Invalid value of currentText for QComboBox (alfa/Cl')
        w1.setReadOnly(bools[0])
        w2.setReadOnly(bools[1])

    def change_prescribed_flow_variables(self, current_text: str):
        w1 = self.widget_dict['MSES']['P']['widget']
        w2 = self.widget_dict['MSES']['T']['widget']
        w3 = self.widget_dict['MSES']['rho']['widget']
        if current_text == 'Specify Pressure, Temperature':
            bools = (False, False, True)
        elif current_text == 'Specify Pressure, Density':
            bools = (False, True, False)
        elif current_text == 'Specify Temperature, Density':
            bools = (True, False, False)
        else:
            raise ValueError('Invalid value of currentText for QComboBox (P/T/rho)')
        w1.setReadOnly(bools[0])
        w2.setReadOnly(bools[1])
        w3.setReadOnly(bools[2])

    def change_Re_active_state(self, state):
        if state == 0 or state is None:
            active = False
        else:
            active = True
        widget_names = ['P', 'T', 'rho', 'L', 'R', 'gam']
        skip_P, skip_T, skip_rho = False, False, False
        if (self.inputs['MSES']['spec_P_T_rho']['current_text'] == 'Specify Pressure, Temperature' and
                self.widget_dict['MSES']['rho']['widget'].isReadOnly()):
            skip_rho = True
        if (self.inputs['MSES']['spec_P_T_rho']['current_text'] == 'Specify Pressure, Density' and
                self.widget_dict['MSES']['T']['widget'].isReadOnly()):
            skip_T = True
        if (self.inputs['MSES']['spec_P_T_rho']['current_text'] == 'Specify Temperature, Density' and
                self.widget_dict['MSES']['P']['widget'].isReadOnly()):
            skip_P = True
        for widget_name in widget_names:
            if not (skip_rho and widget_name == 'rho') and not (skip_P and widget_name == 'P') and not (
                    skip_T and widget_name == 'T'):
                self.widget_dict['MSES'][widget_name]['widget'].setReadOnly(active)
        self.widget_dict['MSES']['REYNIN']['widget'].setReadOnly(not active)
        self.widget_dict['MSES']['spec_P_T_rho']['widget'].setEnabled(not active)
        if not active:
            self.calculate_and_set_Reynolds_number()

    def change_airfoil_order(self, w):
        dialog = AirfoilListDialog(self, current_airfoil_list=[a for a in self.parent().mea.airfoils.keys()])
        if dialog.exec_():
            airfoil_order = dialog.getData()
            self.widget_dict['MSET']['airfoil_order']['widget'].setText(','.join(airfoil_order))

    def setInputs(self):
        self.tab_widget.clear()
        self.widget_dict = {}
        for k, v in self.inputs.items():
            self.widget_dict[k] = {}
            grid_counter = 0
            self.add_tab(k)
            for k_, v_ in v.items():
                if 'label' in v_.keys():
                    label = QLabel(v_['label'], self)
                else:
                    label = None
                widget = getattr(sys.modules[__name__], v_['widget_type'])(self)
                self.widget_dict[k][k_] = {'widget': widget}
                if 'decimals' in v_.keys():
                    widget.setDecimals(v_['decimals'])
                if 'text' in v_.keys():
                    widget.setText(v_['text'])
                    if hasattr(widget, 'textChanged'):
                        widget.textChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'texts' in v_.keys():
                    if k_ == 'additional_data':
                        widget.insertPlainText('\n'.join(v_['texts']))
                    else:
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
                if 'checkbox_callback' in v_.keys():
                    checkbox_callback_action = getattr(self, v_['checkbox_callback'])
                    widget.stateChanged.connect(checkbox_callback_action)
                if 'values' in v_.keys():
                    widget.setValues(v_['values'])
                    if isinstance(widget, GridBounds):
                        widget.boundsChanged.connect(partial(self.dict_connection, widget, k, k_))
                    elif isinstance(widget, MSETMultiGridWidget):
                        widget.multiGridChanged.connect(partial(self.dict_connection, widget, k, k_))
                    elif isinstance(widget, XTRSWidget):
                        widget.XTRSChanged.connect(partial(self.dict_connection, widget, k, k_))
                    elif isinstance(widget, ADWidget):
                        widget.ADChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'editable' in v_.keys():
                    widget.setReadOnly(not v_['editable'])
                if 'text_changed_callback' in v_.keys():
                    action = getattr(self, v_['text_changed_callback'])
                    widget.textChanged.connect(partial(action, widget))
                    action(widget, v_['text'])
                if k_ == 'mea_dir' and self.widget_dict[k]['use_current_mea']['widget'].checkState():
                    widget.setReadOnly(True)
                    self.widget_dict[k][k_]['push_button'].setEnabled(False)
                if k_ == 'additional_data':
                    widget.setMaximumHeight(50)

                if 'restart_grid_counter' in v_.keys() and v_['restart_grid_counter']:
                    grid_counter = 0

                if 'label_col' in v_.keys():
                    label_col = v_['label_col']
                else:
                    label_col = 0
                if label is not None:
                    if 'label_align' in v_.keys():
                        if v_['label_align'] == 'l':
                            label_alignment = Qt.AlignLeft
                        elif v_['label_align'] == 'c':
                            label_alignment = Qt.AlignCenter
                        elif v_['label_align'] == 'r':
                            label_alignment = Qt.AlignRight
                        else:
                            raise ValueError('\'label_align\' must be one of: \'l\', \'c\', or \'r\'')
                    else:
                        label_alignment = Qt.AlignLeft
                    self.grid_layout.addWidget(label, grid_counter, label_col, label_alignment)
                    widget_starting_col = 1
                else:
                    widget_starting_col = 0

                if 'align' in v_.keys():
                    if v_['align'] == 'l':
                        alignment = Qt.AlignLeft
                    elif v_['align'] == 'c':
                        alignment = Qt.AlignCenter
                    elif v_['align'] == 'r':
                        alignment = Qt.AlignRight
                    else:
                        raise ValueError('\'align\' must be one of: \'l\', \'c\', or \'r\'')
                else:
                    alignment = None

                if 'col' in v_.keys():
                    col = v_['col']
                else:
                    col = widget_starting_col
                if 'row_span' in v_.keys():
                    row_span = v_['row_span']
                else:
                    row_span = 1
                if 'col_span' in v_.keys():
                    col_span = v_['col_span']
                else:
                    col_span = 3

                if push_button is None:
                    if checkbox is None:
                        if alignment is not None:
                            self.grid_layout.addWidget(widget, grid_counter, col, row_span, col_span, alignment)
                        else:
                            self.grid_layout.addWidget(widget, grid_counter, col, row_span, col_span)
                    else:
                        if alignment is not None:
                            self.grid_layout.addWidget(widget, grid_counter, col, 1, 2, alignment)
                        else:
                            self.grid_layout.addWidget(widget, grid_counter, col, 1, 2)
                        self.grid_layout.addWidget(checkbox, grid_counter, col + 3)
                else:
                    if alignment is not None:
                        self.grid_layout.addWidget(widget, grid_counter, col, 1, 2, alignment)
                    else:
                        self.grid_layout.addWidget(widget, grid_counter, col, 1, 2)
                    self.grid_layout.addWidget(push_button, grid_counter, col + 2)

                if 'next_on_same_row' in v_.keys() and v_['next_on_same_row']:
                    pass
                else:
                    if 'row_span' in v_.keys():
                        grid_counter += v_['row_span']
                    else:
                        grid_counter += 1

        # self.add_tab("Image Test")
        # image = QSvgWidget(os.path.join(RESOURCE_DIR, 'grid_test.svg'))
        # graphics_scene = QGraphicsScene()
        # temp_widget = graphics_scene.addWidget(image)
        # view = CustomGraphicsView(graphics_scene, parent=self)
        # view.setRenderHint(QPainter.Antialiasing)
        # self.grid_layout.addWidget(view, 0, 0, 4, 4)
        # new_image = QSvgWidget(os.path.join(RESOURCE_DIR, 'sec_34.svg'))
        # temp_widget.setWidget(new_image)

        for k, v in self.inputs.items():
            for k_ in v.keys():
                self.enable_disable_from_checkbox(k, k_)

        # Need to run these functions because self.dict_connection is not run on startup
        self.change_prescribed_aero_parameter(self.inputs['MSES']['spec_alfa_Cl']['current_text'])
        self.change_Re_active_state(self.inputs['MSES']['spec_Re']['state'])
        self.change_prescribed_flow_variables(self.inputs['MSES']['spec_P_T_rho']['current_text'])

    def getInputs(self):
        return self.inputs


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
    def __init__(self, bounds, parent=None, pos_param: bool = False):
        super().__init__(parent)
        self.pos_param = pos_param
        self.setWindowTitle("Parameter Bounds Modification")
        self.setFont(self.parent().font())
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.lower_bound = InftyDoubleSpinBox(lower=True)
        if self.pos_param:
            self.lower_bound.setValue(bounds[0][0])
        else:
            self.lower_bound.setValue(bounds[0])
        self.lower_bound.setDecimals(16)
        if self.pos_param:
            layout.addRow("Lower Bound (x)", self.lower_bound)
        else:
            layout.addRow("Lower Bound", self.lower_bound)

        self.upper_bound = InftyDoubleSpinBox(lower=False)
        if self.pos_param:
            self.upper_bound.setValue(bounds[0][1])
        else:
            self.upper_bound.setValue(bounds[1])
        self.upper_bound.setDecimals(16)
        if self.pos_param:
            layout.addRow("Upper Bound (x)", self.upper_bound)
        else:
            layout.addRow("Upper Bound", self.upper_bound)

        if pos_param:
            self.lower_bound2 = InftyDoubleSpinBox(lower=True)
            self.lower_bound2.setValue(bounds[1][0])
            self.lower_bound2.setDecimals(16)
            layout.addRow("Lower Bound (y)", self.lower_bound2)

            self.upper_bound2 = InftyDoubleSpinBox(lower=False)
            self.upper_bound2.setValue(bounds[1][1])
            self.upper_bound2.setDecimals(16)
            layout.addRow("Upper Bound (y)", self.upper_bound2)

        self.hint_label = QLabel("<Home> key: +/-inf", parent=self)
        self.hint_label.setWordWrap(True)

        layout.addWidget(self.hint_label)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        if self.pos_param:
            return [[self.lower_bound.value(), self.upper_bound.value()],
                    [self.lower_bound2.value(), self.upper_bound2.value()]]
        else:
            return [self.lower_bound.value(), self.upper_bound.value()]

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
    def __init__(self, parent, file_filter: str = "JMEA Files (*.jmea)"):
        super().__init__(parent=parent)
        self.setFileMode(QFileDialog.ExistingFile)
        self.setNameFilter(self.tr(file_filter))
        self.setViewMode(QFileDialog.Detail)


class SaveAsDialog(QFileDialog):
    def __init__(self, parent, file_filter: str = "JMEA Files (*.jmea)"):
        super().__init__(parent=parent)
        self.setFileMode(QFileDialog.AnyFile)
        self.setNameFilter(self.tr(file_filter))
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
            self.inputs = opt_settings_default(self.parent().airfoil_name_list)
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

    def select_existing_jmea_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(self.tr("JMEA Parametrization (*.jmea)"))
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

    def select_coord_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(self.tr("Data Files (*.txt *.dat *.csv)"))
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def select_multiple_coord_files(self, text_edit: QPlainTextEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter(self.tr("Data Files (*.txt *.dat *.csv)"))
        if file_dialog.exec_():
            text_edit.insertPlainText('\n\n'.join(file_dialog.selectedFiles()))

    def select_thickness_file(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(self.tr("Data Files (*.txt *.dat *.csv)"))
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

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
        msg_box.setIcon(QMessageBox.Information)
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
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setFont(self.parent().font())
        msg_box.exec()

    def objectives_changed(self, widget, text: str):
        self.parent().objectives = []
        for obj_func_str in text.split(','):
            objective = Objective(obj_func_str)
            self.parent().objectives.append(objective)
            if text == '':
                widget.setStyleSheet("QLineEdit {background-color: rgba(176,25,25,50)}")
                return
            try:
                function_input_data1 = getattr(cfd_output_templates, self.inputs['Genetic Algorithm'][
                    'tool']['current_text'])
                function_input_data2 = self.convert_text_array_to_dict(
                    self.inputs['Genetic Algorithm']['additional_data']['texts'])
                objective.update({**function_input_data1, **function_input_data2})
                widget.setStyleSheet("QLineEdit {background-color: rgba(16,201,87,50)}")
            except FunctionCompileError:
                widget.setStyleSheet("QLineEdit {background-color: rgba(176,25,25,50)}")
                return

    def constraints_changed(self, widget, text: str):
        self.parent().constraints = []
        for constraint_func_str in text.split(','):
            if len(constraint_func_str) > 0:
                constraint = Constraint(constraint_func_str)
                self.parent().constraints.append(constraint)
                try:
                    function_input_data1 = getattr(cfd_output_templates, self.inputs['Genetic Algorithm'][
                        'tool']['current_text'])
                    function_input_data2 = self.convert_text_array_to_dict(
                        self.inputs['Genetic Algorithm']['additional_data']['texts'])
                    constraint.update({**function_input_data1, **function_input_data2})
                    widget.setStyleSheet("QLineEdit {background-color: rgba(16,201,87,50)}")
                except FunctionCompileError:
                    widget.setStyleSheet("QLineEdit {background-color: rgba(176,25,25,50)}")
                    return

    @staticmethod
    def convert_text_array_to_dict(text_array: list):
        data_dict = {}
        for text in text_array:
            text_split = text.split(': ')
            if len(text_split) > 1:
                k = text_split[0]
                v = float(text_split[1])
                data_dict[k] = v
        return data_dict

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
            if key2 == 'additional_data':
                self.inputs[key1][key2]['texts'] = widget.toPlainText().split('\n')
            else:
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
        elif isinstance(widget, GridBounds):
            self.inputs[key1][key2]['values'] = widget.values()
        elif isinstance(widget, MSETMultiGridWidget):
            self.inputs[key1][key2]['values'] = widget.values()
        elif isinstance(widget, XTRSWidget):
            self.inputs[key1][key2]['values'] = widget.values()
        elif isinstance(widget, ADWidget):
            self.inputs[key1][key2]['values'] = widget.values()

        self.enable_disable_from_checkbox(key1, key2)

        if key2 == 'tool':
            self.objectives_changed(self.widget_dict['Genetic Algorithm']['J']['widget'],
                                    self.inputs['Genetic Algorithm']['J']['text'])
            self.constraints_changed(self.widget_dict['Genetic Algorithm']['G']['widget'],
                                     self.inputs['Genetic Algorithm']['G']['text'])

        if key2 == 'pop_size':
            self.widget_dict[key1]['n_offspring']['widget'].setMinimum(widget.value())

        if key2 in ['P', 'T', 'rho', 'R', 'gam', 'L', 'MACHIN'] and not widget.isReadOnly():
            P = self.inputs['MSES']['P']['value']
            T = self.inputs['MSES']['T']['value']
            rho = self.inputs['MSES']['rho']['value']
            R = self.inputs['MSES']['R']['value']
            # gam = self.inputs['MSES']['gam']['value']
            # L = self.inputs['MSES']['L']['value']
            # M = self.inputs['MSES']['MACHIN']['value']
            P_widget = self.widget_dict['MSES']['P']['widget']
            T_widget = self.widget_dict['MSES']['T']['widget']
            rho_widget = self.widget_dict['MSES']['rho']['widget']
            # Re_widget = self.widget_dict['MSES']['REYNIN']['widget']
            if P_widget.isReadOnly():
                P_widget.setValue(rho * R * T)
            elif T_widget.isReadOnly():
                T_widget.setValue(P / R / rho)
            elif rho_widget.isReadOnly():
                rho_widget.setValue(P / R / T)
            self.calculate_and_set_Reynolds_number()

    def calculate_and_set_Reynolds_number(self):
        T = self.inputs['MSES']['T']['value']
        rho = self.inputs['MSES']['rho']['value']
        R = self.inputs['MSES']['R']['value']
        gam = self.inputs['MSES']['gam']['value']
        L = self.inputs['MSES']['L']['value']
        M = self.inputs['MSES']['MACHIN']['value']
        Re_widget = self.widget_dict['MSES']['REYNIN']['widget']
        nu = viscosity_calculator(T, rho=rho)
        a = np.sqrt(gam * R * T)
        V = M * a
        Re_widget.setValue(V * L / nu)

    def change_prescribed_aero_parameter_xfoil(self, current_text: str):
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

    def change_prescribed_aero_parameter(self, current_text: str):
        w1 = self.widget_dict['MSES']['ALFAIN']['widget']
        w2 = self.widget_dict['MSES']['CLIFIN']['widget']
        if current_text == 'Specify Angle of Attack':
            bools = (False, True)
        elif current_text == 'Specify Lift Coefficient':
            bools = (True, False)
        else:
            raise ValueError('Invalid value of currentText for QComboBox (alfa/Cl')
        w1.setReadOnly(bools[0])
        w2.setReadOnly(bools[1])

    def change_prescribed_flow_variables(self, current_text: str):
        w1 = self.widget_dict['MSES']['P']['widget']
        w2 = self.widget_dict['MSES']['T']['widget']
        w3 = self.widget_dict['MSES']['rho']['widget']
        if current_text == 'Specify Pressure, Temperature':
            bools = (False, False, True)
        elif current_text == 'Specify Pressure, Density':
            bools = (False, True, False)
        elif current_text == 'Specify Temperature, Density':
            bools = (True, False, False)
        else:
            raise ValueError('Invalid value of currentText for QComboBox (P/T/rho)')
        w1.setReadOnly(bools[0])
        w2.setReadOnly(bools[1])
        w3.setReadOnly(bools[2])

    def change_Re_active_state(self, state):
        if state == 0 or state is None:
            active = False
        else:
            active = True
        widget_names = ['P', 'T', 'rho', 'L', 'R', 'gam']
        skip_P, skip_T, skip_rho = False, False, False
        if (self.inputs['MSES']['spec_P_T_rho']['current_text'] == 'Specify Pressure, Temperature' and
                self.widget_dict['MSES']['rho']['widget'].isReadOnly()):
            skip_rho = True
        if (self.inputs['MSES']['spec_P_T_rho']['current_text'] == 'Specify Pressure, Density' and
                self.widget_dict['MSES']['T']['widget'].isReadOnly()):
            skip_T = True
        if (self.inputs['MSES']['spec_P_T_rho']['current_text'] == 'Specify Temperature, Density' and
                self.widget_dict['MSES']['P']['widget'].isReadOnly()):
            skip_P = True
        for widget_name in widget_names:
            if not (skip_rho and widget_name == 'rho') and not (skip_P and widget_name == 'P') and not (
                    skip_T and widget_name == 'T'):
                self.widget_dict['MSES'][widget_name]['widget'].setReadOnly(active)
        self.widget_dict['MSES']['REYNIN']['widget'].setReadOnly(not active)
        self.widget_dict['MSES']['spec_P_T_rho']['widget'].setEnabled(not active)
        if not active:
            self.calculate_and_set_Reynolds_number()

    def select_directory_for_airfoil_analysis(self, line_edit: QLineEdit):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        if file_dialog.exec_():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def setInputs(self):
        self.tab_widget.clear()
        self.widget_dict = {}
        for k, v in self.inputs.items():
            self.widget_dict[k] = {}
            grid_counter = 0
            self.add_tab(k)
            for k_, v_ in v.items():
                # print(f"{k_ = }, {v_ = }")
                if 'label' in v_.keys():
                    label = QLabel(v_['label'], self)
                else:
                    label = None
                widget = getattr(sys.modules[__name__], v_['widget_type'])(self)
                self.widget_dict[k][k_] = {'widget': widget}
                if 'decimals' in v_.keys():
                    widget.setDecimals(v_['decimals'])
                if 'text' in v_.keys():
                    widget.setText(v_['text'])
                    if hasattr(widget, 'textChanged'):
                        widget.textChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'texts' in v_.keys():
                    if k_ == 'additional_data':
                        widget.insertPlainText('\n'.join(v_['texts']))
                    else:
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
                if 'checkbox_callback' in v_.keys():
                    checkbox_callback_action = getattr(self, v_['checkbox_callback'])
                    widget.stateChanged.connect(checkbox_callback_action)
                if 'values' in v_.keys():
                    widget.setValues(v_['values'])
                    if isinstance(widget, GridBounds):
                        widget.boundsChanged.connect(partial(self.dict_connection, widget, k, k_))
                    elif isinstance(widget, MSETMultiGridWidget):
                        widget.multiGridChanged.connect(partial(self.dict_connection, widget, k, k_))
                    elif isinstance(widget, XTRSWidget):
                        widget.XTRSChanged.connect(partial(self.dict_connection, widget, k, k_))
                    elif isinstance(widget, ADWidget):
                        widget.ADChanged.connect(partial(self.dict_connection, widget, k, k_))
                if 'editable' in v_.keys():
                    widget.setReadOnly(not v_['editable'])
                if 'text_changed_callback' in v_.keys():
                    action = getattr(self, v_['text_changed_callback'])
                    widget.textChanged.connect(partial(action, widget))
                    action(widget, v_['text'])
                if k_ == 'mea_dir' and self.widget_dict[k]['use_current_mea']['widget'].checkState():
                    widget.setReadOnly(True)
                    self.widget_dict[k][k_]['push_button'].setEnabled(False)
                if k_ == 'additional_data':
                    widget.setMaximumHeight(50)

                if 'restart_grid_counter' in v_.keys() and v_['restart_grid_counter']:
                    grid_counter = 0

                if 'label_col' in v_.keys():
                    label_col = v_['label_col']
                else:
                    label_col = 0
                if label is not None:
                    if 'label_align' in v_.keys():
                        if v_['label_align'] == 'l':
                            label_alignment = Qt.AlignLeft
                        elif v_['label_align'] == 'c':
                            label_alignment = Qt.AlignCenter
                        elif v_['label_align'] == 'r':
                            label_alignment = Qt.AlignRight
                        else:
                            raise ValueError('\'label_align\' must be one of: \'l\', \'c\', or \'r\'')
                    else:
                        label_alignment = Qt.AlignLeft
                    self.grid_layout.addWidget(label, grid_counter, label_col, label_alignment)
                    widget_starting_col = 1
                else:
                    widget_starting_col = 0

                if 'align' in v_.keys():
                    if v_['align'] == 'l':
                        alignment = Qt.AlignLeft
                    elif v_['align'] == 'c':
                        alignment = Qt.AlignCenter
                    elif v_['align'] == 'r':
                        alignment = Qt.AlignRight
                    else:
                        raise ValueError('\'align\' must be one of: \'l\', \'c\', or \'r\'')
                else:
                    alignment = None

                if 'col' in v_.keys():
                    col = v_['col']
                else:
                    col = widget_starting_col
                if 'row_span' in v_.keys():
                    row_span = v_['row_span']
                else:
                    row_span = 1
                if 'col_span' in v_.keys():
                    col_span = v_['col_span']
                else:
                    col_span = 3

                if push_button is None:
                    if checkbox is None:
                        if alignment is not None:
                            self.grid_layout.addWidget(widget, grid_counter, col, row_span, col_span, alignment)
                        else:
                            self.grid_layout.addWidget(widget, grid_counter, col, row_span, col_span)
                    else:
                        if alignment is not None:
                            self.grid_layout.addWidget(widget, grid_counter, col, 1, 2, alignment)
                        else:
                            self.grid_layout.addWidget(widget, grid_counter, col, 1, 2)
                        self.grid_layout.addWidget(checkbox, grid_counter, col + 3)
                else:
                    if alignment is not None:
                        self.grid_layout.addWidget(widget, grid_counter, col, 1, 2, alignment)
                    else:
                        self.grid_layout.addWidget(widget, grid_counter, col, 1, 2)
                    self.grid_layout.addWidget(push_button, grid_counter, col + 2)

                if 'next_on_same_row' in v_.keys() and v_['next_on_same_row']:
                    pass
                else:
                    if 'row_span' in v_.keys():
                        grid_counter += v_['row_span']
                    else:
                        grid_counter += 1

        for k, v in self.inputs.items():
            for k_ in v.keys():
                self.enable_disable_from_checkbox(k, k_)

        self.change_prescribed_aero_parameter_xfoil(self.inputs['XFOIL']['prescribe']['current_text'])
        self.change_prescribed_aero_parameter(self.inputs['MSES']['spec_alfa_Cl']['current_text'])
        self.change_Re_active_state(self.inputs['MSES']['spec_Re']['state'])
        self.change_prescribed_flow_variables(self.inputs['MSES']['spec_P_T_rho']['current_text'])

    def getInputs(self):
        return self.inputs


class SymmetryDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Set symmetry")
        self.setFont(self.parent().font())
        self.current_param_path = None
        self.current_form_idx = 1

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.inputs = self.setInputs()
        for i in self.inputs:
            layout.addRow(i[0], i[1])

        for idx, input_row in enumerate(self.inputs):
            if not idx % 2 and isinstance(input_row[1], QPushButton):  # only evaluate the odd rows
                input_row[1].clicked.connect(partial(self.switch_row, idx + 1))

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        r0 = ["Target", QPushButton("Select Parameter", self)]
        r1 = ["Selected target", QLineEdit(self)]
        r2 = ["Tool", QPushButton("Select Parameter", self)]
        r3 = ["Selected tool", QLineEdit(self)]
        r4 = ["x1", QPushButton("Select Parameter", self)]
        r5 = ["Selected x1", QLineEdit(self)]
        r6 = ["y1", QPushButton("Select Parameter", self)]
        r7 = ["Selected y1", QLineEdit(self)]
        r8 = ["Line angle", QPushButton("Select Parameter", self)]
        r9 = ["Selected line angle", QLineEdit(self)]
        return [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9]

    @pyqtSlot(int)
    def switch_row(self, row: int):
        self.current_form_idx = row

    def getInputs(self):
        return {'target': self.inputs[1][1].text(),
                'tool': self.inputs[3][1].text(),
                'x1': self.inputs[5][1].text(),
                'y1': self.inputs[7][1].text(),
                'angle': self.inputs[9][1].text()}


class ExportCoordinatesDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.setWindowTitle("Export Airfoil Coordinates")
        self.setFont(self.parent().font())

        self.grid_widget = {}

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton(QDialogButtonBox.Ok)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.grid_layout = QGridLayout(self)

        self.setInputs()

        self.grid_widget['airfoil_order']['line'].setText(','.join([k for k in self.parent().mea.airfoils.keys()]))

        self.grid_layout.addWidget(buttonBox, 7, 1, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        widget_dict = load_data(os.path.join('dialog_widgets', 'export_coordinates_dialog.json'))
        for row_name, row_dict in widget_dict.items():
            self.grid_widget[row_name] = {}
            for w_name, w_dict in row_dict.items():
                widget = getattr(sys.modules[__name__], w_dict["w"])(self)
                self.grid_widget[row_name][w_name] = widget
                if "text" in w_dict.keys() and isinstance(widget, (QLabel, QLineEdit, QPushButton)):
                    widget.setText(w_dict["text"])
                if "func" in w_dict.keys() and isinstance(widget, QPushButton):
                    if row_name == 'choose_dir':
                        widget.clicked.connect(partial(getattr(self, w_dict["func"]),
                                                       self.grid_widget[row_name]["line"]))
                    else:
                        widget.clicked.connect(getattr(self, w_dict["func"]))
                if "tool_tip" in w_dict.keys():
                    widget.setToolTip(w_dict["tool_tip"])
                self.grid_layout.addWidget(widget, w_dict["grid"][0], w_dict["grid"][1], w_dict["grid"][2],
                                           w_dict["grid"][3])

    def getInputs(self):
        inputs = {k: v["line"].text() if "line" in v.keys() else None for k, v in self.grid_widget.items()}

        # Make sure any newline characters are not double-escaped:
        for k, input_ in inputs.items():
            if isinstance(input_, str):
                inputs[k] = input_.replace('\\n', '\n')

        return inputs

    def select_directory(self, line_edit: QLineEdit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory", os.path.expanduser("~"),
                                                        QFileDialog.ShowDirsOnly)
        if selected_dir:
            line_edit.setText(selected_dir)

    def format_mses(self):
        self.grid_widget["header"]["line"].setText("airfoil_name\\n-3.0 3.0 -3.0 3.0")
        self.grid_widget["separator"]["line"].setText("999.0 999.0\\n")
        self.grid_widget["delimiter"]["line"].setText(" ")
        self.grid_widget["file_name"]["line"].setText("blade.airfoil_name")


class ExportControlPointsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.setWindowTitle("Export Control Points")
        self.setFont(self.parent().font())

        self.grid_widget = {}

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton(QDialogButtonBox.Ok)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.grid_layout = QGridLayout(self)

        self.setInputs()

        self.grid_widget['airfoil_order']['line'].setText(','.join([k for k in self.parent().mea.airfoils.keys()]))

        self.grid_layout.addWidget(buttonBox, 7, 1, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        widget_dict = load_data(os.path.join('dialog_widgets', 'export_control_points_dialog.json'))
        for row_name, row_dict in widget_dict.items():
            self.grid_widget[row_name] = {}
            for w_name, w_dict in row_dict.items():
                widget = getattr(sys.modules[__name__], w_dict["w"])(self)
                self.grid_widget[row_name][w_name] = widget
                if "text" in w_dict.keys() and isinstance(widget, (QLabel, QLineEdit, QPushButton)):
                    widget.setText(w_dict["text"])
                if "func" in w_dict.keys() and isinstance(widget, QPushButton):
                    if row_name == 'choose_dir':
                        widget.clicked.connect(partial(getattr(self, w_dict["func"]),
                                                       self.grid_widget[row_name]["line"]))
                    else:
                        widget.clicked.connect(getattr(self, w_dict["func"]))
                if "tool_tip" in w_dict.keys():
                    widget.setToolTip(w_dict["tool_tip"])
                self.grid_layout.addWidget(widget, w_dict["grid"][0], w_dict["grid"][1], w_dict["grid"][2],
                                           w_dict["grid"][3])

    def getInputs(self):
        inputs = {k: v["line"].text() if "line" in v.keys() else None for k, v in self.grid_widget.items()}

        # Make sure any newline characters are not double-escaped:
        for k, input_ in inputs.items():
            if isinstance(input_, str):
                inputs[k] = input_.replace('\\n', '\n')

        return inputs

    def select_directory(self, line_edit: QLineEdit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory", os.path.expanduser("~"),
                                                        QFileDialog.ShowDirsOnly)
        if selected_dir:
            line_edit.setText(selected_dir)


class PosConstraintDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Set Relative Positions")
        self.setFont(self.parent().font())
        self.current_param_path = None
        self.current_form_idx = 1

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.layout = QFormLayout(self)

        self.inputs = self.setInputs()
        for i in self.inputs:
            self.layout.addRow(i[0], i[1])

        for idx, input_row in enumerate(self.inputs):
            if isinstance(input_row[1], QPushButton):
                input_row[1].clicked.connect(partial(self.switch_row, idx + 1))

        self.layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        r0 = ["Target", QPushButton("Select Parameter", self)]
        r1 = ["Selected target", QLineEdit(self)]
        r2 = ["Tool", QPushButton("Select Parameter", self)]
        r3 = ["Selected tool", QLineEdit(self)]
        r4 = ["Mode", QComboBox(self)]
        r4[1].addItems(["distance, angle", "dx, dy"])
        r4[1].currentTextChanged.connect(self.switchMode)
        r5 = ["Distance", QPushButton("Select Parameter", self)]
        r6 = ["Selected distance", QLineEdit(self)]
        r7 = ["Angle", QPushButton("Select Parameter", self)]
        r8 = ["Selected angle", QLineEdit(self)]
        return [r0, r1, r2, r3, r4, r5, r6, r7, r8]

    def switchMode(self, mode: str):
        for idx in reversed(range(5, 9)):
            self.layout.removeRow(idx)
        if 'dx' in mode:
            r5 = ["dx", QPushButton("Select Parameter", self)]
            r6 = ["Selected dx", QLineEdit(self)]
            r7 = ["dy", QPushButton("Select Parameter", self)]
            r8 = ["Selected dy", QLineEdit(self)]
        else:
            r5 = ["Distance", QPushButton("Select Parameter", self)]
            r6 = ["Selected distance", QLineEdit(self)]
            r7 = ["Angle", QPushButton("Select Parameter", self)]
            r8 = ["Selected angle", QLineEdit(self)]
        for idx, r in enumerate([r5, r6, r7, r8]):
            self.inputs[idx + 5] = r
            if isinstance(r[1], QPushButton):
                r[1].clicked.connect(partial(self.switch_row, idx + 6))
            self.layout.insertRow(idx + 5, r[0], r[1])

    @pyqtSlot(int)
    def switch_row(self, row: int):
        self.current_form_idx = row

    def getInputs(self):
        if 'dx' in self.inputs[4][1].currentText():
            return {'target': self.inputs[1][1].text(),
                    'tool': self.inputs[3][1].text(),
                    'dx': self.inputs[6][1].text(),
                    'dy': self.inputs[8][1].text()}
        else:
            return {'target': self.inputs[1][1].text(),
                    'tool': self.inputs[3][1].text(),
                    'dist': self.inputs[6][1].text(),
                    'angle': self.inputs[8][1].text()}


class AirfoilMatchingDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Choose Airfoil to Match")
        self.setFont(self.parent().font())

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.layout = QFormLayout(self)

        self.inputs = self.setInputs()
        for i in self.inputs:
            self.layout.addRow(i[0], i[1])

        self.layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        r0 = ["Airfoil to Match", QLineEdit(self)]
        r0[1].setText('naca0012-il')
        return [r0]

    def getInputs(self):
        return self.inputs[0][1].text()


class AirfoilPlotDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Select Airfoil to Plot")
        self.setFont(self.parent().font())

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.layout = QFormLayout(self)

        self.inputs = self.setInputs()
        for i in self.inputs:
            self.layout.addRow(i[0], i[1])

        self.layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        r0 = ["Airfoil to Plot", QLineEdit(self)]
        r0[1].setText('n0012-il')
        return [r0]

    def getInputs(self):
        return self.inputs[0][1].text()


class AirfoilListView(QListView):
    """Class created from
    https://stackoverflow.com/questions/52873025/pyqt5-qlistview-drag-and-drop-creates-new-hidden-items"""
    def __init__(self, parent: QWidget, airfoil_list: list):
        super().__init__(parent=parent)
        self.airfoil_list = airfoil_list
        self.setDragDropMode(QListView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragEnabled(True)
        model = QStandardItemModel(self)
        for airfoil_name in self.airfoil_list:
            item = QStandardItem(airfoil_name)
            item.setCheckable(True)
            item.setDragEnabled(True)
            item.setDropEnabled(False)
            item.setCheckState(Qt.Checked)
            data = [airfoil_name, item.checkState()]
            item.setData(data)
            model.appendRow(item)

        self.setModel(model)


class AirfoilListDialog(QDialog):
    def __init__(self, parent, current_airfoil_list: list):
        super().__init__(parent)
        self.setWindowTitle("Select Airfoil Order")
        self.setFont(self.parent().font())

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)

        self.q_airfoil_listview = AirfoilListView(self, current_airfoil_list)

        self.central_widget = QWidget(self)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.q_airfoil_listview)
        self.layout.addWidget(buttonBox)
        self.setLayout(self.layout)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getData(self):
        """Function based on
        https://stackoverflow.com/questions/52873025/pyqt5-qlistview-drag-and-drop-creates-new-hidden-items"""
        checked_airfoils = []
        model = self.q_airfoil_listview.model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                checked_airfoils.append(item.text())
        return checked_airfoils
