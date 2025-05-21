import os
import shutil
import sys
import tempfile
import typing
from abc import abstractmethod
from copy import deepcopy
from functools import partial
from typing import List

import numpy as np
import pyqtgraph as pg
import PyQt6.QtWidgets
from PyQt6.QtCore import QEvent, QSignalBlocker
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QRect
from PyQt6.QtCore import pyqtSlot, QStandardPaths
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QWidget, QGridLayout, QLabel, QPushButton, QCheckBox, QTabWidget, QSpinBox,
                             QDoubleSpinBox, QComboBox, QDialog, QVBoxLayout, QSizeGrip, QDialogButtonBox, QMessageBox,
                             QFormLayout, QRadioButton, QGroupBox, QSizePolicy, QColorDialog, QListWidget, QHBoxLayout)
from qframelesswindow import FramelessDialog

from pymead import GUI_DEFAULTS_DIR, q_settings, GUI_DIALOG_WIDGETS_DIR, TargetPathNotFoundError
from pymead.analysis import cfd_output_templates
from pymead.analysis.utils import viscosity_calculator
from pymead.core.airfoil import Airfoil
from pymead.core.bezier import Bezier
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.line import PolyLine
from pymead.core.mea import MEA
from pymead.gui.bounds_values_table import BoundsValuesTable
from pymead.gui.default_settings import xfoil_settings_default
from pymead.gui.file_selection import *
from pymead.gui.infty_doublespinbox import InftyDoubleSpinBox
from pymead.gui.pyqt_vertical_tab_widget.pyqt_vertical_tab_widget import VerticalTabWidget
from pymead.gui.sampling_visualization import SamplingVisualizationWidget
from pymead.gui.scientificspinbox_master.ScientificDoubleSpinBox import ScientificDoubleSpinBox
from pymead.gui.separation_lines import QHSeperationLine
from pymead.gui.side_grip import SideGrip
from pymead.gui.title_bar import DialogTitleBar
from pymead.optimization.objectives_and_constraints import Objective, Constraint, FunctionCompileError
from pymead.optimization.opt_setup import calculate_warm_start_index, read_stencil_from_array
from pymead.utils.dict_recursion import recursive_get
from pymead.utils.misc import get_setting, set_setting, make_ga_opt_dir
from pymead.utils.read_write_files import load_data, save_data, load_documents_path
from pymead.utils.widget_recursion import get_parent


ISMOM_ITEMS = [
    "S-momentum equation",
    "isentropic condition",
    "S-momentum equation, isentropic @ LE",
    "isentropic condition, S-mom. where diss. active"
]
IFFBC_ITEMS = [
    "solid wall airfoil far-field BCs",
    "vortex+source+doublet airfoil far-field BCs",
    "freestream pressure airfoil far-field BCs",
    "supersonic wave freestream BCs",
    "supersonic solid wall far-field BCs"
]
ISMOM_CONVERSION = {item: idx + 1 for idx, item in enumerate(ISMOM_ITEMS)}
IFFBC_CONVERSION = {item: idx + 1 for idx, item in enumerate(IFFBC_ITEMS)}

VALIDATION_SUCCESS = "rgba(16,201,87,50)"
VALIDATION_FAILURE = "rgba(176,25,25,50)"
VALIDATION_DEFAULT = ""


def convert_dialog_to_mset_settings(dialog_input: dict):
    mset_settings = deepcopy(dialog_input)
    mset_settings["airfoils"] = [k for k in mset_settings["multi_airfoil_grid"].keys()]
    mset_settings["n_airfoils"] = len(mset_settings["airfoils"])
    return mset_settings


def convert_dialog_to_mses_settings(dialog_input: dict):
    mses_settings = {
        'ISMOVE': 0,
        'ISPRES': 0,
        'NMODN': 0,
        'NPOSN': 0,
        'viscous_flag': dialog_input['viscous_flag'],
        'inverse_flag': 0,
        'inverse_side': 1,
        'verbose': dialog_input['verbose'],
        'ISMOM': ISMOM_CONVERSION[dialog_input['ISMOM']],
        'IFFBC': IFFBC_CONVERSION[dialog_input['IFFBC']]
    }

    if dialog_input['AD_active']:
        mses_settings['AD_flags'] = [1 for _ in range(dialog_input['AD_number'])]
    else:
        mses_settings['AD_flags'] = [0 for _ in range(dialog_input['AD_number'])]

    values_list = ['REYNIN', 'MACHIN', 'ALFAIN', 'CLIFIN', 'ACRIT', 'MCRIT', 'MUCON',
                   'timeout', 'iter', 'P', 'T', 'rho', 'gam', 'R']
    for value in values_list:
        mses_settings[value] = dialog_input[value]

    if dialog_input['spec_alfa_Cl'] == 'Specify Angle of Attack':
        mses_settings['target'] = 'alfa'
    elif dialog_input['spec_alfa_Cl'] == 'Specify Lift Coefficient':
        mses_settings['target'] = 'Cl'

    for xtrs_key in ("XTRSupper", "XTRSlower"):
        mses_settings[xtrs_key] = dialog_input["xtrs"][xtrs_key]

    for idx, AD_idx in enumerate(dialog_input['AD'].values()):
        for k, v in AD_idx.items():
            if idx == 0:
                mses_settings[k] = [v]
            else:
                mses_settings[k].append(v)

    return mses_settings


def convert_dialog_to_mplot_settings(dialog_input: dict):
    mplot_settings = {
        'timeout': dialog_input['timeout'],
        'Mach': dialog_input['Mach'],
        'Grid': dialog_input['Grid'],
        'Grid_Zoom': dialog_input['Grid_Zoom'],
        'flow_field': dialog_input['Output_Field'],
        "Tecplot": dialog_input["Tecplot"] if "Tecplot" in dialog_input else False,
        "Paraview": dialog_input["Paraview"] if "Paraview" in dialog_input else False,
        'Streamline_Grid': dialog_input["Streamline_Grid"],
        'CPK': dialog_input['CPK'],
    }
    return mplot_settings


def convert_dialog_to_mpolar_settings(dialog_input: dict):
    mplot_settings = {
        "timeout": dialog_input["timeout"],
        "alfa_array": None
    }

    if dialog_input["polar_mode"] == "No Polar Analysis":
        pass
    elif dialog_input["polar_mode"] == "Alpha Sweep from Data File":
        if dialog_input["alfa_array"] != "":
            mplot_settings["alfa_array"] = np.loadtxt(dialog_input["alfa_array"])
    elif dialog_input["polar_mode"] == "Alpha Sweep from Start/Stop/Inc":
        alfa_array = np.arange(dialog_input["alfa_start"], dialog_input["alfa_end"], dialog_input["alfa_inc"])
        if dialog_input["alfa_end"] not in alfa_array:
            alfa_array = np.append(alfa_array, dialog_input["alfa_end"])
        mplot_settings["alfa_array"] = alfa_array
    else:
        raise ValueError("Invalid polar_mode")

    return mplot_settings


get_set_value_names = {'QSpinBox': ('value', 'setValue', 'valueChanged'),
                       'QDoubleSpinBox': ('value', 'setValue', 'valueChanged'),
                       'ScientificDoubleSpinBox': ('value', 'setValue', 'valueChanged'),
                       'QTextArea': ('text', 'setText', 'textChanged'),
                       'QPlainTextArea': ('text', 'setText', 'textChanged'),
                       'QLineEdit': ('text', 'setText', 'textChanged'),
                       'QComboBox': ('currentText', 'setCurrentText', 'currentTextChanged'),
                       'QCheckBox': ('isChecked', 'setChecked', 'stateChanged'),
                       'QPlainTextEdit': ('toPlainText', 'setPlainText', 'textChanged'),
                       'GridBounds': ('value', 'setValue', 'boundsChanged'),
                       'MSETMultiGridWidget': ('value', 'setValue', 'multiGridChanged'),
                       'XTRSWidget': ('values', 'setValues', 'XTRSChanged'),
                       'ADWidget': ('values', 'setValues', 'ADChanged'),
                       'OptConstraintsHTabWidget': ('values', 'setValues', 'OptConstraintsChanged')}
grid_names = {'label': ['label.row', 'label.column', 'label.rowSpan', 'label.columnSpan', 'label.alignment'],
              'widget': ['row', 'column', 'rowSpan', 'columnSpan', 'alignment'],
              'push_button': ['push.row', 'push.column', 'push.rowSpan', 'push.columnSpan', 'push.alignment'],
              'checkbox': ['check.row', 'check.column', 'check.rowSpan', 'check.columnSpan', 'check.alignment']}
# sum(<ragged list>, []) flattens a ragged (or uniform) 2-D list into a 1-D list
reserved_names = ['label', 'widget_type', 'push_button', 'push_button_action', 'clicked_connect', 'active_checkbox',
                  *sum([v for v in grid_names.values()], [])]
msg_modes = {'info': QMessageBox.Icon.Information,
             'warn': QMessageBox.Icon.Warning,
             'question': QMessageBox.Icon.Question,
             "error": QMessageBox.Icon.Critical}


class PymeadDialogWidget(QWidget):
    def __init__(self, settings_file, **kwargs):
        super().__init__()
        self.settings = load_data(settings_file)
        self.widget_dict = {}
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.kwargs = {**kwargs}
        self.setInputs()

    def setInputs(self):
        """This method is used to add Widgets to the Layout"""
        grid_counter = 0
        for w_name, w_dict in self.settings.items():
            self.widget_dict[w_name] = {'label': None, 'widget': None, 'push_button': None, 'checkbox': None}

            # Restart the grid_counter if necessary:
            if 'restart_grid_counter' in w_dict.keys() and w_dict['restart_grid_counter']:
                grid_counter = 0

            # Add the label if necessary:
            if 'label' in w_dict.keys():
                label = QLabel(w_dict['label'], parent=self)
                grid_params_label = {'row': grid_counter, 'column': 0, 'rowSpan': 1, 'columnSpan': 1,
                                     }
                for k, v in w_dict.items():
                    if k in grid_names['label']:
                        grid_params_label[k.split('.')[-1]] = v
                self.layout.addWidget(label, *[v for v in grid_params_label.values()])
                self.widget_dict[w_name]['label'] = label

            # Add the main widget:
            if hasattr(PyQt6.QtWidgets, w_dict['widget_type']):
                # First check if the widget type is found in PyQt5.QtWidgets:
                widget = getattr(PyQt6.QtWidgets, w_dict['widget_type'])(parent=self)
            elif hasattr(sys.modules[__name__], w_dict['widget_type']):
                # If not in PyQt5.QtWidgets, check the modules loaded into this file:
                kwargs = {}
                if w_dict['widget_type'] in ['ADWidget', 'OptConstraintsHTabWidget']:
                    kwargs = self.kwargs
                    if "initial_mea" in kwargs:
                        kwargs.pop("initial_mea")
                elif w_dict["widget_type"] == "XTRSWidget":
                    kwargs = {"initial_mea": self.kwargs.get("initial_mea")}
                if w_dict["widget_type"] != "ADWidget":
                    if "param_list" in kwargs.keys():
                        kwargs.pop("param_list")
                if w_dict["widget_type"] not in ["ADWidget", "OptConstraintsHTabWidget"]:
                    if "geo_col" in kwargs.keys():
                        kwargs.pop("geo_col")
                widget = getattr(sys.modules[__name__], w_dict['widget_type'])(parent=self, **kwargs)
            else:
                raise ValueError(f"Widget type {w_dict['widget_type']} not found in PyQt5.QtWidgets or system modules")
            grid_params_widget = {'row': grid_counter, 'column': 1, 'rowSpan': 1,
                                  'columnSpan': 2 if 'push_button' in w_dict.keys() else 3,
                                  }
            for k, v in w_dict.items():
                if k in grid_names['widget']:
                    grid_params_widget[k] = v
                    if k == 'alignment':
                        grid_params_widget[k] = {'l': Qt.AlignmentFlag.AlignLeft, 
                                                 'c': Qt.AlignmentFlag.AlignCenter, 
                                                 'r': Qt.AlignmentFlag.AlignRight}[v]
            self.layout.addWidget(widget, *[v for v in grid_params_widget.values()])
            self.widget_dict[w_name]['widget'] = widget

            # Add the push button:
            if 'push_button' in w_dict.keys():
                push_button = QPushButton(w_dict['push_button'], parent=self)
                grid_params_push = {'row': grid_counter, 'column': grid_params_widget['column'] + 2, 'rowSpan': 1,
                                    'columnSpan': 1}
                for k, v in w_dict.items():
                    if k in grid_names['push_button']:
                        grid_params_push[k.split('.')[-1]] = v
                push_button.clicked.connect(partial(getattr(self, w_dict['push_button_action']), widget))
                self.layout.addWidget(push_button, *[v for v in grid_params_push.values()])
                self.widget_dict[w_name]['push_button'] = push_button

            if 'active_checkbox' in w_dict.keys():
                checkbox = QCheckBox('Active?', parent=self)
                grid_params_check = {'row': grid_counter, 'column': grid_params_widget['column'] + 2, 'rowSpan': 1,
                                     'columnSpan': 1}
                for k, v in w_dict.items():
                    if k in grid_names['checkbox']:
                        grid_params_check[k.split('.')[-1]] = v
                checkbox.stateChanged.connect(partial(self.activate_deactivate_from_checkbox, widget))
                self.layout.addWidget(checkbox, *[v for v in grid_params_check.values()])
                self.widget_dict[w_name]['checkbox'] = checkbox

            # Connect the button if there is one
            if 'clicked_connect' in w_dict.keys() and isinstance(widget, QPushButton):
                widget.clicked.connect(getattr(self, w_dict['clicked_connect']))

            # Loop through the individual settings of each widget and execute:
            for s_name, s_value in w_dict.items():
                if s_name not in reserved_names and hasattr(widget, s_name):
                    getattr(widget, s_name)(s_value)

            # Increment the counter
            grid_counter += 1 if 'rowSpan' not in w_dict.keys() else w_dict['rowSpan']

        # Add connections for all widgets to dialogChanged (do this in a separate loop so the signals do not get
        # triggered during initialization):
        for w_name, w_dict in self.settings.items():
            widget = self.widget_dict[w_name]['widget']
            if w_dict['widget_type'] in get_set_value_names.keys():
                getattr(widget, get_set_value_names[w_dict['widget_type']][2]).connect(
                    partial(self.dialogChanged, w_name=w_name))

    def value(self):
        """This method is used to extract the data from the Dialog"""
        output_dict = {w_name: None for w_name in self.widget_dict.keys()}
        for w_name, w in self.widget_dict.items():
            if self.settings[w_name]['widget_type'] in get_set_value_names.keys():
                output_dict[w_name] = getattr(w['widget'],
                                              get_set_value_names[self.settings[w_name]['widget_type']][0]
                                              )()
                if w['checkbox'] is not None:
                    state = w['checkbox'].isChecked()
                    output_dict[w_name] = (output_dict[w_name], state)
            else:
                output_dict[w_name] = None
        return output_dict

    @staticmethod
    def activate_deactivate_from_checkbox(widget, state):
        widget.setReadOnly(not state)

    def setValue(self, new_values: dict):
        for k, v in new_values.items():
            if v is not None:
                if k not in self.widget_dict:
                    continue
                if self.widget_dict[k]['checkbox'] is not None:
                    self.widget_dict[k]['checkbox'].setChecked(v[1])
                    getattr(self.widget_dict[k]['widget'], get_set_value_names[self.settings[k]['widget_type']][1])(v[0])
                else:
                    getattr(self.widget_dict[k]['widget'], get_set_value_names[self.settings[k]['widget_type']][1])(v)

    def dialogChanged(self, *_, w_name: str):
        new_inputs = self.value()
        self.updateDialog(new_inputs, w_name)

    @abstractmethod
    def updateDialog(self, new_inputs: dict, w_name: str):
        """Required method which reacts to changes in the dialog inputs. Use the :code:`setValue` method to
        update the dialog at the end of this method if necessary."""
        pass


class PymeadDialogHTabWidget(QTabWidget):

    sigTabsChanged = pyqtSignal(object)

    def __init__(self, parent, widgets: dict, settings_override: dict = None):
        super().__init__()
        self.w_dict = widgets
        self.generateWidgets()
        if settings_override is not None:
            self.setValue(settings_override)

    def generateWidgets(self):
        for k, v in self.w_dict.items():
            self.addTab(v, k)

    def regenerateWidgets(self):
        self.clear()
        self.generateWidgets()
        self.sigTabsChanged.emit([k for k in self.w_dict.keys()])

    def setValue(self, new_values: dict):
        for k, v in new_values.items():
            self.w_dict[k].setValue(v)

    def value(self):
        return {k: v.value() for k, v in self.w_dict.items()}


class PymeadLabeledSpinBox(QObject):

    sigValueChanged = pyqtSignal(int)

    def __init__(self, label: str = "", tool_tip: str = "", minimum: int = None, maximum: int = None,
                 value: int = None, read_only: bool = None):
        self.label = QLabel(label)
        self.widget = QSpinBox()
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        if minimum is not None:
            self.widget.setMinimum(minimum)
        if maximum is not None:
            self.widget.setMaximum(maximum)
        if value is not None:
            self.widget.setValue(value)
        if read_only is not None:
            self.widget.setReadOnly(read_only)
        self.push = None

        super().__init__()
        self.widget.valueChanged.connect(self.sigValueChanged)

    def setValue(self, value: int):
        self.widget.setValue(value)

    def value(self):
        return self.widget.value()

    def setReadOnly(self, read_only: bool):
        self.widget.setReadOnly(read_only)

    def setActive(self, active: int):
        self.widget.setReadOnly(not active)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledDoubleSpinBox(QObject):

    sigValueChanged = pyqtSignal(float)

    def __init__(self, label: str = "", tool_tip: str = "", minimum: float = None, maximum: float = None,
                 value: float = None, decimals: int = None, single_step: float = None, push_label: str = None,
                 read_only: bool = None):
        self.label = QLabel(label)
        self.widget = QDoubleSpinBox()
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        if minimum is not None:
            self.widget.setMinimum(minimum)
        if maximum is not None:
            self.widget.setMaximum(maximum)
        if decimals is not None:
            self.widget.setDecimals(decimals)
        if value is not None:
            self.widget.setValue(value)
        if single_step is not None:
            self.widget.setSingleStep(single_step)
        if read_only is not None:
            self.widget.setReadOnly(read_only)
        self.push = None

        if push_label is not None:
            self.push = QPushButton(push_label)

        super().__init__()
        self.widget.valueChanged.connect(self.sigValueChanged)

    def setValue(self, value: float):
        self.widget.setValue(value)

    def value(self):
        return self.widget.value()

    def setReadOnly(self, read_only: bool):
        self.widget.setReadOnly(read_only)

    def setActive(self, active: bool):
        self.widget.setReadOnly(not active)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledScientificDoubleSpinBox(QObject):

    sigValueChanged = pyqtSignal(float)

    def __init__(self, label: str = "", tool_tip: str = "", minimum: float = None, maximum: float = None,
                 value: float = None, decimals: int = None, single_step: float = None, push_label: str = None,
                 read_only: bool = None):
        self.label = QLabel(label)
        self.widget = ScientificDoubleSpinBox()
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        if minimum is not None:
            self.widget.setMinimum(minimum)
        if maximum is not None:
            self.widget.setMaximum(maximum)
        if decimals is not None:
            self.widget.setDecimals(decimals)
        if value is not None:
            self.widget.setValue(value)
        if single_step is not None:
            self.widget.setSingleStep(single_step)
        if read_only is not None:
            self.widget.setReadOnly(read_only)
        self.push = None

        if push_label is not None:
            self.push = QPushButton(push_label)

        super().__init__()
        self.widget.valueChanged.connect(self.sigValueChanged)

    def setValue(self, value: float):
        self.widget.setValue(value)

    def value(self):
        return self.widget.value()

    def setReadOnly(self, read_only: bool):
        self.widget.setReadOnly(read_only)

    def setActive(self, active: bool):
        self.widget.setReadOnly(not active)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledDoubleSpinSelector(QWidget):

    sigValueChanged = pyqtSignal(float)

    def __init__(self, geo_col: GeometryCollection, parent=None,
                 label: str = "", tool_tip: str = "", minimum: float = None,
                 maximum: float = None,
                 value: float = None,
                 decimals: int = None, single_step: float = None,
                 read_only: bool = None):
        self.geo_col = geo_col
        self._default_value = value
        self.label = QLabel(label)
        self.widget = QWidget()
        self.spin = QDoubleSpinBox()
        self.spin.setMinimumWidth(170)
        self.line = QLineEdit()
        self.line.setReadOnly(True)  # Should always be read-only; only editable by pressing the "plus" button
        self.line.hide()  # Hidden by default unless one or more params/dvs are selected by pressing the "plus" button
        self.lay = QHBoxLayout()
        self.lay.setContentsMargins(0, 0, 0, 0)  # Get rid of the padding around the layout
        self.lay.addWidget(self.spin)
        self.lay.addWidget(self.line)
        self.widget.setLayout(self.lay)
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        if minimum is not None:
            self.spin.setMinimum(minimum)
        if maximum is not None:
            self.spin.setMaximum(maximum)
        if decimals is not None:
            self.spin.setDecimals(decimals)
        if value is not None:
            self.spin.setValue(value)
        if single_step is not None:
            self.spin.setSingleStep(single_step)
        if read_only is not None:
            self.spin.setReadOnly(read_only)
        self.push = QPushButton("+")
        self.push.setMaximumWidth(30)

        super().__init__(parent=parent)
        self.spin.valueChanged.connect(self.sigValueChanged)
        self.push.clicked.connect(self.openSelector)

    def _getFloatFromDVOrParam(self, dv_or_param_name: str) -> float:
        if dv_or_param_name in self.geo_col.container()["desvar"].keys():
            return self.geo_col.container()["desvar"][dv_or_param_name].value()
        elif dv_or_param_name in self.geo_col.container()["params"].keys():
            return self.geo_col.container()["params"][dv_or_param_name].value()
        else:
            raise ValueError(f"Could not find selected key {dv_or_param_name} in the desvar or params containers")

    def setValue(self, value: float or dict):  # Float kept only for backwards compatibility
        if isinstance(value, dict) and value["param"]:
            if isinstance(value["param"], str):
                value["value"] = self._getFloatFromDVOrParam(value["param"])
                self.line.setText(value["param"])
            elif isinstance(value["param"], list):
                if len(value["param"]) > 0 and value["param"][0]:
                    value["value"] = self._getFloatFromDVOrParam(value["param"][0])
                else:
                    value["value"] = self._default_value
                self.line.setText(",".join(value["param"]))
            self.line.show()
            self.spin.setReadOnly(True)
        else:
            self.spin.setReadOnly(False)
            self.line.clear()
            self.line.hide()
        if isinstance(value, dict):
            self.spin.setValue(value["value"])
        else:
            self.spin.setValue(value)

    def value(self) -> dict:
        value = {"value": self.spin.value(), "param": None}
        line_split = self.line.text().split(",")
        if len(line_split) == 1 and line_split[0]:
            value["param"] = line_split[0]
        elif len(line_split) > 1:
            value["param"] = line_split
        return value

    def setReadOnly(self, read_only: bool):
        self.spin.setReadOnly(read_only)

    def setActive(self, active: bool):
        self.spin.setReadOnly(not active)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.spin.show()
            self.line.show()
            self.push.show()
        else:
            self.label.hide()
            self.spin.hide()
            self.line.hide()
            self.push.hide()

    def openSelector(self):
        """
        Opens a selection widget that allows the user to safely set the list of design variables or parameters.
        """
        initial_items = self.line.text().split(",")
        if len(initial_items[0]) == 0:
            initial_items = None
        dialog = DesVarParamListSelectionDialog(
            window_title="DV/Param Selector", geo_col=self.geo_col, parent=self,
            initial_items=initial_items
        )
        # old_dialog_value = dialog.value()
        if dialog.exec():
            new_dialog_value = dialog.value()
            self.setValue({"value": 0.0, "param": new_dialog_value})
            # for dv_name in old_dialog_value["param"]:
            #     if dv_name in new_dialog_value:
            #         continue
            #     if dv_name in self.geo_col.container()["desvar"]:
            #         self.geo_col.container()["desvar"][dv_name].assignable = True
            #     elif dv_name in self.geo_col.container()["params"]:
            #         self.geo_col.container()["params"][dv_name].assignable = True
            # for dv_name in new_dialog_value:
            #     if dv_name in old_dialog_value:
            #         continue
            #     if dv_name in self.geo_col.container()["desvar"]:
            #         self.geo_col.container()["desvar"][dv_name].assignable = False
            #     elif dv_name in self.geo_col.container()["params"]:
            #         self.geo_col.container()["params"][dv_name].assignable = False

class PymeadLabeledLineEdit(QObject):

    sigValueChanged = pyqtSignal(str)

    def __init__(self, label: str = "", tool_tip: str = "", text: str = "", push_label: str = None,
                 read_only: bool = None):

        self.label = QLabel(label)
        self.widget = QLineEdit(text)
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        self.push = None

        if push_label is not None:
            self.push = QPushButton(push_label)
        if read_only is not None:
            self.setReadOnly(read_only)

        super().__init__()
        self.widget.textChanged.connect(self.sigValueChanged)

    def setValue(self, text: str):
        self.widget.setText(text)

    def value(self):
        return self.widget.text()

    def setReadOnly(self, read_only: bool):
        self.widget.setReadOnly(read_only)
        if self.push is not None:
            self.push.setEnabled(not read_only)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledPlainTextEdit(QObject):

    sigValueChanged = pyqtSignal()

    def __init__(self, label: str = "", tool_tip: str = "", text: str = "", push_label: str = None,
                 read_only: bool = None):
        self.label = QLabel(label)
        self.widget = QPlainTextEdit(text)
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        self.push = None

        if push_label is not None:
            self.push = QPushButton(push_label)
        if read_only is not None:
            self.setReadOnly(read_only)

        super().__init__()
        self.widget.textChanged.connect(self.sigValueChanged)

    def setValue(self, text: str):
        self.widget.setPlainText(text)

    def value(self):
        return self.widget.toPlainText()

    def setReadOnly(self, read_only: bool):
        self.widget.setReadOnly(read_only)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()

class PymeadLabeledComboBox(QObject):

    sigValueChanged = pyqtSignal(str)

    def __init__(self, label: str = "", tool_tip: str = "", items: typing.List[str] = None,
                 current_item: str = None):
        self.label = QLabel(label)
        self.widget = QComboBox()
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        self.push = None

        if items is not None:
            self.widget.addItems(items)
        if current_item is not None:
            self.widget.setCurrentText(current_item)

        super().__init__()
        self.widget.currentTextChanged.connect(self.sigValueChanged)

    def setValue(self, text: str):
        self.widget.setCurrentText(text)

    def value(self):
        return self.widget.currentText()

    def setReadOnly(self, read_only: bool):
        self.widget.setEnabled(not read_only)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledCheckbox(QObject):

    sigValueChanged = pyqtSignal(int)

    def __init__(self, label: str = "", tool_tip: str = "", initial_state: int = 0,
                 push_label: str = None):
        self.label = QLabel(label)
        self.widget = QCheckBox()
        self.widget.setChecked(initial_state)
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        self.push = None

        if push_label is not None:
            self.push = QPushButton(push_label)

        super().__init__()
        self.widget.stateChanged.connect(self.sigValueChanged)

    def setValue(self, state: int):
        self.widget.setChecked(state)

    def value(self):
        return self.widget.isChecked()

    def setReadOnly(self, read_only: bool):
        self.widget.setEnabled(not read_only)
        if self.push is not None:
            self.push.setEnabled(not read_only)

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadColorButton(pg.ColorButton):
    def __init__(self, parent=None, color=(128, 128, 128), padding=6):
        super().__init__(parent=parent, color=color, padding=padding)
        self.colorDialog = QColorDialog(parent=parent)
        self.colorDialog.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, True)
        self.colorDialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        self.colorDialog.currentColorChanged.connect(self.dialogColorChanged)
        self.colorDialog.rejected.connect(self.colorRejected)
        self.colorDialog.colorSelected.connect(self.colorSelected)


class PymeadLabeledColorSelector(QObject):

    sigValueChanged = pyqtSignal(tuple)

    def __init__(self, parent=None, label: str = "", tool_tip: str = "", initial_color: tuple = (245, 37, 106),
                 read_only: bool = False):
        self.label = QLabel(label)
        self.widget = PymeadColorButton(color=initial_color, parent=parent)
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        self.push = None
        self.widget.setMinimumHeight(25)
        self.widget.setEnabled(not read_only)

        super().__init__()
        self.widget.sigColorChanged.connect(self.sigValueChanged)

    def setValue(self, color: tuple):
        self.widget.setColor(color)

    def value(self):
        return self.widget.color(mode="byte")

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledPushButton:

    def __init__(self, label: str = "", text: str = "", tool_tip: str = ""):
        self.label = QLabel(label)
        self.widget = QPushButton(text)
        self.label.setToolTip(tool_tip)
        self.widget.setToolTip(tool_tip)
        self.push = None

    def setValue(self, _):
        pass

    @staticmethod
    def value():
        return None

    def setShown(self, shown: bool):
        if shown:
            self.label.show()
            self.widget.show()
            if self.push is not None:
                self.push.show()
        else:
            self.label.hide()
            self.widget.hide()
            if self.push is not None:
                self.push.hide()


class PymeadLabeledListWidget:
    def __init__(self, label: str = "", items: typing.List[str] = None, tool_tip: str = None):
        self.label = QLabel(label)
        self.widget = QListWidget()
        if tool_tip is not None:
            self.label.setToolTip(tool_tip)
            self.widget.setToolTip(tool_tip)
        if items is not None:
            self.widget.addItems(items)

    def value(self) -> list:
        return [self.widget.item(i).text() for i in range(self.widget.count())]

    def setValue(self, items: list):
        self.widget.clear()
        for item in items:
            self.widget.addItem(item)

    def setReadOnly(self, read_only: bool):
        self.widget.setEnabled(not read_only)


class PymeadDialogWidget2(QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.widget_dict = None
        self.lay = QGridLayout()
        self.setLayout(self.lay)
        self.initializeWidgets(**kwargs)
        self.addWidgets(**kwargs)
        self.establishWidgetConnections()
        self._widgets_excluded_from_value = []

    @abstractmethod
    def initializeWidgets(self, *args, **kwargs):
        pass

    def addWidgets(self, *args, **kwargs):
        # Add all the widgets
        row_count = 0
        column = 0
        for widget_name, widget in self.widget_dict.items():
            row_span = 1
            if widget.push is None and widget.label is None:
                col_span = 3
            elif widget.push is None and widget.label is not None:
                col_span = 2
            elif widget.push is not None and widget.label is None:
                col_span = 2
            else:
                col_span = 1

            if widget.label is not None:
                self.lay.addWidget(widget.label, row_count, column, 1, 1)

            if widget.label is None:
                self.lay.addWidget(widget.widget, row_count, column, row_span, col_span)
            else:
                self.lay.addWidget(widget.widget, row_count, column + 1, row_span, col_span)

            if widget.push is not None:
                if widget.label is None:
                    self.lay.addWidget(widget.push, row_count, column + 1, row_span, col_span)
                else:
                    self.lay.addWidget(widget.push, row_count, column + 2, row_span, col_span)

            row_count += 1

    def establishWidgetConnections(self):
        pass

    def excludeWidgetsFromValue(self, widgets: typing.List[str]):
        self._widgets_excluded_from_value = widgets

    def setValue(self, d: dict):
        for d_name, d_value in d.items():
            if d_name in self._widgets_excluded_from_value:
                continue
            try:
                self.widget_dict[d_name].setValue(d_value)
            except KeyError:
                pass

    def value(self) -> dict:
        return {k: v.value() for k, v in self.widget_dict.items() if k not in self._widgets_excluded_from_value}


class PymeadDialogVTabWidget(VerticalTabWidget):
    def __init__(self, parent, widgets: dict, settings_override: dict = None):
        super().__init__()
        self.w_dict = widgets
        for k, v in self.w_dict.items():
            self.addTab(v, k)
        if settings_override is not None:
            self.setValue(settings_override)

    def setValue(self, new_values: dict):
        for k, v in new_values.items():
            self.w_dict[k].setValue(v)

    def value(self):
        return {k: v.value() for k, v in self.w_dict.items()}


class PymeadDialog(FramelessDialog):

    _gripSize = 2

    """
    This subclass of FramelessDialog forces the selection of a WindowTitle and matches the visual 
    format of the GUI
    """
    def __init__(self, parent, window_title: str,
                 widget: PymeadDialogWidget or PymeadDialogVTabWidget or PymeadDialogWidget2,
                 theme: dict, minimum_width: int = None, minimum_height: int = None, fixed_width: int = None,
                 fixed_height: int = None):
        super().__init__(parent=parent)
        self.setWindowTitle(" " + window_title)
        if self.parent() is not None:
            self.setFont(self.parent().font())

        if minimum_width is not None and fixed_width is not None:
            raise ValueError("Cannot specify both minimum width and fixed width")
        if minimum_height is not None and fixed_height is not None:
            raise ValueError("Cannot specify both minimum height and fixed height")
        if minimum_width is not None:
            self.setMinimumWidth(minimum_width)
        if minimum_height is not None:
            self.setMinimumHeight(minimum_height)
        if fixed_width is not None:
            self.setFixedWidth(fixed_width)
        if fixed_height is not None:
            self.setFixedHeight(fixed_height)

        self.vbox_lay = QVBoxLayout()
        self.setLayout(self.vbox_lay)
        self.w = widget

        self.vbox_lay.addWidget(widget)
        self.button_box = self.create_button_box()
        self.vbox_lay.addWidget(self.button_box)

        # mandatory for cursor updates
        self.setMouseTracking(True)

        self.theme = theme

        self.title_bar = DialogTitleBar(self, theme=theme)

        self.sideGrips = [
            SideGrip(self, Qt.Edge.LeftEdge),
            SideGrip(self, Qt.Edge.TopEdge),
            SideGrip(self, Qt.Edge.RightEdge),
            SideGrip(self, Qt.Edge.BottomEdge),
        ]
        # corner grips should be "on top" of everything, otherwise the side grips
        # will take precedence on mouse events, so we are adding them *after*;
        # alternatively, widget.raise_() can be used
        self.cornerGrips = [QSizeGrip(self) for _ in range(4)]

        if fixed_width is not None and fixed_height is None:
            self.resize(self.width(), self.minimumHeight())
        elif fixed_width is None and fixed_height is not None:
            self.resize(self.minimumWidth(), self.height())
        elif fixed_width is None and fixed_height is None:
            self.resize(self.minimumWidth(), self.minimumHeight())
        else:
            pass
        self.resize(self.width(), self.title_bar.height() + self.height())

        self.title_bar.title.setStyleSheet(
            f"""background-color: qlineargradient(x1: 0.0, y1: 0.5, x2: 1.0, y2: 0.5, 
                    stop: 0 {theme['title-gradient-color']}, 
                    stop: 0.6 {theme['background-color']})""")

    def setValue(self, new_inputs):
        self.w.setValue(new_values=new_inputs)

    def value(self):
        return self.w.value()

    def create_button_box(self):
        """Creates a ButtonBox to add to the Layout. Can be overridden to add additional functionality.

        Returns
        -------
        QDialogButtonBox
        """
        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        return buttonBox

    @property
    def gripSize(self):
        return self._gripSize

    def setGripSize(self, size):
        if size == self._gripSize:
            return
        self._gripSize = max(2, size)
        self.updateGrips()

    def updateGrips(self):
        self.setContentsMargins(self.gripSize, self.gripSize + self.title_bar.height(), self.gripSize, self.gripSize)

        outRect = self.rect()
        # an "inner" rect used for reference to set the geometries of size grips
        inRect = outRect.adjusted(self.gripSize, self.gripSize, -self.gripSize, -self.gripSize)

        # top left
        self.cornerGrips[0].setGeometry(QRect(outRect.topLeft(), inRect.topLeft()))
        # top right
        self.cornerGrips[1].setGeometry(QRect(outRect.topRight(), inRect.topRight()).normalized())
        # bottom right
        self.cornerGrips[2].setGeometry(QRect(inRect.bottomRight(), outRect.bottomRight()))
        # bottom left
        self.cornerGrips[3].setGeometry(QRect(outRect.bottomLeft(), inRect.bottomLeft()).normalized())

        # left edge
        self.sideGrips[0].setGeometry(0, inRect.top(), self.gripSize, inRect.height())
        # top edge
        self.sideGrips[1].setGeometry(inRect.left(), 0, inRect.width(), self.gripSize)
        # right edge
        self.sideGrips[2].setGeometry(inRect.left() + inRect.width(), inRect.top(), self.gripSize, inRect.height())
        # bottom edge
        self.sideGrips[3].setGeometry(self.gripSize, inRect.top() + inRect.height(), inRect.width(), self.gripSize)

    def resizeEvent(self, event):
        self.title_bar.resize(self.width(), self.title_bar.height())
        super().resizeEvent(event)
        self.updateGrips()


class PymeadMessageBox(QMessageBox):

    _gripSize = 2

    def __init__(self, parent, msg: str, window_title: str, msg_mode: str, theme: dict):
        super().__init__(parent=parent)
        self.setText(msg)
        self.setWindowTitle(window_title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.setIcon(msg_modes[msg_mode])
        self.setFont(self.parent().font())

        # mandatory for cursor updates
        self.setMouseTracking(True)

        self.theme = theme

        self.title_bar = DialogTitleBar(self, theme=theme)

        self.sideGrips = [
            SideGrip(self, Qt.Edge.LeftEdge),
            SideGrip(self, Qt.Edge.TopEdge),
            SideGrip(self, Qt.Edge.RightEdge),
            SideGrip(self, Qt.Edge.BottomEdge),
        ]
        # corner grips should be "on top" of everything, otherwise the side grips
        # will take precedence on mouse events, so we are adding them *after*;
        # alternatively, widget.raise_() can be used
        self.cornerGrips = [QSizeGrip(self) for _ in range(4)]

        self.resize(self.width(), self.title_bar.height() + self.height())

        self.title_bar.title.setStyleSheet(
            f"""background-color: qlineargradient(x1: 0.0, y1: 0.5, x2: 1.0, y2: 0.5, 
                            stop: 0 {theme['title-gradient-color']}, 
                            stop: 0.6 {theme['background-color']})""")

    @property
    def gripSize(self):
        return self._gripSize

    def setGripSize(self, size):
        if size == self._gripSize:
            return
        self._gripSize = max(2, size)
        self.updateGrips()

    def updateGrips(self):
        self.setContentsMargins(self.gripSize, self.gripSize + self.title_bar.height(), self.gripSize, self.gripSize)

        outRect = self.rect()
        # an "inner" rect used for reference to set the geometries of size grips
        inRect = outRect.adjusted(self.gripSize, self.gripSize, -self.gripSize, -self.gripSize)

        # top left
        self.cornerGrips[0].setGeometry(QRect(outRect.topLeft(), inRect.topLeft()))
        # top right
        self.cornerGrips[1].setGeometry(QRect(outRect.topRight(), inRect.topRight()).normalized())
        # bottom right
        self.cornerGrips[2].setGeometry(QRect(inRect.bottomRight(), outRect.bottomRight()))
        # bottom left
        self.cornerGrips[3].setGeometry(QRect(outRect.bottomLeft(), inRect.bottomLeft()).normalized())

        # left edge
        self.sideGrips[0].setGeometry(0, inRect.top(), self.gripSize, inRect.height())
        # top edge
        self.sideGrips[1].setGeometry(inRect.left(), 0, inRect.width(), self.gripSize)
        # right edge
        self.sideGrips[2].setGeometry(inRect.left() + inRect.width(), inRect.top(), self.gripSize, inRect.height())
        # bottom edge
        self.sideGrips[3].setGeometry(self.gripSize, inRect.top() + inRect.height(), inRect.width(), self.gripSize)

    def resizeEvent(self, event):
        self.title_bar.resize(self.width(), self.title_bar.height())
        super().resizeEvent(event)
        self.updateGrips()


class MSETMultiGridWidget(QTabWidget):

    multiGridChanged = pyqtSignal()

    def __init__(self, parent, initial_mea: MEA = None):
        super().__init__(parent=parent)
        self.airfoil_names = [] if initial_mea is None else [a.name() for a in initial_mea.airfoils]
        self.widget_dict = {}
        self.grid_widgets = {}
        self.grid_widget = None
        self.grid_layout = None
        self.setValue()

    def onMEAChanged(self, mea: MEA or None):
        # Updated the widget based on a new MEA
        if mea is None:
            self.airfoil_names = []
        else:
            self.airfoil_names = [airfoil.name() for airfoil in mea.airfoils]
        self.setValue()

    def add_tab(self, name: str):
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self)
        self.grid_widget.setLayout(self.grid_layout)
        self.widget_dict[name] = {
            "dsLE_dsAvg": PymeadLabeledDoubleSpinBox(label="dsLE/dsAvg", minimum=0.0, maximum=np.inf, value=0.35,
                                                     single_step=0.01),
            "dsTE_dsAvg": PymeadLabeledDoubleSpinBox(label="dsTE/dsAvg", minimum=0.0, maximum=np.inf, value=0.80,
                                                     single_step=0.01),
            "curvature_exp": PymeadLabeledDoubleSpinBox(label="Curvature Exponent", minimum=0.0, maximum=np.inf,
                                                        value=1.30, single_step=0.01),
            "U_s_smax_min": PymeadLabeledDoubleSpinBox(label="U_s_smax_min", minimum=0.0, maximum=np.inf,
                                                       value=1.0, single_step=0.01),
            "U_s_smax_max": PymeadLabeledDoubleSpinBox(label="U_s_smax_max", minimum=0.0, maximum=np.inf,
                                                       value=1.0, single_step=0.01),
            "L_s_smax_min": PymeadLabeledDoubleSpinBox(label="L_s_smax_min", minimum=0.0, maximum=np.inf,
                                                       value=1.0, single_step=0.01),
            "L_s_smax_max": PymeadLabeledDoubleSpinBox(label="L_s_smax_max", minimum=0.0, maximum=np.inf,
                                                       value=1.0, single_step=0.01),
            "U_local_avg_spac_ratio": PymeadLabeledDoubleSpinBox(label="U Local/Max. Density Ratio", minimum=0.0,
                                                                 maximum=np.inf, value=0.0, single_step=0.01),
            "L_local_avg_spac_ratio": PymeadLabeledDoubleSpinBox(label="L Local/Max. Density Ratio", minimum=0.0,
                                                                 maximum=np.inf, value=0.0, single_step=0.01)
        }
        for widget in self.widget_dict[name].values():
            row_count = self.grid_layout.rowCount()
            self.grid_layout.addWidget(widget.label, row_count, 0)
            self.grid_layout.addWidget(widget.widget, row_count, 1)
        self.grid_widgets[name] = self.grid_widget
        self.addTab(self.grid_widget, name)

    def remove_tab(self, name: str):
        self.removeTab(self.indexOf(self.grid_widgets[name]))
        self.grid_widgets.pop(name)
        self.widget_dict.pop(name)

    def remove_tabs(self, names: typing.List[str]):
        for name in names[::-1]:
            self.remove_tab(name)

    def setValue(self, value: dict = None):
        if isinstance(value, dict) and len(value) == 0:
            return

        if value is None:
            for airfoil_name in self.airfoil_names:
                if airfoil_name in self.widget_dict:
                    continue
                self.add_tab(airfoil_name)

            airfoils_to_remove = list(set(self.widget_dict.keys()) - set(self.airfoil_names))
            self.remove_tabs(airfoils_to_remove)
            return

        for airfoil_name, grid_data in value.items():
            for grid_key, grid_val in grid_data.items():
                if airfoil_name not in self.widget_dict:
                    self.add_tab(airfoil_name)
                self.widget_dict[airfoil_name][grid_key].setValue(grid_val)
        airfoils_to_remove = list(set(self.widget_dict.keys()) - set(value.keys()))
        self.remove_tabs(airfoils_to_remove)

    def value(self):
        return {airfoil_name: {grid_key: grid_spin.value() for grid_key, grid_spin in grid_data.items()}
                for airfoil_name, grid_data in self.widget_dict.items()}


class XTRSWidget(QTabWidget):

    XTRSChanged = pyqtSignal()

    def __init__(self, parent, initial_mea: MEA = None):
        super().__init__(parent=parent)
        self.airfoil_names = [] if initial_mea is None else [a.name() for a in initial_mea.airfoils]
        self.widget_dict = {}
        self.grid_widgets = {}
        self.grid_widget = None
        self.grid_layout = None
        self.setValue()

    def onMEAChanged(self, mea: MEA or None):
        # Updated the widget based on a new MEA
        if mea:
            self.airfoil_names = [airfoil.name() for airfoil in mea.airfoils]
        else:
            self.airfoil_names = []
        self.setValue()

    def add_tab(self, name: str):
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self)
        self.grid_widget.setLayout(self.grid_layout)
        self.widget_dict[name] = {
            "XTRSupper": PymeadLabeledDoubleSpinBox(label="XTRSupper", minimum=0.0, maximum=1.0, value=1.0,
                                                    single_step=0.05),
            "XTRSlower": PymeadLabeledDoubleSpinBox(label="XTRSlower", minimum=0.0, maximum=1.0, value=1.0,
                                                    single_step=0.05)
        }
        for widget in self.widget_dict[name].values():
            row_count = self.grid_layout.rowCount()
            self.grid_layout.addWidget(widget.label, row_count, 0)
            self.grid_layout.addWidget(widget.widget, row_count, 1)
        self.grid_widgets[name] = self.grid_widget
        self.addTab(self.grid_widget, name)

    def remove_tab(self, name: str):
        self.removeTab(self.indexOf(self.grid_widgets[name]))
        self.grid_widgets.pop(name)
        self.widget_dict.pop(name)

    def remove_tabs(self, names: typing.List[str]):
        for name in names[::-1]:
            self.remove_tab(name)

    def setValue(self, value: dict = None):
        if isinstance(value, dict) and len(value["XTRSupper"]) == 0:
            return

        if value is None:
            for airfoil_name in self.airfoil_names:
                if airfoil_name in self.widget_dict:
                    continue
                self.add_tab(airfoil_name)

            airfoils_to_remove = list(set(self.widget_dict.keys()) - set(self.airfoil_names))
            self.remove_tabs(airfoils_to_remove)
            return

        for xtrs_key, xtrs_data in value.items():
            for airfoil_name, xtrs_val in xtrs_data.items():
                if airfoil_name not in self.widget_dict:
                    self.add_tab(airfoil_name)
                self.widget_dict[airfoil_name][xtrs_key].setValue(xtrs_val)
        airfoils_to_remove = list(set(self.widget_dict.keys()) - set(value["XTRSupper"].keys()))
        self.remove_tabs(airfoils_to_remove)

    def value(self):
        value = {"XTRSupper": {}, "XTRSlower": {}}
        for airfoil_name, airfoil_data in self.widget_dict.items():
            value["XTRSupper"][airfoil_name] = airfoil_data["XTRSupper"].value()
            value["XTRSlower"][airfoil_name] = airfoil_data["XTRSlower"].value()
        return value


class DesVarListSelectionWidget(PymeadDialogWidget2):
    def __init__(self, geo_col: GeometryCollection, initial_items: list = None, parent=None):
        self.geo_col = geo_col
        initial_items = [] if initial_items is None else initial_items
        super().__init__(parent=parent, initial_items=initial_items)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "main_list": QListWidget(self),
            "selected_list": QListWidget(self),
            "add_button": QPushButton("Add >>", self),
            "remove_button": QPushButton("<< Remove", self)
        }

        self.widget_dict["main_list"].addItems([
            k for k in self.geo_col.container()["desvar"].keys() if k not in kwargs["initial_items"]
        ])
        self.widget_dict["selected_list"].addItems(kwargs["initial_items"])
        self.widget_dict["main_list"].setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.widget_dict["selected_list"].setSelectionMode(QListWidget.SelectionMode.MultiSelection)

    def addWidgets(self, *args, **kwargs):
        self.lay.addWidget(self.widget_dict["main_list"], 0, 0, 4, 1)
        self.lay.addWidget(self.widget_dict["add_button"], 0, 1, 1, 1)
        self.lay.addWidget(self.widget_dict["remove_button"], 1, 1, 1, 1)
        self.lay.addWidget(self.widget_dict["selected_list"], 0, 2, 4, 1)

    def establishWidgetConnections(self):
        self.widget_dict["add_button"].clicked.connect(self.onAddPressed)
        self.widget_dict["remove_button"].clicked.connect(self.onRemovePressed)

    def onAddPressed(self):
        selected_items = self.widget_dict["main_list"].selectedItems()
        for item in selected_items:
            self.widget_dict["selected_list"].addItem(
                self.widget_dict["main_list"].takeItem(self.widget_dict["main_list"].row(item))
            )

    def onRemovePressed(self):
        selected_items = self.widget_dict["selected_list"].selectedItems()
        for item in selected_items:
            self.widget_dict["main_list"].addItem(
                self.widget_dict["selected_list"].takeItem(self.widget_dict["selected_list"].row(item))
            )

    def value(self) -> list:
        selected_list = self.widget_dict["selected_list"]
        return [selected_list.item(i).text() for i in range(selected_list.count())]

    def setValue(self, items: list):
        available_list = self.widget_dict["main_list"]
        available_items = [available_list.item(i).text() for i in range(available_list.count())]
        selected_list = self.widget_dict["selected_list"]
        selected_list.clear()
        for item in items:
            # Only move the item to the list on the right if it was found in the available list (on the left)
            if item not in available_items:
                continue

            selected_list.addItem(item)


class DesVarListSelectionDialog(PymeadDialog):
    def __init__(self, window_title: str, geo_col: GeometryCollection, initial_items: list = None, parent=None):
        theme = geo_col.gui_obj.themes[geo_col.gui_obj.current_theme]
        widget = DesVarListSelectionWidget(geo_col=geo_col, initial_items=initial_items)
        super().__init__(parent=parent, window_title=window_title, widget=widget, theme=theme,
                         minimum_width=400, minimum_height=500)

    def create_button_box(self):
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        return buttonBox


class DesVarParamListSelectionWidget(PymeadDialogWidget2):
    def __init__(self, geo_col: GeometryCollection, initial_items: list = None, parent=None):
        self.geo_col = geo_col
        initial_items = [] if initial_items is None else initial_items
        super().__init__(parent=parent, initial_items=initial_items)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "main_list": QListWidget(self),
            "selected_list": QListWidget(self),
            "add_button": QPushButton("Add >>", self),
            "remove_button": QPushButton("<< Remove", self)
        }

        items_to_add = [k for k in self.geo_col.container()["desvar"].keys() if k not in kwargs["initial_items"]]
        items_to_add.extend([k for k in self.geo_col.container()["params"].keys() if k not in kwargs["initial_items"]])
        self.widget_dict["main_list"].addItems(items_to_add)
        self.widget_dict["selected_list"].addItems(kwargs["initial_items"])
        self.widget_dict["main_list"].setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.widget_dict["selected_list"].setSelectionMode(QListWidget.SelectionMode.MultiSelection)

    def addWidgets(self, *args, **kwargs):
        self.lay.addWidget(self.widget_dict["main_list"], 0, 0, 4, 1)
        self.lay.addWidget(self.widget_dict["add_button"], 0, 1, 1, 1)
        self.lay.addWidget(self.widget_dict["remove_button"], 1, 1, 1, 1)
        self.lay.addWidget(self.widget_dict["selected_list"], 0, 2, 4, 1)

    def establishWidgetConnections(self):
        self.widget_dict["add_button"].clicked.connect(self.onAddPressed)
        self.widget_dict["remove_button"].clicked.connect(self.onRemovePressed)

    def onAddPressed(self):
        selected_items = self.widget_dict["main_list"].selectedItems()
        for item in selected_items:
            self.widget_dict["selected_list"].addItem(
                self.widget_dict["main_list"].takeItem(self.widget_dict["main_list"].row(item))
            )

    def onRemovePressed(self):
        selected_items = self.widget_dict["selected_list"].selectedItems()
        for item in selected_items:
            self.widget_dict["main_list"].addItem(
                self.widget_dict["selected_list"].takeItem(self.widget_dict["selected_list"].row(item))
            )

    def value(self) -> list:
        selected_list = self.widget_dict["selected_list"]
        return [selected_list.item(i).text() for i in range(selected_list.count())]

    def setValue(self, items: list):
        available_list = self.widget_dict["main_list"]
        available_items = [available_list.item(i).text() for i in range(available_list.count())]
        selected_list = self.widget_dict["selected_list"]
        selected_list.clear()
        for item in items:
            # Only move the item to the list on the right if it was found in the available list (on the left)
            if item not in available_items:
                continue

            selected_list.addItem(item)


class DesVarParamListSelectionDialog(PymeadDialog):
    def __init__(self, window_title: str, geo_col: GeometryCollection, initial_items: list = None, parent=None):
        theme = geo_col.gui_obj.themes[geo_col.gui_obj.current_theme]
        widget = DesVarParamListSelectionWidget(geo_col=geo_col, initial_items=initial_items)
        super().__init__(parent=parent, window_title=window_title, widget=widget, theme=theme,
                         minimum_width=400, minimum_height=500)

    def create_button_box(self):
        buttonBox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, self
        )
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        return buttonBox


class ADWidget(QTabWidget):

    ADChanged = pyqtSignal()
    sigXCDELHParamChanged = pyqtSignal(str)

    def __init__(self, parent, param_list: typing.List[str], geo_col: GeometryCollection, number_AD: int):
        super().__init__(parent=parent)
        self.widget_dict = {}
        self.grid_widgets = {}
        self.grid_widget = None
        self.grid_layout = None
        self.param_list = param_list
        self.param_list.insert(0, "")
        self.geo_col = geo_col
        self.number_AD = number_AD
        self.setValue()

    def add_tab(self, name: str):
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self)
        self.grid_widget.setLayout(self.grid_layout)
        self.widget_dict[name] = {
            "ISDELH": PymeadLabeledSpinBox(label="AD Side", minimum=1, maximum=100, value=1),
            "XCDELH": PymeadLabeledDoubleSpinBox(label="AD X-Location", minimum=0.0, maximum=1.0, value=0.1,
                                                 single_step=0.05, decimals=8),
            "XCDELH-Param": PymeadLabeledComboBox(label="AD X-Location Param", items=self.param_list),
            "PTRHIN": PymeadLabeledDoubleSpinBox(label="AD Total Pres. Ratio", minimum=1.0, maximum=np.inf,
                                                 value=1.1, single_step=0.01, decimals=6),
            "ETAH": PymeadLabeledDoubleSpinBox(label="AD Thermal Efficiency", minimum=0.0, maximum=1.0,
                                               value=0.95, single_step=0.01),
            "PTRHIN-DesVar": PymeadLabeledListWidget(
                label="FPR Design Variables",
                tool_tip="Design variables used to define the fan pressure ratio at each point\nin the "
                         "multipoint stencil. Disabled if multipoint is not active."
            ),
            "PTRHIN-DVSelect": PymeadLabeledPushButton(
                label="FPR DV Selector",
                text="Select",
                tool_tip="Design variables used to define the fan pressure ratio at each point\nin the "
                         "multipoint stencil. Disabled if multipoint is not active."
            )
        }

        # Add connections
        self.widget_dict[name]["XCDELH-Param"].sigValueChanged.connect(partial(self.param_changed, name))
        self.widget_dict[name]["PTRHIN-DVSelect"].widget.clicked.connect(self.openSelector)

        for widget in self.widget_dict[name].values():
            row_count = self.grid_layout.rowCount()
            self.grid_layout.addWidget(widget.label, row_count, 0)
            self.grid_layout.addWidget(widget.widget, row_count, 1)
        self.grid_widgets[name] = self.grid_widget
        self.addTab(self.grid_widget, name)

    def openSelector(self):
        """
        Opens a selection widget that allows the user to safely set the list of fan pressure ratio design variables.
        """
        tab_name = self.tabText(self.currentIndex())
        dialog = DesVarListSelectionDialog(
            window_title="Fan Pressure Ratio Design Variables", geo_col=self.geo_col, parent=self,
            initial_items=self.widget_dict[tab_name]["PTRHIN-DesVar"].value()
        )
        old_dialog_value = dialog.value()
        if dialog.exec():
            new_dialog_value = dialog.value()
            self.widget_dict[tab_name]["PTRHIN-DesVar"].setValue(new_dialog_value)
            for dv_name in old_dialog_value:
                if dv_name in new_dialog_value:
                    continue
                self.geo_col.container()["desvar"][dv_name].assignable = True
            for dv_name in new_dialog_value:
                if dv_name in old_dialog_value:
                    continue
                self.geo_col.container()["desvar"][dv_name].assignable = False

    def param_changed(self, ad_idx: str, param_name: str):
        if param_name == "":
            self.widget_dict[ad_idx]["XCDELH"].setReadOnly(False)
            return

        sub_container = "params" if "DV" not in param_name else "desvar"
        param_name = param_name.strip(" (DV)")
        param = self.geo_col.container()[sub_container][param_name]

        if param.value() > 1.0:
            raise ValueError(f"Parameter value ({param.value()}) is greater than the maximum allowable AD x/c "
                             f"location (1.0)")
        if param.value() < 0.0:
            raise ValueError(f"Parameter value ({param.value()}) is less than the minimum allowable AD x/c "
                             f"location (0.0)")

        self.widget_dict[ad_idx]["XCDELH"].setValue(param.value())
        self.widget_dict[ad_idx]["XCDELH"].setReadOnly(True)

    def remove_tab(self, name: str):
        self.removeTab(self.indexOf(self.grid_widgets[name]))
        self.grid_widgets.pop(name)
        self.widget_dict.pop(name)

    def remove_tabs(self, names: typing.List[str]):
        for name in names[::-1]:
            self.remove_tab(name)

    def numberADChanged(self, new_number: int):
        self.number_AD = new_number
        self.setValue()

    def setValue(self, value: dict = None):
        if isinstance(value, dict) and len(value) == 0:
            return

        if value is None:
            for ad_idx in range(self.number_AD):
                if str(ad_idx + 1) in self.widget_dict:
                    continue
                self.add_tab(str(ad_idx + 1))

            ads_to_remove = list(set(self.widget_dict.keys()) - set([str(idx + 1) for idx in range(self.number_AD)]))
            self.remove_tabs(ads_to_remove)
            return

        for ad_name, ad_data in value.items():
            for ad_key, ad_val in ad_data.items():
                if ad_name not in self.widget_dict:
                    self.add_tab(ad_name)
                self.widget_dict[ad_name][ad_key].setValue(ad_val)
        ads_to_remove = list(set(self.widget_dict.keys()) - set(value.keys()))
        self.remove_tabs(ads_to_remove)

    def value(self):
        return {ad_idx: {ad_key: ad_widget.value() for ad_key, ad_widget in ad_data.items()}
                for ad_idx, ad_data in self.widget_dict.items()}

    def setAllActive(self, active: int):
        for ad_idx, widget_set in self.widget_dict.items():
            for key, widget in widget_set.items():
                widget.setReadOnly(not active)


class SingleAirfoilInviscidDialog(QDialog):
    def __init__(self, items: List[tuple], a_list: list, parent=None):
        super().__init__(parent)

        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
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

    def valuesFromWidgets(self):
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
        buttonBox.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        buttonBox.addButton(QDialogButtonBox.StandardButton.Cancel)
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
        buttonBox.addButton("Run", QDialogButtonBox.ButtonRole.AcceptRole)
        buttonBox.addButton(QDialogButtonBox.StandardButton.Cancel)
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
        file_dialog.setFileMode(QFileDialog.FileMode.Directory)
        if file_dialog.exec():
            line_edit.setText(file_dialog.selectedFiles()[0])

    def enable_disable_from_checkbox(self, key: str):
        widget = self.widget_dict[key]['widget']
        if 'widgets_to_enable' in self.inputs[key].keys() and widget.isChecked() or (
                'widgets_to_disable' in self.inputs[key].keys() and not widget.isChecked()
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
        elif 'widgets_to_enable' in self.inputs[key].keys() and not widget.isChecked() or (
                'widgets_to_disable' in self.inputs[key].keys() and widget.isChecked()
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
                self.inputs[key]['active_checkbox'] = widget.isChecked()
            else:
                self.inputs[key]['state'] = widget.isChecked()

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
                widget.setValue(v['value'])
                widget.valueChanged.connect(partial(self.dict_connection, widget, k))
            if 'state' in v.keys():
                widget.setChecked(v['state'])
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
                checkbox.setChecked(v['active_checkbox'])
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
            if k == 'mea_dir' and self.widget_dict[k]['use_current_mea']['widget'].isChecked():
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

    def valuesFromWidgets(self):
        return self.inputs


class DownsamplingGraph:
    def __init__(self, theme: dict, size: tuple = (1000, 300), grid: bool = False):
        pg.setConfigOptions(antialias=True)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)

        self.v = self.w.addPlot()
        self.v.setMenuEnabled(False)
        self.v.setAspectLocked(True)
        self.v.showGrid(x=grid, y=grid)

        # Set the formatting for the graph based on the current theme
        self.set_formatting(theme=theme)

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)

    def set_formatting(self, theme: dict):
        self.w.setBackground(theme["graph-background-color"])
        label_font = f"{get_setting('axis-label-point-size')}pt {get_setting('axis-label-font-family')}"
        self.v.setLabel(axis="bottom", text=f"x/c", font=label_font,
                           color=theme["main-color"])
        self.v.setLabel(axis="left", text=f"y/c", font=label_font, color=theme["main-color"])
        tick_font = QFont(get_setting("axis-tick-font-family"), get_setting("axis-tick-point-size"))
        self.v.getAxis("bottom").setTickFont(tick_font)
        self.v.getAxis("left").setTickFont(tick_font)
        self.v.getAxis("bottom").setTextPen(theme["main-color"])
        self.v.getAxis("left").setTextPen(theme["main-color"])


class DownsamplingPreviewDialog(PymeadDialog):
    def __init__(self, theme: dict, gui_obj, mea_name: str,
                 max_airfoil_points: int = None, curvature_exp: float = 2.0,
                 parent: QWidget or None = None):

        graph = DownsamplingGraph(theme=theme, grid=gui_obj.main_icon_toolbar.buttons["grid"]["button"].isChecked())

        super().__init__(parent=parent, window_title="Airfoil Coordinates Preview", widget=graph.w, theme=theme,
                         minimum_width=800, minimum_height=400)

        # Make a copy of the MEA
        geo_col = GeometryCollection.set_from_dict_rep(gui_obj.geo_col.get_dict_rep())
        if len(geo_col.container()["mea"]) == 0:
            raise ValueError("No multi-element airfoils found in the geometry collection.")
        mea = geo_col.container()["mea"][mea_name]
        if not isinstance(mea, MEA):
            raise TypeError(f"Generated mea was of type {type(mea)} instead of type pymead.core.mea.MEA")

        # Update the curves, using downsampling if specified
        coords_list = mea.get_coords_list(max_airfoil_points, curvature_exp)

        for coords in coords_list:
            graph.v.plot(
                x=coords[:, 0], y=coords[:, 1], symbol="o",
                symbolPen=pg.mkPen(color="indianred"), symbolBrush=pg.mkBrush(color="indianred"),
                pen=pg.mkPen(color="cornflowerblue", width=2)
            )


class MSETDialogWidget(PymeadDialogWidget2):

    sigMEAChanged = pyqtSignal(object)

    def __init__(self, geo_col: GeometryCollection, theme: dict, parent=None):
        self.geo_col = geo_col
        self.theme = theme
        super().__init__(parent=parent, label_column_split="timeout")

    def initializeWidgets(self, label_column_split: str):
        initial_mea_names = [k for k in self.geo_col.container()["mea"].keys()]
        initial_mea = None if len(initial_mea_names) == 0 else self.geo_col.container()["mea"][initial_mea_names[0]]
        self.widget_dict = {
            "mea": PymeadLabeledComboBox(label="MEA", items=initial_mea_names),
            "grid_bounds": GridBounds(self),
            "airfoil_side_points": PymeadLabeledSpinBox(label="Airfoil Side Points",
                                                        minimum=1, maximum=999999, value=180),
            "exp_side_points": PymeadLabeledDoubleSpinBox(label="Side Points Exponent",
                                                          minimum=0.0, maximum=np.inf, value=0.9),
            "inlet_pts_left_stream": PymeadLabeledSpinBox(label="Inlet Points Left",
                                                          minimum=1, maximum=999999, value=41),
            "outlet_pts_right_stream": PymeadLabeledSpinBox(label="Outlet Points Right",
                                                            minimum=1, maximum=999999, value=41),
            "num_streams_top": PymeadLabeledSpinBox(label="Number Top Streamlines",
                                                    minimum=1, maximum=999999, value=17),
            "num_streams_bot": PymeadLabeledSpinBox(label="Number Bottom Streamlines",
                                                    minimum=1, maximum=999999, value=23),
            "max_streams_between": PymeadLabeledSpinBox(label="Max Streamlines Between",
                                                        minimum=1, maximum=999999, value=15),
            "elliptic_param": PymeadLabeledDoubleSpinBox(label="Elliptic Parameter",
                                                         minimum=0.0, maximum=np.inf, value=1.3),
            "stag_pt_aspect_ratio": PymeadLabeledDoubleSpinBox(label="Stag. Pt. Aspect Ratio",
                                                               minimum=0.0, maximum=np.inf, value=2.5),
            "x_spacing_param": PymeadLabeledDoubleSpinBox(label="X-Spacing Parameter",
                                                          minimum=0.0, maximum=np.inf, value=0.85),
            "alf0_stream_gen": PymeadLabeledDoubleSpinBox(label="Streamline Gen. Alpha",
                                                          minimum=-np.inf, maximum=np.inf, value=0.0),
            "timeout": PymeadLabeledDoubleSpinBox(label="MSET Timeout", minimum=0.0, maximum=np.inf,
                                                  value=10.0),
            "multi_airfoil_grid": MSETMultiGridWidget(self, initial_mea=initial_mea),
            "airfoil_analysis_dir": PymeadLabeledLineEdit(label="Analysis Directory", push_label="Choose folder",
                                                          text=tempfile.gettempdir()),
            "airfoil_coord_file_name": PymeadLabeledLineEdit(label="Airfoil Coord. Filename", text="default_airfoil"),
            "save_as_mses_settings": PymeadLabeledPushButton(label="Save As", text="Save MSES Settings As..."),
            "load_mses_settings": PymeadLabeledPushButton(label="Load", text="Load MSES Settings File"),
            "use_downsampling": PymeadLabeledCheckbox(
                label="Use Downsampling?", push_label="Preview",
                tool_tip="Downsample the airfoil coordinates based on the curvature"),
            "downsampling_max_pts": PymeadLabeledSpinBox(
                label="Max Downsampling Points", minimum=20, maximum=9999, value=200,
                tool_tip="Maximum number of airfoil coordinates allowed per airfoil"),
            "downsampling_curve_exp": PymeadLabeledDoubleSpinBox(
                label="Downsampling Curvature Exponent", minimum=0.0001, maximum=9999., value=2.0,
                tool_tip="Importance of curvature in the downsampling scheme.\nValues close to 0 place high emphasis "
                         "on curvature,\nwhile values close to positive infinity place no emphasis\non curvature and "
                         "leave the parameter\nvector effectively uniformly spaced")
        }

    def addWidgets(self, **kwargs):
        # Add all the widgets
        row_count = 0
        column = 0
        for widget_name, widget in self.widget_dict.items():
            if widget_name == kwargs["label_column_split"]:
                column = 4
                row_count = 0
            if widget_name == "multi_airfoil_grid":
                row_span = 7
                col_span = 3
            elif widget_name == "grid_bounds":
                row_span = 3
                col_span = 4
            else:
                row_span = 1
                col_span = 2 if widget.push is None else 1

            if widget_name in ["multi_airfoil_grid", "grid_bounds"]:
                self.lay.addWidget(widget, row_count, column, row_span, col_span)
            else:
                self.lay.addWidget(widget.label, row_count, column, 1, 1)
                self.lay.addWidget(widget.widget, row_count, column + 1, row_span, col_span)

                if widget.push is not None:
                    self.lay.addWidget(widget.push, row_count, column + 2, row_span, col_span)

            row_count += row_span

    def establishWidgetConnections(self):
        # Connect the MEA combobox to the "MEA changed" signal
        self.widget_dict["mea"].widget.currentTextChanged.connect(self.onMEAChanged)
        self.sigMEAChanged.connect(self.widget_dict["multi_airfoil_grid"].onMEAChanged)
        # Show a preview of the downsampling when the button is pushed
        self.widget_dict["use_downsampling"].push.clicked.connect(self.showAirfoilCoordinatesPreview)
        # Connect the airfoil analysis directory button to the choose directory function
        self.widget_dict["airfoil_analysis_dir"].push.clicked.connect(
            partial(select_directory, self, line_edit=self.widget_dict["airfoil_analysis_dir"].widget))
        # Connect the load and save settings buttons
        self.widget_dict["load_mses_settings"].widget.clicked.connect(self.loadMSESSuiteSettings)
        self.widget_dict["save_as_mses_settings"].widget.clicked.connect(self.saveasMSESSuiteSettings)

    def onMEAChanged(self, mea_name: str):
        if mea_name:
            self.sigMEAChanged.emit(self.geo_col.container()["mea"][mea_name])
        else:
            self.sigMEAChanged.emit(None)

    def setValue(self, d: dict):
        for d_name, d_value in d.items():
            try:
                self.widget_dict[d_name].setValue(d_value)
            except KeyError:
                pass

    def value(self) -> dict:
        return {k: v.value() for k, v in self.widget_dict.items()}

    def saveasMSESSuiteSettings(self):
        all_inputs = get_parent(self, 2).value()
        mses_inputs = {k: v for k, v in all_inputs.items() if k in ["MSET", "MSES", "MPLOT", "MPOLAR"]}
        json_dialog = select_json_file(parent=self)
        if json_dialog.exec():
            input_filename = json_dialog.selectedFiles()[0]
            if not os.path.splitext(input_filename)[-1] == '.json':
                input_filename = input_filename + '.json'
            save_data(mses_inputs, input_filename)
            get_parent(self, 3).setStatusTip(f"Saved MSES settings to {input_filename}")

    def loadMSESSuiteSettings(self):
        override_inputs = get_parent(self, 2).value()
        load_dialog = select_existing_json_file(parent=self)
        if load_dialog.exec():
            load_file = load_dialog.selectedFiles()[0]
            new_inputs = load_data(load_file)
            for k, v in new_inputs.items():
                override_inputs[k] = v
            current_window_title = get_parent(self, 3).windowTitle()
            window_title_split = current_window_title.split(" - ")
            new_base_window_title = window_title_split[0]
            get_parent(self, 3).setWindowTitle(f"{new_base_window_title} - {os.path.split(load_file)[-1]}")
            get_parent(self, 2).setValue(override_inputs)  # Overrides the inputs for the whole PymeadDialogVTabWidget
            get_parent(self, 2).setStatusTip(f"Loaded {load_file}")

    def showAirfoilCoordinatesPreview(self):
        mset_settings = self.value()
        gui_obj = self.geo_col.gui_obj
        try:
            preview_dialog = DownsamplingPreviewDialog(
                theme=self.theme, mea_name=self.widget_dict["mea"].widget.currentText(),
                gui_obj=gui_obj,
                max_airfoil_points=mset_settings["downsampling_max_pts"] if bool(mset_settings["use_downsampling"]) else None,
                curvature_exp=mset_settings["downsampling_curve_exp"],
                parent=self
            )
        except ValueError as e:
            self.geo_col.gui_obj.disp_message_box(str(e))
            return

        preview_dialog.exec()


class PanelDialogWidget(PymeadDialogWidget2):
    def __init__(self, settings_override: dict, parent=None):
        super().__init__(parent=parent)
        if settings_override:
            self.setValue(settings_override)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "alfa": PymeadLabeledDoubleSpinBox(label="\u03b1 (\u00b0)", minimum=-np.inf, maximum=np.inf,
                                               value=0.0, decimals=8, single_step=1.0,
                                               tool_tip="Angle of attack (if the geometry is already at an angle"
                                                        "of attack, this value will be added on)")
        }


class MSESDialogWidget2(PymeadDialogWidget2):
    def __init__(self, geo_col: GeometryCollection, parent=None):
        self.geo_col = geo_col
        super().__init__(parent=parent, label_column_split="timeout")

    def initializeWidgets(self, label_column_split: str):

        initial_mea_names = [k for k in self.geo_col.container()["mea"].keys()]
        initial_mea = None if len(initial_mea_names) == 0 else self.geo_col.container()["mea"][initial_mea_names[0]]

        param_list = [param for param in self.geo_col.container()["params"]]
        dv_list = [dv + " (DV)" for dv in self.geo_col.container()["desvar"]]

        self.widget_dict = {
            "viscous_flag": PymeadLabeledCheckbox(label="Viscosity On?", initial_state=2),
            "spec_Re": PymeadLabeledCheckbox(label="Specify Reynolds Number?",
                                             tool_tip="Whether to directly specify the Reynolds number instead of "
                                                      "specifying the thermodynamic state",
                                             initial_state=0),
            "REYNIN": PymeadLabeledDoubleSpinBox(label="Reynolds Number",
                                                 tool_tip="Can only modify this value if 'Specify Reynolds Number?' "
                                                          "is checked",
                                                 minimum=0.0, maximum=np.inf, decimals=16,
                                                 value=15492705.8044970352202654, single_step=10000.0, read_only=True),
            "MACHIN": PymeadLabeledDoubleSpinBox(label="Mach Number", minimum=0.0, maximum=np.inf, value=0.7,
                                                 single_step=0.01, decimals=16),
            "spec_P_T_rho": PymeadLabeledComboBox(label="Specify Flow Variables",
                                                  tool_tip="Which pair of thermodynamic state variables to specify",
                                                  items=["Specify Pressure, Temperature", "Specify Pressure, Density",
                                                         "Specify Temperature, Density"]),
            "P": PymeadLabeledDoubleSpinBox(label="Pressure (Pa)", minimum=0.0, maximum=np.inf, decimals=16,
                                            value=101325.0, single_step=1000.0, read_only=False),
            "T": PymeadLabeledDoubleSpinBox(label="Temperature (K)", minimum=0.001, maximum=np.inf, decimals=16,
                                            value=300.0, single_step=10.0, read_only=False),
            "rho": PymeadLabeledDoubleSpinBox(label="Density (kg/m^3)", minimum=0.001, maximum=np.inf, decimals=16,
                                              value=1.1768292682926829, single_step=0.1, read_only=False),
            "gam": PymeadLabeledDoubleSpinBox(label="Specific Heat Ratio", minimum=0.001, maximum=np.inf, decimals=16,
                                              value=1.4, single_step=0.01),
            "L": PymeadLabeledDoubleSpinBox(label="Length Scale (m)", minimum=0.0, maximum=np.inf, value=1.0,
                                            single_step=0.1, decimals=16),
            "R": PymeadLabeledDoubleSpinBox(label="Gas Constant (J/(kg*K))", minimum=0.001, maximum=np.inf, decimals=16,
                                            single_step=1.0, value=287.05),
            "spec_alfa_Cl": PymeadLabeledComboBox(label="Specify Alpha/Cl", items=["Specify Angle of Attack",
                                                                                   "Specify Lift Coefficient"]),
            "ALFAIN": PymeadLabeledDoubleSpinSelector(self.geo_col, parent=self,
                                                      label="Angle of Attack (deg)", minimum=-np.inf,
                                                      maximum=np.inf,
                                                      decimals=16, value=0.0, single_step=1.0, read_only=False),
            "CLIFIN": PymeadLabeledDoubleSpinBox(label="Lift Coefficient", minimum=-np.inf, maximum=np.inf, decimals=16,
                                                 value=0.0, single_step=0.1, read_only=True),
            "ISMOM": PymeadLabeledComboBox(label="Isentropic/Momentum", tool_tip="The set of MSES equations used to"
                                                                                 " solve the flow problem",
                                           items=["S-momentum equation", "isentropic condition",
                                                  "S-momentum equation, isentropic @ LE",
                                                  "isentropic condition, S-mom. where diss. active"],
                                           current_item="S-momentum equation, isentropic @ LE"),
            "IFFBC": PymeadLabeledComboBox(label="Far-Field Boundary",
                                           items=["solid wall airfoil far-field BCs",
                                                  "vortex+source+doublet airfoil far-field BCs",
                                                  "freestream pressure airfoil far-field BCs",
                                                  "supersonic wave freestream BCs",
                                                  "supersonic solid wall far-field BCs"],
                                           current_item="vortex+source+doublet airfoil far-field BCs"),
            "ACRIT": PymeadLabeledDoubleSpinBox(label="Crit. Amp. Factor",
                                                tool_tip="Critical amplification factor used to determine the boundary"
                                                         " layer transition point",
                                                minimum=0.0, maximum=np.inf, decimals=4, value=9.0),
            "MCRIT": PymeadLabeledDoubleSpinBox(label="Critical Mach Number", minimum=0.0, maximum=1.0,
                                                decimals=4, value=0.95, single_step=0.01),
            "MUCON": PymeadLabeledDoubleSpinBox(label="Artificial Dissipation",
                                                tool_tip="Values close to 1.0 are less stable but generate 'crisper'"
                                                         " shocks, while values much larger than 1.0 are more stable"
                                                         " but generate smeared shocks",
                                                minimum=1.0, maximum=np.inf,
                                                decimals=4, value=1.05, single_step=0.01),
            "timeout": PymeadLabeledDoubleSpinBox(label="Timeout", minimum=0.0, maximum=np.inf, value=15.0),
            "iter": PymeadLabeledSpinBox(label="Maximum Iterations", minimum=1, maximum=1000000, value=100),
            "verbose": PymeadLabeledCheckbox(label="Verbose?", initial_state=0),
            "xtrs": XTRSWidget(parent=self, initial_mea=initial_mea),
            "AD_active": PymeadLabeledCheckbox(label="Actuator Disks Active",
                                               tool_tip="If this box is not checked, the actuator disk parameters"
                                                        " will not be used",
                                               initial_state=0),
            "AD_number": PymeadLabeledSpinBox(label="Num. Actuator Disks", minimum=0, maximum=5, value=0,
                                              read_only=True, tool_tip="'Actuator disks active' must be checked "
                                                                       "to enable modification of the number of AD's"),
            "AD": ADWidget(parent=self, param_list=param_list + dv_list, geo_col=self.geo_col, number_AD=0)
        }

    def addWidgets(self, *args, **kwargs):
        # Add all the widgets
        row_count = 0
        column = 0
        for widget_name, widget in self.widget_dict.items():
            if widget_name == kwargs["label_column_split"]:
                column = 4
                row_count = 0
            if widget_name == "AD":
                row_span = 5
                col_span = 4
            elif widget_name == "xtrs":
                row_span = 3
                col_span = 4
            else:
                row_span = 1
                col_span = 2 if widget.push is None else 1

            if widget_name in ["AD", "xtrs"]:
                self.lay.addWidget(widget, row_count, column, row_span, col_span)
            else:
                self.lay.addWidget(widget.label, row_count, column, 1, 1)
                self.lay.addWidget(widget.widget, row_count, column + 1, row_span, col_span)

                if widget.push is not None:
                    self.lay.addWidget(widget.push, row_count, column + 2, row_span, col_span)

            row_count += row_span

    def establishWidgetConnections(self):
        # Create the required connections
        self.widget_dict["AD_active"].sigValueChanged.connect(self.widget_dict["AD_number"].setActive)
        self.widget_dict["AD_active"].sigValueChanged.connect(self.widget_dict["AD"].setAllActive)
        self.widget_dict["AD_number"].sigValueChanged.connect(self.widget_dict["AD"].numberADChanged)
        self.widget_dict["spec_P_T_rho"].sigValueChanged.connect(self.spec_P_T_rho_changed)
        for thermo_var in ("P", "T", "rho", "gam", "R", "MACHIN", "L"):
            self.widget_dict[thermo_var].sigValueChanged.connect(self.recalculateThermoState)
        self.widget_dict["spec_Re"].sigValueChanged.connect(self.spec_Re_changed)
        self.widget_dict["spec_alfa_Cl"].sigValueChanged.connect(self.spec_alfa_Cl_changed)

    def spec_P_T_rho_changed(self, spec_P_T_rho: str):
        if "Pressure" in spec_P_T_rho:
            self.widget_dict["P"].setReadOnly(False)
        else:
            self.widget_dict["P"].setReadOnly(True)
        if "Temperature" in spec_P_T_rho:
            self.widget_dict["T"].setReadOnly(False)
        else:
            self.widget_dict["T"].setReadOnly(True)
        if "Density" in spec_P_T_rho:
            self.widget_dict["rho"].setReadOnly(False)
        else:
            self.widget_dict["rho"].setReadOnly(True)

    def spec_Re_changed(self, state: int):
        if state:
            for thermo_var in ("P", "T", "rho", "gam", "R", "L", "spec_P_T_rho"):
                self.widget_dict[thermo_var].setReadOnly(True)
            self.widget_dict["REYNIN"].setReadOnly(False)
        else:
            for thermo_var in ("P", "T", "rho", "gam", "R", "L", "spec_P_T_rho"):
                self.widget_dict[thermo_var].setReadOnly(False)
            self.widget_dict["REYNIN"].setReadOnly(True)
            self.spec_P_T_rho_changed(self.widget_dict["spec_P_T_rho"].value())
            self.recalculateThermoState(None)

    def spec_alfa_Cl_changed(self, alfa_Cl: str):
        if "Angle" in alfa_Cl:
            self.widget_dict["ALFAIN"].setReadOnly(False)
            self.widget_dict["CLIFIN"].setReadOnly(True)
        else:
            self.widget_dict["ALFAIN"].setReadOnly(True)
            self.widget_dict["CLIFIN"].setReadOnly(False)

    def recalculateThermoState(self, _):
        if self.widget_dict["spec_Re"].value():
            return
        spec_P_T_rho = self.widget_dict["spec_P_T_rho"].value()
        R = self.widget_dict["R"].value()
        gam = self.widget_dict["gam"].value()
        if "Pressure" in spec_P_T_rho and "Temperature" in spec_P_T_rho:
            self.widget_dict["rho"].setValue(self.widget_dict["P"].value() / (R * self.widget_dict["T"].value()))
        elif "Pressure" in spec_P_T_rho and "Density" in spec_P_T_rho:
            self.widget_dict["T"].setValue(self.widget_dict["P"].value() / (R * self.widget_dict["rho"].value()))
        else:
            self.widget_dict["P"].setValue(self.widget_dict["rho"].value() * R * self.widget_dict["T"].value())
        nu = viscosity_calculator(self.widget_dict["T"].value(), rho=self.widget_dict["rho"].value())
        a = np.sqrt(gam * R * self.widget_dict["T"].value())
        V = self.widget_dict["MACHIN"].value() * a
        self.widget_dict['REYNIN'].setValue(V * self.widget_dict["L"].value() / nu)

    def setValue(self, d: dict):
        for d_name, d_value in d.items():
            try:
                self.widget_dict[d_name].setValue(d_value)
            except KeyError:
                pass

    def value(self) -> dict:
        return {k: v.value() for k, v in self.widget_dict.items()}


class MPLOTDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, "mplot_settings.json"))

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class MPOLARDialogWidget(PymeadDialogWidget2):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "timeout": PymeadLabeledDoubleSpinBox(label="Timeout (s)",
                                                  tool_tip="Maximum time allotted for the entire polar run",
                                                  minimum=0.0, maximum=np.inf, value=300.0, decimals=1,
                                                  single_step=10.0),
            "polar_mode": PymeadLabeledComboBox(label="Polar Mode",
                                                tool_tip="Specify whether to run a polar or which mode to run in",
                                                items=["No Polar Analysis",
                                                       "Alpha Sweep from Data File",
                                                       "Alpha Sweep from Start/Stop/Inc"],
                                                current_item="No Polar Analysis"),
            "alfa_array": PymeadLabeledLineEdit(label="Angle of Attack Sweep",
                                                tool_tip="Specify a text or dat file with a single column specifying "
                                                         "the angles of attack to run in degrees",
                                                push_label="Choose file"),
            "alfa_start": PymeadLabeledDoubleSpinBox(label="\u03b1 Sweep Start (deg)",
                                                     tool_tip="Starting angle of attack for the sweep. Ignored unless"
                                                              " 'Polar Mode' is 'Alpha Sweep from Start/Stop/Inc'",
                                                     minimum=-180.0, maximum=180.0, value=-2.0, decimals=2,
                                                     single_step=1.0),
            "alfa_end": PymeadLabeledDoubleSpinBox(label="\u03b1 Sweep End (deg)",
                                                   tool_tip="Final angle of attack for the sweep. Ignored unless"
                                                            " 'Polar Mode' is 'Alpha Sweep from Start/Stop/Inc'",
                                                   minimum=-180.0, maximum=180.0, value=14.0, decimals=2,
                                                   single_step=1.0),
            "alfa_inc": PymeadLabeledDoubleSpinBox(label="\u03b1 Increment (deg)",
                                                   tool_tip="Angle of attack increment for the sweep. Ignored unless"
                                                            " 'Polar Mode' is 'Alpha Sweep from Start/Stop/Inc'",
                                                   minimum=0.01, maximum=10.0, value=1.0, decimals=2,
                                                   single_step=0.1)
        }

    def establishWidgetConnections(self):
        self.widget_dict["alfa_array"].push.clicked.connect(
            partial(select_data_file, self, line_edit=self.widget_dict["alfa_array"].widget))


class OptConstraintsMultiCheckbox(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.widget_dict = {
            "min_val_of_max_thickness": QCheckBox("Minimum Thickness", self),
            "min_radius_curvature": QCheckBox("Min. Radius of Curvature", self),
            "thickness_at_points": QCheckBox("Thickness Distribution", self),
            "min_area": QCheckBox("Minimum Area", self),
            "internal_geometry": QCheckBox("Internal Geometry", self),
        }
        self.lay = QGridLayout()

        grid_mapping = {
            0: [0, 0],
            1: [0, 1],
            2: [0, 2],
            3: [1, 0],
            4: [1, 1],
        }

        for idx, widget in enumerate(self.widget_dict.values()):
            self.lay.addWidget(widget, grid_mapping[idx][0], grid_mapping[idx][1])
        self.setLayout(self.lay)
        self.setTitle("Active Constraints")
        self.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed))

    def value(self) -> dict:
        return {k: v.isChecked() for k, v in self.widget_dict.items()}

    def setValue(self, d: dict):
        for k, v in d.items():
            self.widget_dict[k].setChecked(v)


class VisualizeButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__("Visualize Constraints", parent=parent)
        self.setStyleSheet("color: #1ed8e6; font-size: 18px;")
        # self.setMaximumWidth(300)


class OptConstraintsDialogWidget(PymeadDialogWidget2):
    def __init__(self, geo_col: GeometryCollection, airfoil_name: str, theme: dict, grid: bool, parent=None):
        self.grid = grid
        self.theme = theme
        self.airfoil_name = airfoil_name
        self.geo_col = geo_col
        self.sub_dialog = None
        super().__init__(parent=parent)
        self.graph = None

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "active_constraints": OptConstraintsMultiCheckbox(self),
            "visualize": VisualizeButton(self),
            "min_val_of_max_thickness": PymeadLabeledDoubleSpinBox(
                label="Minimum Thickness", decimals=16, minimum=0.0, maximum=np.inf, single_step=0.01, value=0.1
            ),
            "min_val_of_max_thickness_policy": PymeadLabeledComboBox(
                label="Min. Thickness Policy", items=["Non-Dimensional", "Dimensional"], current_item="Non-Dimensional"
            ),
            "min_radius_curvature": PymeadLabeledDoubleSpinBox(
                label="Min. Radius of Curvature", decimals=16, minimum=0.0, maximum=np.inf, single_step=0.001,
                value=0.004
            ),
            "min_radius_curvature_policy": PymeadLabeledComboBox(
                label="Min. Radius of Curvature Policy", items=["Non-Dimensional", "Dimensional"],
                current_item="Non-Dimensional"
            ),
            "thickness_at_points": PymeadLabeledLineEdit(label="Thickness Dist. File", push_label="Select File"),
            "thickness_at_points_policy": PymeadLabeledComboBox(
                label="Thickness Distribution Policy",
                items=[
                    "Non-Dimensional, Chord-Perpendicular",
                    "Dimensional, Chord-Perpendicular",
                    "Non-Dimensional, Vertical",
                    "Dimensional, Vertical"
                ]
            ),
            "min_area": PymeadLabeledDoubleSpinBox(
                label="Minimum Area", decimals=16, minimum=0.0, maximum=np.inf, single_step=0.01, value=0.04
            ),
            "min_area_policy": PymeadLabeledComboBox(
                label="Minimum Area Policy", items=["Non-Dimensional", "Dimensional"], current_item="Non-Dimensional"
            ),
            "internal_geometry": PymeadLabeledLineEdit(
                label="Internal Geometry File", push_label="Select File",
                tool_tip="Ensure that the given convex hull of x-y coordinates (preferably closed) "
                         "fits inside the airfoil"
            ),
            "internal_geometry_policy": PymeadLabeledComboBox(
                label="Int. Geometry Timing",
                items=[
                    "Rotate/Translate/Scale w/ Airfoil, Eval. Before Aerodynamic Evaluation",
                    "Rotate/Translate w/ Airfoil, Eval. Before Aerodynamic Evaluation",
                    "Absolute, Eval. Before Aerodynamic Evaluation",
                    "Absolute, Eval. After Aerodynamic Evaluation",
                ],
                current_item="Rotate/Translate/Scale w/ Airfoil, Eval. Before Aerodynamic Evaluation",
                tool_tip="The timing of the internal geometry fit check can be important\ndepending on whether the "
                         "internal geometry should be rotated\nwith the airfoil angle of attack. If the internal "
                         "geometry\ncan move with the airfoil angle of attack, choose 'Before Aerodynamic Evaluation' "
                         "(faster).\nOtherwise, choose 'After Aerodynamic Evaluation' (slower)."
            )
        }

        # Hide all the constraint widgets on initialization
        for k, v in self.widget_dict.items():
            if k in ["active_constraints", "visualize"]:
                continue
            v.setShown(False)

        # Set the tool tip from each of the constraint types to the checkboxes
        for k, v in self.widget_dict["active_constraints"].widget_dict.items():
            v.setToolTip(self.widget_dict[k].widget.toolTip())

    def addWidgets(self, *args, **kwargs):
        # Add the active constraint and visualization button widgets together and a single VBoxLayout
        vbox_layout = QVBoxLayout()
        group_visualize_widget = QWidget(self)
        group_visualize_widget.setLayout(vbox_layout)
        vbox_layout.addWidget(self.widget_dict["active_constraints"])
        vbox_layout.addWidget(self.widget_dict["visualize"], alignment=Qt.AlignmentFlag.AlignHCenter)
        self.lay.addWidget(group_visualize_widget, 0, 0, 1, 3, alignment=Qt.AlignmentFlag.AlignTop)

        row_count = 1
        column = 0
        for widget_name, widget in self.widget_dict.items():
            if widget_name in ["active_constraints", "visualize"]:
                continue
            row_span = 1
            col_span = 2 if widget.push is None else 1
            self.lay.addWidget(widget.label, row_count, column, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
            self.lay.addWidget(widget.widget, row_count, column + 1, row_span, col_span, alignment=Qt.AlignmentFlag.AlignTop)

            if widget.push is not None:
                self.lay.addWidget(widget.push, row_count, column + 2, row_span, col_span, alignment=Qt.AlignmentFlag.AlignTop)

            row_count += 1

    def establishWidgetConnections(self):
        for k, v in self.widget_dict["active_constraints"].widget_dict.items():
            v.stateChanged.connect(self.widget_dict[k].setShown)
            v.stateChanged.connect(self.widget_dict[k + "_policy"].setShown)
        self.widget_dict["thickness_at_points"].push.clicked.connect(partial(
            self.select_data_file, line_edit=self.widget_dict["thickness_at_points"].widget))
        self.widget_dict["internal_geometry"].push.clicked.connect(partial(
            self.select_data_file, line_edit=self.widget_dict["internal_geometry"].widget))
        self.widget_dict["visualize"].clicked.connect(self.visualize)

    def visualize(self):
        if self.sub_dialog is not None:
            return

        def _visualize_min_radius():
            min_radius_data = airfoil.visualize_min_radius()
            self.sub_dialog.graph.plot_items["min_radius_point"].setData(
                [min_radius_data["xy"][0]], [min_radius_data["xy"][1]]
            )
            text = (f"min(|R|) = {min_radius_data['R_abs_min']:.4f} {self.geo_col.units.current_length_unit()}<br>"
                    f"min(|R|)/c = {min_radius_data['R_abs_min_over_c']:.4f}")
            self.sub_dialog.graph.plot_items["min_radius_text"].setHtml(
                f"<span style='font-family: DejaVu Sans Mono; "
                f"color: #E8C205; size: 10pt'>{text}</span>"
            )
            text_pos = (min_radius_data["xy"][0] - 0.01 * airfoil.measure_chord(),
                        min_radius_data["xy"][1] - 0.01 * airfoil.measure_chord())
            self.sub_dialog.graph.plot_items["min_radius_text"].setPos(text_pos[0], text_pos[1])

        def _visualize_max_thickness():
            max_thickness_data = airfoil.visualize_max_thickness()
            self.sub_dialog.graph.plot_items["max_thickness_line"].setData(
                max_thickness_data["xy"][:, 0], max_thickness_data["xy"][:, 1]
            )
            text = (f"t_max = {max_thickness_data['t_max']:.6f} {self.geo_col.units.current_length_unit()}<br>"
                    f"t/c_max = {max_thickness_data['t/c_max']:.6f}<br>")
            self.sub_dialog.graph.plot_items["max_thickness_text"].setHtml(
                f"<span style='font-family: DejaVu Sans Mono; "
                f"color: #32A852; size: 10pt'>{text}</span>"
            )
            text_pos = (max_thickness_data["xy"][0, 0], max_thickness_data["xy"][0, 1] + 0.05 * airfoil.measure_chord())
            self.sub_dialog.graph.plot_items["max_thickness_text"].setPos(text_pos[0], text_pos[1])

        def _visualize_min_area():
            area_normalized = airfoil.compute_area(airfoil_frame_relative=True)
            area_absolute = airfoil.compute_area(airfoil_frame_relative=False)
            text = (f"A = {area_absolute:.6f} {self.geo_col.units.current_length_unit()}<sup>2</sup><br>"
                    f"A/c<sup>2</sup> = {area_normalized:.6f}")
            self.sub_dialog.graph.plot_items["min_area_text"].setHtml(
                f"<span style='font-family: DejaVu Sans Mono; "
                f"color: cornflowerblue; size: 10pt'>{text}</span>"
            )
            text_pos = np.mean((airfoil.leading_edge.as_array(),
                                airfoil.trailing_edge.as_array()), axis=0) + np.array(
                [0.0, 0.3 * airfoil.measure_chord()]
            )
            self.sub_dialog.graph.plot_items["min_area_text"].setPos(text_pos[0], text_pos[1])

        def _visualize_thickness_at_points():
            cnstr_spec = dialog_value["thickness_at_points"]
            try:
                if cnstr_spec[0] == "":
                    self.geo_col.gui_obj.disp_message_box(f"Must specify thickness distribution file")
                    return
                xt_arr = np.loadtxt(cnstr_spec[0])
            except FileNotFoundError:
                self.geo_col.gui_obj.disp_message_box(f"File {cnstr_spec[0]} not found")
                return

            airfoil_frame_relative = "Non-Dimensional" in cnstr_spec[2]
            x_arr, t_arr = xt_arr[:, 0], xt_arr[:, 1]
            data = airfoil.visualize_thickness_at_points(
                x_arr, t_arr, airfoil_frame_relative=airfoil_frame_relative,
                vertical_check="Vertical" in cnstr_spec[2]
            )
            for sl in data["slices"]:
                self.sub_dialog.graph.v.plot(sl[:, 0], sl[:, 1], pen=pg.mkPen(color="magenta", width=1.5))
            if len(data["warning_x_vals"]) > 0:
                self.geo_col.gui_obj.disp_message_box(
                    f"Thickness check intersection not found for x-values: {data['warning_x_vals']}",message_mode="warn"
                )

        def _visualize_internal_geometry():
            cnstr_spec = dialog_value["internal_geometry"]
            try:
                if cnstr_spec[0] == "":
                    self.geo_col.gui_obj.disp_message_box(f"Must specify internal geometry file")
                    return
                xy_polyline = np.loadtxt(cnstr_spec[0])
            except FileNotFoundError:
                self.geo_col.gui_obj.disp_message_box(f"File {cnstr_spec[0]} not found")
                return

            airfoil_frame_relative, rotate_with_airfoil, translate_with_airfoil, scale_with_airfoil = True, True, True, True
            if len(cnstr_spec) > 1:
                if "Rotate/Translate w/ Airfoil" in cnstr_spec[2]:
                    scale_with_airfoil = False
                    airfoil_frame_relative = False
                elif "Absolute" in cnstr_spec[2]:
                    airfoil_frame_relative = False
                    rotate_with_airfoil = False
                    translate_with_airfoil = False
                    scale_with_airfoil = False

            data = airfoil.visualize_contains_line_string(
                xy_polyline,
                airfoil_frame_relative=airfoil_frame_relative,
                rotate_with_airfoil=rotate_with_airfoil,
                scale_with_airfoil=scale_with_airfoil,
                translate_with_airfoil=translate_with_airfoil
            )

            self.sub_dialog.graph.plot_items["internal_geometry_line"].setData(
                data["xy_polyline"][:, 0], data["xy_polyline"][:, 1]
            )

        dialog_value = self.value()
        airfoil = self.geo_col.container()["airfoils"][self.airfoil_name]
        airfoil_coords = airfoil.get_coords_selig_format()
        self.sub_dialog = GeometricConstraintGraphDialog(
            theme=self.theme, geo_col=self.geo_col, grid=self.grid, airfoil_name=self.airfoil_name, parent=self
        )
        self.sub_dialog.sigConstraintGraphClosed.connect(self.onVisualizeClosed)
        self.sub_dialog.graph.plot_items["airfoil"].setData(airfoil_coords[:, 0], airfoil_coords[:, 1])
        if dialog_value["min_radius_curvature"][1]:
            _visualize_min_radius()
        if dialog_value["min_val_of_max_thickness"][1]:
            _visualize_max_thickness()
        if dialog_value["min_area"][1]:
            _visualize_min_area()
        if dialog_value["thickness_at_points"][1]:
            _visualize_thickness_at_points()
        if dialog_value["internal_geometry"][1]:
            _visualize_internal_geometry()
        self.sub_dialog.show()

    def onVisualizeClosed(self):
        self.sub_dialog = None

    def select_data_file(self, line_edit: QLineEdit):
        select_data_file(parent=self.parent(), line_edit=line_edit)

    def value(self) -> dict:
        d = {}
        for k, v in self.widget_dict["active_constraints"].widget_dict.items():
            d[k] = [self.widget_dict[k].value(), v.isChecked(), self.widget_dict[k + "_policy"].value()]
        return d

    def setValue(self, d: dict):
        # Used to ensure backwards compatibility with pymead < 2.0.0-b12
        if "check_thickness_at_points" in d.keys():
            d["thickness_at_points"] = [d["thickness_at_points"], d.pop("check_thickness_at_points")]
        if "use_internal_geometry" in d.keys():
            d["internal_geometry"] = [d["internal_geometry"], d.pop("use_internal_geometry")]
        if "external_geometry" in d.keys():
            d.pop("external_geometry")

        for k, v in d.items():
            if k in ["active_constraints", "visualize"]:
                continue
            if v is None:
                continue
            self.widget_dict["active_constraints"].widget_dict[k].setChecked(v[1])
            self.widget_dict[k].setValue(v[0])
            if len(v) > 2:  # Used to ensure backwards compatibility with pymead < 2.0.0-b12
                self.widget_dict[k + "_policy"].setValue(v[2])


class OptConstraintsHTabWidget(PymeadDialogHTabWidget):

    OptConstraintsChanged = pyqtSignal()

    def __init__(self, parent, geo_col: GeometryCollection, theme: dict, grid: bool):
        self.geo_col = geo_col
        self.theme = theme
        self.grid = grid
        super().__init__(parent=parent,
                         widgets={k: OptConstraintsDialogWidget(geo_col=geo_col, airfoil_name=k, theme=self.theme, grid=self.grid)
                                  for k in geo_col.container()["airfoils"]})
        self.label = None
        self.widget = self
        self.push = None

    def reorderRegenerateWidgets(self, new_airfoil_name_list: list):
        temp_dict = {}
        for airfoil_name in new_airfoil_name_list:
            if airfoil_name not in self.w_dict:
                self.w_dict[airfoil_name] = OptConstraintsDialogWidget(geo_col=self.geo_col, airfoil_name=airfoil_name,
                                                                       theme=self.theme, grid=self.grid)
            temp_dict[airfoil_name] = self.w_dict[airfoil_name]
        self.w_dict = temp_dict
        self.regenerateWidgets()

    def setValues(self, values: dict):
        self.setValue(new_values=values)

    def values(self):
        return self.value()

    def valueChanged(self, k1, k2, v2):
        self.OptConstraintsChanged.emit()


class XFOILDialogWidget(PymeadDialogWidget):
    def __init__(self, current_airfoils: typing.List[str], settings_override: dict = None):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'xfoil_settings.json'))
        self.widget_dict["base_dir"]["widget"].setText(tempfile.gettempdir())
        self.widget_dict["airfoil"]["widget"].addItems(current_airfoils)
        if settings_override:
            self.setValue(new_values=settings_override)

    def calculate_and_set_Reynolds_number(self, new_inputs: dict):
        Re_widget = self.widget_dict['Re']['widget']
        nu = viscosity_calculator(new_inputs['T'], rho=new_inputs['rho'])
        a = np.sqrt(new_inputs['gam'] * new_inputs['R'] * new_inputs['T'])
        V = new_inputs['Ma'] * a
        Re_widget.setValue(V * new_inputs['L'] / nu)

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

    def change_prescribed_flow_variables(self, current_text: str):
        w1 = self.widget_dict['P']['widget']
        w2 = self.widget_dict['T']['widget']
        w3 = self.widget_dict['rho']['widget']
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

    def change_Re_active_state(self, new_inputs: dict):
        active = new_inputs['spec_Re']
        widget_names = ['P', 'T', 'rho', 'L', 'R', 'gam']
        skip_P, skip_T, skip_rho = False, False, False
        if new_inputs['spec_P_T_rho'] == 'Specify Pressure, Temperature' and self.widget_dict['rho'][
            'widget'].isReadOnly():
            skip_rho = True
        if new_inputs['spec_P_T_rho'] == 'Specify Pressure, Density' and self.widget_dict['T']['widget'].isReadOnly():
            skip_T = True
        if new_inputs['spec_P_T_rho'] == 'Specify Temperature, Density' and self.widget_dict['P'][
            'widget'].isReadOnly():
            skip_P = True
        for widget_name in widget_names:
            if not (skip_rho and widget_name == 'rho') and not (skip_P and widget_name == 'P') and not (
                    skip_T and widget_name == 'T'):
                self.widget_dict[widget_name]['widget'].setReadOnly(active)
        self.widget_dict['Re']['widget'].setReadOnly(not active)
        self.widget_dict['spec_P_T_rho']['widget'].setEnabled(not active)
        if not active:
            self.calculate_and_set_Reynolds_number(new_inputs)

    def updateDialog(self, new_inputs: dict, w_name: str):
        if w_name == 'spec_Re':
            self.change_Re_active_state(new_inputs)
        if w_name == 'spec_P_T_rho':
            self.change_prescribed_flow_variables(new_inputs['spec_P_T_rho'])
        if w_name == 'prescribe':
            self.change_prescribed_aero_parameter(new_inputs['prescribe'])
        if w_name in ['P', 'T', 'rho', 'R', 'gam', 'L', 'Ma'] and not self.widget_dict[w_name]['widget'].isReadOnly():
            if self.widget_dict['P']['widget'].isReadOnly():
                self.widget_dict['P']['widget'].setValue(new_inputs['rho'] * new_inputs['R'] * new_inputs['T'])
            elif self.widget_dict['T']['widget'].isReadOnly():
                self.widget_dict['T']['widget'].setValue(new_inputs['P'] / new_inputs['R'] / new_inputs['rho'])
            elif self.widget_dict['rho']['widget'].isReadOnly():
                self.widget_dict['rho']['widget'].setValue(new_inputs['P'] / new_inputs['R'] / new_inputs['T'])
            new_inputs = self.value()
            if not (w_name == 'Ma' and new_inputs['spec_Re']):
                self.calculate_and_set_Reynolds_number(new_inputs)

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit, starting_dir=tempfile.gettempdir())


class GAGeneralSettingsDialogWidget(PymeadDialogWidget2):

    sigMEAFileChanged = pyqtSignal(list, list, object, object)

    def __init__(self, parent=None):
        self.current_save_file = None
        super().__init__(parent=parent)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "save": PymeadLabeledPushButton(label="Save", text="Save Settings"),
            "save_as": PymeadLabeledPushButton(label="Save As", text="Save Settings As..."),
            "load": PymeadLabeledPushButton(label="Load", text="Load Settings File"),
            "warm_start_active": PymeadLabeledCheckbox(label="Warm Start Active?", initial_state=0),
            "warm_start_generation": PymeadLabeledSpinBox(label="Warm Start Generation", value=-1, minimum=-2147483647,
                                                          maximum=2147483647, read_only=True),
            "warm_start_dir": PymeadLabeledLineEdit(label="Warm Start Directory", push_label="Choose folder",
                                                    read_only=True,
                                                    tool_tip="Choose '-1' to start from the most recent generation"),
            "use_initial_settings": PymeadLabeledCheckbox(label="Use Initial Settings?", initial_state=2),
            "mea_file": PymeadLabeledLineEdit(label="MEA File", push_label="Choose file"),
            "batch_mode_active": PymeadLabeledCheckbox(
                label="Batch Mode Active?", initial_state=0,
                tool_tip="If this box is checked, all settings in this dialog will be\noverridden by the settings "
                         "in the selected JSON settings files."),
            "batch_mode_files": PymeadLabeledPlainTextEdit(label="Batch Settings Files", push_label="Choose files",
                                                           read_only=True)
        }

    def establishWidgetConnections(self):
        self.widget_dict["save"].widget.clicked.connect(self.save_opt_settings)
        self.widget_dict["save_as"].widget.clicked.connect(self.saveas_opt_settings)
        self.widget_dict["load"].widget.clicked.connect(self.load_opt_settings)
        self.widget_dict["warm_start_active"].sigValueChanged.connect(self.warm_start_active_state_changed)
        self.widget_dict["warm_start_dir"].push.clicked.connect(partial(self.select_directory,
                                                                        self.widget_dict["warm_start_dir"].widget))
        self.widget_dict["mea_file"].push.clicked.connect(partial(self.select_existing_jmea_file,
                                                                  self.widget_dict["mea_file"].widget))
        self.widget_dict["batch_mode_active"].sigValueChanged.connect(self.batch_mode_active_state_changed)
        self.widget_dict["batch_mode_files"].push.clicked.connect(partial(self.select_multiple_json_files,
                                                                          self.widget_dict["batch_mode_files"].widget))

    def warm_start_active_state_changed(self, state: int):
        self.widget_dict["warm_start_generation"].widget.setReadOnly(not bool(state))
        self.widget_dict["warm_start_dir"].setReadOnly(not bool(state))

    def batch_mode_active_state_changed(self, state: int):
        self.widget_dict["batch_mode_files"].widget.setReadOnly(not bool(state))

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit)

    def select_existing_jmea_file(self, line_edit: QLineEdit):
        jmea_file = select_existing_jmea_file(parent=self.parent(), line_edit=line_edit)
        if jmea_file is not None:
            gui_obj = get_parent(self, 4)
            gui_obj.load_geo_col(file_name=jmea_file)
            self.sigMEAFileChanged.emit([airfoil for airfoil in gui_obj.geo_col.container()["airfoils"].values()],
                                        [mea for mea in gui_obj.geo_col.container()["mea"].values()],
                                        None, None)

    def select_multiple_json_files(self, text_edit: QPlainTextEdit):
        select_multiple_json_files(parent=self.parent(), text_edit=text_edit)

    def save_opt_settings(self):
        opt_file = None
        if self.current_save_file is not None:
            opt_file = self.current_save_file
        if get_parent(self, 4) and get_parent(self, 4).current_settings_save_file is not None:
            opt_file = get_parent(self, 4).current_settings_save_file
        if opt_file is not None:
            new_inputs = get_parent(self, 2).value()  # Gets the inputs from the PymeadDialogVTabWidget
            save_data(new_inputs, opt_file)
            get_parent(self, 2).setStatusTip(f"Settings saved ({opt_file})")
            # msg_box = PymeadMessageBox(parent=self, msg=f"Settings saved as {self.current_save_file}",
            #                            window_title='Save Notification', msg_mode='info')
            # msg_box.exec()
        else:
            self.saveas_opt_settings()

    def load_opt_settings(self):
        load_dialog = select_existing_json_file(parent=self, starting_dir=load_documents_path("ga-settings-dir"))
        if load_dialog.exec():
            load_file = load_dialog.selectedFiles()[0]
            q_settings.setValue("ga-settings-dir", os.path.split(load_file)[0])
            new_inputs = load_data(load_file)
            great_great_grandparent = get_parent(self, depth=4)
            if great_great_grandparent:
                great_great_grandparent.current_settings_save_file = load_file
            else:
                self.current_save_file = load_file

            gui_obj = get_parent(self, 4)
            geo_col_file = new_inputs["General Settings"]["mea_file"]
            if geo_col_file:
                gui_obj.load_geo_col(file_name=geo_col_file)

            # Pass the new GeometryCollection reference to the actuator disk
            get_parent(self, 3).mses_widget.widget_dict["AD"].geo_col = deepcopy(gui_obj.geo_col)

            self.sigMEAFileChanged.emit([airfoil for airfoil in gui_obj.geo_col.container()["airfoils"].values()],
                                        [mea for mea in gui_obj.geo_col.container()["mea"].values()],
                                        new_inputs["XFOIL"]["airfoil"],
                                        new_inputs["MSET"]["mea"])

            get_parent(self, 3).setWindowTitle(f"Optimization Setup - {os.path.split(load_file)[-1]}")
            get_parent(self, 2).setValue(new_inputs)  # Overrides the inputs for the whole PymeadDialogVTabWidget
            get_parent(self, 2).setStatusTip(f"Loaded {load_file}")

    def saveas_opt_settings(self):
        inputs_to_save = get_parent(self, 2).value()
        json_dialog = select_json_file(parent=self)
        if json_dialog.exec():
            input_filename = json_dialog.selectedFiles()[0]
            if not os.path.splitext(input_filename)[-1] == '.json':
                input_filename = input_filename + '.json'
            save_data(inputs_to_save, input_filename)
            if get_parent(self, 4):
                get_parent(self, 4).current_settings_save_file = input_filename
            else:
                self.current_save_file = input_filename
            get_parent(self, 3).setStatusTip(f"Saved optimization settings to {input_filename}")

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class GAConstraintsTerminationDialogWidget(PymeadDialogWidget2):
    def __init__(self, geo_col: GeometryCollection, theme: dict, grid: bool, parent=None):
        self.geo_col = geo_col
        self.theme = theme
        self.grid = grid
        super().__init__(parent=parent)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "constraints": OptConstraintsHTabWidget(parent=None, geo_col=self.geo_col, theme=self.theme, grid=self.grid),
            "f_tol": PymeadLabeledScientificDoubleSpinBox(label="Function Tolerance", value=1e-6, minimum=0.0,
                                                          maximum=100000.0),
            "cv_tol": PymeadLabeledScientificDoubleSpinBox(label="Constraint Violation Tol.", value=1.0e-6,
                                                           minimum=0.0, maximum=100000.0),
            "x_tol": PymeadLabeledScientificDoubleSpinBox(label="Parameter Tolerance", value=1.0e-8,
                                                          minimum=0.0, maximum=100000.0),
            "nth_gen": PymeadLabeledSpinBox(label="Termination Calc. Frequency", value=10, minimum=1, maximum=100000),
            "n_last": PymeadLabeledSpinBox(label="Num. Prev. Gens. to Check", value=30, minimum=1, maximum=100000),
            "n_max_gen": PymeadLabeledSpinBox(label="Maximum Generations", value=500, minimum=1, maximum=100000),
            "n_max_evals": PymeadLabeledSpinBox(label="Maximum Function Calls", value=100000, minimum=1,
                                                maximum=10000000)
        }


class MultiPointOptDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'multi_point_opt_settings.json'))
        self.current_save_file = None

    def select_data_file(self, line_edit: QLineEdit):
        select_data_file(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class GeneticAlgorithmDialogWidget(PymeadDialogWidget2):
    def __init__(self, gui_obj, multi_point_dialog_widget: MultiPointOptDialogWidget, parent=None):
        self.multi_point_dialog_widget = multi_point_dialog_widget
        self.gui_obj = gui_obj
        super().__init__(parent=parent)
        self.multi_point = self.multi_point_dialog_widget.widget_dict["multi_point_active"]["widget"].isChecked()

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "tool": PymeadLabeledComboBox(label="CFD Tool", items=["XFOIL", "MSES"], current_item="XFOIL"),
            "J": PymeadLabeledLineEdit(
                label="Objective Functions",
                tool_tip="Enter the objective functions to be minimized, separated by commas.\n"
                         "Variables can be started with the dollar sign ($).",
                text="$Cd"
            ),
            "G": PymeadLabeledLineEdit(
                label="Aerodynamic Constraints ",
                tool_tip="Enter the constraints to be applied, separated by commas.\n"
                         "Variables can be started with the dollar sign ($).",
                text=""
            ),
            "pop_size": PymeadLabeledSpinBox(label="Population Size", minimum=1, maximum=2147483647, value=50),
            "n_offspring": PymeadLabeledSpinBox(
                label="Number of Offspring",
                tool_tip="Number of offspring to generate to fill out the population.\nOffspring with converged "
                         "objective functions become members of the\ncurrent population until the population is "
                         "full. Must be greater\nthan or equal to the population size",
                minimum=1, maximum=2147483647, value=1500
            ),
            "max_sampling_width": PymeadLabeledDoubleSpinBox(
                label="Max. Sampling Width", decimals=8, minimum=0, maximum=1.0, value=0.2, single_step=0.01,
                push_label="Visualize sampling",
                tool_tip="Maximum random distance from original (seed) parameter\n value for each active and "
                         "unlinked parameter"
            ),
            "eta_crossover": PymeadLabeledDoubleSpinBox(
                label="\u03b7 (crossover)", decimals=3, minimum=0.0, maximum=100000.0, value=20.0,
                tool_tip="Simulated Binary Crossover parameter as described in\n"
                         "https://pymoo.org/operators/crossover.html?highlight=eta%20crossover"
            ),
            "eta_mutation": PymeadLabeledDoubleSpinBox(
                label="\u03b7 (mutation)", decimals=3, minimum=0.0, maximum=100000.0, value=15.0,
                tool_tip="Polynomial Mutation parameter as descrbied in\n"
                         "https://pymoo.org/operators/mutation.html"
            ),
            "random_seed": PymeadLabeledSpinBox(label="Random Seed", minimum=0, maximum=2147483647, value=1),
            "num_processors": PymeadLabeledSpinBox(
                label="Number of Processors", minimum=1, maximum=100000, value=os.cpu_count(),
                tool_tip="Number of logical processors to use for the optimization. The default value is the "
                         "total number of logical processors available on your machine."
            ),
            "algorithm_save_frequency": PymeadLabeledSpinBox(
                label="State Save Frequency", minimum=1, maximum=1000, value=1,
                tool_tip="How often to save the setCheckState of the genetic algorithm.\n"
                         "A setValue of '1' enforces a setCheckState save every generation"
            ),
            "root_dir": PymeadLabeledLineEdit(label="Opt. Root Directory", text="", push_label="Choose folder"),
            "opt_dir_name": PymeadLabeledLineEdit(
                label="Opt. Directory Name", text="ga_opt",
                tool_tip="A unique directory will be created inside 'root_dir' with this name\n"
                         "and followed by an underscore and a number if necessary"
            ),
            "temp_analysis_dir_name": PymeadLabeledLineEdit(
                label="Temp. Analysis Dir. Name", text="analysis_temp",
                tool_tip="The directory 'root_dir/temp_analysis_dir_name' will be created\n"
                         "to hold the temporary XFOIL or MSES analysis files"
            )
        }
        self.num_processors_changed(self.widget_dict["num_processors"].value())
        self.root_dir_changed(self.widget_dict["root_dir"].value())

    def establishWidgetConnections(self):
        self.widget_dict["max_sampling_width"].push.clicked.connect(self.visualize_sampling)
        self.widget_dict["root_dir"].push.clicked.connect(
            partial(self.select_directory, self.widget_dict["root_dir"].widget)
        )
        self.widget_dict["J"].sigValueChanged.connect(self.objectives_changed)
        self.widget_dict["G"].sigValueChanged.connect(self.constraints_changed)
        self.widget_dict["num_processors"].sigValueChanged.connect(self.num_processors_changed)
        self.widget_dict["root_dir"].sigValueChanged.connect(self.root_dir_changed)
        self.widget_dict["tool"].sigValueChanged.connect(self.update_objectives_and_constraints)
        self.multi_point_dialog_widget.widget_dict["multi_point_active"]["widget"].stateChanged.connect(
            self.update_objectives_and_constraints
        )

    def setValue(self, new_values: dict):
        super().setValue(new_values)
        self.update_objectives_and_constraints()

    def update_objectives_and_constraints(self, *args, **kwargs):
        self.multi_point = self.multi_point_dialog_widget.widget_dict["multi_point_active"]["widget"].isChecked()
        value = self.value()
        self.objectives_changed(value["J"])
        self.constraints_changed(value["G"])

    def visualize_sampling(self):
        starting_value = self.widget_dict["max_sampling_width"].value()
        background_color = self.gui_obj.themes[self.gui_obj.current_theme]["graph-background-color"]
        theme = self.gui_obj.themes[self.gui_obj.current_theme]
        geo_col_dict = self.gui_obj.geo_col.get_dict_rep()

        dialog = SamplingVisualizationDialog(geo_col_dict=geo_col_dict, initial_sampling_width=starting_value,
                                             initial_n_samples=20, background_color=background_color, theme=theme,
                                             parent=self)
        dialog.exec()

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass

    def multi_point_changed(self, state: int or bool):
        self.multi_point = state
        self.objectives_changed(self.widget_dict["J"].widget.text())
        self.constraints_changed(self.widget_dict["G"].widget.text())

    def get_template(self) -> dict:
        dialog_value = self.value()
        tool = dialog_value["tool"]
        assert isinstance(tool, str)
        if self.multi_point:
            tool += "_MULTIPOINT"
        return getattr(cfd_output_templates, tool)

    def objectives_changed(self, text: str):
        widget = self.widget_dict["J"].widget
        template = self.get_template()
        self.gui_obj.objectives.clear()
        any_constraint_failed = False
        for obj_func_str in text.split(','):
            objective = Objective(obj_func_str)
            self.gui_obj.objectives.append(objective)
            if text == '':
                widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_FAILURE}}}")
                return
            try:
                objective.update(template)
                if not any_constraint_failed:
                    widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_SUCCESS}}}")
            except FunctionCompileError:
                widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_FAILURE}}}")
                any_constraint_failed = True

    def constraints_changed(self, text: str):
        widget = self.widget_dict["G"].widget
        template = self.get_template()
        self.gui_obj.constraints.clear()
        any_constraint_failed = False
        for constraint_func_str in text.split(','):
            if len(constraint_func_str) > 0:
                constraint = Constraint(constraint_func_str)
                self.gui_obj.constraints.append(constraint)
                try:
                    constraint.update(template)
                    if not any_constraint_failed:
                        widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_SUCCESS}}}")
                except FunctionCompileError:
                    widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_FAILURE}}}")
                    any_constraint_failed = True

    def num_processors_changed(self, value: int):
        widget = self.widget_dict["num_processors"].widget
        if value <= os.cpu_count():
            widget.setStyleSheet(f"QSpinBox {{background-color: {VALIDATION_DEFAULT}}}")
        else:
            widget.setStyleSheet(f"QSpinBox {{background-color: {VALIDATION_FAILURE}}}")

    def root_dir_changed(self, value: str):
        widget = self.widget_dict["root_dir"].widget
        if len(value) == 0:
            widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_FAILURE}}}")
            return
        if not os.path.exists(value):
            widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_FAILURE}}}")
            return
        widget.setStyleSheet(f"QLineEdit {{background-color: {VALIDATION_SUCCESS}}}")

    @staticmethod
    def convert_text_array_to_dict(multi_line_text: str):
        text_array = multi_line_text.split('\n')
        data_dict = {}
        for text in text_array:
            text_split = text.split(': ')
            if len(text_split) > 1:
                k = text_split[0]
                v = float(text_split[1])
                data_dict[k] = v
        return data_dict


class PanelDialog(PymeadDialog):
    def __init__(self, parent: QWidget, theme: dict, settings_override: dict = None):
        self.w = PanelDialogWidget(settings_override)
        super().__init__(parent=parent, window_title="Basic Panel Code Analysis", widget=self.w, theme=theme,
                         fixed_width=270, fixed_height=130)


class XFOILDialog(PymeadDialog):
    def __init__(self, parent: QWidget, current_airfoils: typing.List[str], theme: dict,
                 settings_override: dict = None):
        self.w = XFOILDialogWidget(current_airfoils=current_airfoils, settings_override=settings_override)
        super().__init__(parent=parent, window_title="Single Airfoil Viscous Analysis", widget=self.w,
                         theme=theme, minimum_width=900, minimum_height=500)


class MultiAirfoilDialog(PymeadDialog):
    def __init__(self, parent: QWidget, geo_col: GeometryCollection, theme: dict, settings_override: dict = None):
        mset_dialog_widget = MSETDialogWidget(geo_col=geo_col, theme=theme)
        mses_dialog_widget = MSESDialogWidget2(geo_col=geo_col)
        mset_dialog_widget.sigMEAChanged.connect(mses_dialog_widget.widget_dict["xtrs"].onMEAChanged)
        mplot_dialog_widget = MPLOTDialogWidget()
        mpolar_dialog_widget = MPOLARDialogWidget()
        tab_widgets = {
            "MSET": mset_dialog_widget,
            "MSES": mses_dialog_widget,
            "MPLOT": mplot_dialog_widget,
            "MPOLAR": mpolar_dialog_widget
        }
        widget = PymeadDialogVTabWidget(parent=None, widgets=tab_widgets, settings_override=settings_override)
        super().__init__(parent=parent, window_title="Multi-Element-Airfoil Analysis", widget=widget, theme=theme,
                         minimum_width=900, minimum_height=700)


class ScreenshotDialog(PymeadDialog):
    def __init__(self, parent: QWidget, theme: dict, windows: typing.List[str]):

        widget = QWidget()
        super().__init__(parent=parent, window_title="Screenshot", widget=widget, theme=theme,
                         minimum_width=400, fixed_height=175)
        self.grid_widget = {}
        self.grid_layout = QGridLayout()

        self.setInputs()
        self.grid_widget["window"]["combobox"].addItems(windows)
        widget.setLayout(self.grid_layout)
        self.window_list = windows

    def setInputs(self):
        widget_dict = load_data(os.path.join(GUI_DIALOG_WIDGETS_DIR, "screenshot_dialog.json"))
        for row_name, row_dict in widget_dict.items():
            self.grid_widget[row_name] = {}
            for w_name, w_dict in row_dict.items():
                widget = getattr(sys.modules[__name__], w_dict["w"])(self)
                self.grid_widget[row_name][w_name] = widget
                if "text" in w_dict.keys() and isinstance(widget, (QLabel, QLineEdit, QPushButton)):
                    widget.setText(w_dict["text"])
                if "func" in w_dict.keys() and isinstance(widget, QPushButton):
                    if "choose" in row_name:
                        widget.clicked.connect(partial(getattr(self, w_dict["func"]),
                                                       self.grid_widget[row_name]["line"]))
                    else:
                        widget.clicked.connect(getattr(self, w_dict["func"]))
                if "tool_tip" in w_dict.keys():
                    widget.setToolTip(w_dict["tool_tip"])
                for attr_name, attr_value in w_dict.items():
                    if attr_name not in ["text", "func", "tool_tip", "grid", "w"]:
                        getattr(widget, attr_name)(attr_value)
                self.grid_layout.addWidget(widget, w_dict["grid"][0], w_dict["grid"][1], w_dict["grid"][2],
                                           w_dict["grid"][3])

    def value(self):
        inputs = {
            "image_file": self.grid_widget["choose_image_file"]["line"].text(),
            "window": self.grid_widget["window"]["combobox"].currentText()
        }
        return inputs

    def setValue(self, new_inputs):
        if new_inputs["window"] in self.window_list:
            combo = self.grid_widget["window"]["combobox"]
            combo.setCurrentIndex(combo.findText(new_inputs["window"]))
        self.grid_widget["choose_image_file"]["line"].setText(new_inputs["image_file"])

    def select_jpg_file(self, line_edit: QLineEdit):
        select_jpg_file(parent=self.parent(), line_edit=line_edit)


class BoundsDialog(QDialog):
    def __init__(self, bounds, parent=None, pos_param: bool = False):
        super().__init__(parent)
        self.pos_param = pos_param
        self.setWindowTitle("Parameter Bounds Modification")
        self.setFont(self.parent().font())
        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
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

    def valuesFromWidgets(self):
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


class SamplingVisualizationDialog(PymeadDialog):
    def __init__(self, geo_col_dict: dict, initial_sampling_width: float, initial_n_samples: int, background_color: str,
                 theme: dict, parent=None):
        self.sampling_widget = SamplingVisualizationWidget(None, geo_col_dict,
                                                           initial_sampling_width=initial_sampling_width,
                                                           initial_n_samples=initial_n_samples,
                                                           background_color=background_color)

        super().__init__(parent=parent, window_title="Sampling Visualization", widget=self.sampling_widget, theme=theme)


class LoadDialog(QFileDialog):
    def __init__(self, parent, settings_var: str, file_filter: str = "JMEA Files (*.jmea)"):
        super().__init__(parent=parent)

        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilter(self.tr(file_filter))
        self.setViewMode(QFileDialog.ViewMode.Detail)
        self.settings_var = settings_var

        # Get default open location
        if q_settings.contains(settings_var):
            path = q_settings.value(settings_var)
        else:
            path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)

        self.setDirectory(path)


class LoadAirfoilAlgFileWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        layout = QGridLayout(self)

        self.opt_dir_line = QLineEdit("", self)
        self.pkl_selection_button = QPushButton("Choose Opt Dir", self)
        self.generation_label = QLabel("Generation", self)
        self.generation_spin = QSpinBox(self)
        self.generation_spin.setMinimum(1)
        self.generation_spin.setMaximum(1000000)
        self.index_button = QRadioButton("Use Index", self)
        self.weight_button = QRadioButton("Use Weights", self)
        self.index_spin = QSpinBox(self)
        self.index_spin.setValue(0)
        self.index_spin.setMaximum(9999)
        self.weight_line = QLineEdit(self)

        self.index_button.toggled.connect(self.index_selected)
        self.weight_button.toggled.connect(self.weight_selected)

        self.weight_line.textChanged.connect(self.weights_changed)
        self.pkl_selection_button.clicked.connect(partial(select_directory, parent=self, line_edit=self.opt_dir_line))

        self.weight_line.setText("0.5,0.5")

        self.index_button.toggle()

        layout.addWidget(self.opt_dir_line, 0, 0)
        layout.addWidget(self.pkl_selection_button, 0, 1)
        layout.addWidget(self.generation_label, 1, 0)
        layout.addWidget(self.generation_spin, 1, 1)
        layout.addWidget(self.index_button, 2, 0)
        layout.addWidget(self.weight_button, 2, 1)
        layout.addWidget(self.index_spin, 3, 0)
        layout.addWidget(self.weight_line, 3, 1)

        self.setLayout(layout)

        inputs = self.valuesFromWidgets()
        for setting_var in ("opt_load_dir", "opt_load_generation",
                            "opt_load_use_index", "opt_load_use_weights", "opt_load_index", "opt_load_weights"):
            if q_settings.contains(setting_var):
                inputs[setting_var] = q_settings.value(setting_var)
                if setting_var in ("opt_load_use_index", "opt_load_use_weights",
                                   "opt_load_index", "opt_load_generation"):
                    inputs[setting_var] = int(inputs[setting_var])
                if setting_var == "opt_load_weights":
                    inputs[setting_var] = [float(s) for s in inputs[setting_var]]
        self.setInputs(inputs)

    def index_selected(self):
        self.index_spin.setReadOnly(False)
        self.weight_line.setReadOnly(True)

    def weight_selected(self):
        self.index_spin.setReadOnly(True)
        self.weight_line.setReadOnly(False)

    def weights_changed(self, new_text: str):
        weights = self.validate_weights(new_text)
        if len(weights) > 0:
            self.weight_line.setStyleSheet("QLineEdit { background: rgba(16,201,87,50) }")
        else:
            self.weight_line.setStyleSheet("QLineEdit { background: rgba(176,25,25,50) }")

    @staticmethod
    def validate_weights(text: str):
        text = text.strip()
        text_list = text.split(",")
        try:
            weights = [float(t) for t in text_list]
            weight_sum = sum(weights)
            if not np.isclose(weight_sum, 1.0, rtol=1e-12):
                return []
        except:
            return []

        return weights

    def valuesFromWidgets(self):
        inputs = {
            "opt_load_dir": self.opt_dir_line.text(),
            "opt_load_generation": self.generation_spin.value(),
            "opt_load_use_index": int(self.index_button.isChecked()),
            "opt_load_use_weights": int(self.weight_button.isChecked()),
            "opt_load_index": self.index_spin.value(),
            "opt_load_weights": self.validate_weights(self.weight_line.text())
        }
        return inputs

    def setInputs(self, inputs: dict):
        self.opt_dir_line.setText(inputs["opt_load_dir"])
        self.generation_spin.setValue(inputs["opt_load_generation"])
        if inputs["opt_load_use_index"]:
            self.index_button.toggle()
        elif inputs["opt_load_use_weights"]:
            self.weight_button.toggle()
        self.index_spin.setValue(inputs["opt_load_index"])

        if isinstance(inputs["opt_load_weights"], list):
            self.weight_line.setText(",".join([str(w) for w in inputs["opt_load_weights"]]))
        elif isinstance(inputs["opt_load_weights"], str):
            self.weight_line.setText(inputs["opt_load_weights"])

    @staticmethod
    def assignQSettings(inputs: dict):
        q_settings.setValue("opt_load_dir", inputs["opt_load_dir"])
        q_settings.setValue("opt_load_generation", inputs["opt_load_generation"])
        q_settings.setValue("opt_load_use_index", inputs["opt_load_use_index"])
        q_settings.setValue("opt_load_use_weights", inputs["opt_load_use_weights"])
        q_settings.setValue("opt_load_index", inputs["opt_load_index"])
        q_settings.setValue("opt_load_weights", inputs["opt_load_weights"])


class LoadAirfoilAlgFile(PymeadDialog):
    def __init__(self, theme: dict, parent=None):
        self.load_airfoil_alg_file_widget = LoadAirfoilAlgFileWidget()
        super().__init__(parent=parent, window_title="Load Optimized Airfoil",
                         widget=self.load_airfoil_alg_file_widget, theme=theme)

    def valuesFromWidgets(self):
        return self.load_airfoil_alg_file_widget.valuesFromWidgets()


class SaveAsDialog(QFileDialog):
    def __init__(self, parent, file_filter: str = "JMEA Files (*.jmea)"):
        super().__init__(parent=parent)
        self.setFileMode(QFileDialog.FileMode.AnyFile)
        self.setNameFilter(self.tr(file_filter))
        self.setViewMode(QFileDialog.ViewMode.Detail)


class NewGeoColDialog(PymeadDialog):
    def __init__(self, theme: dict, parent=None, window_title: str or None = None, message: str or None = None):
        w = QWidget() if message is None else QLabel(message)
        window_title = window_title if window_title is not None else "Save Changes?"
        super().__init__(parent=parent, window_title=window_title, widget=w, theme=theme)
        self.reject_changes = False
        self.save_successful = False

    def create_button_box(self):

        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No | QDialogButtonBox.StandardButton.Cancel, self)
        buttonBox.button(QDialogButtonBox.StandardButton.Yes).clicked.connect(self.yes)
        buttonBox.button(QDialogButtonBox.StandardButton.Yes).clicked.connect(self.accept)
        buttonBox.button(QDialogButtonBox.StandardButton.No).clicked.connect(self.no)
        buttonBox.button(QDialogButtonBox.StandardButton.No).clicked.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        return buttonBox

    @pyqtSlot()
    def yes(self):
        try:
            save_successful = self.parent().save_geo_col()
            self.save_successful = save_successful
        except:
            self.save_successful = False

    @pyqtSlot()
    def no(self):
        self.reject_changes = True


class ExitDialog(PymeadDialog):
    def __init__(self, theme: dict, parent=None, window_title: str or None = None, message: str or None = None):
        window_title = window_title if window_title is not None else "Exit?"
        message = message if message is not None else "Airfoil not saved.\nAre you sure you want to exit?"
        w = QLabel(message)
        super().__init__(parent=parent, window_title=window_title, widget=w, theme=theme)

    def create_button_box(self):
        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No, self)
        buttonBox.button(QDialogButtonBox.StandardButton.Yes).clicked.connect(self.accept)
        buttonBox.button(QDialogButtonBox.StandardButton.No).clicked.connect(self.reject)
        return buttonBox


class EditBoundsDialog(PymeadDialog):
    def __init__(self, geo_col: GeometryCollection, theme: dict, parent=None):
        self.bv_table = BoundsValuesTable(geo_col=geo_col)
        super().__init__(parent=parent, window_title="Edit Bounds", widget=self.bv_table, theme=theme,
                         minimum_width=466, minimum_height=497)

    def resizetoFit(self):
        self.bv_table.resizeColumnsToContents()
        self.resize(self.bv_table.sizeHint())


class OptimizationDialogVTabWidget(PymeadDialogVTabWidget):
    def __init__(self, parent, widgets: dict, settings_override: dict):
        super().__init__(parent=parent, widgets=widgets, settings_override=settings_override)
        self.objectives = None
        self.constraints = None


class OptimizationSetupDialog(PymeadDialog):
    def __init__(self, parent, geo_col: GeometryCollection, theme: dict, grid: bool, settings_override: dict = None):
        w0 = GAGeneralSettingsDialogWidget()
        w3 = XFOILDialogWidget(current_airfoils=[k for k in geo_col.container()["airfoils"]])
        w4 = MSETDialogWidget(geo_col=geo_col, theme=theme)
        w2 = GAConstraintsTerminationDialogWidget(geo_col=geo_col, theme=theme, grid=grid)
        w7 = MultiPointOptDialogWidget()
        w5 = MSESDialogWidget2(geo_col=geo_col)
        w1 = GeneticAlgorithmDialogWidget(gui_obj=geo_col.gui_obj, multi_point_dialog_widget=w7)
        w6 = PymeadDialogWidget(os.path.join(GUI_DEFAULTS_DIR, 'mplot_settings.json'))
        w = OptimizationDialogVTabWidget(parent=self, widgets={'General Settings': w0,
                                                        'Genetic Algorithm': w1,
                                                        'Constraints/Termination': w2,
                                                               'Multi-Point Optimization': w7,
                                                        'XFOIL': w3, 'MSET': w4, 'MSES': w5, 'MPLOT': w6},
                                         settings_override=settings_override)
        super().__init__(parent=parent, window_title='Optimization Setup', widget=w, theme=theme)
        w.objectives = geo_col.gui_obj.objectives
        w.constraints = geo_col.gui_obj.constraints

        w1.update_objectives_and_constraints()  # IMPORTANT: makes sure that the objectives/constraints get stored

        self.geo_col = geo_col
        self.mset_widget = w4
        self.xfoil_widget = w3
        self.mses_widget = w5
        self.constraints_widget = w2
        self.mset_widget.sigMEAChanged.connect(self.mses_widget.widget_dict["xtrs"].onMEAChanged)
        w0.sigMEAFileChanged.connect(self.onMEAFileChanged)

    def onMEAFileChanged(self, airfoil_objs: typing.List[Airfoil], mea_objs: typing.List[MEA],
                         current_airfoil: str or None, current_mea: str or None):
        self.geo_col = get_parent(self, 1).geo_col
        self.mset_widget.geo_col = self.geo_col
        self.mses_widget.geo_col = self.geo_col
        self.constraints_widget.geo_col = self.geo_col
        self.xfoil_widget.widget_dict["airfoil"]["widget"].clear()
        self.mset_widget.widget_dict["mea"].widget.clear()

        # Add the new list of parameters to the actuator disk MSES widget
        param_list = [param for param in self.geo_col.container()["params"]]
        dv_list = [dv + " (DV)" for dv in self.geo_col.container()["desvar"]]
        AD_widget = self.mses_widget.widget_dict["AD"]
        AD_widget.param_list = [""] + param_list + dv_list
        for ad in AD_widget.widget_dict.values():
            ad["XCDELH-Param"].widget.clear()
            ad["XCDELH-Param"].widget.addItems(AD_widget.param_list)

        airfoil_names = [airfoil.name() for airfoil in airfoil_objs]
        mea_names = [mea.name() for mea in mea_objs]
        self.xfoil_widget.widget_dict["airfoil"]["widget"].addItems(airfoil_names)
        self.mset_widget.widget_dict["mea"].widget.addItems(mea_names)
        self.constraints_widget.widget_dict["constraints"].reorderRegenerateWidgets(airfoil_names)

        if current_airfoil is not None:
            self.xfoil_widget.widget_dict["airfoil"]["widget"].setCurrentText(current_airfoil)
        else:
            if len(airfoil_names) > 0:
                self.xfoil_widget.widget_dict["airfoil"]["widget"].setCurrentText(airfoil_names[0])

        if current_mea is not None:
            self.mset_widget.widget_dict["mea"].widget.setCurrentText(current_mea)
        else:
            if len(mea_names) > 0:
                self.mset_widget.widget_dict["mea"].widget.setCurrentText(mea_names[0])


class ExportCoordinatesDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        w = QWidget()
        self.grid_widget = {}
        self.grid_layout = QGridLayout()
        self.setInputs()
        w.setLayout(self.grid_layout)
        super().__init__(parent=parent, window_title="Export Airfoil Coordinates", widget=w, theme=theme,
                         minimum_width=626, minimum_height=379)
        self.grid_widget["airfoil_order"]["line"].setText(
            ",".join([k for k in self.parent().geo_col.container()["airfoils"].keys()]))

    def setInputs(self):
        widget_dict = load_data(os.path.join(GUI_DIALOG_WIDGETS_DIR, "export_coordinates_dialog.json"))
        for row_name, row_dict in widget_dict.items():
            self.grid_widget[row_name] = {}
            for w_name, w_dict in row_dict.items():
                widget = getattr(sys.modules[__name__], w_dict["w"])()
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
                if "checkstate" in w_dict.keys() and isinstance(widget, QCheckBox):
                    widget.setChecked(w_dict["checkstate"])
                if "lower_bound" in w_dict.keys() and (isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox)):
                    widget.setMinimum(w_dict["lower_bound"])
                if "upper_bound" in w_dict.keys() and (isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox)):
                    widget.setMaximum(w_dict["upper_bound"])
                if "value" in w_dict.keys() and (isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox)):
                    widget.setValue(w_dict["value"])
                self.grid_layout.addWidget(widget, w_dict["grid"][0], w_dict["grid"][1], w_dict["grid"][2],
                                           w_dict["grid"][3])

    def value(self):
        inputs = {}
        for k, v in self.grid_widget.items():
            if "line" in v.keys():
                inputs[k] = v["line"].text()
            elif "spinbox" in v.keys():
                inputs[k] = v["spinbox"].value()
            elif "checkbox" in v.keys():
                inputs[k] = bool(v["checkbox"].isChecked())
            else:
                inputs[k] = None

        # Make sure any newline characters are not double-escaped:
        for k, input_ in inputs.items():
            if isinstance(input_, str):
                inputs[k] = input_.replace('\\n', '\n')

        return inputs

    def select_directory(self, line_edit: QLineEdit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory", os.path.expanduser("~"),
                                                        QFileDialog.Option.ShowDirsOnly)
        if selected_dir:
            line_edit.setText(selected_dir)

    def format_mses(self):
        self.grid_widget["header"]["line"].setText("airfoil_name\\n-3.0 3.0 -3.0 3.0")
        self.grid_widget["separator"]["line"].setText("999.0 999.0\\n")
        self.grid_widget["delimiter"]["line"].setText(" ")
        self.grid_widget["file_name"]["line"].setText("blade.airfoil_name")


class ExportCSVDialogWidget(PymeadDialogWidget2):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "csv_file": PymeadLabeledLineEdit("Select CSV File", push_label="Select Filename")
        }

    def establishWidgetConnections(self):
        self.widget_dict["csv_file"].push.clicked.connect(self.select_data_file)

    def select_data_file(self):
        select_data_file(
            parent=self,
            line_edit=self.widget_dict["csv_file"].widget,
            starting_dir=get_setting("export_csv_default_open_location")
        )
        file_name = self.widget_dict["csv_file"].widget.text()
        if file_name:
            set_setting("export_csv_default_open_location", file_name)


class ExportCSVDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        widget = ExportCSVDialogWidget()
        super().__init__(parent, window_title="Export CSV", widget=widget, theme=theme,
                         minimum_width=450, fixed_height=125)


class ExportControlPointsDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        w = QWidget()
        self.grid_widget = {}
        self.grid_layout = QGridLayout()
        self.setInputs()
        w.setLayout(self.grid_layout)
        w.setMinimumWidth(400)
        super().__init__(parent=parent, window_title="Export Control Points", widget=w, theme=theme,
                         minimum_width=500, fixed_height=185)
        self.grid_widget["airfoil_order"]["line"].setText(
            ",".join([k for k in self.parent().geo_col.container()["airfoils"].keys()]))

    def setInputs(self):
        widget_dict = load_data(os.path.join(GUI_DIALOG_WIDGETS_DIR, "export_control_points_dialog.json"))
        for row_name, row_dict in widget_dict.items():
            self.grid_widget[row_name] = {}
            for w_name, w_dict in row_dict.items():
                widget = getattr(sys.modules[__name__], w_dict["w"])()
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

    def value(self):
        inputs = {k: v["line"].text() if "line" in v.keys() else None for k, v in self.grid_widget.items()}

        # Make sure any newline characters are not double-escaped:
        for k, input_ in inputs.items():
            if isinstance(input_, str):
                inputs[k] = input_.replace('\\n', '\n')

        return inputs

    def select_directory(self, line_edit: QLineEdit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory", os.path.expanduser("~"),
                                                        QFileDialog.Option.ShowDirsOnly)
        if selected_dir:
            line_edit.setText(selected_dir)


class ExportIGESDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        widget = QWidget()
        super().__init__(parent=parent, window_title="Export IGES", widget=widget, theme=theme,
                         minimum_width=600, fixed_height=271)

        self.grid_widget = {}
        self.grid_layout = QGridLayout()

        self.setInputs()
        widget.setLayout(self.grid_layout)

    def setInputs(self):
        widget_dict = load_data(os.path.join(GUI_DIALOG_WIDGETS_DIR, "export_IGES.json"))
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
                for attr_name, attr_value in w_dict.items():
                    if attr_name not in ["text", "func", "tool_tip", "grid", "w"]:
                        getattr(widget, attr_name)(attr_value)
                if isinstance(widget, QDoubleSpinBox):
                    widget.setMinimum(-np.inf)
                    widget.setMaximum(np.inf)
                self.grid_layout.addWidget(widget, w_dict["grid"][0], w_dict["grid"][1], w_dict["grid"][2],
                                           w_dict["grid"][3])

    def value(self):
        inputs = {
            "dir": self.grid_widget["choose_dir"]["line"].text(),
            "file_name": self.grid_widget["file_name"]["line"].text(),
            "rotation": [self.grid_widget["rotation"][xyz].value() for xyz in ["x", "y", "z"]],
            "scaling": [self.grid_widget["scaling"][xyz].value() for xyz in ["x", "y", "z"]],
            "translation": [self.grid_widget["translation"][xyz].value() for xyz in ["x", "y", "z"]],
            "transformation_order": self.grid_widget["transform_order"]["line"].text()
        }

        # Make sure any newline characters are not double-escaped:
        for k, input_ in inputs.items():
            if isinstance(input_, str):
                inputs[k] = input_.replace('\\n', '\n')

        return inputs

    def select_directory(self, line_edit: QLineEdit):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select a directory", os.path.expanduser("~"),
                                                        QFileDialog.Option.ShowDirsOnly)
        if selected_dir:
            line_edit.setText(selected_dir)


class AirfoilMatchingDialog(PymeadDialog):
    def __init__(self, parent, airfoil_names: typing.List[str], theme: dict):

        widget = QWidget()
        super().__init__(parent, window_title="Choose Airfoil to Match", widget=widget, theme=theme,
                         minimum_width=300, minimum_height=200)

        self.airfoil_names = airfoil_names

        self.lay = QGridLayout()

        self.inputs = self.setInputs()
        widget.setLayout(self.lay)

        for i in self.inputs:
            row_count = self.lay.rowCount()
            self.lay.addWidget(i.label, row_count, 0)
            if i.push is None:
                self.lay.addWidget(i.widget, row_count, 1, 1, 2)
            else:
                self.lay.addWidget(i.widget, row_count, 1, 1, 1)
                self.lay.addWidget(i.push, row_count, 2, 1, 1)

    def setInputs(self):
        inputs = [
            PymeadLabeledComboBox(label="Tool Airfoil", tool_tip="pymead Airfoil object name; the parametrization used "
                                                                 "to match the target airfoil",
                                  items=self.airfoil_names),
            PymeadLabeledComboBox(label="Airfoil type", tool_tip="Choose whether to use an AirfoilTools airfoil or a "
                                                                 "coordinate-file airfoil",
                                  items=["AirfoilTools", "Coordinate File"]),
            PymeadLabeledLineEdit(label="Web Airfoil", tool_tip="URL-name of the AirfoilTools airfoil (name in the "
                                                                "parentheses on airfoiltools.com)", text="n0012-il"),
            PymeadLabeledLineEdit(label="Airfoil from File", tool_tip="Absolute file path of a Selig-format "
                                                                      "(counter-clockwise starting with upper trailing "
                                                                      "edge, space-delimited, airfoil coordinates file",
                                  text="", push_label="Select Airfoil")
        ]

        # Run the connection once to show or hide the correct widgets
        self.inputs = inputs
        self.airfoilTypeChanged(inputs[1].value())

        inputs[1].sigValueChanged.connect(self.airfoilTypeChanged)
        inputs[3].push.clicked.connect(self.selectDatFile)

        return inputs

    def airfoilTypeChanged(self, airfoil_type: str):
        if airfoil_type == "AirfoilTools":
            self.inputs[2].label.show()
            self.inputs[2].widget.show()
            self.inputs[3].label.hide()
            self.inputs[3].widget.hide()
            self.inputs[3].push.hide()
        else:
            self.inputs[2].label.hide()
            self.inputs[2].widget.hide()
            self.inputs[3].label.show()
            self.inputs[3].widget.show()
            self.inputs[3].push.show()

    def selectDatFile(self):
        select_data_file(parent=self, line_edit=self.inputs[3].widget)

    def value(self):
        airfoil_type = self.inputs[1].value()
        if airfoil_type == "AirfoilTools":
            target_airfoil = self.inputs[2].value()
        else:
            data_file = self.inputs[3].value()
            if not os.path.exists(data_file):
                raise TargetPathNotFoundError(f"Could not find airfoil file {data_file} in the system")
            target_airfoil = np.loadtxt(data_file)
        return {"tool_airfoil": self.inputs[0].value(), "target_airfoil": target_airfoil}


class BSplineDegreeDialogWidget(PymeadDialogWidget2):
    def __init__(self, parent=None):
        super().__init__(parent)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "degree": PymeadLabeledSpinBox(
                label="Degree",
                tool_tip="Degree of the B-spline basis functions",
                minimum=2,
                maximum=67,
                value=3
            )
        }


class BSplineDegreeDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        widget = BSplineDegreeDialogWidget()
        super().__init__(
            parent,
            window_title="Select Basis Function Degree",
            widget=widget,
            theme=theme,
            minimum_width=300
        )


class LoadPointsDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        widget = QWidget()
        super().__init__(parent, window_title="Load Points", widget=widget, theme=theme,
                         minimum_width=450, minimum_height=202)
        self.lay = QGridLayout()
        widget.setLayout(self.lay)
        self.inputs = self.setInputs()

        for i in self.inputs:
            row_count = self.lay.rowCount()
            self.lay.addWidget(i.label, row_count, 0)
            if i.push is None:
                self.lay.addWidget(i.widget, row_count, 1, 1, 2)
            else:
                self.lay.addWidget(i.widget, row_count, 1, 1, 1)
                self.lay.addWidget(i.push, row_count, 2, 1, 1)

    def setInputs(self):
        explanation = PymeadLabeledPlainTextEdit(
            label=None,
            text="Load points from a .txt/.dat/.csv file in space- or comma-delimited format with two columns: x and y",
            read_only=True
        )
        file_name_widget = PymeadLabeledLineEdit(
            label="Data File",
            text="",
            read_only=False,
            push_label="Choose File"
        )
        file_name_widget.push.clicked.connect(self.select_data_file)

        inputs = [explanation, file_name_widget]
        return inputs

    def select_data_file(self):
        select_data_file(
            parent=self,
            line_edit=self.inputs[1].widget,
            starting_dir=get_setting("load_points_default_open_location")
        )
        file_name = self.inputs[1].widget.text()
        if file_name:
            set_setting("load_points_default_open_location", file_name)

    def value(self):
        return self.inputs[1].widget.text()


class MakeAirfoilRelativeDialog(PymeadDialog):
    def __init__(self, theme: dict, geo_col: GeometryCollection, parent=None):
        if len(geo_col.container()["airfoils"]) == 0:
            raise ValueError("An airfoil must be created first before a point can be made airfoil-relative")
        widget = QWidget()
        super().__init__(parent=parent, window_title="Airfoil-Relative Point", widget=widget, theme=theme,
                         minimum_width=250, fixed_height=125)
        self.lay = QGridLayout()
        widget.setLayout(self.lay)
        self.geo_col = geo_col
        self.inputs = self.setInputs()

        for i in self.inputs:
            row_count = self.lay.rowCount()
            self.lay.addWidget(i.label, row_count, 0)
            if i.push is None:
                self.lay.addWidget(i.widget, row_count, 1, 1, 2)
            else:
                self.lay.addWidget(i.widget, row_count, 1, 1, 1)
                self.lay.addWidget(i.push, row_count, 2, 1, 1)

    def setInputs(self):
        airfoil_list = list(self.geo_col.container()["airfoils"].keys())
        inputs = [
            PymeadLabeledComboBox(label="Airfoil",
                                  tool_tip="Select the airfoil to which to bind the selected points",
                                  items=airfoil_list),
        ]
        self.inputs = inputs
        return inputs

    def value(self):
        return {"airfoil": self.inputs[0].value()}


class AirfoilDialog(PymeadDialog):
    def __init__(self, theme: dict, geo_col: GeometryCollection, parent=None):
        if len(geo_col.container()["points"]) == 0:
            raise ValueError("An airfoil cannot be created until at least one Point has been added.")
        widget = QWidget()
        super().__init__(parent=parent, window_title="Create Airfoil", widget=widget, theme=theme,
                         minimum_width=300, fixed_height=234)
        self.lay = QGridLayout()
        widget.setLayout(self.lay)
        self.geo_col = geo_col
        self.inputs = self.setInputs()

        for i in self.inputs:
            row_count = self.lay.rowCount()
            self.lay.addWidget(i.label, row_count, 0)
            if i.push is None:
                self.lay.addWidget(i.widget, row_count, 1, 1, 2)
            else:
                self.lay.addWidget(i.widget, row_count, 1, 1, 1)
                self.lay.addWidget(i.push, row_count, 2, 1, 1)

    def setInputs(self):
        point_list = list(self.geo_col.container()["points"].keys())
        inputs = [
            PymeadLabeledComboBox(label="Leading Edge",
                                  tool_tip="Select the point corresponding to the airfoil's leading edge",
                                  items=point_list),
            PymeadLabeledComboBox(label="Trailing Edge",
                                  tool_tip="Select the point corresponding to the airfoil's trailing edge.\n"
                                           "This point does not have to be physically present on the airfoil,\n"
                                           "but is only used to determine the chord length and angle of attack",
                                  items=point_list),
            PymeadLabeledCheckbox(label="Thin Airfoil",
                                  tool_tip="A thin airfoil has coincident trailing edge,\nupper surface end, and "
                                           "lower surface end",
                                  initial_state=0),
            PymeadLabeledComboBox(label="Upper Surface End",
                                  tool_tip="Select the trailing edge point that lies on the upper surface.\nThis "
                                           "is the first airfoil point using counter-clockwise ordering",
                                  items=point_list),
            PymeadLabeledComboBox(label="Lower Surface End",
                                  tool_tip="Select the trailing edge point that lies on the lower surface.\nThis"
                                           " is the last airfoil point using counter-clockwise ordering",
                                  items=point_list)
        ]
        inputs[2].sigValueChanged.connect(self.thinAirfoilChecked)
        for combo_idx in [0, 1, 3, 4]:
            inputs[combo_idx].sigValueChanged.connect(self.pointObjectChanged)
        self.inputs = inputs
        self.pointObjectChanged(None)
        return inputs

    def thinAirfoilChecked(self, state: int):
        self.inputs[3].setReadOnly(bool(state))
        self.inputs[4].setReadOnly(bool(state))

    def pointObjectChanged(self, _):
        self.geo_col.clear_selected_objects()
        for combo_idx in [0, 1, 3, 4]:
            self.geo_col.select_object(self.geo_col.container()["points"][self.inputs[combo_idx].value()])

    def value(self):
        return {
            "leading_edge": self.geo_col.container()["points"][self.inputs[0].value()] if self.inputs[0].value() in self.geo_col.container()["points"] else None,
            "trailing_edge": self.geo_col.container()["points"][self.inputs[1].value()] if self.inputs[1].value() in self.geo_col.container()["points"] else None,
            "thin_airfoil": self.inputs[2].value(),
            "upper_surf_end": self.geo_col.container()["points"][self.inputs[3].value()] if self.inputs[3].value() in self.geo_col.container()["points"] else None,
            "lower_surf_end": self.geo_col.container()["points"][self.inputs[4].value()] if self.inputs[4].value() in self.geo_col.container()["points"] else None
        }


class WebAirfoilDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        widget = QWidget()
        super().__init__(parent, window_title="Load Airfoil from Coordinates", widget=widget, theme=theme,
                         minimum_width=400, fixed_height=236)
        self.lay = QGridLayout()
        widget.setLayout(self.lay)
        self.inputs = self.setInputs()

        for i in self.inputs:
            row_count = self.lay.rowCount()
            self.lay.addWidget(i.label, row_count, 0)
            if i.push is None:
                self.lay.addWidget(i.widget, row_count, 1, 1, 2)
            else:
                self.lay.addWidget(i.widget, row_count, 1, 1, 1)
                self.lay.addWidget(i.push, row_count, 2, 1, 1)

        self.setMinimumWidth(300)

    def setInputs(self):
        inputs = [
            PymeadLabeledComboBox(label="Airfoil type", tool_tip="Choose whether to use an AirfoilTools airfoil or a "
                                                                 "coordinate-file airfoil",
                                  items=["AirfoilTools", "Coordinate File"]),
            PymeadLabeledLineEdit(label="Web Airfoil", tool_tip="URL-name of the AirfoilTools airfoil (name in the "
                                                                "parentheses on airfoiltools.com)", text="n0012-il"),
            PymeadLabeledLineEdit(label="Airfoil from File", tool_tip="Absolute file path of a Selig-format "
                                                                      "(counter-clockwise starting with upper trailing "
                                                                      "edge, space-delimited, airfoil coordinates file",
                                  text="", push_label="Select Airfoil", read_only=True),
            PymeadLabeledCheckbox(label="Reference?", tool_tip="Whether to add this airfoil as a reference polyline "
                                                               "rather than an airfoil object. This produces a static "
                                                               "plot of the airfoil coordinates in the background that "
                                                               "can be used for comparison"),
            PymeadLabeledColorSelector(label="Reference Color", tool_tip="Color used for the airfoil coordinates; only "
                                                                         "used if 'Reference' is selected",
                                       read_only=True, parent=self)
        ]
        inputs[0].sigValueChanged.connect(self.airfoilTypeChanged)
        inputs[2].push.clicked.connect(self.selectDatFile)
        inputs[3].sigValueChanged.connect(self.referenceSelectionChanged)
        return inputs

    def airfoilTypeChanged(self, airfoil_type: str):
        if airfoil_type == "AirfoilTools":
            self.inputs[1].setReadOnly(False)
            self.inputs[2].setReadOnly(True)
        else:
            self.inputs[1].setReadOnly(True)
            self.inputs[2].setReadOnly(False)

    def selectDatFile(self):
        select_data_file(parent=self, line_edit=self.inputs[2].widget)

    def referenceSelectionChanged(self, reference: bool):
        self.inputs[4].widget.setEnabled(reference)

    def value(self):
        if self.inputs[0].value() == "AirfoilTools":
            return [self.inputs[1].value(), self.inputs[3].value(), self.inputs[4].value()]
        else:
            return [self.inputs[2].value(), self.inputs[3].value(), self.inputs[4].value()]


class MSESFieldPlotDialogWidget(PymeadDialogWidget):
    def __init__(self, default_field_dir: str = None):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'mses_field_plot_settings.json'))
        if default_field_dir is not None:
            self.widget_dict['analysis_dir']['widget'].setText(default_field_dir)
        self.widget_dict["analysis_dir"]["widget"].textChanged.connect(partial(set_setting, "plot-field-dir"))

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self, line_edit=line_edit, starting_dir=tempfile.gettempdir())

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class MSESFieldPlotDialog(PymeadDialog):
    def __init__(self, parent: QWidget, theme: dict, default_field_dir: str = None):
        w = MSESFieldPlotDialogWidget(default_field_dir=default_field_dir)
        super().__init__(parent=parent, window_title="MSES Field Plot Settings", widget=w, theme=theme,
                         minimum_width=450, fixed_height=155)


class GridBounds(QWidget):
    boundsChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent=parent)
        layout = QGridLayout()
        self.setLayout(layout)
        label_names = ['Left', 'Right', 'Bottom', 'Top']
        self.labels = {k: QLabel(k, self) for k in label_names}
        label_positions = {
            'Left': [1, 0],
            'Right': [1, 2],
            'Bottom': [2, 0],
            'Top': [2, 2],
        }
        self.widgets = {k: QDoubleSpinBox() for k in label_positions}
        defaults = {
            'Left': [-5.0, 1, 1],
            'Right': [5.0, 1, 3],
            'Bottom': [-5.0, 2, 1],
            'Top': [5.0, 2, 3],
        }
        for k, v in defaults.items():
            self.widgets[k].setMinimum(-np.inf)
            self.widgets[k].setMaximum(np.inf)
            self.widgets[k].setValue(v[0])
            self.widgets[k].setMinimumWidth(75)
        layout.addWidget(QHSeperationLine(self), 0, 0, 1, 1)
        layout.addWidget(QLabel('Grid Bounds', self), 0, 1, 1, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QHSeperationLine(self), 0, 3, 1, 1)
        for label_name in label_names:
            layout.addWidget(self.labels[label_name], label_positions[label_name][0], label_positions[label_name][1],
                             1, 1, Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self.widgets[label_name], defaults[label_name][1], defaults[label_name][2], 1, 1,
                             Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(QHSeperationLine(self), 3, 0, 1, 4)
        for w in self.widgets.values():
            w.valueChanged.connect(self.valueChanged)

    def setValue(self, value_list: list):
        if len(value_list) != 4:
            raise ValueError('Length of input value list must be 4')
        self.widgets['Left'].setValue(value_list[0])
        self.widgets['Right'].setValue(value_list[1])
        self.widgets['Bottom'].setValue(value_list[2])
        self.widgets['Top'].setValue(value_list[3])

    def value(self):
        return [self.widgets['Left'].value(), self.widgets['Right'].value(), self.widgets['Bottom'].value(),
                self.widgets['Top'].value()]

    def valueChanged(self, _):
        self.boundsChanged.emit()


class PlotExportDialogWidget(PymeadDialogWidget2):
    def __init__(self, gui_obj, current_min_level: float, current_max_level: float, parent=None):
        self.gui_obj = gui_obj
        self.current_min_level = current_min_level
        self.current_max_level = current_max_level
        super().__init__(parent=parent)

    def initializeWidgets(self):
        available_fonts = ["DejaVu Sans Mono", "DejaVu Serif", "DejaVu Sans"]
        self.widget_dict = {
            "save_name": PymeadLabeledLineEdit(label="File Name", text="airfoil.png",
                                               tool_tip="Name of the image. If the '.png' extension is not included,"
                                                        " it will be appended automatically"),
            "save_dir": PymeadLabeledLineEdit(label="Save Directory", push_label="Choose Directory",
                                              text=get_setting("plot-field-export-dir")),
            "width": PymeadLabeledSpinBox(label="Image Width", minimum=100, maximum=10000, value=1920,
                                          tool_tip="Increasing this value increases the plot resolution but also "
                                                   "increases the image file size"),
            "tick_font_family": PymeadLabeledComboBox(label="Tick Font", items=available_fonts,
                                                      current_item=get_setting("axis-tick-font-family")),
            "tick_font_size": PymeadLabeledSpinBox(label="Tick Point Size", minimum=1, maximum=100,
                                                   value=get_setting("axis-tick-point-size")),
            "label_font_family": PymeadLabeledComboBox(label="Label Font", items=available_fonts,
                                                       current_item=get_setting("axis-label-font-family")),
            "label_font_size": PymeadLabeledSpinBox(label="Label Point Size", minimum=1, maximum=100,
                                                    value=get_setting("axis-label-point-size")),


        }
        if self.current_min_level is not None and self.current_max_level is not None:
            self.widget_dict["min_level"] = PymeadLabeledDoubleSpinBox(
                label="Minimum Contour Level", minimum=-10000, maximum=10000, value=self.current_min_level,
                decimals=4, single_step=0.1)
            self.widget_dict["max_level"] = PymeadLabeledDoubleSpinBox(
                label="Maximum Contour Level", minimum=-10000, maximum=10000, value=self.current_max_level,
                decimals=4, single_step=0.1)

    def addWidgets(self, *args, **kwargs):
        # Add all the widgets
        for widget_name, widget in self.widget_dict.items():
            row_count = self.lay.rowCount()
            self.lay.addWidget(widget.label, row_count, 0)
            self.lay.addWidget(widget.widget, row_count, 1)
            if widget.push is not None:
                self.lay.addWidget(widget.push, row_count, 2)

    def establishWidgetConnections(self):
        self.widget_dict["save_dir"].push.clicked.connect(
            partial(select_directory, self, line_edit=self.widget_dict["save_dir"].widget))

        self.widget_dict["tick_font_family"].sigValueChanged.connect(self.tickFontChanged)
        self.widget_dict["tick_font_size"].sigValueChanged.connect(self.tickFontChanged)
        self.widget_dict["label_font_family"].sigValueChanged.connect(self.labelFontChanged)
        self.widget_dict["label_font_size"].sigValueChanged.connect(self.labelFontChanged)
        self.widget_dict["save_dir"].sigValueChanged.connect(partial(set_setting, "plot-field-export-dir"))
        if "min_level" in self.widget_dict and "max_level" in self.widget_dict:
            self.widget_dict["min_level"].sigValueChanged.connect(self.minLevelChanged)
            self.widget_dict["max_level"].sigValueChanged.connect(self.maxLevelChanged)

    def tickFontChanged(self, _):
        widget_values = self.value()
        tick_font = QFont(widget_values["tick_font_family"], widget_values["tick_font_size"])
        set_setting("cbar-tick-font-family", widget_values["tick_font_family"])
        set_setting("cbar-tick-point-size", widget_values["tick_font_size"])
        set_setting("axis-tick-font-family", widget_values["tick_font_family"])
        set_setting("axis-tick-point-size", widget_values["tick_font_size"])
        if self.gui_obj.cbar is not None:
            self.gui_obj.cbar.axis.setStyle(tickFont=tick_font)
            self.gui_obj.cbar.getAxis("right").setWidth(20 + 2 * widget_values["label_font_size"] +
                                                        2 * widget_values["tick_font_size"])
        self.gui_obj.airfoil_canvas.plot.getAxis("bottom").setTickFont(tick_font)
        self.gui_obj.airfoil_canvas.plot.getAxis("left").setTickFont(tick_font)

    def labelFontChanged(self, _):
        widget_values = self.value()
        theme_color = self.gui_obj.themes[self.gui_obj.current_theme]["main-color"]
        new_font = f"{widget_values['label_font_size']}pt {widget_values['label_font_family']}"
        set_setting("axis-label-font-family", widget_values["label_font_family"])
        set_setting("axis-label-point-size", widget_values["label_font_size"])
        for axis in ("left", "bottom"):
            label_text = self.gui_obj.airfoil_canvas.plot.getAxis(axis).label.toPlainText()
            self.gui_obj.airfoil_canvas.plot.setLabel(
                axis=axis, text=label_text, color=theme_color, font=new_font)
        if self.gui_obj.cbar is not None:
            cbar_text = self.gui_obj.cbar.getAxis("right").label.toPlainText()
            self.gui_obj.cbar.setLabel(axis="right", text=cbar_text, color=theme_color, font=new_font)
            self.gui_obj.cbar.getAxis("right").setWidth(20 + 2 * widget_values["label_font_size"] +
                                                        2 * widget_values["tick_font_size"])

    def setValue(self, d: dict):
        for d_name, d_value in d.items():
            try:
                self.widget_dict[d_name].setValue(d_value)
            except KeyError:
                pass

    def value(self) -> dict:
        return {k: v.value() for k, v in self.widget_dict.items()}

    def minLevelChanged(self, min_level: float):
        self.widget_dict["max_level"].widget.setMinimum(min_level + 0.0001)
        self.gui_obj.airfoil_canvas.setColorBarLevels(min_level, self.widget_dict["max_level"].widget.value())

    def maxLevelChanged(self, max_level: float):
        self.widget_dict["min_level"].widget.setMaximum(max_level - 0.0001)
        self.gui_obj.airfoil_canvas.setColorBarLevels(self.widget_dict["min_level"].widget.value(), max_level)


class PlotExportDialog(PymeadDialog):
    def __init__(self, parent, gui_obj, theme: dict, current_min_level: float, current_max_level: float):
        widget = PlotExportDialogWidget(gui_obj=gui_obj, current_min_level=current_min_level,
                                        current_max_level=current_max_level)
        super().__init__(parent, window_title="Plot Export", widget=widget, theme=theme,
                         minimum_width=500, fixed_height=300)


class ExitOptimizationDialog(PymeadDialog):
    def __init__(self, parent, theme: dict):
        widget = QLabel("An optimization task is running. Quit?")
        super().__init__(parent=parent, window_title="Terminate Optimization?", widget=widget, theme=theme,
                         fixed_width=264, fixed_height=100)


class SplitBezierDialogWidget(PymeadDialogWidget2):
    def __init__(self, bezier: Bezier, parent=None):
        self.bezier = bezier
        self.geo_col = self.bezier.geo_col
        self.split_point = None
        super().__init__(parent=parent)

    def initializeWidgets(self, *args, **kwargs):
        t = 0.5
        x = self.bezier.evaluate_xy(np.array([t]))[0, 0]
        assert isinstance(x, float)
        self.widget_dict = {
            "t": PymeadLabeledDoubleSpinBox(label="t", tool_tip="Parameter value where the curve will be split",
                                            minimum=0.0, maximum=1.0, value=t, decimals=15, single_step=0.01),
            "x": PymeadLabeledDoubleSpinBox(label="x", tool_tip="Value along the x-axis where the curve will be split",
                                            minimum=-np.inf, maximum=np.inf, value=x, decimals=15, single_step=0.01)
        }

    def establishWidgetConnections(self):
        self.widget_dict["t"].sigValueChanged.connect(self.onTChanged)
        self.widget_dict["x"].sigValueChanged.connect(self.onXChanged)

    def onTChanged(self, t: float):
        x = self.bezier.evaluate_xy(np.array([t]))[0, 0]
        assert isinstance(x, float)
        with QSignalBlocker(self.widget_dict["x"]):
            self.widget_dict["x"].setValue(x)
        self.onSpinValueChanged(t)

    def onXChanged(self, x: float):
        t = self.bezier.compute_t_corresponding_to_x(x)
        with QSignalBlocker(self.widget_dict["t"]):
            self.widget_dict["t"].setValue(t)
        self.onSpinValueChanged(t)

    def removeSplitPoint(self):
        if self.split_point is None:
            return
        self.geo_col.remove_pymead_obj(self.split_point)

    def onSpinValueChanged(self, t: float):
        self.removeSplitPoint()
        point_loc = self.bezier.evaluate_xy(np.array([t]))[0]
        self.split_point = self.geo_col.add_point(point_loc[0], point_loc[1])


class SplitBezierDialog(PymeadDialog):
    def __init__(self, bezier: Bezier, parent=None):
        w = SplitBezierDialogWidget(bezier)
        theme = bezier.gui_obj.themes[bezier.gui_obj.current_theme]
        super().__init__(parent=parent, window_title="Split Bézier", widget=w, theme=theme)

    def accept(self):
        self.w.removeSplitPoint()
        super().accept()

    def reject(self):
        self.w.removeSplitPoint()
        super().reject()

    def close(self):
        self.w.removeSplitPoint()
        super().close()


class SplitPolylineDialog(PymeadDialog):
    def __init__(self, parent, theme: dict, polyline: PolyLine, geo_col: GeometryCollection):
        self.polyline = polyline
        self.geo_col = geo_col
        widget = QWidget()
        super().__init__(parent=parent, window_title="Split PolyLine", widget=widget, theme=theme,
                         fixed_width=192, minimum_height=126)
        self.lay = QFormLayout()
        self.spin = QSpinBox()
        self.spin.setMinimum(3)
        self.spin.setMaximum(len(polyline.coords) - 3)
        self.lay.addRow("Split Index", self.spin)
        widget.setLayout(self.lay)
        self.split_point = None
        self.spin.valueChanged.connect(self.onSpinValueChanged)

    def removeSplitPoint(self):
        if self.split_point is None:
            return
        self.geo_col.remove_pymead_obj(self.split_point)

    def onSpinValueChanged(self, new_value: int):
        self.removeSplitPoint()
        self.split_point = self.geo_col.add_point(self.polyline.coords[new_value, 0],
                                                  self.polyline.coords[new_value, 1])

    def accept(self):
        self.removeSplitPoint()
        super().accept()

    def reject(self):
        self.removeSplitPoint()
        super().reject()

    def close(self):
        self.removeSplitPoint()
        super().close()

    def value(self):
        return self.spin.value()


class GeneralSettingsDialogWidget(PymeadDialogWidget2):
    def __init__(self, geo_col: GeometryCollection, parent=None):
        self.geo_col = geo_col
        super().__init__(parent=parent)

    def initializeWidgets(self, *args, **kwargs):
        self.widget_dict = {
            "length_unit": PymeadLabeledComboBox(label="Length Unit", items=["m", "mm", "in", "cm"],
                                                 current_item=self.geo_col.units.current_length_unit()),
            "angle_unit": PymeadLabeledComboBox(label="Angle Unit", items=["rad", "deg"],
                                                current_item=self.geo_col.units.current_angle_unit())
        }

    def establishWidgetConnections(self):
        self.widget_dict["length_unit"].sigValueChanged.connect(self.onLengthUnitChanged)
        self.widget_dict["angle_unit"].sigValueChanged.connect(self.onAngleUnitChanged)

    def onLengthUnitChanged(self, new_unit: str):
        self.geo_col.switch_units("length", old_unit=self.geo_col.units.current_length_unit(), new_unit=new_unit)
        q_settings.setValue("length_unit", new_unit)
        if self.geo_col.gui_obj.cbar is not None:
            self.geo_col.gui_obj.plot_field(show_dialog=False)

    def onAngleUnitChanged(self, new_unit: str):
        self.geo_col.switch_units("angle", old_unit=self.geo_col.units.current_angle_unit(), new_unit=new_unit)
        q_settings.setValue("angle_unit", new_unit)


class SettingsDialog(PymeadDialog):

    def __init__(self, parent, geo_col: GeometryCollection, theme: dict, settings_override: dict = None):
        widgets = {
            "General": GeneralSettingsDialogWidget(geo_col=geo_col)
        }
        w = PymeadDialogVTabWidget(parent=None, widgets=widgets, settings_override=settings_override)
        super().__init__(parent=parent, window_title="Settings", widget=w, theme=theme,
                         minimum_width=256, minimum_height=210)


def convert_opt_settings_to_param_dict(opt_settings: dict, n_var: int, bypass_exe_check: bool = False,
                                       bypass_num_proc_check: bool = False) -> dict:
    param_dict = {'tool': opt_settings['Genetic Algorithm']['tool'],
                  'algorithm_save_frequency': opt_settings['Genetic Algorithm']['algorithm_save_frequency'],
                  'n_obj': len(opt_settings['Genetic Algorithm']['J'].split(',')),
                  'n_constr': len(opt_settings['Genetic Algorithm']['G'].split(',')) if opt_settings['Genetic Algorithm']['G'] != '' else 0,
                  'population_size': opt_settings['Genetic Algorithm']['pop_size'],
                  'n_ref_dirs': opt_settings['Genetic Algorithm']['pop_size'],
                  'n_offsprings': opt_settings['Genetic Algorithm']['n_offspring'],
                  'max_sampling_width': opt_settings['Genetic Algorithm']['max_sampling_width'],
                  'xl': 0.0,
                  'xu': 1.0,
                  'seed': opt_settings['Genetic Algorithm']['random_seed'],
                  'multi_point': opt_settings['Multi-Point Optimization']['multi_point_active'],
                  'design_idx': opt_settings['Multi-Point Optimization']['design_idx'],
                  'num_processors': opt_settings['Genetic Algorithm']['num_processors'],
                  'x_tol': opt_settings['Constraints/Termination']['x_tol'],
                  'cv_tol': opt_settings['Constraints/Termination']['cv_tol'],
                  'f_tol': opt_settings['Constraints/Termination']['f_tol'],
                  'nth_gen': opt_settings['Constraints/Termination']['nth_gen'],
                  'n_last': opt_settings['Constraints/Termination']['n_last'],
                  'n_max_gen': opt_settings['Constraints/Termination']['n_max_gen'],
                  'n_max_evals': opt_settings['Constraints/Termination']['n_max_evals'],
                  'xfoil_settings': {
                      'Re': opt_settings['XFOIL']['Re'],
                      'Ma': opt_settings['XFOIL']['Ma'],
                      'xtr': [opt_settings['XFOIL']['xtr_upper'], opt_settings['XFOIL']['xtr_lower']],
                      'N': opt_settings['XFOIL']['N'],
                      'iter': opt_settings['XFOIL']['iter'],
                      'visc': opt_settings['XFOIL']['viscous_flag'],
                      'timeout': opt_settings['XFOIL']['timeout'],
                      'prescribe': opt_settings['XFOIL']['prescribe'],
                      'airfoil': opt_settings['XFOIL']['airfoil'],
                  },
                  'mset_settings': convert_dialog_to_mset_settings(opt_settings['MSET']),
                  'mses_settings': convert_dialog_to_mses_settings(opt_settings['MSES']),
                  'mplot_settings': convert_dialog_to_mplot_settings(opt_settings['MPLOT']),
                  'constraints': opt_settings['Constraints/Termination']['constraints'],
                  'multi_point_active': opt_settings['Multi-Point Optimization']['multi_point_active'],
                  'multi_point_stencil': opt_settings['Multi-Point Optimization']['multi_point_stencil'],
                  'verbose': True,
                  'eta_crossover': opt_settings['Genetic Algorithm']['eta_crossover'],
                  'eta_mutation': opt_settings['Genetic Algorithm']['eta_mutation'],
                  }

    if opt_settings['XFOIL']['prescribe'] == 'Angle of Attack (deg)':
        param_dict['xfoil_settings']['alfa'] = opt_settings['XFOIL']['alfa']
    elif opt_settings['XFOIL']['prescribe'] == 'Viscous Cl':
        param_dict['xfoil_settings']['Cl'] = opt_settings['XFOIL']['Cl']
    elif opt_settings['XFOIL']['prescribe'] == 'Viscous Cl':
        param_dict['xfoil_settings']['CLI'] = opt_settings['XFOIL']['CLI']
    param_dict['mses_settings']['n_airfoils'] = param_dict['mset_settings']['n_airfoils']

    # First check to make sure MSET, MSES, and MPLOT can be found on system path and marked as executable:
    if not bypass_exe_check:
        if param_dict["tool"] == "XFOIL" and shutil.which('xfoil') is None:
            raise ValueError('XFOIL executable \'xfoil\' not found on system path')
        if param_dict["tool"] == "MSES" and shutil.which('mset') is None:
            raise ValueError('MSES suite executable \'mset\' not found on system path')
        if param_dict["tool"] == "MSES" and shutil.which('mses') is None:
            raise ValueError('MSES suite executable \'mses\' not found on system path')
        if param_dict["tool"] == "MSES" and shutil.which('mplot') is None:
            raise ValueError('MPLOT suite executable \'mplot\' not found on system path')

    if not bypass_num_proc_check and param_dict["num_processors"] > os.cpu_count():
        raise ValueError(f"Number of processors specified ({param_dict['num_processors']}) is greater than "
                         f"the number of logical processors available on this machine ({os.cpu_count()})")

    # TODO: reimplement this logic
    # norm_val_list = geo_col.extract_design_variable_values(bounds_normalized=True)
    # if isinstance(norm_val_list, str):
    #     error_message = norm_val_list
    #     self.disp_message_box(error_message, message_mode='error')
    #     exit_the_dialog = True
    #     early_return = True
    #     continue

    param_dict["n_var"] = n_var

    # CONSTRAINTS
    # First set the param dict constraints as a deepcopy to ensure that the thickness distribution constraint
    # file paths do not get overridden by the data.tolist() call
    opt_settings["Constraints/Termination"]["constraints"] = deepcopy(param_dict["constraints"])

    for airfoil_name, constraint_set in param_dict["constraints"].items():

        # Thickness distribution check parameters
        if constraint_set["thickness_at_points"][1]:
            thickness_file = constraint_set["thickness_at_points"][0]
            try:
                data = np.loadtxt(thickness_file)
                constraint_set["thickness_at_points"][0] = data.tolist()
            except FileNotFoundError:
                message = f"Thickness file {thickness_file} not found"
                raise FileNotFoundError(message)
        else:
            constraint_set['thickness_at_points'] = None

        # Internal geometry check parameters
        if constraint_set["internal_geometry"][1]:
            internal_geometry_file = constraint_set["internal_geometry"][0]
            try:
                data = np.loadtxt(internal_geometry_file)
                constraint_set["internal_geometry"][0] = data.tolist()
            except FileNotFoundError:
                message = f"Internal geometry file {internal_geometry_file} not found"
                raise FileNotFoundError(message)
        else:
            constraint_set["internal_geometry"] = None

    # MULTI-POINT OPTIMIZATION
    multi_point_stencil = None
    if opt_settings['Multi-Point Optimization']['multi_point_active']:
        try:
            multi_point_data = np.loadtxt(param_dict['multi_point_stencil'], delimiter=',')
            multi_point_stencil = read_stencil_from_array(multi_point_data, tool=param_dict["tool"])
        except FileNotFoundError:
            message = f'Multi-point stencil file {param_dict["multi_point_stencil"]} not found'
            raise FileNotFoundError(message)
    if param_dict['tool'] == 'MSES':
        param_dict['mses_settings']['multi_point_stencil'] = multi_point_stencil
    elif param_dict['tool'] == 'XFOIL':
        param_dict['xfoil_settings']['multi_point_stencil'] = multi_point_stencil
    else:
        raise ValueError(f"Currently only MSES and XFOIL are supported as analysis tools for "
                         f"aerodynamic shape optimization. Tool selected was {param_dict['tool']}")

    # Warm start parameters
    if opt_settings['General Settings']['warm_start_active']:
        opt_dir = opt_settings['General Settings']['warm_start_dir']
    else:
        opt_dir = make_ga_opt_dir(opt_settings['Genetic Algorithm']['root_dir'],
                                  opt_settings['Genetic Algorithm']['opt_dir_name'])

    param_dict['opt_dir'] = opt_dir
    # self.current_opt_folder = opt_dir.replace(os.sep, "/")

    name_base = 'ga_airfoil'
    name = [f"{name_base}_{i}" for i in range(opt_settings['Genetic Algorithm']['n_offspring'])]
    param_dict['name'] = name

    base_folder = os.path.join(opt_settings['Genetic Algorithm']['root_dir'],
                               opt_settings['Genetic Algorithm']['temp_analysis_dir_name'])
    param_dict['base_folder'] = base_folder
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    if opt_settings['General Settings']['warm_start_active']:
        param_dict['warm_start_generation'] = calculate_warm_start_index(
            opt_settings['General Settings']['warm_start_generation'], opt_dir)
        if param_dict['warm_start_generation'] == 0:
            opt_settings['General Settings']['warm_start_active'] = False
    param_dict_save = deepcopy(param_dict)
    if not opt_settings['General Settings']['warm_start_active']:
        save_data(param_dict_save, os.path.join(opt_dir, 'param_dict.json'))
        save_data(opt_settings, os.path.join(opt_dir, "opt_settings.json"))
    else:
        save_data(param_dict_save, os.path.join(
            opt_dir, f'param_dict_{param_dict["warm_start_generation"]}.json'))
        save_data(opt_settings, os.path.join(
            opt_dir, f"opt_settings_{param_dict['warm_start_generation']}.json"))

    return param_dict


class GeometricConstraintGraph:
    def __init__(self, theme: dict, geo_col: GeometryCollection,
                 pen=None, size: tuple = (1000, 300), grid: bool = False):
        self.geo_col = geo_col
        pg.setConfigOptions(antialias=True)

        if pen is None:
            pen = pg.mkPen(color="cornflowerblue", width=2)

        self.w = pg.GraphicsLayoutWidget(show=True, size=size)
        self.w.setMinimumWidth(800)
        self.w.setMinimumHeight(400)

        self.v = self.w.addPlot(pen=pen)
        self.v.setMenuEnabled(False)
        self.v.showGrid(x=grid, y=grid)
        self.v.setAspectLocked(True)
        self.legend = self.v.addLegend(offset=(-5, 5))

        self.plot_items = {
            "airfoil": self.v.plot(pen=pg.mkPen(color="cornflowerblue", width=2), name="Airfoil"),
            "min_radius_point": self.v.plot(
                symbol="star", symbolSize=20, symbolBrush=pg.mkBrush(color="#e8c205"),
                symbolPen=pg.mkPen(color="#787366")
            ),
            "min_radius_text": pg.TextItem(anchor=(1, 0)),
            "max_thickness_line": self.v.plot(pen=pg.mkPen(color="#32a852", width=1.5), name="Max Thickness"),
            "max_thickness_text": pg.TextItem(anchor=(0.5, 0.5)),
            "min_area_text": pg.TextItem(anchor=(0.5, 0.5)),
            "internal_geometry_line": self.v.plot(pen=pg.mkPen(color="#bb76c4", width=1.5), name="Internal Geometry")
        }
        # Some of the item types must be added explicitly to the plot
        for item in ["min_radius_text", "max_thickness_text", "min_area_text"]:
            self.v.addItem(self.plot_items[item])

        # Set the formatting for the graph based on the current theme
        self.set_formatting(theme=theme)

    def set_background(self, background_color: str):
        self.w.setBackground(background_color)

    def set_formatting(self, theme: dict):
        self.w.setBackground(theme["graph-background-color"])
        label_font = f"{get_setting('axis-label-point-size')}pt {get_setting('axis-label-font-family')}"
        self.v.setLabel(axis="bottom", text=f"x [{self.geo_col.units.current_length_unit()}]", font=label_font,
                           color=theme["main-color"])
        self.v.setLabel(axis="left", text=f"y [{self.geo_col.units.current_length_unit()}]",
                        font=label_font, color=theme["main-color"])
        tick_font = QFont(get_setting("axis-tick-font-family"), get_setting("axis-tick-point-size"))
        self.v.getAxis("bottom").setTickFont(tick_font)
        self.v.getAxis("left").setTickFont(tick_font)
        self.v.getAxis("bottom").setTextPen(theme["main-color"])
        self.v.getAxis("left").setTextPen(theme["main-color"])
        self.set_legend_label_format(theme)

    def set_legend_label_format(self, theme: dict):
        for _, label in self.legend.items:
            label.item.setHtml(f"<span style='font-family: DejaVu Sans; color: "
                               f"{theme['main-color']}; size: 8pt'>{label.text}</span>")


class GeometricConstraintGraphDialog(PymeadDialog):

    sigConstraintGraphClosed = pyqtSignal()

    def __init__(self, theme: dict, geo_col: GeometryCollection, grid: bool, airfoil_name: str, parent=None):
        self.graph = GeometricConstraintGraph(theme=theme, geo_col=geo_col, grid=grid)
        self.widget = self.graph.w
        super().__init__(parent=parent, window_title=f"Geometric Constraint Visualization - {airfoil_name}",
                         widget=self.widget, theme=theme)
        self.accepted.connect(self.onAccepted)
        self.rejected.connect(self.onRejected)

    def closeEvent(self, a0):
        self.sigConstraintGraphClosed.emit()
        super().closeEvent(a0)

    def onAccepted(self):
        self.sigConstraintGraphClosed.emit()

    def onRejected(self):
        self.sigConstraintGraphClosed.emit()


class FileOverwriteDialog(PymeadDialog):
    def __init__(self, theme: dict, parent=None):
        widget = QLabel("File already exists. Overwrite?")
        super().__init__(parent=parent, window_title="Overwrite?", widget=widget, theme=theme)
