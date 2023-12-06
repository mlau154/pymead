import typing
from abc import abstractmethod

import PyQt5.QtWidgets
import numpy as np
from typing import List
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QDoubleSpinBox, QComboBox, QSpinBox, \
    QTabWidget, QLabel, QMessageBox, QCheckBox, QVBoxLayout, QWidget, QGridLayout, QPushButton, QListView, QRadioButton
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QStandardItem, QStandardItemModel
import tempfile
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QStandardPaths

from pymead.core.mea import MEA
from pymead.gui.sampling_visualization import SamplingVisualizationWidget
from pymead.gui.infty_doublespinbox import InftyDoubleSpinBox
from pymead.gui.pyqt_vertical_tab_widget.pyqt_vertical_tab_widget import VerticalTabWidget
from pymead.gui.scientificspinbox_master.ScientificDoubleSpinBox import ScientificDoubleSpinBox
from pymead.gui.file_selection import *
from pymead.gui.separation_lines import QHSeperationLine
from pymead.utils.widget_recursion import get_parent
import sys
import os
from copy import deepcopy
from functools import partial
from pymead.utils.read_write_files import load_data, save_data, load_documents_path
from pymead.utils.dict_recursion import recursive_get
from pymead.gui.default_settings import xfoil_settings_default
from pymead.gui.bounds_values_table import BoundsValuesTable
from pymead.optimization.objectives_and_constraints import Objective, Constraint, FunctionCompileError
from pymead.analysis import cfd_output_templates
from pymead.analysis.utils import viscosity_calculator
from pymead import GUI_DEFAULTS_DIR, GUI_DIALOG_WIDGETS_DIR, q_settings
import pyqtgraph as pg
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtGui import QContextMenuEvent


mses_settings_json = load_data(os.path.join(GUI_DEFAULTS_DIR, 'mses_settings.json'))


ISMOM_CONVERSION = {item: idx + 1 for idx, item in enumerate(mses_settings_json['ISMOM']['addItems'])}
IFFBC_CONVERSION = {item: idx + 1 for idx, item in enumerate(mses_settings_json['IFFBC']['addItems'])}


get_set_value_names = {'QSpinBox': ('value', 'setValue', 'valueChanged'),
                       'QDoubleSpinBox': ('value', 'setValue', 'valueChanged'),
                       'ScientificDoubleSpinBox': ('value', 'setValue', 'valueChanged'),
                       'QTextArea': ('text', 'setText', 'textChanged'),
                       'QPlainTextArea': ('text', 'setText', 'textChanged'),
                       'QLineEdit': ('text', 'setText', 'textChanged'),
                       'QComboBox': ('currentText', 'setCurrentText', 'currentTextChanged'),
                       'QCheckBox': ('checkState', 'setCheckState', 'stateChanged'),
                       'QPlainTextEdit': ('toPlainText', 'setPlainText', 'textChanged'),
                       'GridBounds': ('values', 'setValues', 'boundsChanged'),
                       'MSETMultiGridWidget': ('values', 'setValues', 'multiGridChanged'),
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
msg_modes = {'info': QMessageBox.Information, 'warn': QMessageBox.Warning, 'question': QMessageBox.Question}


def convert_dialog_to_mset_settings(dialog_input: dict):
    # mset_settings = {
    #     'airfoil_order': dialog_input['airfoil_order']['text'].split(','),
    #     'grid_bounds': dialog_input['grid_bounds']['values'],
    #     'verbose': dialog_input['verbose']['state'],
    #     'airfoil_analysis_dir': dialog_input['airfoil_analysis_dir']['text'],
    #     'airfoil_coord_file_name': dialog_input['airfoil_coord_file_name']['text'],
    # }
    # values_list = ['airfoil_side_points', 'exp_side_points', 'inlet_pts_left_stream', 'outlet_pts_right_stream',
    #                'num_streams_top', 'num_streams_bot', 'max_streams_between', 'elliptic_param',
    #                'stag_pt_aspect_ratio', 'x_spacing_param', 'alf0_stream_gen', 'timeout']
    # for value in values_list:
    #     mset_settings[value] = dialog_input[value]['value']
    # for idx, airfoil in enumerate(dialog_input['multi_airfoil_grid']['values'].values()):
    #     for k, v in airfoil.items():
    #         if idx == 0:
    #             mset_settings[k] = [v]
    #         else:
    #             mset_settings[k].append(v)
    mset_settings = deepcopy(dialog_input)
    mset_settings['multi_airfoil_grid'].pop('airfoil_order')
    mset_settings['airfoil_order'] = dialog_input['airfoil_order'].split(',')
    mset_settings['n_airfoils'] = len(mset_settings['airfoil_order'])
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
    }

    mses_settings['ISMOM'] = ISMOM_CONVERSION[dialog_input['ISMOM']]
    mses_settings['IFFBC'] = IFFBC_CONVERSION[dialog_input['IFFBC']]

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

    for idx, (airfoil_name, airfoil) in enumerate(dialog_input['xtrs'].items()):
        if airfoil_name != 'airfoil_order':
            for k, v in airfoil.items():
                if idx == 0:
                    mses_settings[k] = [v]
                else:
                    mses_settings[k].append(v)

    for idx, AD_idx in enumerate(dialog_input['AD'].values()):
        for k, v in AD_idx.items():
            if idx == 0:
                mses_settings[k] = [v]
            else:
                mses_settings[k].append(v)

    # print(f"{mses_settings = }")

    return mses_settings


def convert_dialog_to_mplot_settings(dialog_input: dict):
    mplot_settings = {
        'timeout': dialog_input['timeout'],
        'Mach': dialog_input['Mach'],
        'Grid': dialog_input['Grid'],
        'Grid_Zoom': dialog_input['Grid_Zoom'],
        'flow_field': dialog_input['Output_Field'],
        'Streamline_Grid': dialog_input["Streamline_Grid"],
        'CPK': dialog_input['CPK'],
        'capSS': dialog_input['capSS'],
        'epma': dialog_input['epma']
    }
    return mplot_settings


def default_input_dict():
    input_dict = {
        'A0': {
            'dsLE_dsAvg': 0.35,
            'dsTE_dsAvg': 0.8,
            'curvature_exp': 1.3,
            'U_s_smax_min': 1,
            'U_s_smax_max': 1,
            'L_s_smax_min': 1,
            'L_s_smax_max': 1,
            'U_local_avg_spac_ratio': 0,
            'L_local_avg_spac_ratio': 0,
        },
        'airfoil_order': ['A0'],
    }
    return input_dict


def default_inputs_XTRS():
    input_dict = {
        'A0': {
            'XTRSupper': 1.0,
            'XTRSlower': 1.0,
        },
        'airfoil_order': ['A0'],
    }
    return input_dict


def default_inputs_AD():
    input_dict = {
        '1': {
            'ISDELH': 1,
            'XCDELH': 0.1,
            'PTRHIN': 1.1,
            'ETAH': 0.95,
            'from_geometry': {}
        },
    }
    return input_dict


class MSETMultiGridWidget(QTabWidget):

    multiGridChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.labels = {
            'dsLE_dsAvg': 'dsLE/dsAvg',
            'dsTE_dsAvg': 'dsTE/dsAvg',
            'curvature_exp': 'Curvature Exponent',
            'U_s_smax_min': 'U_s_smax_min',
            'U_s_smax_max': 'U_s_smax_max',
            'L_s_smax_min': 'L_s_smax_min',
            'L_s_smax_max': 'L_s_smax_max',
            'U_local_avg_spac_ratio': 'U Local Avg. Spac. Ratio',
            'L_local_avg_spac_ratio': 'L Local Avg. Spac. Ratio',
        }
        self.input_dict = default_input_dict()
        self.tab_names = self.input_dict['airfoil_order']
        self.widget_dict = {}
        self.grid_widget = None
        self.grid_layout = None
        self.generateWidgets()
        self.setTabs()

    def generateWidgets(self):
        for k1, v1 in self.input_dict.items():
            if k1 != 'airfoil_order':
                self.widget_dict[k1] = {}
                for k2, v2 in v1.items():
                    w = QDoubleSpinBox(self)
                    w.setMinimum(0.0)
                    w.setMaximum(np.inf)
                    w.setValue(v2)
                    w.setSingleStep(0.01)
                    w.valueChanged.connect(partial(self.valueChanged, k1, k2))
                    w_label = QLabel(self.labels[k2], self)
                    self.widget_dict[k1][k2] = {
                        'widget': w,
                        'label': w_label,
                    }

    def regenerateWidgets(self):
        self.generateWidgets()
        self.setTabs()

    def onAirfoilAdded(self, new_airfoil_name_list: list):
        for airfoil_name in new_airfoil_name_list:
            if airfoil_name not in self.input_dict.keys():
                self.input_dict[airfoil_name] = deepcopy(default_input_dict()['A0'])
        self.tab_names = new_airfoil_name_list
        self.regenerateWidgets()

    def onAirfoilRemoved(self, new_airfoil_name_list: list):
        names_to_remove = []
        for airfoil_name in self.input_dict.keys():
            if airfoil_name not in new_airfoil_name_list:
                names_to_remove.append(airfoil_name)
        for airfoil_name in names_to_remove:
            self.input_dict.pop(airfoil_name)
        self.tab_names = new_airfoil_name_list
        self.regenerateWidgets()

    def onAirfoilListChanged(self, new_airfoil_name_list: list):
        self.input_dict['airfoil_order'] = new_airfoil_name_list
        if len(new_airfoil_name_list) > len(self.tab_names):
            self.onAirfoilAdded(new_airfoil_name_list)
        elif len(new_airfoil_name_list) < len(self.tab_names):
            self.onAirfoilRemoved(new_airfoil_name_list)
        else:
            self.tab_names = new_airfoil_name_list
            self.regenerateWidgets()
        self.multiGridChanged.emit()

    def setTabs(self):
        self.clear()
        for tab_name in self.tab_names:
            self.add_tab(tab_name)
            grid_row_counter = 0
            for k, v in self.widget_dict[tab_name].items():
                self.grid_layout.addWidget(v['label'], grid_row_counter, 0)
                self.grid_layout.addWidget(v['widget'], grid_row_counter, 1)
                grid_row_counter += 1

    def updateTabNames(self, tab_name_list: list):
        self.tab_names = tab_name_list

    def add_tab(self, name: str):
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self)
        self.grid_widget.setLayout(self.grid_layout)
        self.addTab(self.grid_widget, name)

    def setValues(self, values: dict):
        self.input_dict = deepcopy(values)
        if self.input_dict['airfoil_order'] != self.tab_names:  # This only happens when re-loading the dialog
            self.updateTabNames(self.input_dict['airfoil_order'])
            self.regenerateWidgets()  # This function already sets the values, thus the else statement
        else:
            for k1, v1 in values.items():
                if k1 != 'airfoil_order':
                    for k2, v2 in v1.items():
                        self.widget_dict[k1][k2]['widget'].setValue(v2)

    def values(self):
        return self.input_dict

    def valueChanged(self, k1, k2, v2):
        self.input_dict[k1][k2] = v2
        self.multiGridChanged.emit()


class XTRSWidget(QTabWidget):

    XTRSChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.labels = {
            'XTRSupper': 'XTRSupper',
            'XTRSlower': 'XTRSlower',
        }
        self.input_dict = default_inputs_XTRS()
        self.tab_names = self.input_dict['airfoil_order']
        self.widget_dict = {}
        self.grid_widget = None
        self.grid_layout = None
        self.generateWidgets()
        self.setTabs()

    def generateWidgets(self):
        for k1, v1 in self.input_dict.items():
            if k1 != 'airfoil_order':
                self.widget_dict[k1] = {}
                for k2, v2 in v1.items():
                    w = QDoubleSpinBox(self)
                    w.setMinimum(0.0)
                    w.setMaximum(1.0)
                    w.setValue(v2)
                    w.setSingleStep(0.05)
                    w.valueChanged.connect(partial(self.valueChanged, k1, k2))
                    w_label = QLabel(self.labels[k2], self)
                    self.widget_dict[k1][k2] = {
                        'widget': w,
                        'label': w_label,
                    }

    def regenerateWidgets(self):
        self.generateWidgets()
        self.setTabs()

    def onAirfoilAdded(self, new_airfoil_name_list: list):
        for airfoil_name in new_airfoil_name_list:
            if airfoil_name not in self.input_dict.keys():
                self.input_dict[airfoil_name] = deepcopy(default_inputs_XTRS()['A0'])
        self.tab_names = new_airfoil_name_list
        self.regenerateWidgets()

    def onAirfoilRemoved(self, new_airfoil_name_list: list):
        names_to_remove = []
        for airfoil_name in self.input_dict.keys():
            if airfoil_name not in new_airfoil_name_list:
                names_to_remove.append(airfoil_name)
        for airfoil_name in names_to_remove:
            self.input_dict.pop(airfoil_name)
        self.tab_names = new_airfoil_name_list
        self.regenerateWidgets()

    def onAirfoilListChanged(self, new_airfoil_name_list: list):
        self.input_dict['airfoil_order'] = new_airfoil_name_list
        if len(new_airfoil_name_list) > len(self.tab_names):
            self.onAirfoilAdded(new_airfoil_name_list)
        elif len(new_airfoil_name_list) < len(self.tab_names):
            self.onAirfoilRemoved(new_airfoil_name_list)
        else:
            self.tab_names = new_airfoil_name_list
            self.regenerateWidgets()
        self.XTRSChanged.emit()

    def setTabs(self):
        self.clear()
        for tab_name in self.tab_names:
            self.add_tab(tab_name)
            grid_row_counter = 0
            for k, v in self.widget_dict[tab_name].items():
                self.grid_layout.addWidget(v['label'], grid_row_counter, 0)
                self.grid_layout.addWidget(v['widget'], grid_row_counter, 1)
                grid_row_counter += 1

    def updateTabNames(self, tab_name_list: list):
        self.tab_names = tab_name_list

    def add_tab(self, name: str):
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self)
        self.grid_widget.setLayout(self.grid_layout)
        self.addTab(self.grid_widget, name)

    def setValues(self, values: dict):
        self.input_dict = deepcopy(values)
        if self.input_dict['airfoil_order'] != self.tab_names:  # This only happens when re-loading the dialog
            self.updateTabNames(self.input_dict['airfoil_order'])
            self.regenerateWidgets()  # This function already sets the values, thus the else statement
        else:
            for k1, v1 in values.items():
                if k1 != 'airfoil_order':
                    for k2, v2 in v1.items():
                        self.widget_dict[k1][k2]['widget'].setValue(v2)

    def values(self):
        return self.input_dict

    def valueChanged(self, k1, k2, v2):
        self.input_dict[k1][k2] = v2
        self.XTRSChanged.emit()


class ADDoubleSpinBox(QDoubleSpinBox):

    sigEquationChanged = pyqtSignal(str)

    def __init__(self, parent, AD_tab: str, design_tree_widget):
        super().__init__(parent=parent)
        self.equation_edit = None
        self.AD_tab = AD_tab
        # design_tree_widget.sigSelChanged.connect(self.set_value_from_param_tree)

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        action = QAction('Define by equation', parent=self)
        action.triggered.connect(self.on_define_by_equation)
        menu.addAction(action)
        if menu.exec_(event.globalPos()):
            pass

    def on_define_by_equation(self):
        self.equation_edit = QLineEdit(self)
        self.equation_edit.setMinimumWidth(100)
        get_parent(self, depth=3).grid_layout[self.AD_tab].addWidget(self.equation_edit, 1, 3, 1, 1)
        self.equation_edit.textChanged.connect(self.on_equation_changed)

    def on_equation_changed(self):
        self.sigEquationChanged.emit(self.equation_edit.text())

    def set_value_from_param_tree(self, data: tuple):
        if str(get_parent(self, depth=3).currentIndex() + 1) == self.AD_tab and self.equation_edit is not None:
            # (Make sure we are writing to the widget corresponding to the current tab)
            self.equation_edit.setText(data[0])
            self.setValue(data[1])
            get_parent(self, depth=3).input_dict[self.AD_tab]['from_geometry']['XCDELH'] = data[0]


class ADWidget(QTabWidget):

    ADChanged = pyqtSignal()

    def __init__(self, parent, design_tree_widget=None):
        super().__init__(parent=parent)
        self.labels = {
            'ISDELH': 'AD Side',
            'XCDELH': 'AD X-Location',
            'PTRHIN': 'AD Total Pres. Ratio',
            'ETAH': 'AD Thermal Efficiency'
        }
        self.tab_names = ['1']
        self.input_dict = default_inputs_AD()
        self.widget_dict = {}
        self.grid_widget = {'1': None}
        self.grid_layout = {'1': None}
        self.design_tree_widget = design_tree_widget
        self.generateWidgets()
        self.setTabs()

    def generateWidgets(self):
        for k1, v1 in self.input_dict.items():
            self.widget_dict[k1] = {}
            for k2, v2 in v1.items():
                if k2 != 'from_geometry':
                    if k2 in ['PTRHIN', 'ETAH']:
                        w = QDoubleSpinBox(self)
                    elif k2 == 'XCDELH':
                        w = ADDoubleSpinBox(self, k1, design_tree_widget=self.design_tree_widget)
                    else:
                        w = QSpinBox(self)
                    if k2 in ['XCDELH', 'ETAH']:
                        w.setMinimum(0.0)
                        w.setMaximum(1.0)
                        w.setSingleStep(0.01)
                        w.setDecimals(12)
                    elif k2 == 'ISDELH':
                        w.setMinimum(1)
                        w.setMaximum(100)
                    elif k2 == 'PTRHIN':
                        w.setMinimum(1.0)
                        w.setMaximum(np.inf)
                        w.setSingleStep(0.05)
                        w.setDecimals(12)

                    w.setValue(v2)

                    w.valueChanged.connect(partial(self.valueChanged, k1, k2))
                    if isinstance(w, ADDoubleSpinBox):
                        w.sigEquationChanged.connect(partial(self.equationChanged, k1, k2))
                    w_label = QLabel(self.labels[k2], self)
                    self.widget_dict[k1][k2] = {
                        'widget': w,
                        'label': w_label,
                    }
            for k2, v2 in v1['from_geometry'].items():
                if len(v2) == 0:
                    continue
                w = QLineEdit(self.widget_dict[k1][k2]['widget'])
                w.setText(v2)
                w.setMinimumWidth(100)
                self.widget_dict[k1][k2]['widget'].equation_edit = w
                w.textChanged.connect(partial(self.equationChanged, k1, k2))
                self.widget_dict[k1][k2]['from_geometry'] = w

    def regenerateWidgets(self):
        self.generateWidgets()
        self.setTabs()

    def onADAdded(self, new_AD_list: list):
        for ad in new_AD_list:
            if ad not in self.input_dict.keys():
                self.input_dict[ad] = deepcopy(default_inputs_AD()['1'])
        self.tab_names = new_AD_list
        self.regenerateWidgets()

    def onADRemoved(self, new_AD_list: list):
        ads_to_remove = []
        for k in self.input_dict.keys():
            if k not in new_AD_list:
                ads_to_remove.append(k)
        for ad_to_remove in ads_to_remove:
            self.input_dict.pop(ad_to_remove)
        self.tab_names = new_AD_list
        self.regenerateWidgets()

    def onADListChanged(self, new_AD_list: list):
        if len(new_AD_list) > len(self.tab_names):
            self.onADAdded(new_AD_list)
        elif len(new_AD_list) < len(self.tab_names):
            self.onADRemoved(new_AD_list)
        else:
            self.tab_names = new_AD_list
            self.regenerateWidgets()
        self.ADChanged.emit()

    def setTabs(self):
        self.clear()
        for tab_name in self.tab_names:
            self.add_tab(tab_name)
            grid_row_counter = 0
            for k, v in self.widget_dict[tab_name].items():
                self.grid_layout[tab_name].addWidget(v['label'], grid_row_counter, 0)
                self.grid_layout[tab_name].addWidget(v['widget'], grid_row_counter, 1)
                if 'from_geometry' in v.keys():
                    self.grid_layout[tab_name].addWidget(v['from_geometry'], grid_row_counter, 2)
                grid_row_counter += 1

    def updateTabNames(self, tab_name_list: list):
        self.tab_names = tab_name_list

    def add_tab(self, name: str):
        self.grid_widget[name] = QWidget()
        self.grid_layout[name] = QGridLayout(self)
        self.grid_widget[name].setLayout(self.grid_layout[name])
        self.addTab(self.grid_widget[name], name)

    def setValues(self, values: dict):
        for k1, v1 in values.items():
            for k2, v2 in v1['from_geometry'].items():
                if len(v2) == 0:
                    continue
                self.input_dict[k1]['from_geometry'][k2] = v2
        self.updateTabNames([k for k in values.keys()])
        self.regenerateWidgets()
        for k1, v1 in values.items():
            for k2, v2 in v1.items():
                if k2 != 'from_geometry':
                    self.widget_dict[k1][k2]['widget'].setValue(v2)
                    self.input_dict[k1][k2] = v2
            for k2, v2 in v1['from_geometry'].items():
                if len(v2) == 0:
                    continue
                self.widget_dict[k1][k2]['from_geometry'].setText(v2)

    def values(self):
        return self.input_dict

    def equationChanged(self, k1, k2, v2):
        self.input_dict[k1]["from_geometry"][k2] = v2
        self.ADChanged.emit()

    def valueChanged(self, k1, k2, v2):
        # print(f"Value changed! {k1 = }, {k2 = }, {v2 = }")
        if k2 == 'from_geometry':
            print("Setting from_geometry!")
            self.input_dict[k1]["from_geometry"][k1] = v2
        else:
            self.input_dict[k1][k2] = v2
        self.ADChanged.emit()

    def setReadOnly(self, read_only: bool):
        for k1, v1 in self.widget_dict.items():
            for k2, v2 in v1.items():
                v2['widget'].setReadOnly(read_only)


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
                                     'alignment': Qt.Alignment()}
                for k, v in w_dict.items():
                    if k in grid_names['label']:
                        grid_params_label[k.split('.')[-1]] = v
                self.layout.addWidget(label, *[v for v in grid_params_label.values()])
                self.widget_dict[w_name]['label'] = label

            # Add the main widget:
            if hasattr(PyQt5.QtWidgets, w_dict['widget_type']):
                # First check if the widget type is found in PyQt5.QtWidgets:
                widget = getattr(PyQt5.QtWidgets, w_dict['widget_type'])(parent=self)
            elif hasattr(sys.modules[__name__], w_dict['widget_type']):
                # If not in PyQt5.QtWidgets, check the modules loaded into this file:
                kwargs = {}
                if w_dict['widget_type'] in ['ADWidget', 'OptConstraintsHTabWidget']:
                    kwargs = self.kwargs
                widget = getattr(sys.modules[__name__], w_dict['widget_type'])(parent=self, **kwargs)
            else:
                raise ValueError(f"Widget type {w_dict['widget_type']} not found in PyQt5.QtWidgets or system modules")
            grid_params_widget = {'row': grid_counter, 'column': 1, 'rowSpan': 1,
                                  'columnSpan': 2 if 'push_button' in w_dict.keys() else 3, 'alignment': Qt.Alignment()}
            for k, v in w_dict.items():
                if k in grid_names['widget']:
                    grid_params_widget[k] = v
                    if k == 'alignment':
                        grid_params_widget[k] = {'l': Qt.AlignLeft, 'c': Qt.AlignCenter, 'r': Qt.AlignRight}[v]
            self.layout.addWidget(widget, *[v for v in grid_params_widget.values()])
            self.widget_dict[w_name]['widget'] = widget

            # Add the push button:
            if 'push_button' in w_dict.keys():
                push_button = QPushButton(w_dict['push_button'], parent=self)
                grid_params_push = {'row': grid_counter, 'column': grid_params_widget['column'] + 2, 'rowSpan': 1,
                                    'columnSpan': 1, 'alignment': Qt.Alignment()}
                for k, v in w_dict.items():
                    if k in grid_names['push_button']:
                        grid_params_push[k.split('.')[-1]] = v
                push_button.clicked.connect(partial(getattr(self, w_dict['push_button_action']), widget))
                self.layout.addWidget(push_button, *[v for v in grid_params_push.values()])
                self.widget_dict[w_name]['push_button'] = push_button

            if 'active_checkbox' in w_dict.keys():
                checkbox = QCheckBox('Active?', parent=self)
                grid_params_check = {'row': grid_counter, 'column': grid_params_widget['column'] + 2, 'rowSpan': 1,
                                     'columnSpan': 1, 'alignment': Qt.Alignment()}
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

    def getInputs(self):
        """This method is used to extract the data from the Dialog"""
        output_dict = {w_name: None for w_name in self.widget_dict.keys()}
        for w_name, w in self.widget_dict.items():
            if self.settings[w_name]['widget_type'] in get_set_value_names.keys():
                output_dict[w_name] = getattr(w['widget'],
                                              get_set_value_names[self.settings[w_name]['widget_type']][0]
                                              )()
                if w['checkbox'] is not None:
                    state = w['checkbox'].checkState()
                    output_dict[w_name] = (output_dict[w_name], state)
            else:
                output_dict[w_name] = None
        return output_dict

    @staticmethod
    def activate_deactivate_from_checkbox(widget, state):
        widget.setReadOnly(not state)

    def overrideInputs(self, new_values: dict):
        for k, v in new_values.items():
            if v is not None:
                if self.widget_dict[k]['checkbox'] is not None:
                    self.widget_dict[k]['checkbox'].setCheckState(v[1])
                    getattr(self.widget_dict[k]['widget'], get_set_value_names[self.settings[k]['widget_type']][1])(v[0])
                else:
                    getattr(self.widget_dict[k]['widget'], get_set_value_names[self.settings[k]['widget_type']][1])(v)

    def dialogChanged(self, *_, w_name: str):
        new_inputs = self.getInputs()
        self.updateDialog(new_inputs, w_name)

    @abstractmethod
    def updateDialog(self, new_inputs: dict, w_name: str):
        """Required method which reacts to changes in the dialog inputs. Use the :code:`overrideInputs` method to
        update the dialog at the end of this method if necessary."""
        pass


class PymeadDialogHTabWidget(QTabWidget):

    sigTabsChanged = pyqtSignal(object)

    def __init__(self, parent, widgets: dict, settings_override: dict = None):
        super().__init__()
        self.w_dict = widgets
        self.generateWidgets()
        if settings_override is not None:
            self.overrideInputs(settings_override)

    def generateWidgets(self):
        for k, v in self.w_dict.items():
            self.addTab(v, k)

    def regenerateWidgets(self):
        self.clear()
        self.generateWidgets()
        self.sigTabsChanged.emit([k for k in self.w_dict.keys()])

    def overrideInputs(self, new_values: dict):
        for k, v in new_values.items():
            self.w_dict[k].overrideInputs(new_values=v)

    def getInputs(self):
        return {k: v.getInputs() for k, v in self.w_dict.items()}


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


class DownsamplingPreviewDialog(QDialog):
    def __init__(self, use_downsampling: bool, downsampling_max_pts: int, downsampling_curve_exp: float,
                 parent: QWidget or None = None):
        super().__init__(parent=parent)

        self.setWindowTitle("Airfoil Coordinates Preview")
        self.setFont(self.parent().font())
        self.setGeometry(300, 300, 700, 250)

        self.grid_widget = {}

        # buttonBox = QDialogButtonBox(self)
        # buttonBox.addButton(QDialogButtonBox.Ok)
        # buttonBox.addButton(QDialogButtonBox.Cancel)
        self.grid_layout = QGridLayout(self)

        # Add pyqtgraph widget
        self.w = pg.GraphicsLayoutWidget(parent=self, size=(700, 250))
        # self.w.setBackground('#2a2a2b')
        self.v = self.w.addPlot()
        self.v.setAspectLocked()

        gui_object = get_parent(self, depth=5)

        theme = gui_object.themes[gui_object.current_theme]
        self.w.setBackground(theme["graph-background-color"])

        # Make a copy of the MEA
        mea = gui_object.mea.deepcopy()
        if not isinstance(mea, MEA):
            raise TypeError(f"Generated mea was of type {type(mea)} instead of type pymead.core.mea.MEA")

        # Update the curves, using downsampling if specified
        for a in mea.airfoils.values():
            new_param_vec_list = None
            if use_downsampling:
                new_param_vec_list = a.downsample(max_airfoil_points=downsampling_max_pts,
                                                  curvature_exp=downsampling_curve_exp)
            for c_idx, curve in enumerate(a.curve_list):
                if new_param_vec_list is not None:
                    curve.update(curve.P, t=new_param_vec_list[c_idx])

        for a in mea.airfoils.values():
            for c in a.curve_list:
                self.v.plot(x=c.x, y=c.y, symbol="o")

        self.grid_layout.addWidget(self.w, 0, 0, 3, 3)
        # self.grid_layout.addWidget(buttonBox, 1, 1, 1, 2)
        #
        # buttonBox.accepted.connect(self.accept)
        # buttonBox.rejected.connect(self.reject)


class MSETDialogWidget(PymeadDialogWidget):
    airfoilOrderChanged = pyqtSignal(object)

    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, "mset_settings.json"))
        self.widget_dict['airfoil_analysis_dir']['widget'].setText(tempfile.gettempdir())

    def change_airfoil_order(self, _):
        if not all([a in self.widget_dict['airfoil_order']['widget'].text().split(',') for a in get_parent(self, 4).mea.airfoils.keys()]):
            current_airfoil_list = [a for a in get_parent(self, 4).mea.airfoils.keys()]
        else:
            current_airfoil_list = self.widget_dict['airfoil_order']['widget'].text().split(',')
        dialog = AirfoilListDialog(self, current_airfoil_list=current_airfoil_list)
        if dialog.exec_():
            airfoil_order = dialog.getData()
            self.widget_dict['airfoil_order']['widget'].setText(','.join(airfoil_order))
            self.airfoilOrderChanged.emit(','.join(airfoil_order))

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        if w_name == 'airfoil_order':
            self.widget_dict['multi_airfoil_grid']['widget'].onAirfoilListChanged(
                new_inputs['airfoil_order'].split(','))

    def saveas_mset_mses_mplot_settings(self):
        all_inputs = get_parent(self, 2).getInputs()
        mses_inputs = {k: v for k, v in all_inputs.items() if k in ["MSET", "MSES", "MPLOT"]}
        json_dialog = select_json_file(parent=self)
        if json_dialog.exec_():
            input_filename = json_dialog.selectedFiles()[0]
            if not os.path.splitext(input_filename)[-1] == '.json':
                input_filename = input_filename + '.json'
            save_data(mses_inputs, input_filename)
            # if get_parent(self, 4):
            #     get_parent(self, 4).current_settings_save_file = input_filename
            # else:
            #     self.current_save_file = input_filename
            get_parent(self, 3).setStatusTip(f"Saved MSES settings to {input_filename}")
            # print(f"{get_parent(self, 3) = }")

    def load_mset_mses_mplot_settings(self):
        override_inputs = get_parent(self, 2).getInputs()
        load_dialog = select_existing_json_file(parent=self)
        if load_dialog.exec_():
            load_file = load_dialog.selectedFiles()[0]
            new_inputs = load_data(load_file)
            for k, v in new_inputs.items():
                override_inputs[k] = v
            # great_great_grandparent = get_parent(self, depth=4)
            # if great_great_grandparent:
            #     great_great_grandparent.current_settings_save_file = load_file
            # else:
            #     self.current_save_file = load_file
            get_parent(self, 3).setWindowTitle(f"Optimization Setup - {os.path.split(load_file)[-1]}")
            get_parent(self, 2).overrideInputs(override_inputs)  # Overrides the inputs for the whole PymeadDialogVTabWidget
            get_parent(self, 2).setStatusTip(f"Loaded {load_file}")

    def show_airfoil_coordinates_preview(self, _):
        override_inputs = get_parent(self, 2).getInputs()
        use_downsampling = bool(override_inputs["MSET"]["use_downsampling"])
        downsampling_max_pts = override_inputs["MSET"]["downsampling_max_pts"]
        downsampling_curve_exp = override_inputs["MSET"]["downsampling_curve_exp"]
        preview_dialog = DownsamplingPreviewDialog(use_downsampling=use_downsampling,
                                                   downsampling_max_pts=downsampling_max_pts,
                                                   downsampling_curve_exp=downsampling_curve_exp,
                                                   parent=self)
        preview_dialog.exec_()


class MSESDialogWidget(PymeadDialogWidget):
    def __init__(self, mset_dialog_widget: MSETDialogWidget, design_tree_widget):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'mses_settings.json'),
                         design_tree_widget=design_tree_widget)
        mset_dialog_widget.airfoilOrderChanged.connect(self.onAirfoilOrderChanged)

    def onAirfoilOrderChanged(self, airfoil_list: str):
        self.widget_dict['xtrs']['widget'].onAirfoilListChanged(airfoil_list.split(','))

    def deactivate_AD(self, read_only: bool):
        self.widget_dict['AD']['widget'].setReadOnly(read_only)
        self.widget_dict['AD_number']['widget'].setReadOnly(read_only)
        if not read_only:
            self.widget_dict['AD']['widget'].setToolTip("")
            self.widget_dict['AD_number']['widget'].setToolTip("")
        else:
            self.widget_dict['AD']['widget'].setToolTip(self.settings['AD']['setToolTip'])
            self.widget_dict['AD_number']['widget'].setToolTip(self.settings['AD_number']['setToolTip'])

    def calculate_and_set_Reynolds_number(self, new_inputs: dict):
        Re_widget = self.widget_dict['REYNIN']['widget']
        nu = viscosity_calculator(new_inputs['T'], rho=new_inputs['rho'])
        a = np.sqrt(new_inputs['gam'] * new_inputs['R'] * new_inputs['T'])
        V = new_inputs['MACHIN'] * a
        Re_widget.setValue(V * new_inputs['L'] / nu)

    def change_prescribed_aero_parameter(self, current_text: str):
        w1 = self.widget_dict['ALFAIN']['widget']
        w2 = self.widget_dict['CLIFIN']['widget']
        if current_text == 'Specify Angle of Attack':
            bools = (False, True)
        elif current_text == 'Specify Lift Coefficient':
            bools = (True, False)
        else:
            raise ValueError('Invalid value of currentText for QComboBox (alfa/Cl')
        w1.setReadOnly(bools[0])
        w2.setReadOnly(bools[1])

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
        self.widget_dict['REYNIN']['widget'].setReadOnly(not active)
        self.widget_dict['spec_P_T_rho']['widget'].setEnabled(not active)
        if not active:
            self.calculate_and_set_Reynolds_number(new_inputs)

    def updateDialog(self, new_inputs: dict, w_name: str):
        if w_name == 'AD_number':
            self.widget_dict['AD']['widget'].onADListChanged([str(j + 1) for j in range(new_inputs['AD_number'])])
        if w_name == 'AD_active':
            self.deactivate_AD(not new_inputs['AD_active'])
        if w_name == 'spec_Re':
            self.change_Re_active_state(new_inputs)
        if w_name == 'spec_P_T_rho':
            self.change_prescribed_flow_variables(new_inputs['spec_P_T_rho'])
        if w_name == 'spec_alfa_Cl':
            self.change_prescribed_aero_parameter(new_inputs['spec_alfa_Cl'])
        if w_name in ['P', 'T', 'rho', 'R', 'gam', 'L', 'MACHIN'] and not self.widget_dict[w_name][
            'widget'].isReadOnly():
            if self.widget_dict['P']['widget'].isReadOnly():
                self.widget_dict['P']['widget'].setValue(new_inputs['rho'] * new_inputs['R'] * new_inputs['T'])
            elif self.widget_dict['T']['widget'].isReadOnly():
                self.widget_dict['T']['widget'].setValue(new_inputs['P'] / new_inputs['R'] / new_inputs['rho'])
            elif self.widget_dict['rho']['widget'].isReadOnly():
                self.widget_dict['rho']['widget'].setValue(new_inputs['P'] / new_inputs['R'] / new_inputs['T'])
            new_inputs = self.getInputs()
            if not (w_name == 'MACHIN' and new_inputs['spec_Re']):
                self.calculate_and_set_Reynolds_number(new_inputs)


class OptConstraintsDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'opt_constraints_settings.json'))

    def select_data_file(self, line_edit: QLineEdit):
        select_data_file(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class OptConstraintsHTabWidget(PymeadDialogHTabWidget):

    OptConstraintsChanged = pyqtSignal()

    def __init__(self, parent, mset_dialog_widget: MSETDialogWidget = None):
        super().__init__(parent=parent,
                         widgets={'A0': OptConstraintsDialogWidget()})
        mset_dialog_widget.airfoilOrderChanged.connect(self.onAirfoilListChanged)

    def reorderRegenerateWidgets(self, new_airfoil_name_list: list):
        temp_dict = {}
        for airfoil_name in new_airfoil_name_list:
            temp_dict[airfoil_name] = self.w_dict[airfoil_name]
        self.w_dict = temp_dict
        self.regenerateWidgets()

    def onAirfoilAdded(self, new_airfoil_name_list: list):
        for airfoil_name in new_airfoil_name_list:
            if airfoil_name not in self.w_dict.keys():
                self.w_dict[airfoil_name] = OptConstraintsDialogWidget()
        self.reorderRegenerateWidgets(new_airfoil_name_list=new_airfoil_name_list)

    def onAirfoilRemoved(self, new_airfoil_name_list: list):
        names_to_remove = []
        for airfoil_name in self.w_dict.keys():
            if airfoil_name not in new_airfoil_name_list:
                names_to_remove.append(airfoil_name)
        for airfoil_name in names_to_remove:
            self.w_dict.pop(airfoil_name)
        self.reorderRegenerateWidgets(new_airfoil_name_list=new_airfoil_name_list)

    def onAirfoilListChanged(self, new_airfoil_name_list_str: str):
        new_airfoil_name_list = new_airfoil_name_list_str.split(',')
        if len(new_airfoil_name_list) > len([k for k in self.w_dict.keys()]):
            self.onAirfoilAdded(new_airfoil_name_list)
        elif len(new_airfoil_name_list) < len([k for k in self.w_dict.keys()]):
            self.onAirfoilRemoved(new_airfoil_name_list)
        else:
            self.reorderRegenerateWidgets(new_airfoil_name_list=new_airfoil_name_list)

    def setValues(self, values: dict):
        self.onAirfoilListChanged(new_airfoil_name_list_str=','.join([k for k in values.keys()]))
        self.overrideInputs(new_values=values)

    def values(self):
        return self.getInputs()

    def valueChanged(self, k1, k2, v2):
        self.OptConstraintsChanged.emit()


class XFOILDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'xfoil_settings.json'))
        self.widget_dict['airfoil_analysis_dir']['widget'].setText(tempfile.gettempdir())

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
            new_inputs = self.getInputs()
            if not (w_name == 'Ma' and new_inputs['spec_Re']):
                self.calculate_and_set_Reynolds_number(new_inputs)

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit, starting_dir=tempfile.gettempdir())


class GAGeneralSettingsDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'ga_general_settings.json'))
        self.current_save_file = None

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit)

    def select_existing_jmea_file(self, line_edit: QLineEdit):
        select_existing_jmea_file(parent=self.parent(), line_edit=line_edit)

    def select_multiple_json_files(self, text_edit: QPlainTextEdit):
        select_multiple_json_files(parent=self.parent(), text_edit=text_edit)

    def save_opt_settings(self):
        opt_file = None
        if self.current_save_file is not None:
            opt_file = self.current_save_file
        if get_parent(self, 4) and get_parent(self, 4).current_settings_save_file is not None:
            opt_file = get_parent(self, 4).current_settings_save_file
        if opt_file is not None:
            new_inputs = get_parent(self, 2).getInputs()  # Gets the inputs from the PymeadDialogVTabWidget
            save_data(new_inputs, opt_file)
            get_parent(self, 2).setStatusTip(f"Settings saved ({opt_file})")
            # msg_box = PymeadMessageBox(parent=self, msg=f"Settings saved as {self.current_save_file}",
            #                            window_title='Save Notification', msg_mode='info')
            # msg_box.exec()
        else:
            self.saveas_opt_settings()

    def load_opt_settings(self):
        load_dialog = select_existing_json_file(parent=self, starting_dir=load_documents_path("ga-settings-dir"))
        if load_dialog.exec_():
            load_file = load_dialog.selectedFiles()[0]
            q_settings.setValue("ga-settings-dir", os.path.split(load_file)[0])
            new_inputs = load_data(load_file)
            great_great_grandparent = get_parent(self, depth=4)
            if great_great_grandparent:
                great_great_grandparent.current_settings_save_file = load_file
            else:
                self.current_save_file = load_file
            get_parent(self, 3).setWindowTitle(f"Optimization Setup - {os.path.split(load_file)[-1]}")
            get_parent(self, 2).overrideInputs(new_inputs)  # Overrides the inputs for the whole PymeadDialogVTabWidget
            get_parent(self, 2).setStatusTip(f"Loaded {load_file}")

    def saveas_opt_settings(self):
        inputs_to_save = get_parent(self, 2).getInputs()
        json_dialog = select_json_file(parent=self)
        if json_dialog.exec_():
            input_filename = json_dialog.selectedFiles()[0]
            if not os.path.splitext(input_filename)[-1] == '.json':
                input_filename = input_filename + '.json'
            save_data(inputs_to_save, input_filename)
            if get_parent(self, 4):
                get_parent(self, 4).current_settings_save_file = input_filename
            else:
                self.current_save_file = input_filename
            get_parent(self, 3).setStatusTip(f"Saved optimization settings to {input_filename}")
            # print(f"{get_parent(self, 3) = }")

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class GAConstraintsTerminationDialogWidget(PymeadDialogWidget):
    def __init__(self, mset_dialog_widget: MSETDialogWidget = None):
        self.mset_dialog_widget = mset_dialog_widget
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'ga_constraints_termination_settings.json'),
                         mset_dialog_widget=mset_dialog_widget)

    def select_data_file(self, line_edit: QLineEdit):
        select_data_file(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class GASaveLoadDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'ga_save_load_settings.json'))
        self.current_save_file = None

    def select_json_file(self, line_edit: QLineEdit):
        select_json_file(parent=self.parent(), line_edit=line_edit)

    def select_existing_json_file(self, line_edit: QLineEdit):
        select_existing_json_file(parent=self.parent(), line_edit=line_edit)

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit)

    def save_opt_settings(self):
        if self.current_save_file is not None:
            new_inputs = self.parent().getInputs()  # Gets the inputs from the PymeadDialogVTabWidget
            save_data(new_inputs, self.current_save_file)
            msg_box = PymeadMessageBox(parent=self, msg=f"Settings saved as {self.current_save_file}",
                                       window_title='Save Notification', msg_mode='info')
            msg_box.exec()
        else:
            self.saveas_opt_settings()

    def load_opt_settings(self):
        new_inputs = load_data(self.widget_dict['settings_load_dir']['widget'].text())
        self.current_save_file = new_inputs['Save/Load']['settings_save_dir']
        self.parent().overrideInputs(new_inputs)  # Overrides the inputs for the whole PymeadDialogVTabWidget

    def saveas_opt_settings(self):
        inputs_to_save = self.parent().getInputs()
        input_filename = os.path.join(self.widget_dict['settings_saveas_dir']['widget'].text(),
                                      self.widget_dict['settings_saveas_filename']['widget'].text())
        save_data(inputs_to_save, input_filename)
        self.current_save_file = input_filename
        msg_box = PymeadMessageBox(parent=self, msg=f"Settings saved as {input_filename}",
                                   window_title='Save Notification', msg_mode='info')
        msg_box.exec()

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class MultiPointOptDialogWidget(PymeadDialogWidget):
    def __init__(self):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'multi_point_opt_settings.json'))
        self.current_save_file = None

    def select_data_file(self, line_edit: QLineEdit):
        select_data_file(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class GeneticAlgorithmDialogWidget(PymeadDialogWidget):
    def __init__(self, multi_point_dialog_widget: MultiPointOptDialogWidget):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'genetic_algorithm_settings.json'))
        self.widget_dict['J']['widget'].textChanged.connect(partial(self.objectives_changed,
                                                                    self.widget_dict['J']['widget']))
        self.widget_dict['G']['widget'].textChanged.connect(partial(self.constraints_changed,
                                                                    self.widget_dict['G']['widget']))
        multi_point_active_widget = multi_point_dialog_widget.widget_dict['multi_point_active']['widget']
        self.multi_point = multi_point_active_widget.checkState()
        tool = self.getInputs()['tool']
        self.cfd_template = tool
        if self.multi_point:
            self.cfd_template += '_MULTIPOINT'
        multi_point_active_widget.stateChanged.connect(self.multi_point_changed)

    def overrideInputs(self, new_values: dict):
        super().overrideInputs(new_values)
        self.update_objectives_and_constraints()

    def update_objectives_and_constraints(self):
        inputs = self.getInputs()
        self.objectives_changed(self.widget_dict['J']['widget'], inputs['J'])
        self.constraints_changed(self.widget_dict['G']['widget'], inputs['G'])

    def visualize_sampling(self, ws_widget, _):
        general_settings = self.parent().findChild(GAGeneralSettingsDialogWidget).getInputs()
        starting_value = ws_widget.value()
        use_current_mea = bool(general_settings["use_current_mea"])
        mea_file = general_settings["mea_file"]
        gui_obj = get_parent(self, depth=4)
        background_color = gui_obj.themes[gui_obj.current_theme]["graph-background-color"]
        if use_current_mea or len(mea_file) == 0 or not os.path.exists(mea_file):
            jmea_dict = gui_obj.mea.copy_as_param_dict(deactivate_airfoil_graphs=True)
        else:
            jmea_dict = load_data(mea_file)

        dialog = SamplingVisualizationDialog(jmea_dict=jmea_dict, initial_sampling_width=starting_value,
                                             initial_n_samples=20, background_color=background_color, parent=self)
        dialog.exec_()

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self.parent(), line_edit=line_edit)

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass

    def multi_point_changed(self, state: int or bool):
        # print("Multi point changed!")
        self.multi_point = state
        self.objectives_changed(self.widget_dict['J']['widget'], self.widget_dict['J']['widget'].text())
        self.constraints_changed(self.widget_dict['G']['widget'], self.widget_dict['G']['widget'].text())

    def objectives_changed(self, widget, text: str):
        objective_container = get_parent(self, depth=4)
        if objective_container is None:
            objective_container = get_parent(self, depth=1)
        inputs = self.getInputs()
        tool = inputs['tool']
        if self.multi_point:
            tool += '_MULTIPOINT'
        objective_container.objectives = []
        for obj_func_str in text.split(','):
            objective = Objective(obj_func_str)
            objective_container.objectives.append(objective)
            if text == '':
                widget.setStyleSheet("QLineEdit {background-color: rgba(176,25,25,50)}")
                return
            try:
                function_input_data1 = getattr(cfd_output_templates, tool)
                function_input_data2 = self.convert_text_array_to_dict(inputs['additional_data'])
                objective.update({**function_input_data1, **function_input_data2})
                widget.setStyleSheet("QLineEdit {background-color: rgba(16,201,87,50)}")
            except FunctionCompileError:
                widget.setStyleSheet("QLineEdit {background-color: rgba(176,25,25,50)}")
                # print("Function compile error!")
                return

    def constraints_changed(self, widget, text: str):
        constraint_container = get_parent(self, depth=4)
        if constraint_container is None:
            constraint_container = get_parent(self, depth=1)
        inputs = self.getInputs()
        tool = inputs['tool']
        if self.multi_point:
            tool += '_MULTIPOINT'
        constraint_container.constraints = []
        for constraint_func_str in text.split(','):
            if len(constraint_func_str) > 0:
                constraint = Constraint(constraint_func_str)
                constraint_container.constraints.append(constraint)
                try:
                    function_input_data1 = getattr(cfd_output_templates, tool)
                    function_input_data2 = self.convert_text_array_to_dict(inputs['additional_data'])
                    constraint.update({**function_input_data1, **function_input_data2})
                    widget.setStyleSheet("QLineEdit {background-color: rgba(16,201,87,50)}")
                except FunctionCompileError:
                    widget.setStyleSheet("QLineEdit {background-color: rgba(176,25,25,50)}")
                    return

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


class SamplingVisualizationDialog(QDialog):
    def __init__(self, jmea_dict: dict, initial_sampling_width: float, initial_n_samples: int, background_color: str,
                 parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Visualize Sampling")
        self.setFont(self.parent().font())
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.sampling_widget = SamplingVisualizationWidget(self, jmea_dict,
                                                           initial_sampling_width=initial_sampling_width,
                                                           initial_n_samples=initial_n_samples,
                                                           background_color=background_color)

        layout.addWidget(self.sampling_widget)

        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)


class PymeadDialogVTabWidget(VerticalTabWidget):
    def __init__(self, parent, widgets: dict, settings_override: dict = None):
        super().__init__()
        self.w_dict = widgets
        for k, v in self.w_dict.items():
            self.addTab(v, k)
        if settings_override is not None:
            self.overrideInputs(settings_override)

    def overrideInputs(self, new_values: dict):
        for k, v in new_values.items():
            self.w_dict[k].overrideInputs(new_values=v)

    def getInputs(self):
        return {k: v.getInputs() for k, v in self.w_dict.items()}


class PymeadDialog(QDialog):
    """This subclass of QDialog forces the selection of a WindowTitle and matches the visual format of the GUI"""
    def __init__(self, parent, window_title: str, widget: PymeadDialogWidget or PymeadDialogVTabWidget):
        super().__init__(parent=parent)
        self.setWindowTitle(window_title)
        if self.parent() is not None:
            self.setFont(self.parent().font())

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.w = widget

        self.layout.addWidget(widget)
        self.layout.addWidget(self.create_button_box())

    # def setInputs(self):
    #     self.w.setInputs()

    def overrideInputs(self, new_inputs):
        self.w.overrideInputs(new_values=new_inputs)

    def getInputs(self):
        return self.w.getInputs()

    def create_button_box(self):
        """Creates a ButtonBox to add to the Layout. Can be overridden to add additional functionality.

        Returns
        =======
        QDialogButtonBox
        """
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        return buttonBox


class XFOILDialog(PymeadDialog):
    def __init__(self, parent: QWidget, settings_override: dict = None):
        w = XFOILDialogWidget()
        super().__init__(parent=parent, window_title="Single Airfoil Viscous Analysis", widget=w)


class MultiAirfoilDialog(PymeadDialog):
    def __init__(self, parent: QWidget, design_tree_widget, settings_override: dict = None):
        w3 = PymeadDialogWidget(os.path.join(GUI_DEFAULTS_DIR, 'mplot_settings.json'))
        w2 = MSETDialogWidget()
        # w2.airfoilOrderChanged.connect(self.airfoil_order_changed)
        w4 = MSESDialogWidget(mset_dialog_widget=w2, design_tree_widget=design_tree_widget)
        w = PymeadDialogVTabWidget(parent=None, widgets={'MSET': w2, 'MSES': w4, 'MPLOT': w3},
                                   settings_override=settings_override)
        super().__init__(parent=parent, window_title="Multi-Element-Airfoil Analysis", widget=w)


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


class ScreenshotDialog(QDialog):
    def __init__(self, parent: QWidget):

        super().__init__(parent=parent)

        self.setWindowTitle("Screenshot")
        self.setFont(self.parent().font())

        self.grid_widget = {}

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton(QDialogButtonBox.Ok)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.grid_layout = QGridLayout(self)

        self.setInputs()

        self.grid_layout.addWidget(buttonBox, self.grid_layout.rowCount(), 1, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        widget_dict = load_data(os.path.join('dialog_widgets', 'screenshot_dialog.json'))
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

    def getInputs(self):
        inputs = {
            "image_file": self.grid_widget["choose_image_file"]["line"].text(),
            "window": self.grid_widget["window"]["combobox"].currentText()
        }
        return inputs

    def select_jpg_file(self, line_edit: QLineEdit):
        select_jpg_file(parent=self.parent(), line_edit=line_edit)


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
    def __init__(self, parent, settings_var: str, file_filter: str = "JMEA Files (*.jmea)"):
        super().__init__(parent=parent)

        self.setFileMode(QFileDialog.ExistingFile)
        self.setNameFilter(self.tr(file_filter))
        self.setViewMode(QFileDialog.Detail)
        self.settings_var = settings_var

        # Get default open location
        if q_settings.contains(settings_var):
            path = q_settings.value(settings_var)
        else:
            path = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)

        self.setDirectory(path)


class LoadAirfoilAlgFileWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        layout = QGridLayout(self)

        self.pkl_line = QLineEdit("", self)
        self.pkl_selection_button = QPushButton("Choose File", self)
        self.index_button = QRadioButton("Use Index", self)
        self.weight_button = QRadioButton("Use Weights", self)
        self.index_spin = QSpinBox(self)
        self.index_spin.setValue(0)
        self.index_spin.setMaximum(9999)
        self.weight_line = QLineEdit(self)

        self.index_button.toggled.connect(self.index_selected)
        self.weight_button.toggled.connect(self.weight_selected)

        self.weight_line.textChanged.connect(self.weights_changed)
        self.pkl_selection_button.clicked.connect(self.choose_file_clicked)

        self.weight_line.setText("0.5,0.5")

        self.index_button.toggle()

        layout.addWidget(self.pkl_line, 0, 0)
        layout.addWidget(self.pkl_selection_button, 0, 1)
        layout.addWidget(self.index_button, 1, 0)
        layout.addWidget(self.weight_button, 1, 1)
        layout.addWidget(self.index_spin, 2, 0)
        layout.addWidget(self.weight_line, 2, 1)

        self.setLayout(layout)

        inputs = self.getInputs()
        for setting_var in ("pkl_file", "pkl_use_index", "pkl_use_weights", "pkl_index", "pkl_weights"):
            if q_settings.contains(setting_var):
                inputs[setting_var] = q_settings.value(setting_var)
                if setting_var in ("pkl_use_index", "pkl_use_weights", "pkl_index"):
                    inputs[setting_var] = int(inputs[setting_var])
                if setting_var == "pkl_weights":
                    inputs[setting_var] = [float(s) for s in inputs[setting_var]]
        self.setInputs(inputs)

    def choose_file_clicked(self):
        dialog = LoadDialog(self, settings_var="pkl_file_dir", file_filter="PKL files (*.pkl)")

        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            self.pkl_line.setText(file_name)
            file_name_parent_dir = os.path.dirname(file_name)
            q_settings.setValue(dialog.settings_var, file_name_parent_dir)

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

    def getInputs(self):
        inputs = {
            "pkl_file": self.pkl_line.text(),
            "pkl_use_index": int(self.index_button.isChecked()),
            "pkl_use_weights": int(self.weight_button.isChecked()),
            "pkl_index": self.index_spin.value(),
            "pkl_weights": self.validate_weights(self.weight_line.text())
        }
        return inputs

    def setInputs(self, inputs: dict):
        self.pkl_line.setText(inputs["pkl_file"])
        if inputs["pkl_use_index"]:
            self.index_button.toggle()
        elif inputs["pkl_use_weights"]:
            self.weight_button.toggle()
        self.index_spin.setValue(inputs["pkl_index"])

        if isinstance(inputs["pkl_weights"], list):
            self.weight_line.setText(",".join([str(w) for w in inputs["pkl_weights"]]))
        elif isinstance(inputs["pkl_weights"], str):
            self.weight_line.setText(inputs["pkl_weights"])

    @staticmethod
    def assignQSettings(inputs: dict):
        q_settings.setValue("pkl_file", inputs["pkl_file"])
        q_settings.setValue("pkl_use_index", inputs["pkl_use_index"])
        q_settings.setValue("pkl_use_weights", inputs["pkl_use_weights"])
        q_settings.setValue("pkl_index", inputs["pkl_index"])
        q_settings.setValue("pkl_weights", inputs["pkl_weights"])


class LoadAirfoilAlgFile(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.setWindowTitle("Load Optimized Airfoil")
        self.setFont(self.parent().font())

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton(QDialogButtonBox.Ok)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.grid_layout = QGridLayout(self)

        self.load_airfoil_alg_file_widget = LoadAirfoilAlgFileWidget(self)
        self.grid_layout.addWidget(self.load_airfoil_alg_file_widget, 0, 0)
        self.grid_layout.addWidget(buttonBox, self.grid_layout.rowCount(), 0)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return self.load_airfoil_alg_file_widget.getInputs()


class SaveAsDialog(QFileDialog):
    def __init__(self, parent, file_filter: str = "JMEA Files (*.jmea)"):
        super().__init__(parent=parent)
        self.setFileMode(QFileDialog.AnyFile)
        self.setNameFilter(self.tr(file_filter))
        self.setViewMode(QFileDialog.Detail)


class NewMEADialog(QDialog):
    def __init__(self, parent=None, window_title: str or None = None, message: str or None = None):
        super().__init__(parent=parent)
        window_title = window_title if window_title is not None else "Save Changes?"
        self.setWindowTitle(window_title)
        self.setFont(self.parent().font())
        self.reject_changes = False
        self.save_successful = False
        buttonBox = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        if message is not None:
            label = QLabel(message, parent=self)
            layout.addWidget(label)

        layout.addWidget(buttonBox)

        buttonBox.button(QDialogButtonBox.Yes).clicked.connect(self.yes)
        buttonBox.button(QDialogButtonBox.Yes).clicked.connect(self.accept)
        buttonBox.button(QDialogButtonBox.No).clicked.connect(self.no)
        buttonBox.button(QDialogButtonBox.No).clicked.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    @pyqtSlot()
    def yes(self):
        try:
            save_successful = self.parent().save_mea()
            self.save_successful = save_successful
        except:
            self.save_successful = False

    @pyqtSlot()
    def no(self):
        self.reject_changes = True


class ExitDialog(QDialog):
    def __init__(self, parent=None, window_title: str or None = None, message: str or None = None):
        super().__init__(parent=parent)
        window_title = window_title if window_title is not None else "Exit?"
        self.setWindowTitle(window_title)
        self.setFont(self.parent().font())
        buttonBox = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No, self)
        layout = QFormLayout(self)

        message = message if message is not None else "Airfoil not saved.\nAre you sure you want to exit?"
        label = QLabel(message, parent=self)

        layout.addWidget(label)
        layout.addWidget(buttonBox)

        buttonBox.button(QDialogButtonBox.Yes).clicked.connect(self.accept)
        buttonBox.button(QDialogButtonBox.No).clicked.connect(self.reject)


class EditBoundsDialog(QDialog):
    def __init__(self, jmea_dict: dict, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("Edit Bounds")
        self.setFont(self.parent().font())
        buttonBox = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        layout = QFormLayout(self)

        self.bv_table = BoundsValuesTable(jmea_dict=jmea_dict)

        layout.addWidget(self.bv_table)

        layout.addWidget(buttonBox)

        buttonBox.button(QDialogButtonBox.Save).clicked.connect(self.accept)
        buttonBox.rejected.connect(self.reject)


class OptimizationDialogVTabWidget(PymeadDialogVTabWidget):
    def __init__(self, parent, widgets: dict, settings_override: dict):
        super().__init__(parent=parent, widgets=widgets, settings_override=settings_override)
        self.objectives = None
        self.constraints = None


class OptimizationSetupDialog(PymeadDialog):
    def __init__(self, parent, design_tree_widget, settings_override: dict = None):
        w0 = GAGeneralSettingsDialogWidget()
        w3 = XFOILDialogWidget()
        w4 = MSETDialogWidget()
        w2 = GAConstraintsTerminationDialogWidget(mset_dialog_widget=w4)
        w7 = MultiPointOptDialogWidget()
        w5 = MSESDialogWidget(mset_dialog_widget=w4, design_tree_widget=design_tree_widget)
        w1 = GeneticAlgorithmDialogWidget(multi_point_dialog_widget=w7)
        w6 = PymeadDialogWidget(os.path.join(GUI_DEFAULTS_DIR, 'mplot_settings.json'))
        w = OptimizationDialogVTabWidget(parent=self, widgets={'General Settings': w0,
                                                        'Genetic Algorithm': w1,
                                                        'Constraints/Termination': w2,
                                                               'Multi-Point Optimization': w7,
                                                        'XFOIL': w3, 'MSET': w4, 'MSES': w5, 'MPLOT': w6},
                                         settings_override=settings_override)
        super().__init__(parent=parent, window_title='Optimization Setup', widget=w)
        w.objectives = self.parent().objectives
        w.constraints = self.parent().constraints

        w1.update_objectives_and_constraints()  # IMPORTANT: makes sure that the objectives/constraints get stored


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

        self.grid_layout.addWidget(buttonBox, 10, 1, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        widget_dict = load_data(os.path.join(GUI_DIALOG_WIDGETS_DIR, 'export_coordinates_dialog.json'))
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
                if "checkstate" in w_dict.keys() and isinstance(widget, QCheckBox):
                    widget.setCheckState(w_dict["checkstate"])
                if "lower_bound" in w_dict.keys() and (isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox)):
                    widget.setMinimum(w_dict["lower_bound"])
                if "upper_bound" in w_dict.keys() and (isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox)):
                    widget.setMaximum(w_dict["upper_bound"])
                if "value" in w_dict.keys() and (isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox)):
                    widget.setValue(w_dict["value"])
                self.grid_layout.addWidget(widget, w_dict["grid"][0], w_dict["grid"][1], w_dict["grid"][2],
                                           w_dict["grid"][3])

    def getInputs(self):
        inputs = {}
        for k, v in self.grid_widget.items():
            if "line" in v.keys():
                inputs[k] = v["line"].text()
            elif "spinbox" in v.keys():
                inputs[k] = v["spinbox"].value()
            elif "checkbox" in v.keys():
                inputs[k] = bool(v["checkbox"].checkState())
            else:
                inputs[k] = None

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


class ExportIGESDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent=parent)

        self.setWindowTitle("Export IGES")
        self.setFont(self.parent().font())

        self.grid_widget = {}

        buttonBox = QDialogButtonBox(self)
        buttonBox.addButton(QDialogButtonBox.Ok)
        buttonBox.addButton(QDialogButtonBox.Cancel)
        self.grid_layout = QGridLayout(self)

        self.setInputs()

        self.grid_layout.addWidget(buttonBox, 6, 1, 1, 2)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def setInputs(self):
        widget_dict = load_data(os.path.join('dialog_widgets', 'export_IGES.json'))
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

    def getInputs(self):
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
        r0[1].setText('n0012-il')
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


class MSESFieldPlotDialogWidget(PymeadDialogWidget):
    def __init__(self, default_field_dir: str = None):
        super().__init__(settings_file=os.path.join(GUI_DEFAULTS_DIR, 'mses_field_plot_settings.json'))
        if default_field_dir is not None:
            self.widget_dict['analysis_dir']['widget'].setText(default_field_dir)

    def select_directory(self, line_edit: QLineEdit):
        select_directory(parent=self, line_edit=line_edit, starting_dir=tempfile.gettempdir())

    def updateDialog(self, new_inputs: dict, w_name: str):
        pass


class MSESFieldPlotDialog(PymeadDialog):
    def __init__(self, parent: QWidget, default_field_dir: str = None):
        w = MSESFieldPlotDialogWidget(default_field_dir=default_field_dir)
        super().__init__(parent=parent, window_title="MSES Field Plot Settings", widget=w)


class PymeadMessageBox(QMessageBox):
    def __init__(self, parent, msg: str, window_title: str, msg_mode: str):
        super().__init__(parent=parent)
        self.setText(msg)
        self.setWindowTitle(window_title)
        self.setIcon(msg_modes[msg_mode])
        self.setFont(self.parent().font())


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
            'Bottom': [5.0, 2, 1],
            'Top': [-5.0, 2, 3],
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
