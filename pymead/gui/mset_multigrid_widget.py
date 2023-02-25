from PyQt5.QtWidgets import QTabWidget, QWidget, QGridLayout, QDoubleSpinBox, QLabel, QSpinBox
from PyQt5.QtCore import pyqtSignal
import numpy as np
from copy import deepcopy
from functools import partial


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
        }
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


class ADWidget(QTabWidget):

    ADChanged = pyqtSignal()

    def __init__(self, parent):
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
        self.grid_widget = None
        self.grid_layout = None
        self.generateWidgets()
        self.setTabs()

    def generateWidgets(self):
        for k1, v1 in self.input_dict.items():
            self.widget_dict[k1] = {}
            for k2, v2 in v1.items():
                if k2 != 'ISDELH':
                    w = QDoubleSpinBox(self)
                else:
                    w = QSpinBox(self)
                if k2 in ['XCDELH', 'ETAH']:
                    w.setMinimum(0.0)
                    w.setMaximum(1.0)
                    w.setSingleStep(0.01)
                elif k2 == 'ISDELH':
                    w.setMinimum(1)
                    w.setMaximum(100)
                elif k2 == 'PTRHIN':
                    w.setMinimum(1.0)
                    w.setMaximum(np.inf)
                    w.setSingleStep(0.05)

                w.setValue(v2)

                w.valueChanged.connect(partial(self.valueChanged, k1, k2))
                w_label = QLabel(self.labels[k2], self)
                self.widget_dict[k1][k2] = {
                    'widget': w,
                    'label': w_label,
                }

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
        self.updateTabNames([k for k in values.keys()])
        self.regenerateWidgets()
        for k1, v1 in values.items():
            for k2, v2 in v1.items():
                self.widget_dict[k1][k2]['widget'].setValue(v2)
                self.input_dict[k1][k2] = v2

    def values(self):
        return self.input_dict

    def valueChanged(self, k1, k2, v2):
        self.input_dict[k1][k2] = v2
        self.ADChanged.emit()

    def setReadOnly(self, read_only: bool):
        for k1, v1 in self.widget_dict.items():
            for k2, v2 in v1.items():
                v2['widget'].setReadOnly(read_only)
