from PyQt5.QtWidgets import QTabWidget, QWidget, QGridLayout, QDoubleSpinBox, QLabel, QSpinBox
from PyQt5.QtCore import pyqtSignal
import numpy as np


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
        }
    }
    return input_dict


def default_inputs_XTRS():
    input_dict = {
        'A0': {
            'XTRSupper': 1.0,
            'XTRSlower': 1.0,
        }
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
        self.tab_names = ['A0']
        self.input_dict = default_input_dict()
        self.widget_dict = {}
        self.grid_widget = None
        self.grid_layout = None
        self.generateWidgets()
        self.setTabs()

    def generateWidgets(self):
        for k1, v1 in self.input_dict.items():
            self.widget_dict[k1] = {}
            for k2, v2 in v1.items():
                w = QDoubleSpinBox(self)
                w.setMinimum(0.0)
                w.setMaximum(np.inf)
                w.setValue(v2)
                w.valueChanged.connect(self.valueChanged)
                w_label = QLabel(self.labels[k2], self)
                self.widget_dict[k1][k2] = {
                    'widget': w,
                    'label': w_label,
                }

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
        for k1, v1 in values.items():
            for k2, v2 in v1.items():
                self.widget_dict[k1][k2]['widget'].setValue(v2)

    def values(self):
        return {k1: {k2: v2['widget'].value() for k2, v2 in v1.items()} for k1, v1 in self.widget_dict.items()}

    def valueChanged(self, _):
        self.multiGridChanged.emit()


class XTRSWidget(QTabWidget):

    XTRSChanged = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.labels = {
            'XTRSupper': 'XTRSupper',
            'XTRSlower': 'XTRSlower',
        }
        self.tab_names = ['A0']
        self.input_dict = default_inputs_XTRS()
        self.widget_dict = {}
        self.grid_widget = None
        self.grid_layout = None
        self.generateWidgets()
        self.setTabs()

    def generateWidgets(self):
        for k1, v1 in self.input_dict.items():
            self.widget_dict[k1] = {}
            for k2, v2 in v1.items():
                w = QDoubleSpinBox(self)
                w.setMinimum(0.0)
                w.setMaximum(1.0)
                w.setValue(v2)
                w.valueChanged.connect(self.valueChanged)
                w_label = QLabel(self.labels[k2], self)
                self.widget_dict[k1][k2] = {
                    'widget': w,
                    'label': w_label,
                }

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
        for k1, v1 in values.items():
            for k2, v2 in v1.items():
                self.widget_dict[k1][k2]['widget'].setValue(v2)

    def values(self):
        return {k1: {k2: v2['widget'].value() for k2, v2 in v1.items()} for k1, v1 in self.widget_dict.items()}

    def valueChanged(self, _):
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
                if v2 != 'ISDELH':
                    w = QDoubleSpinBox(self)
                else:
                    w = QSpinBox(self)
                if v2 in ['XCDELH', 'ETAH']:
                    w.setMinimum(0.0)
                    w.setMaximum(1.0)
                elif v2 == 'ISDELH':
                    w.setMinimum(1)
                    w.setMaximum(100)
                elif v2 == 'PTRHIN':
                    w.setMinimum(1.0)
                    w.setMaximum(np.inf)

                w.setValue(v2)

                w.valueChanged.connect(self.valueChanged)
                w_label = QLabel(self.labels[k2], self)
                self.widget_dict[k1][k2] = {
                    'widget': w,
                    'label': w_label,
                }

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
        for k1, v1 in values.items():
            for k2, v2 in v1.items():
                self.widget_dict[k1][k2]['widget'].setValue(v2)

    def values(self):
        return {k1: {k2: v2['widget'].value() for k2, v2 in v1.items()} for k1, v1 in self.widget_dict.items()}

    def valueChanged(self, _):
        self.ADChanged.emit()
