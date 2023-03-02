from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QWidget, QGridLayout, QLabel, QPushButton, \
    QCheckBox, QMessageBox
from pymead.gui.mset_multigrid_widget import MSETMultiGridWidget, XTRSWidget, ADWidget
from pymead.gui.grid_bounds_widget import GridBounds
from pymead.gui.scientificspinbox_master.ScientificDoubleSpinBox import ScientificDoubleSpinBox
from abc import abstractmethod
from functools import partial

import PyQt5.QtWidgets
from PyQt5.QtCore import Qt
from pymead.utils.read_write_files import load_data
from pymead.gui.pyqt_vertical_tab_widget.pyqt_vertical_tab_widget.verticalTabWidget import VerticalTabWidget
import sys


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
                       'ADWidget': ('values', 'setValues', 'ADChanged')}
grid_names = {'label': ['label.row', 'label.column', 'label.rowSpan', 'label.columnSpan', 'label.alignment'],
              'widget': ['row', 'column', 'rowSpan', 'columnSpan', 'alignment'],
              'push_button': ['push.row', 'push.column', 'push.rowSpan', 'push.columnSpan', 'push.alignment'],
              'checkbox': ['check.row', 'check.column', 'check.rowSpan', 'check.columnSpan', 'check.alignment']}
# sum(<ragged list>, []) flattens a ragged (or uniform) 2-D list into a 1-D list
reserved_names = ['label', 'widget_type', 'push_button', 'push_button_action', 'clicked_connect', 'active_checkbox',
                  *sum([v for v in grid_names.values()], [])]
msg_modes = {'info': QMessageBox.Information, 'warn': QMessageBox.Warning, 'question': QMessageBox.Question}


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
                if w_dict['widget_type'] == 'ADWidget':
                    kwargs = self.kwargs
                widget = getattr(sys.modules[__name__], w_dict['widget_type'])(parent=self, **kwargs)
                print(f"{widget = }")
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


class PymeadDialogTabWidget(VerticalTabWidget):
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
    def __init__(self, parent, window_title: str, widget: PymeadDialogWidget or PymeadDialogTabWidget):
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


class PymeadMessageBox(QMessageBox):
    def __init__(self, parent, msg: str, window_title: str, msg_mode: str):
        super().__init__(parent=parent)
        self.setText(msg)
        self.setWindowTitle(window_title)
        self.setIcon(msg_modes[msg_mode])
        self.setFont(self.parent().font())
