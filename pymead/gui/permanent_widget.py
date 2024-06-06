from PyQt5.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel, QComboBox

from pymead.version import __version__


class PymeadProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setStyleSheet(
        #     '''
        #     QProgressBar {border: 2px solid; border-color: #8E9091;}
        #     QProgressBar::chunk {background-color: #6495ED; width: 10px; margin: 0.5px;}
        #     '''
        # )
        self.setMaximum(100)
        self.setValue(0)
        self.hide()


class PymeadInfoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText(f"pymead {__version__}")


class PermanentWidget(QWidget):
    def __init__(self, geo_col, parent=None):
        super().__init__(parent)
        self.geo_col = geo_col
        self.lay = QHBoxLayout(self)
        self.inviscid_cl_label = QLabel("Inviscid CL", self)
        self.inviscid_cl_combo = QComboBox(self)
        self.inviscid_cl_combo.setMinimumWidth(80)
        self.progress_bar = PymeadProgressBar(self)
        self.info_label = PymeadInfoLabel(self)
        self.lay.addWidget(self.inviscid_cl_label)
        self.lay.addWidget(self.inviscid_cl_combo)
        self.lay.addWidget(self.progress_bar)
        self.lay.addWidget(self.info_label)
        self.setLayout(self.lay)

    def updateAirfoils(self):
        previous_item = self.inviscid_cl_combo.currentText()
        new_item_list = [""] + [a for a in self.geo_col.container()["airfoils"]]

        # Clear the combo box and add the new items
        self.inviscid_cl_combo.clear()
        self.inviscid_cl_combo.addItems(new_item_list)

        # If the previous airfoil is still present, set it as the current selected item. This will
        # prevent the current airfoil being analyzed from changing when airfoils are added
        if previous_item in new_item_list:
            self.inviscid_cl_combo.setCurrentText(previous_item)
