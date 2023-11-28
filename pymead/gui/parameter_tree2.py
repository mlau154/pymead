from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout, QHeaderView, QDialog, QGridLayout, \
    QDoubleSpinBox, QLineEdit, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QModelIndex

from pymead.core.bezier2 import Bezier
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.param2 import Param


class HeaderButtonRow(QHeaderView):
    sigExpandPressed = pyqtSignal()
    sigCollapsePressed = pyqtSignal()

    def __init__(self, parent):
        super().__init__(Qt.Horizontal, parent)
        self.lay = QHBoxLayout(self)
        self.expandButton = QPushButton("Expand All", self)
        self.collapseButton = QPushButton("Collapse All", self)
        self.expandButton.clicked.connect(self.expandButtonPressed)
        self.collapseButton.clicked.connect(self.collapseButtonPressed)
        self.lay.addWidget(self.expandButton)
        self.lay.addWidget(self.collapseButton)
        self.setLayout(self.lay)
        self.setFixedHeight(40)

    @pyqtSlot()
    def expandButtonPressed(self):
        self.sigExpandPressed.emit()

    @pyqtSlot()
    def collapseButtonPressed(self):
        self.sigCollapsePressed.emit()


class ValueSpin(QDoubleSpinBox):
    def __init__(self, parent, param: Param):
        super().__init__(parent)
        self.setDecimals(6)
        self.param = param
        if self.param.lower() is not None:
            self.setMinimum(self.param.lower())
        if self.param.upper() is not None:
            self.setMaximum(self.param.upper())
        self.setValue(self.param.value())
        self.valueChanged.connect(self.onValueChanged)

    def onValueChanged(self, value: float):
        self.param.set_value(value)


class NameEdit(QLineEdit):
    def __init__(self, parent, param: Param):
        super().__init__(parent)
        self.param = param
        self.setText(self.param.name())
        self.textChanged.connect(self.onTextChanged)

    def onTextChanged(self, name: float):
        self.param.set_name(name)


class LowerSpin(QDoubleSpinBox):
    def __init__(self, parent, param: Param):
        super().__init__(parent)
        self.setDecimals(6)
        self.param = param
        self.setMinimum(-1e9)
        self.setMaximum(1e9)
        self.setValue(self.param.lower())
        self.valueChanged.connect(self.onValueChanged)

    def onValueChanged(self, lower: float):
        self.param.set_lower(lower)


class UpperSpin(QDoubleSpinBox):
    def __init__(self, parent, param: Param):
        super().__init__(parent)
        self.setDecimals(6)
        self.param = param
        self.setMinimum(-1e9)
        self.setMaximum(1e9)
        self.setValue(self.param.upper())
        self.valueChanged.connect(self.onValueChanged)

    def onValueChanged(self, upper: float):
        self.param.set_upper(upper)


class ParamButton(QPushButton):
    def __init__(self, param):
        self.param = param
        super().__init__(param.name())
        self.setMaximumWidth(100)
        self.dialog = None
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(f"Param - {self.param.name()}")
        layout = QGridLayout()
        value_label = QLabel("Value", self)
        value_spin = ValueSpin(self, self.param)
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.param)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(value_label, 0, 0)
        layout.addWidget(value_spin, 0, 1)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        if self.param.lower() is not None:
            lower_label = QLabel("Lower Bound", self)
            lower_spin = LowerSpin(self, self.param)
            layout.addWidget(lower_label, layout.rowCount(), 0)
            layout.addWidget(lower_spin, layout.rowCount(), 1)
        if self.param.upper() is not None:
            upper_label = QLabel("Upper Bound", self)
            upper_spin = LowerSpin(self, self.param)
            layout.addWidget(upper_label, layout.rowCount(), 0)
            layout.addWidget(upper_spin, layout.rowCount(), 1)
        self.dialog.setLayout(layout)
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"Param - {name}")
        self.setText(name)


class ParameterTree(QTreeWidget):
    def __init__(self, geo_col: GeometryCollection, parent):
        super().__init__(parent)

        self.geo_col = geo_col
        self.geo_col.geo_tree = self

        self.setColumnCount(1)

        self.items = [QTreeWidgetItem(None, [f"{k}"]) for k in self.geo_col.container().keys()]
        self.topLevelDict = {k: i for i, k in enumerate(self.geo_col.container().keys())}
        self.insertTopLevelItems(0, self.items)

        # Make the header
        self.setHeaderLabel("")
        self.headerRow = HeaderButtonRow(self)
        self.headerRow.sigExpandPressed.connect(self.onExpandPressed)
        self.headerRow.sigCollapsePressed.connect(self.onCollapsePressed)
        self.setHeader(self.headerRow)

        # self.header().resizeSection(0, 320)
        self.setMinimumWidth(200)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)

    def addParam(self, param: Param):
        top_level_item = self.items[self.topLevelDict["params"]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        param.tree_item = child_item
        # index = top_level_item.indexOfChild(child_item)
        self.setItemWidget(child_item, 0, ParamButton(param))

    def removeParam(self, param: Param):
        top_level_item = self.items[self.topLevelDict["params"]]
        top_level_item.removeChild(param.tree_item)
        param.tree_item = None

    def addBezier(self, bezier: Bezier):
        pass

    def onExpandPressed(self):
        self.expandAll()

    def onCollapsePressed(self):
        self.collapseAll()
