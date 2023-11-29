from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout, QHeaderView, QDialog, QGridLayout, \
    QDoubleSpinBox, QLineEdit, QLabel, QDialogButtonBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QRegularExpression

from pymead.core.airfoil2 import Airfoil
from pymead.core.point import Point
from pymead.core.bezier2 import Bezier
from pymead.core.line2 import LineSegment
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
        self.setSingleStep(0.01)
        self.param = param
        if self.param.lower() is not None:
            self.setMinimum(self.param.lower())
        else:
            self.setMinimum(-1.0e9)
        if self.param.upper() is not None:
            self.setMaximum(self.param.upper())
        else:
            self.setMaximum(1.0e9)
        self.setValue(self.param.value())
        self.valueChanged.connect(self.onValueChanged)

    def onValueChanged(self, value: float):
        self.param.set_value(value)


class NameValidator(QValidator):

    def __init__(self, parent, tree, sub_container: str):
        super().__init__(parent)
        self.geo_col = tree.geo_col
        self.sub_container = sub_container
        self.regex = QRegularExpression("^[a-z-A-Z_0-9]+$")

    def validate(self, a0, a1):
        if a0 in self.geo_col.container()[self.sub_container].keys():
            return QValidator.Invalid, a0, a1

        if not self.regex.match(a0).hasMatch():
            return QValidator.Invalid, a0, a1

        return QValidator.Acceptable, a0, a1

    def fixup(self, a0):
        pass


class NameEdit(QLineEdit):
    def __init__(self, parent, obj: Param or LineSegment or Bezier or Point, tree):
        super().__init__(parent)
        self.obj = obj
        self.tree = tree

        if isinstance(obj, Point):
            sub_container = "points"
        elif isinstance(obj, Param):
            sub_container = "params"
        elif isinstance(obj, Bezier):
            sub_container = "bezier"
        elif isinstance(obj, LineSegment):
            sub_container = "lines"
        elif isinstance(obj, Airfoil):
            sub_container = "airfoils"
        else:
            raise ValueError("Invalid NameEdit input object")

        validator = NameValidator(self, tree, sub_container=sub_container)
        self.setValidator(validator)
        self.setText(self.obj.name())
        self.textChanged.connect(self.onTextChanged)

    def onTextChanged(self, name: float):
        self.obj.set_name(name)


class LowerSpin(QDoubleSpinBox):
    def __init__(self, parent, param: Param):
        super().__init__(parent)
        self.setDecimals(6)
        self.setSingleStep(0.01)
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
        self.setSingleStep(0.01)
        self.param = param
        self.setMinimum(-1e9)
        self.setMaximum(1e9)
        self.setValue(self.param.upper())
        self.valueChanged.connect(self.onValueChanged)

    def onValueChanged(self, upper: float):
        self.param.set_upper(upper)


class ParamButton(QPushButton):
    sigValueChanged = pyqtSignal(float)  # value

    def __init__(self, param, tree, name_editable: bool = True):
        self.param = param
        super().__init__(param.name())
        self.setMaximumWidth(100)
        self.tree = tree
        self.dialog = None
        self.name_editable = name_editable
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(f"Param - {self.param.name()}")
        layout = QGridLayout()
        value_label = QLabel("Value", self)
        value_spin = ValueSpin(self, self.param)
        value_spin.valueChanged.connect(self.onValueChange)
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.param, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        if not self.name_editable:
            name_edit.setReadOnly(True)
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

        # Add the button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox, layout.rowCount(), 1)
        buttonBox.accepted.connect(self.dialog.accept)
        buttonBox.rejected.connect(self.dialog.reject)

        self.dialog.setLayout(layout)
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"Param - {name}")
        self.setText(name)

    def onValueChange(self, value: float):
        self.sigValueChanged.emit(value)


class PointButton(QPushButton):
    sigNameChanged = pyqtSignal(str, object)

    def __init__(self, point: Point, tree):
        self.point = point
        super().__init__(point.name())
        self.setMaximumWidth(100)
        self.tree = tree
        self.dialog = None
        self.x_button = None
        self.y_button = None
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(f"Point - {self.point.name()}")
        layout = QGridLayout()
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.point, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        x_label = QLabel("x", self)
        self.x_button = ParamButton(self.point.x(), self.tree, name_editable=False)
        self.x_button.sigValueChanged.connect(self.onXChanged)
        y_label = QLabel("y", self)
        self.y_button = ParamButton(self.point.y(), self.tree, name_editable=False)
        self.y_button.sigValueChanged.connect(self.onYChanged)
        layout.addWidget(name_label, 0, 0)
        layout.addWidget(name_edit, 0, 1)
        layout.addWidget(x_label, 1, 0)
        layout.addWidget(self.x_button, 1, 1)
        layout.addWidget(y_label, 2, 0)
        layout.addWidget(self.y_button, 2, 1)

        # Add the button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox, layout.rowCount(), 1)
        buttonBox.accepted.connect(self.dialog.accept)
        buttonBox.rejected.connect(self.dialog.reject)

        self.dialog.setLayout(layout)
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onXChanged(self, x: float):
        self.point.request_move(x, self.point.y().value())

    def onYChanged(self, y: float):
        self.point.request_move(self.point.x().value(), y)

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"Point - {name}")
        self.setText(name)
        self.x_button.setText(f"{name}.x")
        self.y_button.setText(f"{name}.y")
        self.sigNameChanged.emit(name, self.point)

    def enterEvent(self, a0):
        self.point.gui_obj.setScatterStyle("hovered")

    def leaveEvent(self, a0):
        self.point.gui_obj.setScatterStyle("default")


class BezierButton(QPushButton):
    def __init__(self, bezier: Bezier, tree):
        self.bezier = bezier
        super().__init__(bezier.name())
        self.setMaximumWidth(100)
        self.tree = tree
        self.dialog = None
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(f"Bezier - {self.bezier.name()}")
        layout = QGridLayout()
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.bezier, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        for point in self.bezier.point_sequence().points():
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            layout.addWidget(point_button, layout.rowCount(), 0)

        # Add the button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox, layout.rowCount(), 1)
        buttonBox.accepted.connect(self.dialog.accept)
        buttonBox.rejected.connect(self.dialog.reject)

        self.dialog.setLayout(layout)
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"Bezier - {name}")
        self.setText(name)
        # for point in self.bezier.point_sequence().points():
        #     if point.tree_item is not None:
        #         self.tree.itemWidget(point.tree_item, 0).setText(name)

    def enterEvent(self, a0):
        self.bezier.gui_obj.setCurveStyle("hovered")

    def leaveEvent(self, a0):
        self.bezier.gui_obj.setCurveStyle("default")


class LineButton(QPushButton):
    def __init__(self, line: LineSegment, tree):
        self.line = line
        super().__init__(line.name())
        self.setMaximumWidth(100)
        self.tree = tree
        self.dialog = None
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(f"Line - {self.line.name()}")
        layout = QGridLayout()
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.line, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        for point in self.line.point_sequence().points():
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            layout.addWidget(point_button, layout.rowCount(), 0)

        # Add the button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox, layout.rowCount(), 1)
        buttonBox.accepted.connect(self.dialog.accept)
        buttonBox.rejected.connect(self.dialog.reject)

        self.dialog.setLayout(layout)
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"Line - {name}")
        self.setText(name)

    def enterEvent(self, a0):
        self.line.gui_obj.setCurveStyle("hovered")

    def leaveEvent(self, a0):
        self.line.gui_obj.setCurveStyle("default")


class AirfoilButton(QPushButton):
    def __init__(self, airfoil: Airfoil, tree):
        self.airfoil = airfoil
        super().__init__(airfoil.name())
        self.setMaximumWidth(100)
        self.tree = tree
        self.dialog = None
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle(f"Line - {self.airfoil.name()}")
        layout = QGridLayout()
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.airfoil, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        labels = ["Leading Edge", "Trailing Edge", "Upper Surface End", "Lower Surface End"]
        points = [self.airfoil.leading_edge, self.airfoil.trailing_edge, self.airfoil.upper_surf_end,
                  self.airfoil.lower_surf_end]

        for label, point in zip(labels, points):
            q_label = QLabel(label)
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            row_count = layout.rowCount()
            layout.addWidget(q_label, row_count, 0)
            layout.addWidget(point_button, row_count, 1)

        # Add the button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox, layout.rowCount(), 1)
        buttonBox.accepted.connect(self.dialog.accept)
        buttonBox.rejected.connect(self.dialog.reject)

        self.dialog.setLayout(layout)
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"Airfoil - {name}")
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
        self.setItemWidget(child_item, 0, ParamButton(param, self))

    def removeParam(self, param: Param):
        top_level_item = self.items[self.topLevelDict["params"]]
        top_level_item.removeChild(param.tree_item)
        param.tree_item = None

    def addPoint(self, point: Point):
        top_level_item = self.items[self.topLevelDict["points"]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        point.tree_item = child_item
        self.setItemWidget(child_item, 0, PointButton(point, self))

    def removePoint(self, point: Point):
        top_level_item = self.items[self.topLevelDict["points"]]
        top_level_item.removeChild(point.tree_item)
        point.tree_item = None

    def addBezier(self, bezier: Bezier):
        top_level_item = self.items[self.topLevelDict["bezier"]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        bezier.tree_item = child_item
        bezier_button = BezierButton(bezier, self)
        self.setItemWidget(child_item, 0, bezier_button)

    def removeBezier(self, bezier: Bezier):
        top_level_item = self.items[self.topLevelDict["bezier"]]
        top_level_item.removeChild(bezier.tree_item)
        bezier.tree_item = None

    def addLine(self, line: LineSegment):
        top_level_item = self.items[self.topLevelDict["lines"]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        line.tree_item = child_item
        line_button = LineButton(line, self)
        self.setItemWidget(child_item, 0, line_button)

    def removeLine(self, line: LineSegment):
        top_level_item = self.items[self.topLevelDict["lines"]]
        top_level_item.removeChild(line.tree_item)
        line.tree_item = None

    def addAirfoil(self, airfoil: Airfoil):
        top_level_item = self.items[self.topLevelDict["airfoils"]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        airfoil.tree_item = child_item
        airfoil_button = AirfoilButton(airfoil, self)
        self.setItemWidget(child_item, 0, airfoil_button)

    def removeAirfoil(self, airfoil: Airfoil):
        top_level_item = self.items[self.topLevelDict["airfoils"]]
        top_level_item.removeChild(airfoil.tree_item)
        airfoil.tree_item = None

    def onExpandPressed(self):
        self.expandAll()

    def onCollapsePressed(self):
        self.collapseAll()
