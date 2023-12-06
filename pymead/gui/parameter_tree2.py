import sys
from abc import abstractmethod

import numpy as np
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout, QHeaderView, QDialog, QGridLayout, \
    QDoubleSpinBox, QLineEdit, QLabel, QDialogButtonBox, QMenu
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QRegularExpression

from pymead.core.airfoil2 import Airfoil
from pymead.core.constraints import GeoCon, CollinearConstraint, CurvatureConstraint
from pymead.core import UNITS
from pymead.core.dimensions import LengthDimension, AngleDimension
from pymead.core.mea2 import MEA
from pymead.core.point import Point
from pymead.core.bezier2 import Bezier
from pymead.core.line2 import LineSegment
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.param2 import Param, DesVar, LengthParam, AngleParam, LengthDesVar, AngleDesVar
from pymead.core.pymead_obj import PymeadObj


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
        if isinstance(param, LengthParam) or isinstance(param, AngleParam):
            self.setSuffix(f" {param.unit()}")
        self.param = param
        if self.param.lower() is not None:
            self.setMinimum(self.param.lower())
        else:
            # if isinstance(self.param, LengthParam) or isinstance(self.param, AngleParam):
            #     self.setMinimum(0.0)
            # else:
            self.setMinimum(-1.0e9)
        if self.param.upper() is not None:
            self.setMaximum(self.param.upper())
        else:
            # if isinstance(self.param, AngleParam):
            #     self.setMaximum(UNITS.convert_angle_to_base(2 * np.pi, self.param.unit()))
            # else:
            self.setMaximum(1.0e9)
        self.setValue(self.param.value())
        self.valueChanged.connect(self.onValueChanged)

    # def setValue(self, val):
    #
    #     print(f"{val = }")
    #
    #     if isinstance(self.param, LengthParam) and self.param.point is None and val < 0.0:
    #         return
    #     elif isinstance(self.param, AngleParam):
    #         val = val % (2 * np.pi)
    #
    #     super().setValue(val)

    # def validate(self, inp, pos):
    #     if not hasattr(self, "param"):
    #         return QValidator.Acceptable
    #     elif isinstance(self.param, LengthParam) or isinstance(self.param, AngleParam) and len(inp.split()) > 1:
    #
    #         print(f"{inp = }, {pos = }")
    #         val = float(inp.split()[0])
    #         print(f"{val = }")
    #
    #         if isinstance(self.param, LengthParam) and val > 0.0:
    #             return QValidator.Acceptable
    #
    #         if isinstance(self.param, AngleParam) and 0.0 <= val < UNITS.convert_angle_to_base(2 * np.pi,
    #                                                                                            self.param.unit()):
    #             return QValidator.Acceptable
    #
    #         return QValidator.Intermediate
    #     else:
    #         return QValidator.Acceptable
    #
    # def fixup(self, s):
    #     suffix = None
    #     s_split = s.split()
    #     number = float(s_split[0])
    #     if len(s_split) > 1:
    #         suffix = s_split[1]
    #     print(f"{s = }")
    #
    #     if (isinstance(self.param, LengthParam) or isinstance(self.param, AngleParam)) and number < 0.0:
    #         return f"0.0 {suffix}"
    #
    #     if isinstance(self.param, AngleParam) and number > UNITS.convert_angle_from_base(2 * np.pi, self.param.unit()):
    #         return f"{UNITS.convert_angle_from_base(2 * np.pi)} {suffix}"

    def onValueChanged(self, value: float):
        if self.param.point is None:
            self.param.set_value(value)
        else:
            if self.param is self.param.point.x():
                self.param.point.request_move(value, self.param.point.y().value())
            elif self.param is self.param.point.y():
                self.param.point.request_move(self.param.point.x().value(), value)
        self.setValue(self.param.value())


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
    def __init__(self, parent, pymead_obj: PymeadObj, tree):
        super().__init__(parent)
        self.pymead_obj = pymead_obj
        self.tree = tree

        validator = NameValidator(self, tree, sub_container=pymead_obj.sub_container)
        self.setValidator(validator)
        self.setText(self.pymead_obj.name())
        self.textChanged.connect(self.onTextChanged)

    def onTextChanged(self, name: str):
        self.pymead_obj.set_name(name)


class LowerSpin(QDoubleSpinBox):
    def __init__(self, parent, param: Param):
        super().__init__(parent)
        self.setDecimals(6)
        self.setSingleStep(0.01)
        self.param = param
        if (isinstance(param, LengthParam) and self.param.point is None) or isinstance(param, AngleParam):
            self.setMinimum(0.0)
        else:
            self.setMinimum(-1e9)

        if isinstance(param, AngleParam):
            self.setMaximum(UNITS.convert_angle_to_base(2 * np.pi, self.param.unit()))
        else:
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


class TreeButton(QPushButton):
    sigNameChanged = pyqtSignal(str, object)

    def __init__(self, pymead_obj: PymeadObj, tree):
        super().__init__(pymead_obj.name())
        self.setMaximumWidth(100)
        self.pymead_obj = pymead_obj
        self.tree = tree
        self.dialog = None
        self.clicked.connect(self.onClicked)

    def onClicked(self):
        self.dialog = self.createDialog()
        if self.dialog.exec_():
            pass
        self.dialog = None

    def onNameChange(self, name: str):
        if self.dialog is not None:
            self.dialog.setWindowTitle(f"{name}")
        self.setText(name)
        self.sigNameChanged.emit(name, self.pymead_obj)

    def createDialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{self.pymead_obj.name()}")
        layout = QGridLayout()
        dialog.setLayout(layout)
        self.modifyDialogInternals(dialog, layout)
        self.addButtonBoxToDialog(dialog, layout)
        return dialog

    @abstractmethod
    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        pass

    def addButtonBoxToDialog(self, dialog: QDialog, layout: QGridLayout):
        # Add the button box
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox, layout.rowCount(), 1)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)


class ParamButton(TreeButton):
    sigValueChanged = pyqtSignal(float)  # value

    def __init__(self, param: Param, tree, name_editable: bool = True):
        super().__init__(pymead_obj=param, tree=tree)
        self.name_editable = name_editable
        self.param = param

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
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
            row_count = layout.rowCount()
            layout.addWidget(lower_label, row_count, 0)
            layout.addWidget(lower_spin, row_count, 1)
        if self.param.upper() is not None:
            upper_label = QLabel("Upper Bound", self)
            upper_spin = UpperSpin(self, self.param)
            row_count = layout.rowCount()
            layout.addWidget(upper_label, row_count, 0)
            layout.addWidget(upper_spin, row_count, 1)

    def onValueChange(self, value: float):
        self.sigValueChanged.emit(value)


class LengthParamButton(ParamButton):
    pass


class AngleParamButton(ParamButton):
    pass


class DesVarButton(TreeButton):
    sigValueChanged = pyqtSignal(float)  # value

    def __init__(self, desvar: DesVar, tree, name_editable: bool = True):
        super().__init__(pymead_obj=desvar, tree=tree)
        self.name_editable = name_editable
        self.desvar = desvar

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        value_label = QLabel("Value", self)
        value_spin = ValueSpin(self, self.desvar)
        value_spin.valueChanged.connect(self.onValueChange)
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.desvar, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        if not self.name_editable:
            name_edit.setReadOnly(True)
        layout.addWidget(value_label, 0, 0)
        layout.addWidget(value_spin, 0, 1)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        lower_label = QLabel("Lower Bound", self)
        lower_spin = LowerSpin(self, self.desvar)
        row_count = layout.rowCount()
        layout.addWidget(lower_label, row_count, 0)
        layout.addWidget(lower_spin, row_count, 1)
        upper_label = QLabel("Upper Bound", self)
        upper_spin = UpperSpin(self, self.desvar)
        row_count = layout.rowCount()
        layout.addWidget(upper_label, row_count, 0)
        layout.addWidget(upper_spin, row_count, 1)

    def onValueChange(self, value: float):
        self.sigValueChanged.emit(value)


class LengthDesVarButton(DesVarButton):
    pass


class AngleDesVarButton(DesVarButton):
    pass


class PointButton(TreeButton):

    def __init__(self, point: Point, tree):
        super().__init__(pymead_obj=point, tree=tree)
        self.point = point
        self.x_button = None
        self.y_button = None

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
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

    def onXChanged(self, x: float):
        self.point.request_move(x, self.point.y().value())

    def onYChanged(self, y: float):
        self.point.request_move(self.point.x().value(), y)

    def onNameChange(self, name: str):
        self.x_button.setText(f"{name}.x")
        self.y_button.setText(f"{name}.y")
        super().onNameChange(name=name)

    def enterEvent(self, a0):
        self.point.canvas_item.setScatterStyle("hovered")

    def leaveEvent(self, a0):
        self.point.canvas_item.setScatterStyle("default")


class BezierButton(TreeButton):

    def __init__(self, bezier: Bezier, tree):
        super().__init__(pymead_obj=bezier, tree=tree)
        self.bezier = bezier

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.bezier, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        for point in self.bezier.point_sequence().points():
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            layout.addWidget(point_button, layout.rowCount(), 0)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)

    def enterEvent(self, a0):
        self.bezier.canvas_item.setCurveStyle("hovered")

    def leaveEvent(self, a0):
        self.bezier.canvas_item.setCurveStyle("default")


class LineSegmentButton(TreeButton):

    def __init__(self, line: LineSegment, tree):
        super().__init__(pymead_obj=line, tree=tree)
        self.line = line

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.line, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        for point in self.line.point_sequence().points():
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            layout.addWidget(point_button, layout.rowCount(), 0)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)

    def enterEvent(self, a0):
        self.line.canvas_item.setCurveStyle("hovered")

    def leaveEvent(self, a0):
        self.line.canvas_item.setCurveStyle("default")


class AirfoilButton(TreeButton):
    def __init__(self, airfoil: Airfoil, tree):
        super().__init__(pymead_obj=airfoil, tree=tree)
        self.airfoil = airfoil

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
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

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)


class MEAButton(TreeButton):
    def __init__(self, mea: MEA, tree):
        super().__init__(pymead_obj=mea, tree=tree)
        self.mea = mea

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.mea, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        for airfoil in self.mea.airfoils:
            airfoil_button = AirfoilButton(airfoil, self.tree)
            airfoil_button.sigNameChanged.connect(self.onAirfoilNameChange)
            layout.addWidget(airfoil_button, layout.rowCount(), 0)

    def onAirfoilNameChange(self, name: str, airfoil: Airfoil):
        if airfoil.tree_item is not None:
            self.tree.itemWidget(airfoil.tree_item, 0).setText(name)


class LengthDimensionButton(TreeButton):

    def __init__(self, length_dim: LengthDimension, tree):
        super().__init__(pymead_obj=length_dim, tree=tree)
        self.length_dimension = length_dim

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.length_dimension, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        labels = ["Tool Point", "Target Point"]
        points = [self.length_dimension.tool(), self.length_dimension.target()]
        for label, point in zip(labels, points):
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            q_label = QLabel(label, self)
            row_count = layout.rowCount()
            layout.addWidget(q_label, row_count, 0)
            layout.addWidget(point_button, row_count, 1)
        row_count = layout.rowCount()
        layout.addWidget(QLabel("Length Param", self), row_count, 0)
        layout.addWidget(LengthParamButton(self.length_dimension.param(), self.tree), row_count, 1)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)


class AngleDimensionButton(TreeButton):

    def __init__(self, angle_dim: AngleDimension, tree):
        super().__init__(pymead_obj=angle_dim, tree=tree)
        self.angle_dimension = angle_dim

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.angle_dimension, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        labels = ["Tool Point", "Target Point"]
        points = [self.angle_dimension.tool(), self.angle_dimension.target()]
        for label, point in zip(labels, points):
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            q_label = QLabel(label, self)
            row_count = layout.rowCount()
            layout.addWidget(q_label, row_count, 0)
            layout.addWidget(point_button, row_count, 1)
        row_count = layout.rowCount()
        layout.addWidget(QLabel("Angle Param", self), row_count, 0)
        layout.addWidget(LengthParamButton(self.angle_dimension.param(), self.tree), row_count, 1)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)


class CollinearConstraintButton(TreeButton):

    def __init__(self, collinear_constraint: CollinearConstraint, tree):
        super().__init__(pymead_obj=collinear_constraint, tree=tree)
        self.collinear_constraint = collinear_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.collinear_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        labels = ["Start Point", "Middle Point", "End Point"]
        points = [self.collinear_constraint.target().points()[0], self.collinear_constraint.tool(),
                  self.collinear_constraint.target().points()[1]]
        for label, point in zip(labels, points):
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            q_label = QLabel(label, self)
            row_count = layout.rowCount()
            layout.addWidget(q_label, row_count, 0)
            layout.addWidget(point_button, row_count, 1)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)


class CurvatureConstraintButton(TreeButton):

    def __init__(self, curvature_constraint: CurvatureConstraint, tree):
        super().__init__(pymead_obj=curvature_constraint, tree=tree)
        self.curvature_constraint = curvature_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.curvature_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        labels = ["Curve 1 G2 Point", "Curve 1 G1 Point", "Curve Joint", "Curve 2 G1 Point", "Curve 2 G2 Point"]
        points = [self.curvature_constraint.target().points()[0], self.curvature_constraint.target().points()[1],
                  self.curvature_constraint.tool(),
                  self.curvature_constraint.target().points()[2], self.curvature_constraint.target().points()[3]]
        for label, point in zip(labels, points):
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            q_label = QLabel(label, self)
            row_count = layout.rowCount()
            layout.addWidget(q_label, row_count, 0)
            layout.addWidget(point_button, row_count, 1)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)


class ParameterTree(QTreeWidget):
    def __init__(self, geo_col: GeometryCollection, parent):
        super().__init__(parent)

        self.geo_col = geo_col
        self.geo_col.tree = self

        self.setColumnCount(1)

        self.container_titles = {
            "desvar": "Design Variables",
            "params": "Parameters",
            "points": "Points",
            "lines": "Lines",
            "bezier": "Bézier Curves",
            "airfoils": "Airfoils",
            "mea": "Multi-Element Airfoils",
            "geocon": "Geometric Constraints",
            "dims": "Dimensions"
        }

        self.items = [QTreeWidgetItem(None, [f"{self.container_titles[k]}"]) for k in self.geo_col.container().keys()]
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

        # Set the tree to be expanded by default
        self.expandAll()

    def addPymeadTreeItem(self, pymead_obj: PymeadObj):
        top_level_item = self.items[self.topLevelDict[pymead_obj.sub_container]]
        child_item = QTreeWidgetItem([""])
        top_level_item.addChild(child_item)
        pymead_obj.tree_item = child_item

        button_args = (pymead_obj, self)
        button_mappings = {"Param": "ParamButton",
                           "LengthParam": "LengthParamButton",
                           "AngleParam": "AngleParamButton",
                           "DesVar": "DesVarButton",
                           "LengthDesVar": "LengthDesVarButton",
                           "AngleDesVar": "AngleDesVarButton",
                           "Point": "PointButton",
                           "Bezier": "BezierButton",
                           "LineSegment": "LineSegmentButton",
                           "Airfoil": "AirfoilButton",
                           "MEA": "MEAButton",
                           "LengthDimension": "LengthDimensionButton",
                           "AngleDimension": "AngleDimensionButton",
                           "CollinearConstraint": "CollinearConstraintButton",
                           "CurvatureConstraint": "CurvatureConstraintButton",
                           }
        button = getattr(sys.modules[__name__], button_mappings[type(pymead_obj).__name__])(*button_args)

        self.setItemWidget(child_item, 0, button)

    def removePymeadTreeItem(self, pymead_obj: PymeadObj):
        top_level_item = self.items[self.topLevelDict[pymead_obj.sub_container]]
        top_level_item.removeChild(pymead_obj.tree_item)
        pymead_obj.tree_item = None

    def onExpandPressed(self):
        self.expandAll()

    def onCollapsePressed(self):
        self.collapseAll()

    def contextMenuEvent(self, a0):
        item = self.itemAt(a0.x(), a0.y())
        if item is None:
            return

        item_text = item.text(0)

        if item_text == "Design Variables":
            menu = QMenu(self)
            addDesVarAction = menu.addAction("Add Design Variable")
            res = menu.exec_(a0.globalPos())

            if res == addDesVarAction:
                self.geo_col.add_desvar(0.0, "dv")

        elif item_text == "Parameters":
            menu = QMenu(self)
            addParameterAction = menu.addAction("Add Parameter")
            res = menu.exec_(a0.globalPos())

            if res == addParameterAction:
                self.geo_col.add_param(0.0, "param")

        elif item_text == "":
            button = self.itemWidget(item, 0)
            if isinstance(button, TreeButton):
                menu = QMenu(self)

                promoteAction = None
                demoteAction = None

                pymead_obj = button.pymead_obj
                pymead_obj_type = type(pymead_obj)

                if pymead_obj_type in [Param, LengthParam, AngleParam]:
                    promoteAction = menu.addAction("Promote to Design Variable")
                elif pymead_obj_type in [DesVar, LengthDesVar, AngleDesVar]:
                    demoteAction = menu.addAction("Demote to Parameter")

                removeObjectAction = menu.addAction("Delete")
                res = menu.exec_(a0.globalPos())

                if res == removeObjectAction:
                    self.geo_col.remove_pymead_obj(button.pymead_obj)
                elif res == promoteAction:
                    self.geo_col.promote_param_to_desvar(pymead_obj)
                elif res == demoteAction:
                    self.geo_col.demote_desvar_to_param(pymead_obj)
