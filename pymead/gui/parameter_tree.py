import sys
import time
import typing
from abc import abstractmethod

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QValidator, QFont, QBrush, QColor
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout, QHeaderView, QDialog, QGridLayout, \
    QDoubleSpinBox, QLineEdit, QLabel, QDialogButtonBox, QMenu, QAbstractItemView, QTreeWidgetItemIterator
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QRegularExpression

from pymead.core.airfoil import Airfoil
from pymead.core.constraints import *
from pymead.core import UNITS
from pymead.core.dimensions import LengthDimension, AngleDimension
from pymead.core.mea import MEA
from pymead.core.point import Point
from pymead.core.bezier import Bezier
from pymead.core.line import LineSegment
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.param import Param, DesVar, LengthParam, AngleParam, LengthDesVar, AngleDesVar
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
        self.pymead_obj = param
        self.setMaximumWidth(150)
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

    def __init__(self, pymead_obj: PymeadObj, tree, top_level: bool = False):
        label = "Edit" if top_level else pymead_obj.name()
        self.top_level = top_level
        super().__init__(label)
        self.setMaximumWidth(150)
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
        if self.top_level:
            self.pymead_obj.tree_item.setText(0, name)
        else:
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

    def __init__(self, param: Param, tree, name_editable: bool = True, top_level: bool = False):
        super().__init__(pymead_obj=param, tree=tree, top_level=top_level)
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

    def __init__(self, desvar: DesVar, tree, name_editable: bool = True, top_level: bool = False):
        super().__init__(pymead_obj=desvar, tree=tree, top_level=top_level)
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

    def __init__(self, point: Point, tree, top_level: bool = False):
        super().__init__(pymead_obj=point, tree=tree, top_level=top_level)
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
        if self.top_level:
            return
        if self.pymead_obj.tree_item.hoverable:
            self.tree.geo_col.hover_enter_obj(self.pymead_obj)

    def leaveEvent(self, a0):
        if self.top_level:
            return
        if self.pymead_obj.tree_item.hoverable:
            self.tree.geo_col.hover_leave_obj(self.pymead_obj)


class BezierButton(TreeButton):

    def __init__(self, bezier: Bezier, tree, top_level: bool = False):
        super().__init__(pymead_obj=bezier, tree=tree, top_level=top_level)
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

    def __init__(self, line: LineSegment, tree, top_level: bool = False):
        super().__init__(pymead_obj=line, tree=tree, top_level=top_level)
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
    def __init__(self, airfoil: Airfoil, tree, top_level: bool = False):
        super().__init__(pymead_obj=airfoil, tree=tree, top_level=top_level)
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
    def __init__(self, mea: MEA, tree, top_level: bool = False):
        super().__init__(pymead_obj=mea, tree=tree, top_level=top_level)
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

    def __init__(self, length_dim: LengthDimension, tree, top_level: bool = False):
        super().__init__(pymead_obj=length_dim, tree=tree, top_level=top_level)
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


class DistanceConstraintButton(TreeButton):

    def __init__(self, distance_constraint: DistanceConstraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=distance_constraint, tree=tree, top_level=top_level)
        self.distance_constraint = distance_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.distance_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)
        labels = ["Start Point", "End Point"]
        points = [self.distance_constraint.p1, self.distance_constraint.p2]
        for label, point in zip(labels, points):
            point_button = PointButton(point, self.tree)
            point_button.sigNameChanged.connect(self.onPointNameChange)
            q_label = QLabel(label, self)
            row_count = layout.rowCount()
            layout.addWidget(q_label, row_count, 0)
            layout.addWidget(point_button, row_count, 1)
        row_count = layout.rowCount()
        layout.addWidget(QLabel("Length Param", self), row_count, 0)
        layout.addWidget(LengthParamButton(self.distance_constraint.param(), self.tree), row_count, 1)

    def onPointNameChange(self, name: str, point: Point):
        if point.tree_item is not None:
            self.tree.itemWidget(point.tree_item, 0).setText(name)


class AngleDimensionButton(TreeButton):

    def __init__(self, angle_dim: AngleDimension, tree, top_level: bool = False):
        super().__init__(pymead_obj=angle_dim, tree=tree, top_level=top_level)
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

    def __init__(self, collinear_constraint: AntiParallel3Constraint, tree, top_level: bool = False):
        # TODO: either remove this constraint or create it (Collinear instead of AntiParallel3)
        super().__init__(pymead_obj=collinear_constraint, tree=tree, top_level=top_level)
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


class RelAngle3ConstraintButton(TreeButton):

    def __init__(self, rel_angle_constraint: RelAngle3Constraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=rel_angle_constraint, tree=tree, top_level=top_level)
        self.rel_angle_constraint = rel_angle_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.rel_angle_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        row_count = layout.rowCount()
        layout.addWidget(QLabel("Angle Param", self), row_count, 0)
        layout.addWidget(AngleParamButton(self.rel_angle_constraint.param(), self.tree), row_count, 1)

        labels = ["Start", "Vertex", "End"]
        points = [self.rel_angle_constraint.start_point, self.rel_angle_constraint.vertex,
                  self.rel_angle_constraint.end_point]
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


class Perp3ConstraintButton(TreeButton):

    def __init__(self, perp3_constraint: Perp3Constraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=perp3_constraint, tree=tree, top_level=top_level)
        self.perp3_constraint = perp3_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.perp3_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        labels = ["Start", "Vertex", "End"]
        points = [self.perp3_constraint.start_point, self.perp3_constraint.vertex,
                  self.perp3_constraint.end_point]
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


class AntiParallel3ConstraintButton(TreeButton):

    def __init__(self, antiparallel3_constraint: AntiParallel3Constraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=antiparallel3_constraint, tree=tree, top_level=top_level)
        self.antiparallel3_constraint = antiparallel3_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.antiparallel3_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        labels = ["Start", "Vertex", "End"]
        points = [self.antiparallel3_constraint.p1, self.antiparallel3_constraint.p2,
                  self.antiparallel3_constraint.p3]
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


class RelAngle4ConstraintButton(TreeButton):

    def __init__(self, rel_angle_constraint: RelAngle4Constraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=rel_angle_constraint, tree=tree, top_level=top_level)
        self.rel_angle_constraint = rel_angle_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.rel_angle_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        row_count = layout.rowCount()
        layout.addWidget(QLabel("Angle Param", self), row_count, 0)
        layout.addWidget(AngleParamButton(self.rel_angle_constraint.param(), self.tree), row_count, 1)

        labels = ["Tool Start", "Tool End", "Target Start", "Target End"]
        points = [self.rel_angle_constraint.tool().points()[0], self.rel_angle_constraint.tool().points()[1],
                  self.rel_angle_constraint.target().points()[0], self.rel_angle_constraint.target().points()[1]]
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


class PerpendicularConstraintButton(TreeButton):

    def __init__(self, perpendicular_constraint: Perp4Constraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=perpendicular_constraint, tree=tree, top_level=top_level)
        self.perpendicular_constraint = perpendicular_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.perpendicular_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        labels = ["Tool Start", "Tool End", "Target Start", "Target End"]
        points = [self.perpendicular_constraint.tool().points()[0], self.perpendicular_constraint.tool().points()[1],
                  self.perpendicular_constraint.target().points()[0], self.perpendicular_constraint.target().points()[1]]
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


class Parallel4ConstraintButton(TreeButton):

    def __init__(self, parallel_constraint: Parallel4Constraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=parallel_constraint, tree=tree, top_level=top_level)
        self.parallel_constraint = parallel_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.parallel_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        labels = ["Line 1 Start", "Line 1 End", "Line 2 Start", "Line 2 End"]
        points = [self.parallel_constraint.p1, self.parallel_constraint.p2,
                  self.parallel_constraint.p3, self.parallel_constraint.p4]
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


class SymmetryConstraintButton(TreeButton):

    def __init__(self, symmetry_constraint: SymmetryConstraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=symmetry_constraint, tree=tree, top_level=top_level)
        self.symmetry_constraint = symmetry_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.symmetry_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        labels = ["Mirror Start", "Mirror End", "Tool Point", "Target Point"]
        points = [self.symmetry_constraint.p1, self.symmetry_constraint.p2,
                  self.symmetry_constraint.p3, self.symmetry_constraint.p4]
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


class ROCurvatureConstraintButton(TreeButton):

    def __init__(self, curvature_constraint: ROCurvatureConstraint, tree, top_level: bool = False):
        super().__init__(pymead_obj=curvature_constraint, tree=tree, top_level=top_level)
        self.curvature_constraint = curvature_constraint

    def modifyDialogInternals(self, dialog: QDialog, layout: QGridLayout) -> None:
        name_label = QLabel("Name", self)
        name_edit = NameEdit(self, self.curvature_constraint, self.tree)
        name_edit.textChanged.connect(self.onNameChange)
        layout.addWidget(name_label, 1, 0)
        layout.addWidget(name_edit, 1, 1)

        row_count = layout.rowCount()
        layout.addWidget(QLabel("Radius of Curvature", self), row_count, 0)
        layout.addWidget(LengthParamButton(self.curvature_constraint.param(), self.tree), row_count, 1)

        labels = ["Curve 1 G2 Point", "Curve 1 G1 Point", "Curve Joint", "Curve 2 G1 Point", "Curve 2 G2 Point"]
        points = [self.curvature_constraint.g2_point_curve_1, self.curvature_constraint.g1_point_curve_1,
                  self.curvature_constraint.curve_joint,
                  self.curvature_constraint.g1_point_curve_2, self.curvature_constraint.g2_point_curve_2]
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


class PymeadTreeWidgetItem(QTreeWidgetItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hoverable = True


class ParameterTree(QTreeWidget):
    def __init__(self, geo_col: GeometryCollection, parent):
        super().__init__(parent)

        # Exchange references with the geometry collection
        self.geo_col = geo_col
        self.geo_col.tree = self

        # Single column for the tree
        self.setColumnCount(2)

        # Aliases (suitable for display) for the sub-containers
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

        # Set the top-level items (sub_containers)
        self.items = None
        self.topLevelDict = None
        self.addContainers()

        # Make the header
        self.setHeaderLabel("")
        self.headerRow = HeaderButtonRow(self)
        self.headerRow.sigExpandPressed.connect(self.onExpandPressed)
        self.headerRow.sigCollapsePressed.connect(self.onCollapsePressed)
        self.setHeader(self.headerRow)

        # Set the tree widget geometry
        self.setMinimumWidth(300)
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.setColumnWidth(1, 120)

        # Set the tree to be expanded by default
        self.expandAll()

        # Set the selection mode to extended. This allows the user to perform the usual operations of Shift-Click,
        # Ctrl-Click, or drag to select multiple tree items at once
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Item selection changed connection
        self.previous_items = None
        self.itemSelectionChanged.connect(self.onItemSelectionChanged)

        # Allow mouse tracking so we can implement a hover method
        self.setMouseTracking(True)

        # Previous item hovered
        self.previous_item_hovered = None

    def onItemSelectionChanged(self):
        if self.previous_items is not None:
            for item in self.previous_items:
                if item.parent() is not None and item.parent().text(0) == "Points" and item not in self.selectedItems():
                    button = self.itemWidget(item, 1)
                    if button is not None:
                        point = button.point
                        self.geo_col.deselect_object(point)
                elif item.parent() is not None and item.parent().text(0) == "Airfoils" and item not in self.selectedItems():
                    button = self.itemWidget(item, 1)
                    if button is not None:
                        airfoil = button.airfoil
                        self.geo_col.deselect_object(airfoil)

        for item in self.selectedItems():
            if item.parent() is not None and item.parent().text(0) == "Points":
                button = self.itemWidget(item, 1)
                if button is not None:
                    point = button.point
                    self.geo_col.select_object(point)
            elif item.parent() is not None and item.parent().text(0) == "Airfoils":
                button = self.itemWidget(item, 1)
                if button is not None:
                    airfoil = button.airfoil
                    self.geo_col.select_object(airfoil)

        self.previous_items = self.selectedItems()

    def addContainers(self):
        self.items = [PymeadTreeWidgetItem(
            None, [f"{self.container_titles[k]}"]) for k in self.geo_col.container().keys()]
        self.topLevelDict = {k: i for i, k in enumerate(self.geo_col.container().keys())}
        self.insertTopLevelItems(0, self.items)

        # Sort the items in ascending order (A to Z)
        self.sortItems(0, Qt.SortOrder.AscendingOrder)

    def mouseMoveEvent(self, event):
        """
        Since the QTreeWidget does not emit a Hover signal, we effectively create one here by tracking the position
        of the mouse when it is inside the Parameter Tree and check whether there is a PymeadTreeWidgetItem under
        the mouse.
        """
        # Tracks the tree widget for a hover event, since a hover signal is not implemented in QTreeWidget
        tree_item = self.itemAt(event.x(), event.y())

        # Hover leave
        if (self.previous_item_hovered is not None and tree_item is not self.previous_item_hovered and
                self.previous_item_hovered.hoverable):
            button = self.itemWidget(self.previous_item_hovered, 1)
            if button is not None:
                self.geo_col.hover_leave_obj(button.pymead_obj)
            else:
                self.setItemStyle(self.previous_item_hovered, "default")

        if not isinstance(tree_item, PymeadTreeWidgetItem):
            self.previous_item_hovered = tree_item
            return

        if tree_item.hoverable:
            # Hover enter
            if tree_item is not None:
                right_column_widget = self.itemWidget(tree_item, 1)
                if right_column_widget is not None:
                    self.geo_col.hover_enter_obj(right_column_widget.pymead_obj)
                else:
                    self.setItemStyle(tree_item, "hovered")

        # Assign the current tree widget item to the previous item hovered
        self.previous_item_hovered = tree_item

    def leaveEvent(self, a0):
        """
        Reimplement the leave event to handle the case where the mouse exits directly sideways through the "Edit"
        button. In this case, the mouseMoveEvent will not catch the hover leave, so we need to put that logic here.
        """
        if self.previous_item_hovered is not None and self.previous_item_hovered.hoverable:
            button = self.itemWidget(self.previous_item_hovered, 1)
            if button is not None:
                self.geo_col.hover_leave_obj(button.pymead_obj)
            else:
                self.setItemStyle(self.previous_item_hovered, "default")
            self.previous_item_hovered = None

    def addPymeadTreeItem(self, pymead_obj: PymeadObj):
        top_level_item = self.items[self.topLevelDict[pymead_obj.sub_container]]
        child_item = PymeadTreeWidgetItem([pymead_obj.name()])
        top_level_item.addChild(child_item)
        pymead_obj.tree_item = child_item

        if isinstance(pymead_obj, Param):
            right_column_widget = ValueSpin(self, pymead_obj)
        else:
            button_args = (pymead_obj, self)
            right_column_widget = getattr(sys.modules[__name__],
                                          f"{type(pymead_obj).__name__}Button")(*button_args, top_level=True)

        self.setItemWidget(child_item, 1, right_column_widget)

    def removePymeadTreeItem(self, pymead_obj: PymeadObj):
        top_level_item = self.items[self.topLevelDict[pymead_obj.sub_container]]
        top_level_item.removeChild(pymead_obj.tree_item)
        pymead_obj.tree_item = None

    def onExpandPressed(self):
        self.expandAll()

    def onCollapsePressed(self):
        self.collapseAll()

    def setItemStyle(self, item: PymeadTreeWidgetItem, style: str):
        valid_styles = ["default", "hovered"]
        if style not in ["default", "hovered"]:
            raise ValueError(f"Style found ({style}) is not a valid style. Must be one of {valid_styles}.")

        background_color = self.palette().color(self.backgroundRole())
        if style == "default":
            # item.setBackground(0, background_color)
            # item.setBackground(1, background_color)
            brush = QBrush(QColor(self.parent().parent().themes[self.parent().parent().current_theme]['main-color']))
            item.setForeground(0, brush)
        elif style == "hovered" and item.hoverable:
            # gradient = QtGui.QLinearGradient(0, 0, 150, 0)
            # gradient.setColorAt(0, QColor("#2678c9aa"))
            # gradient.setColorAt(1, self.palette().color(self.backgroundRole()))
            # item.setBackground(0, gradient)
            # item.setBackground(1, self.palette().color(self.backgroundRole()))
            brush = QBrush(QColor("#edb126"))
            item.setForeground(0, brush)

    def setForegroundColorAllItems(self, color: str):
        it = QTreeWidgetItemIterator(self)
        while it.value():
            it.value().setForeground(0, QBrush(QColor(color)))
            it += 1

    def contextMenuEvent(self, a0):
        # item = self.itemAt(a0.x(), a0.y())
        # if item is None:
        #     return

        items = self.selectedItems()
        if len(items) == 0:
            return

        if len(items) == 1 and items[0].text(0) == "Design Variables":
            menu = QMenu(self)
            addDesVarAction = menu.addAction("Add Design Variable")
            res = menu.exec_(a0.globalPos())

            if res is None:
                return

            if res is addDesVarAction:
                self.geo_col.add_desvar(0.0, "dv")

        elif len(items) == 1 and items[0].text(0) == "Parameters":
            menu = QMenu(self)
            addParameterAction = menu.addAction("Add Parameter")
            res = menu.exec_(a0.globalPos())

            if res is None:
                return

            if res is addParameterAction:
                self.geo_col.add_param(0.0, "param")

        elif all([item.parent() is not None for item in items]) and all(
                [item.parent() is items[0].parent() for item in items]):
            # button = self.itemWidget(item, 1)
            pymead_objs = [self.itemWidget(item, 1).pymead_obj for item in items]

            promoteAction = None
            demoteAction = None

            pymead_obj_type = type(pymead_objs[0])

            if pymead_obj_type in [Param, LengthParam, AngleParam]:
                menu = QMenu(self)
                promoteAction = menu.addAction("Promote to Design Variable")
                removeObjectAction = menu.addAction("Delete")
            elif pymead_obj_type in [DesVar, LengthDesVar, AngleDesVar]:
                menu = QMenu(self)
                demoteAction = menu.addAction("Demote to Parameter")
                removeObjectAction = menu.addAction("Delete")
            else:
                menu = QMenu(self)
                removeObjectAction = menu.addAction("Delete")

            res = menu.exec_(a0.globalPos())

            if res is None:
                return

            if res is removeObjectAction:
                for pymead_obj in pymead_objs:
                    # If the object is a Point, set the style to default first so that the text item gets removed
                    # if isinstance(pymead_obj, Point):
                    #     self.geo_col.canvas.setItemStyle(pymead_obj.canvas_item, "default")
                    self.geo_col.remove_pymead_obj(pymead_obj)
            elif res is promoteAction:
                for pymead_obj in pymead_objs:
                    self.geo_col.promote_param_to_desvar(pymead_obj)
            elif res is demoteAction:
                for pymead_obj in pymead_objs:
                    self.geo_col.demote_desvar_to_param(pymead_obj)

            self.geo_col.clear_selected_objects()
