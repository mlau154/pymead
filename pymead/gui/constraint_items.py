from abc import abstractmethod

import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPen, QColor

from pymead.core.constraints import *
from pymead.utils.misc import get_setting


class ConstraintItem(QObject):

    sigItemHovered = pyqtSignal()
    sigItemLeaveHovered = pyqtSignal()

    def __init__(self, constraint: GeoCon, canvas_items: list, theme: dict):
        super().__init__()
        self.constraint = constraint
        self.constraint.canvas_item = self
        self.canvas_items = canvas_items
        self.setStyle(theme=theme, mode="default")

    def addItems(self, canvas):
        for item in self.canvas_items:
            canvas.addItem(item)
            # TODO: connect hover leave/enter signals here

    def hide(self):
        for canvas_item in self.canvas_items:
            canvas_item.hide()

    def show(self):
        for canvas_item in self.canvas_items:
            canvas_item.show()

    @abstractmethod
    def update(self):
        pass

    def setStyle(self, theme: dict = None, mode: str = "default"):
        if mode == "default" and theme is None:
            raise ValueError("Must specify theme for default constraint style")
        for item in self.canvas_items:
            if isinstance(item, pg.ArrowItem):
                if mode == "default":
                    item.setStyle(brush=theme["main-color"])
                else:
                    item.setStyle(brush=get_setting(f"curve_{mode}_pen_color"))
            elif isinstance(item, pg.TextItem):
                if mode == "default":
                    item.setColor(theme["main-color"])
                else:
                    item.setColor(get_setting(f"curve_{mode}_pen_color"))
            elif isinstance(item, pg.PlotDataItem) or isinstance(item, pg.PlotCurveItem):
                pen = item.opts["pen"]
                if isinstance(pen, QPen):
                    if mode == "default":
                        pen.setColor(QColor(theme["main-color"]))
                    else:
                        pen.setColor(QColor(get_setting(f"curve_{mode}_pen_color")))
                item.setPen(pen)


class DistanceConstraintItem(ConstraintItem):
    def __init__(self, constraint: DistanceConstraint, theme: dict):
        self.arrow_style = {"headLen": 10}
        self.text_style = {"anchor": (0.5, 0.5)}
        canvas_items = [
            pg.ArrowItem(**self.arrow_style),
            pg.ArrowItem(**self.arrow_style),
            pg.TextItem(**self.text_style),
            pg.PlotCurveItem(),
            pg.PlotCurveItem(),
            pg.PlotCurveItem(),
        ]
        super().__init__(constraint=constraint, canvas_items=canvas_items, theme=theme)
        self.canvas_items[2].setFont(QFont("DejaVu Sans Mono", 10))
        for item in canvas_items:
            if isinstance(item, pg.ArrowItem):
                item.setZValue(-8)
            else:
                item.setZValue(-10)
        self.update()

    def update(self):
        angle = self.constraint.p1.measure_angle(self.constraint.p2)
        dist = self.constraint.p1.measure_distance(self.constraint.p2)

        handle_angle = angle + np.pi / 2
        arrow_offset = 0.04
        p1_arrow = (self.constraint.p1.x().value() + arrow_offset * np.cos(handle_angle),
                    self.constraint.p1.y().value() + arrow_offset * np.sin(handle_angle))
        p2_arrow = (self.constraint.p2.x().value() + arrow_offset * np.cos(handle_angle),
                    self.constraint.p2.y().value() + arrow_offset * np.sin(handle_angle))

        text_offset = 0.06
        text_angle = np.rad2deg((angle + np.pi / 2) % np.pi - np.pi / 2)
        text_pos = (self.constraint.p1.x().value() + text_offset * np.cos(handle_angle) + 0.5 * dist * np.cos(angle),
                    self.constraint.p1.y().value() + text_offset * np.sin(handle_angle) + 0.5 * dist * np.sin(angle))

        line_x = [self.constraint.p1.x().value() + arrow_offset * np.cos(handle_angle),
                  self.constraint.p1.x().value() + arrow_offset * np.cos(handle_angle) + dist * np.cos(angle)]
        line_y = [self.constraint.p1.y().value() + arrow_offset * np.sin(handle_angle),
                  self.constraint.p1.y().value() + arrow_offset * np.sin(handle_angle) + dist * np.sin(angle)]

        handle_offset = 0.05
        handle1_x = [self.constraint.p1.x().value(),
                     self.constraint.p1.x().value() + handle_offset * np.cos(handle_angle)]
        handle1_y = [self.constraint.p1.y().value(),
                     self.constraint.p1.y().value() + handle_offset * np.sin(handle_angle)]

        handle2_x = [self.constraint.p2.x().value(),
                     self.constraint.p2.x().value() + handle_offset * np.cos(handle_angle)]
        handle2_y = [self.constraint.p2.y().value(),
                     self.constraint.p2.y().value() + handle_offset * np.sin(handle_angle)]

        self.canvas_items[0].setPos(*p1_arrow)
        self.canvas_items[0].setStyle(angle=-np.rad2deg(angle))
        self.canvas_items[1].setPos(*p2_arrow)
        self.canvas_items[1].setStyle(angle=-np.rad2deg(angle) + 180)
        self.canvas_items[2].setPos(*text_pos)
        self.canvas_items[2].setText(f"{self.constraint.param().value():.4f}")
        self.canvas_items[2].setAngle(text_angle)
        self.canvas_items[3].setData(x=line_x, y=line_y)
        self.canvas_items[4].setData(x=handle1_x, y=handle1_y)
        self.canvas_items[5].setData(x=handle2_x, y=handle2_y)


class SymmetryConstraintItem(ConstraintItem):
    def __init__(self, constraint: SymmetryConstraint, theme: dict):
        canvas_items = [
            pg.TextItem("\u24c2"),
            pg.TextItem("\u24c2")
        ]
        canvas_items[0].setFont(QFont("DejaVu Sans", 10))
        canvas_items[1].setFont(QFont("DejaVu Sans", 10))
        for item in canvas_items:
            item.setZValue(-10)
        super().__init__(constraint=constraint, canvas_items=canvas_items, theme=theme)
        self.update()

    def update(self):
        self.canvas_items[0].setPos(self.constraint.p3.x().value() + 0.01, self.constraint.p3.y().value() - 0.01)
        self.canvas_items[1].setPos(self.constraint.p4.x().value() + 0.01, self.constraint.p4.y().value() - 0.01)


class ROCurvatureConstraintItem(ConstraintItem):
    def __init__(self, constraint: ROCurvatureConstraint, theme: dict):
        canvas_items = [
            pg.TextItem("\u24c7"),
        ]
        canvas_items[0].setFont(QFont("DejaVu Sans", 10))
        for item in canvas_items:
            item.setZValue(-10)
        super().__init__(constraint=constraint, canvas_items=canvas_items, theme=theme)
        self.update()

    def update(self):
        self.canvas_items[0].setPos(self.constraint.curve_joint.x().value() - 0.01,
                                    self.constraint.curve_joint.y().value() - 0.01)


class AntiParallel3ConstraintItem(ConstraintItem):
    def __init__(self, constraint: AntiParallel3Constraint, theme: dict):
        self.text_style = dict(anchor=(0.5, 0.5))
        pen = pg.mkPen(width=1, style=Qt.DashLine)
        canvas_items = [
            pg.TextItem("\u2225", **self.text_style),
            pg.TextItem("\u2225", **self.text_style),
            pg.PlotDataItem(pen=pen),
            pg.PlotDataItem(pen=pen)
        ]
        canvas_items[0].setFont(QFont("DejaVu Sans", 10))
        canvas_items[1].setFont(QFont("DejaVu Sans", 10))
        for item in canvas_items:
            item.setZValue(-10)
        super().__init__(constraint=constraint, canvas_items=canvas_items, theme=theme)
        self.update()

    def update(self):
        midpoint1 = [np.mean([self.constraint.p1.x().value(), self.constraint.p2.x().value()]),
                     np.mean([self.constraint.p1.y().value(), self.constraint.p2.y().value()])]
        midpoint2 = [np.mean([self.constraint.p3.x().value(), self.constraint.p2.x().value()]),
                     np.mean([self.constraint.p3.y().value(), self.constraint.p2.y().value()])]
        line1_x = [self.constraint.p2.x().value(), self.constraint.p1.x().value()]
        line1_y = [self.constraint.p2.y().value(), self.constraint.p1.y().value()]

        line2_x = [self.constraint.p2.x().value(), self.constraint.p3.x().value()]
        line2_y = [self.constraint.p2.y().value(), self.constraint.p3.y().value()]
        self.canvas_items[0].setPos(*midpoint1)
        self.canvas_items[1].setPos(*midpoint2)
        self.canvas_items[2].setData(x=line1_x, y=line1_y)
        self.canvas_items[3].setData(x=line2_x, y=line2_y)


class RelAngle3ConstraintItem(ConstraintItem):
    def __init__(self, constraint: RelAngle3Constraint, theme: dict):
        pen = pg.mkPen(width=1, style=Qt.DashLine)
        canvas_items = [
            pg.PlotDataItem(),
            pg.PlotDataItem(pen=pen),
            pg.PlotDataItem(pen=pen),
            pg.TextItem(anchor=(0, 0.5)),
        ]
        canvas_items[3].setFont(QFont("DejaVu Sans Mono", 10))
        super().__init__(constraint=constraint, canvas_items=canvas_items, theme=theme)
        for item in canvas_items:
            item.setZValue(-10)
        self.update()

    def update(self):
        dist1 = self.constraint.p2.measure_distance(self.constraint.p1)
        dist2 = self.constraint.p2.measure_distance(self.constraint.p3)
        angle1 = self.constraint.p2.measure_angle(self.constraint.p1)
        angle2 = angle1 - self.constraint.param().rad()
        mean_angle = np.mean([angle1, angle2])
        text_distance = 0.15 * np.mean([dist1, dist2])
        text_x = self.constraint.p2.x().value() + text_distance * np.cos(mean_angle)
        text_y = self.constraint.p2.y().value() + text_distance * np.sin(mean_angle)

        theta = np.linspace(angle1, angle2, 30)
        x = self.constraint.p2.x().value() + np.mean([dist1, dist2]) * 0.1 * np.cos(theta)
        y = self.constraint.p2.y().value() + np.mean([dist1, dist2]) * 0.1 * np.sin(theta)

        line1_x = [self.constraint.p2.x().value(), self.constraint.p1.x().value()]
        line1_y = [self.constraint.p2.y().value(), self.constraint.p1.y().value()]

        line2_x = [self.constraint.p2.x().value(), self.constraint.p3.x().value()]
        line2_y = [self.constraint.p2.y().value(), self.constraint.p3.y().value()]

        self.canvas_items[0].setData(x=x, y=y)
        self.canvas_items[1].setData(x=line1_x, y=line1_y)
        self.canvas_items[2].setData(x=line2_x, y=line2_y)
        self.canvas_items[3].setPos(text_x, text_y)
        self.canvas_items[3].setText(f"{np.rad2deg(self.constraint.param().rad()):.2f}\u00b0")


class Perp3ConstraintItem(ConstraintItem):
    def __init__(self, constraint: Perp3Constraint, theme: dict):
        pen = pg.mkPen(width=1, style=Qt.DashLine)
        canvas_items = [
            pg.PlotDataItem(),
            pg.PlotDataItem(),
            pg.PlotDataItem(pen=pen),
            pg.PlotDataItem(pen=pen)
        ]
        super().__init__(constraint=constraint, canvas_items=canvas_items, theme=theme)
        for item in canvas_items:
            item.setZValue(-10)
        self.update()

    def update(self):
        angle1 = self.constraint.p2.measure_angle(self.constraint.p1)
        angle2 = angle1 - np.pi / 2

        square_side = 0.05

        x1 = [self.constraint.p2.x().value() + square_side * np.cos(angle1),
              self.constraint.p2.x().value() + square_side * np.cos(angle1) + square_side * np.cos(angle2)]
        y1 = [self.constraint.p2.y().value() + square_side * np.sin(angle1),
              self.constraint.p2.y().value() + square_side * np.sin(angle1) + square_side * np.sin(angle2)]
        x2 = [self.constraint.p2.x().value() + square_side * np.cos(angle2),
              self.constraint.p2.x().value() + square_side * np.cos(angle2) + square_side * np.cos(angle1)]
        y2 = [self.constraint.p2.y().value() + square_side * np.sin(angle2),
              self.constraint.p2.y().value() + square_side * np.sin(angle2) + square_side * np.sin(angle1)]

        line1_x = [self.constraint.p2.x().value(), self.constraint.p1.x().value()]
        line1_y = [self.constraint.p2.y().value(), self.constraint.p1.y().value()]

        line2_x = [self.constraint.p2.x().value(), self.constraint.p3.x().value()]
        line2_y = [self.constraint.p2.y().value(), self.constraint.p3.y().value()]

        self.canvas_items[0].setData(x=x1, y=y1)
        self.canvas_items[1].setData(x=x2, y=y2)
        self.canvas_items[2].setData(x=line1_x, y=line1_y)
        self.canvas_items[3].setData(x=line2_x, y=line2_y)
