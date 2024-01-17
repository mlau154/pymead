import os

import pyqtgraph as pg
import numpy as np
from PyQt5.QtGui import QFont, QImage
import PIL

from pymead.core.constraints import *
from pymead import ICON_DIR


class ConstraintItem:

    def __init__(self, constraint: GeoCon, canvas_items: list):
        self.constraint = constraint
        self.constraint.canvas_item = self
        self.canvas_items = canvas_items

    def addItems(self, canvas):
        for item in self.canvas_items:
            canvas.addItem(item)

    @abstractmethod
    def update(self):
        pass


class DistanceConstraintItem(ConstraintItem):
    def __init__(self, constraint: DistanceConstraint):
        self.arrow_style = {"brush": (150, 150, 150), "headLen": 10}
        self.text_style = {"color": (255, 255, 255), "anchor": (0.5, 0.5)}
        canvas_items = [
            pg.ArrowItem(**self.arrow_style),
            pg.ArrowItem(**self.arrow_style),
            pg.TextItem(**self.text_style),
            pg.PlotCurveItem(),
            pg.PlotCurveItem(),
            pg.PlotCurveItem(),
        ]
        super().__init__(constraint=constraint, canvas_items=canvas_items)
        self.canvas_items[2].setFont(QFont("DejaVu Sans Mono", 10))
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

        text_offset = 0.05
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
    def __init__(self, constraint: SymmetryConstraint):
        image = np.array(PIL.Image.open(os.path.join(ICON_DIR, "symmetry_constraint.drawio.png")))
        canvas_items = [
            pg.ImageItem(image=image)
        ]
        super().__init__(constraint=constraint, canvas_items=canvas_items)
        self.update()

    def update(self):
        self.canvas_items[0].setPos(self.constraint.p1.x().value() + 0.01, self.constraint.p1.y().value() - 0.01)


class RelAngle3ConstraintItem(ConstraintItem):
    def __init__(self, constraint: RelAngle3Constraint):
        image = np.array(PIL.Image.open(os.path.join(ICON_DIR, "symmetry_constraint.drawio.png")))
        canvas_items = [
            pg.ImageItem(image=image)
        ]
        super().__init__(constraint=constraint, canvas_items=canvas_items)
        self.update()

    def update(self):
        self.canvas_items[0].setPos(self.constraint.vertex.x().value() + 0.01, self.constraint.vertex.y().value() - 0.01)
