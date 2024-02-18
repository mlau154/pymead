import typing

import numpy as np

from pymead.core.point import PointSequence, Point
from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.utils.get_airfoil import extract_data_from_airfoiltools


class LineSegment(ParametricCurve):

    def __init__(self, point_sequence: PointSequence, name: str or None = None, **kwargs):
        super().__init__(sub_container="lines", **kwargs)
        self._point_sequence = None
        self.set_point_sequence(point_sequence)
        name = "Line-1" if name is None else name
        self.set_name(name)
        self._add_references()

    def _add_references(self):
        for idx, point in enumerate(self.point_sequence().points()):
            # Add the object reference to each point in the curve
            if self not in point.curves:
                point.curves.append(self)

    def point_sequence(self):
        return self._point_sequence

    def set_point_sequence(self, point_sequence: PointSequence):
        if len(point_sequence) != 2:
            raise ValueError("Point sequence must contain exactly two points")
        self._point_sequence = point_sequence

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def point_removal_deletes_curve(self):
        return True

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)

        if len(self.point_sequence()) > 1:
            delete_curve = False
        else:
            delete_curve = True

        return delete_curve

    def remove(self):
        if self.canvas_item is not None:
            self.canvas_item.sigRemove.emit(self.canvas_item)

    def update(self):
        p_curve_data = self.evaluate()
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(curve_data=p_curve_data)

    def evaluate(self, t: np.ndarray or None = None, **kwargs):
        if "nt" not in kwargs.keys() and t is None:
            kwargs["nt"] = 2  # Set the default parameter vector for the line to be [0.0, 1.0]
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t
        p1 = self.point_sequence().points()[0]
        p2 = self.point_sequence().points()[1]
        x1 = p1.x().value()
        y1 = p1.y().value()
        x2 = p2.x().value()
        y2 = p2.y().value()
        theta = np.arctan2(y2 - y1, x2 - x1)
        r = np.hypot(x2 - x1, y2 - y1)
        x = x1 + t * r * np.cos(theta)
        y = y1 + t * r * np.sin(theta)
        xy = np.column_stack((x, y))
        xpyp = np.repeat(np.array([r * np.cos(theta), r * np.sin(theta)]), t.shape[0])
        xppypp = np.repeat(np.array([0.0, 0.0]), t.shape[0])
        k = np.zeros(t.shape)
        R = np.inf * np.ones(t.shape)
        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()]}


class PolyLine(ParametricCurve):

    def __init__(self, point_sequence: PointSequence = None, te: Point = None, web_airfoil_name: str = None,
                 breaks: typing.List[list] = None, name: str or None = None, **kwargs):

        if point_sequence is None and web_airfoil_name is None:
            raise ValueError("Must specify at least one of either point_sequence or web_airfoil_name")

        super().__init__(sub_container="polylines", **kwargs)
        self._point_sequence = None
        self.coords = None
        self.te = te if te is not None else None
        self.web_airfoil_name = web_airfoil_name
        if web_airfoil_name is not None:
            if point_sequence is None:
                point_sequence, self.coords = self.convert_airfoil_tools_airfoil_to_sequence_and_coords(
                    web_airfoil_name)
            else:
                _, self.coords = self.convert_airfoil_tools_airfoil_to_sequence_and_coords(web_airfoil_name)
        self.set_point_sequence(point_sequence)
        if name is None:
            if web_airfoil_name:
                name = f"{web_airfoil_name}-1"
            else:
                name = "PolyLine-1"
        self.set_name(name)
        self.breaks = breaks
        self._add_references()

    def convert_airfoil_tools_airfoil_to_sequence_and_coords(self, point_sequence: str):
        # TODO: add breaks functionality here
        coords = extract_data_from_airfoiltools(point_sequence)
        coords_dist_from_origin = np.hypot(coords[:, 0], coords[:, 1])
        if coords[0, 1] >= 0.0 >= coords[-1, 1]:
            # For the usual case where the trailing edge upper point is at or above 0 and the trailing edge lower
            # point is at or below 0, just use (1, 0) as the trailing edge
            self.te = Point(1.0, 0.0)
        else:
            # Otherwise, use the mean of the upper and lower points as the trailing edge point
            self.te = Point(0.5 * (coords[0, 0] + coords[-1, 0]), 0.5 * (coords[0, 1] + coords[-1, 1]))
        le_row = np.argmin(coords_dist_from_origin)
        if np.hypot(coords[0, 0] - coords[-1, 0], coords[0, 1] - coords[-1, 1]) < 1e-6:  # sharp trailing edge
            points = [self.te] + [Point(coords[row, 0], coords[row, 1]) for row in [1, le_row, -2]] + [self.te]
        else:
            points = [Point(coords[row, 0], coords[row, 1]) for row in [0, 1, le_row, -2, -1]]
        point_sequence = PointSequence(points=points)
        return point_sequence, coords

    def _add_references(self):
        for idx, point in enumerate(self.point_sequence().points()):
            # Add the object reference to each point in the curve
            if self not in point.curves:
                point.curves.append(self)

    def point_sequence(self):
        return self._point_sequence

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def point_removal_deletes_curve(self):
        return True

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)

        if len(self.point_sequence()) > 1:
            delete_curve = False
        else:
            delete_curve = True

        return delete_curve

    def remove(self):
        if self.canvas_item is not None:
            self.canvas_item.sigRemove.emit(self.canvas_item)

    def update(self):
        p_curve_data = self.evaluate()
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(curve_data=p_curve_data)

    def evaluate(self, t: np.ndarray or None = None, **kwargs):
        xy = self.coords
        t = np.linspace(0.0, 1.0, xy.shape[0])
        xp = np.gradient(xy[:, 0], t)
        yp = np.gradient(xy[:, 1], t)
        xpyp = np.column_stack((xp, yp))
        xpp = np.gradient(xp, t)
        ypp = np.gradient(yp, t)
        xppypp = np.column_stack((xpp, ypp))
        k = np.true_divide(xpyp[:, 0] * xppypp[:, 1] - xpyp[:, 1] * xppypp[:, 0], np.hypot(xpyp[:, 0], xpyp[:, 1])**1.5)
        R = np.true_divide(1, k)
        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()], "te": self.te.name(),
                "web_airfoil_name": self.web_airfoil_name, "breaks": self.breaks}
