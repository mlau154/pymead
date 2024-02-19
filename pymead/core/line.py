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

    AirfoilTools = 0
    DatFile = 1

    def __init__(self, source: str, start: int or float = None, end: int or float = None,
                 point_sequence: PointSequence = None, name: str or None = None, num_header_rows: int = 0,
                 delimiter: str or None = None, **kwargs):

        super().__init__(sub_container="polylines", **kwargs)
        self._point_sequence = None
        self.source = source
        self.source_type = self.DatFile if ("/" in source or "\\" in source) else self.AirfoilTools
        self.start = start
        self.end = end
        self.num_header_rows = num_header_rows
        self.delimiter = delimiter
        self.coords = None
        if self.source_type == self.AirfoilTools:
            self.coords = self._load_coords_from_airfoil_tools()
        elif self.source_type == self.DatFile:
            self.coords = self._load_coords_from_dat_file(self.num_header_rows, self.delimiter)
        if start is None and isinstance(end, int):
            self.coords = self.coords[:end, :]
        elif end is None and isinstance(start, int):
            self.coords = self.coords[start:, :]
        elif isinstance(start, int) and isinstance(end, int):
            self.coords = self.coords[start:end, :]
        point_sequence = self._extract_point_sequence_from_coords() if point_sequence is None else point_sequence
        self.set_point_sequence(point_sequence)
        if name is None:
            if self.source_type == self.AirfoilTools:
                name = f"{self.source}-1"
            else:
                name = "PolyLine-1"
        self.set_name(name)
        self._add_references()

    def split(self, split: int or Point):
        if split < 3 or split > len(self.coords) - 3:
            raise ValueError("Found split value out of bounds for PolyLine")
        polyline1, polyline2 = None, None
        if isinstance(split, int):
            end_1 = self.start + split + 1 if self.start is not None else split + 1
            start_2 = self.start + split if self.start is not None else split
            polyline1 = PolyLine(source=self.source, start=self.start, end=end_1)
            polyline2 = PolyLine(source=self.source, start=start_2, end=self.end)
            polyline2.point_sequence().points()[0] = polyline1.point_sequence().points()[-1]
            polyline2.point_sequence().points()[0].curves.append(polyline2)
        return [polyline1, polyline2]

    def _load_coords_from_dat_file(self, num_header_rows: int = 0, delimiter: str or None = None):
        return np.loadtxt(self.source, skiprows=num_header_rows, delimiter=delimiter)

    def _load_coords_from_airfoil_tools(self):
        return extract_data_from_airfoiltools(self.source)

    def _extract_point_sequence_from_coords(self):
        return PointSequence(points=[Point(self.coords[row, 0], self.coords[row, 1]) for row in [0, -1]])

    def convert_airfoil_tools_airfoil_to_sequence_and_coords(self):
        points = [Point(self.coords[row, 0], self.coords[row, 1]) for row in [0, 1, -2, -1]]
        return PointSequence(points=points)

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
        with np.errstate(divide="ignore"):
            k = np.true_divide(xpyp[:, 0] * xppypp[:, 1] - xpyp[:, 1] * xppypp[:, 0],
                               np.hypot(xpyp[:, 0], xpyp[:, 1])**1.5)
            R = np.true_divide(1, k)
        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()], "source": self.source,
                "start": self.start, "end": self.end}
