import sys
import typing

import networkx
import numpy as np

from pymead.core.param import LengthParam
from pymead.core.pymead_obj import PymeadObj


class Point(PymeadObj):
    def __init__(self, x: float, y: float, name: str or None = None, setting_from_geo_col: bool = False,
                 fixed: bool = False):
        super().__init__(sub_container="points")
        self._x = None
        self._y = None
        self._fixed = fixed
        self._fixed_weak = False
        self.gcs = None
        self.root = False
        self.geo_cons = []
        self.dims = []
        self.curves = []
        self.setting_from_geo_col = setting_from_geo_col
        name = "Point-1" if name is None else name
        self.set_name(name)
        self.set_x(x)
        self.set_y(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def fixed(self):
        return self._fixed

    def fixed_weak(self):
        return self._fixed_weak

    def set_x(self, x: LengthParam or float):
        self._x = x if isinstance(x, LengthParam) else LengthParam(
            value=x, name=self.name() + ".x", setting_from_geo_col=self.setting_from_geo_col, point=self)
        if self not in self._x.geo_objs:
            self._x.geo_objs.append(self)
        self._x.point = self

    def set_y(self, y: LengthParam or float):
        self._y = y if isinstance(y, LengthParam) else LengthParam(
            value=y, name=self.name() + ".y", setting_from_geo_col=self.setting_from_geo_col, point=self)
        if self not in self._y.geo_objs:
            self._y.geo_objs.append(self)
        self._y.point = self

    def set_fixed(self, fixed: bool):
        self._fixed = fixed

    def set_fixed_weak(self, fixed_weak: bool):
        self._fixed_weak = fixed_weak

    def set_name(self, name: str):
        # Rename the x and y parameters of the Point
        if self.x() is not None:
            self.x().set_name(f"{name}.x")
        if self.y() is not None:
            self.y().set_name(f"{name}.y")

        super().set_name(name)

    def as_array(self):
        return np.array([self.x().value(), self.y().value()])

    @classmethod
    def generate_from_array(cls, arr: np.ndarray, name: str or None = None):
        if arr.ndim != 1:
            raise ValueError("Points can only be generated from 1-dimensional arrays")
        return cls(x=arr[0], y=arr[1], name=name)

    def measure_distance(self, other: "Point"):
        # Note: the quotes around "Point" are necessary here because this type hint is a forward reference.
        # This means that an object of type <Class> is specified as a hint somewhere inside the definition for <Class>
        return np.hypot(other.x().value() - self.x().value(), other.y().value() - self.y().value())

    def measure_angle(self, other: "Point"):
        return np.arctan2(other.y().value() - self.y().value(), other.x().value() - self.x().value())

    def request_move(self, xp: float, yp: float, force: bool = False):

        if (self.gcs is None or (self.gcs is not None and len(self.geo_cons) == 0) or force or
                (self.gcs is not None and self.root)):

            if self.root:
                points_to_update = self.gcs.move_root(self, dx=xp-self.x().value(), dy=yp-self.y().value())
                constraints_to_update = []
                for point in networkx.dfs_preorder_nodes(self.gcs, source=self):
                    for geo_con in point.geo_cons:
                        if geo_con not in constraints_to_update:
                            constraints_to_update.append(geo_con)

                for geo_con in constraints_to_update:
                    if geo_con.canvas_item is not None:
                        geo_con.canvas_item.update()
            else:
                self.x().set_value(xp)
                self.y().set_value(yp)
                points_to_update = [self]

            # Update the GUI object, if there is one
            if self.canvas_item is not None:
                self.canvas_item.updateCanvasItem(self.x().value(), self.y().value())

            curves_to_update = []
            for point in points_to_update:
                if point.canvas_item is not None:
                    point.canvas_item.updateCanvasItem(point.x().value(), point.y().value())

                for curve in point.curves:
                    if curve not in curves_to_update:
                        curves_to_update.append(curve)

            airfoils_to_update = []
            for curve in curves_to_update:
                if curve.airfoil is not None and curve.airfoil not in airfoils_to_update:
                    airfoils_to_update.append(curve.airfoil)
                curve.update()

            for airfoil in airfoils_to_update:
                airfoil.update_coords()
                if airfoil.canvas_item is not None:
                    airfoil.canvas_item.generatePicture()

    def __repr__(self):
        return f"Point {self.name()}<x={self.x().value():.6f}, y={self.y().value():.6f}>"

    def get_dict_rep(self):
        return {"x": float(self.x().value()), "y": float(self.y().value()), "fixed": self.fixed()}


class PointSequence:
    def __init__(self, points: typing.List[Point]):
        self._points = None
        self.set_points(points)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.generate_from_slice(self, idx)
        else:
            return self.points()[idx]

    def __setitem__(self, idx, val):
        self.points()[idx] = val

    def __len__(self):
        return len(self.points())

    @classmethod
    def generate_from_slice(cls, original_point_seq, s):
        return cls(points=original_point_seq.points()[s].copy())

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.shape[1] != 2:
            raise ValueError(f"Array must have two columns, x and y. Found {arr.shape[1]} columns.")
        return cls(points=[Point(x=x, y=y, name=f"PointFromArray-Index{idx}")
                           for idx, (x, y) in enumerate(zip(arr[:, 0], arr[:, 1]))])

    def points(self):
        return self._points

    def point_idx_from_ref(self, point: Point):
        return self.points().index(point)

    def reverse(self):
        self.points().reverse()

    def set_points(self, points: typing.List[Point]):
        self._points = points

    def insert_point(self, idx: int, point: Point):
        self._points.insert(idx, point)

    def append_point(self, point: Point):
        self._points.append(point)

    def remove_point(self, idx: int):
        self._points.pop(idx)

    def as_array(self):
        return np.array([[p.x().value(), p.y().value()] for p in self.points()])

    def extract_subsequence(self, indices: list):
        return PointSequence(points=[self.points()[idx] for idx in indices])
