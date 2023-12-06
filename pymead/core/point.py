import sys
import typing

import numpy as np

from pymead.core.param2 import LengthParam
from pymead.core.pymead_obj import PymeadObj


class Point(PymeadObj):
    def __init__(self, x: float, y: float, name: str or None = None, setting_from_geo_col: bool = False):
        super().__init__(sub_container="points")
        self._x = None
        self._y = None
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

    def request_move(self, xp: float, yp: float, requestor: PymeadObj or None = None):
        # Initialize variables which may or may not be used
        initial_x = None  # x-location of the current point before the movement
        initial_y = None  # y-location of the current point before the movement
        initial_psi1 = None  # Curvature control arm 1 angle before the current point movement
        initial_psi2 = None  # Curvature control arm 2 angle before the current point movement
        initial_R = None  # Radius of curvature before the current point movement

        # Get the concrete class names of all the constraints associated with this point so that we calculate the
        # minimum required amount of information
        geo_con_classes = [str(geo_con.__class__.__name__) for geo_con in self.geo_cons]
        if "CollinearConstraint" in geo_con_classes or "CurvatureConstraint" in geo_con_classes:
            initial_x = self.x().value()
            initial_y = self.y().value()
        if "CurvatureConstraint" in geo_con_classes:
            for idx, geo_con in enumerate(self.geo_cons):
                if geo_con_classes[idx] == "CurvatureConstraint":
                    data = geo_con.calculate_curvature_data()
                    initial_psi1 = data.psi1
                    initial_psi2 = data.psi2
                    initial_R = data.R1
                    break

        self.x().set_value(xp)
        self.y().set_value(yp)
        for geo_con in self.geo_cons:
            kwargs = {}

            # Get class by name to avoid circular import
            class_name = str(geo_con.__class__)
            if "PositionConstraint" in class_name:
                kwargs = dict(calling_point=self)
            elif "CollinearConstraint" in class_name:
                kwargs = dict(calling_point=self, initial_x=initial_x, initial_y=initial_y)
            elif "CurvatureConstraint" in class_name:
                kwargs = dict(calling_point=self, initial_x=initial_x, initial_y=initial_y, initial_psi1=initial_psi1,
                              initial_psi2=initial_psi2, initial_R=initial_R)

            # Enforce the constraint
            geo_con.enforce(**kwargs)

        for dim in self.dims:
            print(f"{dim = }, {requestor = }")
            if requestor is not None and dim is requestor:
                continue
            if requestor is not None and dim in requestor.associated_dims:
                continue
            dim.update_param_from_points()

        # Update the GUI object, if there is one
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(self.x().value(), self.y().value())

        # TODO: add tree object update here as well

        for curve in self.curves:
            curve.update()

    def force_move(self, xp: float, yp: float):
        self.x().set_value(xp)
        self.y().set_value(yp)

        # Update the GUI object, if there is one
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(self.x().value(), self.y().value())

        # TODO: add tree object update here as well

        for curve in self.curves:
            curve.update()


class PointSequence:
    def __init__(self, points: typing.List[Point]):
        self._points = None
        self.set_points(points)

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

    def remove_point(self, idx: int):
        self._points.pop(idx)

    def as_array(self):
        return np.array([[p.x().value(), p.y().value()] for p in self.points()])

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.shape[1] != 2:
            raise ValueError(f"Array must have two columns, x and y. Found {arr.shape[1]} columns.")
        return cls(points=[Point(x=x, y=y, name=f"PointFromArray-Index{idx}")
                           for idx, (x, y) in enumerate(zip(arr[:, 0], arr[:, 1]))])

    def __len__(self):
        return len(self.points())

    def extract_subsequence(self, indices: list):
        return PointSequence(points=[self.points()[idx] for idx in indices])
