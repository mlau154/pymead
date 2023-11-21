import typing

import numpy as np

from pymead.core.param2 import Param


class Point:
    def __init__(self, x: float, y: float, name: str or None = None, setting_from_geo_col: bool = False):
        self._name = None
        self._x = None
        self._y = None
        self.geo_col = None
        self.setting_from_geo_col = setting_from_geo_col
        self.set_name(name)
        self.set_x(x)
        self.set_y(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def name(self):
        return self._name

    def xy(self):
        return [self._x, self._y]

    def set_x(self, x: Param or float):
        self._x = x if isinstance(x, Param) else Param(
            value=x, name=self.name() + ".x", setting_from_geo_col=self.setting_from_geo_col)
        if self not in self._x.geo_objs:
            self._x.geo_objs.append(self)

    def set_y(self, y: Param or float):
        self._y = y if isinstance(y, Param) else Param(
            value=y, name=self.name() + ".y", setting_from_geo_col=self.setting_from_geo_col)
        if self not in self._y.geo_objs:
            self._y.geo_objs.append(self)

    def set_name(self, name: str or None = None):
        self._name = "Point" if name is None else name
        if self.x() is not None:
            self.x().set_name(f"{self.name()}.x")
        if self.y() is not None:
            self.y().set_name(f"{self.name()}.y")

    def set_xy(self, xy: typing.List[Param]):
        self._x = xy[0]
        self._y = xy[1]

    def as_array(self):
        return np.array([self.x().value(), self.y().value()])

    @classmethod
    def generate_from_array(cls, arr: np.ndarray, name: str or None = None):
        if arr.ndim != 1:
            raise ValueError("Points can only be generated from 1-dimensional arrays")
        return cls(x=arr[0], y=arr[1], name=name)

    def measure_distance(self, other: "Point"):
        # Interesting note: the quotes around "Point" are necessary here because this type hint is a forward reference.
        # This means that an object of type <Class> is specified as a hint somewhere inside the definition for <Class>
        return np.hypot(other.x().value() - self.x().value(), other.y().value() - self.y().value())


class PointSequence:
    def __init__(self, points: typing.List[Point]):
        self._points = None
        self.set_points(points)

    def points(self):
        return self._points

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


class SurfPoint(Point):
    pass


class SurfPointSequence(PointSequence):
    def __init__(self, surf_points: typing.List[SurfPoint]):
        super().__init__(points=surf_points)

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.shape[1] != 2:
            raise ValueError(f"Array must have two columns, x and y. Found {arr.shape[1]} columns.")
        return cls(surf_points=[SurfPoint(x=x, y=y, name=f"SurfPointFromArray-Index{idx}")
                                for idx, (x, y) in enumerate(zip(arr[:, 0], arr[:, 1]))])
