import typing

import numpy as np

from pymead.core.param2 import Param


class Point:
    def __init__(self, x: Param, y: Param):
        self._x = None
        self._y = None
        self.set_x(x)
        self.set_y(y)

    def x(self):
        return self._x

    def y(self):
        return self._y

    def xy(self):
        return [self._x, self._y]

    def set_x(self, x: Param):
        self._x = x

    def set_y(self, y: Param):
        self._y = y

    def set_xy(self, xy: typing.List[Param]):
        self._x = xy[0]
        self._y = xy[1]


class PointSequence:
    def __init__(self, points: typing.List[Point]):
        self._points = None
        self.set_points(points)

    def points(self):
        return self._points

    def set_points(self, points: typing.List[Point]):
        self._points = points

    def as_array(self):
        return np.array([[p.x().value(), p.y().value()] for p in self.points()])

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.shape[1] != 2:
            raise ValueError(f"Array must have two columns, x and y. Found {arr.shape[1]} columns.")
        return cls(points=[Point(x=Param(x), y=Param(y)) for x, y in zip(arr[:, 0], arr[:, 1])])

    def __len__(self):
        return len(self.points())


class SurfPoint(Point):
    pass


class SurfPointSequence(PointSequence):
    def __init__(self, surf_points: typing.List[SurfPoint]):
        super().__init__(points=surf_points)

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.shape[1] != 2:
            raise ValueError(f"Array must have two columns, x and y. Found {arr.shape[1]} columns.")
        return cls(surf_points=[SurfPoint(x=Param(x), y=Param(y)) for x, y in zip(arr[:, 0], arr[:, 1])])
