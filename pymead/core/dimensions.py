from abc import abstractmethod

import numpy as np

from pymead.core.param2 import Param
from pymead.core.point import PointSequence, Point


class Dimension:
    def __init__(self, tool: PointSequence or Point, target: PointSequence or Point, param: Param):
        self._tool = None
        self._target = None
        self._param = None
        self.set_tool(tool)
        self.set_target(target)
        self.set_param(param)

    def tool(self):
        return self._tool

    def target(self):
        return self._target

    def param(self):
        return self._param

    def set_tool(self, tool: PointSequence or Point):
        self._tool = tool

    def set_target(self, target: PointSequence or Point):
        self._target = target

    def set_param(self, param: Param):
        self._param = param

    @abstractmethod
    def update_points_from_param(self):
        pass

    @abstractmethod
    def update_param_from_points(self):
        pass


class LengthDimension(Dimension):
    def __init__(self, tool_point: Point, target_point: Point, length_param: Param):
        super().__init__(tool=tool_point, target=target_point, param=length_param)

    def update_points_from_param(self):
        target_angle = self.tool().measure_angle(self.target())
        self.target().request_move(self.tool().x().value() + self.param().value() * np.cos(target_angle),
                                   self.tool().y().value() + self.param().value() * np.sin(target_angle))

    def update_param_from_points(self):
        self.param().set_value(self.tool().measure_distance(self.target()))
