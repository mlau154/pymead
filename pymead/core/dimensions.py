from abc import abstractmethod

import numpy as np

from pymead.core import UNITS
from pymead.core.param2 import Param, LengthParam, AngleParam
from pymead.core.point import PointSequence, Point


class Dimension:
    def __init__(self, tool: PointSequence or Point, target: PointSequence or Point, param: Param or None = None):
        self._tool = None
        self._target = None
        self._param = None
        self.geo_col = None
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
        if isinstance(tool, Point):
            if self not in tool.dims:
                tool.dims.append(self)
        elif isinstance(tool, PointSequence):
            for point in tool.points():
                if self not in point.dims:
                    point.dims.append(self)
        else:
            raise TypeError(f"tool must be either Point or PointSequence for Dimension. Found type: {type(tool)}")

    def set_target(self, target: PointSequence or Point):
        self._target = target
        if isinstance(target, Point):
            if self not in target.dims:
                target.dims.append(self)
        elif isinstance(target, PointSequence):
            for point in target.points():
                if self not in point.dims:
                    point.dims.append(self)
        else:
            raise TypeError(f"tool must be either Point or PointSequence for Dimension. Found type: {type(target)}")

    def set_param(self, param: Param or None = None):
        if param is None:
            self._param = self.update_param_from_points()
        else:
            self._param = param
            self.update_param_from_points()

    @abstractmethod
    def update_points_from_param(self):
        pass

    @abstractmethod
    def update_param_from_points(self):
        pass


class LengthDimension(Dimension):
    def __init__(self, tool_point: Point, target_point: Point, length_param: LengthParam or None = None):
        super().__init__(tool=tool_point, target=target_point, param=length_param)

    def update_points_from_param(self):
        target_angle = self.tool().measure_angle(self.target())
        self.target().request_move(self.tool().x().value() + self.param().value() * np.cos(target_angle),
                                   self.tool().y().value() + self.param().value() * np.sin(target_angle))

    def update_param_from_points(self):
        length = self.tool().measure_distance(self.target())
        if self.param() is None:
            if self.geo_col is None:
                param = LengthParam(value=length, name="LengthDim")
            else:
                param = self.geo_col.add_param(value=length, name="LengthDim", unit_type="length")
        else:
            param = self.param()
        param.set_value(self.tool().measure_distance(self.target()))
        return param


class AngleDimension(Dimension):
    def __init__(self, tool_point: Point, target_point: Point, angle_param: AngleParam or None = None):
        super().__init__(tool=tool_point, target=target_point, param=angle_param)

    def update_points_from_param(self):
        target_length = self.tool().measure_distance(self.target())
        self.target().request_move(self.tool().x().value() + target_length * np.cos(self.param().rad()),
                                   self.tool().y().value() + target_length * np.sin(self.param().rad()))

    def update_param_from_points(self):
        angle = self.tool().measure_angle(self.target())
        if self.param() is None:
            if self.geo_col is None:
                param = AngleParam(value=angle, name="AngleDim")
            else:
                param = self.geo_col.add_param(value=angle, name="LengthDim", unit_type="angle")
        else:
            param = self.param()
        param.set_value(UNITS.convert_angle_from_base(self.tool().measure_angle(self.target()),
                                                    unit=UNITS.current_angle_unit()))
        return param
