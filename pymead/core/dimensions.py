import typing
from abc import abstractmethod

import numpy as np

from pymead.core import UNITS
from pymead.core.param import Param, LengthParam, AngleParam
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj


class Dimension(PymeadObj):
    def __init__(self, tool: PointSequence or Point, target: PointSequence or Point, param: Param or None = None,
                 name: str or None = None):
        self._tool = None
        self._target = None
        self._param = None
        self.geo_col = None
        self.set_tool(tool)
        self.set_target(target)
        self.set_param(param)

        # Add associated dimensions (usually an AngleDimension if this object is a LengthDimension, or vice versa).
        # This code block helps prevents recursion errors where a dimension parameter update triggers a point
        # movement, which in turn triggers a dimension parameter update, which triggers a point movement, etc.
        # This dimension along with any associated dimensions get passed to the Point.request_move() method
        # as a "requestor," which prevents more than one Dimension.update().set_value() from occurring.
        self.associated_dims = []
        for point in [self.target(), self.tool()]:
            for dim in point.dims:
                if dim is self:
                    continue
                if (dim.tool() is self.tool() and dim.target() is self.target()) or (
                        dim.tool() is self.target() and dim.target() is self.tool()):
                    if dim not in self.associated_dims:
                        self.associated_dims.append(dim)
                    if self not in dim.associated_dims:
                        dim.associated_dims.append(self)

        super().__init__(sub_container="dims")
        self.set_name(name)

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
        if self not in self._param.dims:
            self._param.dims.append(self)

    @abstractmethod
    def update_points_from_param(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_param_from_points(self, *args, **kwargs):
        pass


class LengthDimension(Dimension):
    def __init__(self, tool_point: Point, target_point: Point, length_param: LengthParam or None = None,
                 name: str or None = None):
        name = "LengthDim-1" if name is None else name
        super().__init__(tool=tool_point, target=target_point, param=length_param, name=name)

    def update_points_from_param(self, updated_objs: typing.List[PymeadObj] = None):
        updated_objs = [] if updated_objs is None else updated_objs

        target_angle = self.tool().measure_angle(self.target())
        new_x = self.tool().x().value() + self.param().value() * np.cos(target_angle)
        new_y = self.tool().y().value() + self.param().value() * np.sin(target_angle)
        if self in updated_objs:
            self.target().force_move(new_x, new_y)
        else:
            updated_objs.append(self)
            self.target().request_move(new_x, new_y, updated_objs=updated_objs)

    def update_param_from_points(self, updated_objs: typing.List[PymeadObj] = None):
        updated_objs = [] if updated_objs is None else updated_objs

        length = self.tool().measure_distance(self.target())
        if self.param() is None:
            if self.geo_col is None:
                param = LengthParam(value=length, name="Length-1")
            else:
                param = self.geo_col.add_param(value=length, name="Length-1", unit_type="length")
        else:
            param = self.param()

        if self not in updated_objs:
            updated_objs.append(self)

        param.set_value(length, updated_objs=updated_objs)

        return param

    def get_dict_rep(self):
        return {"tool_point": self.tool().name(), "target_point": self.target().name(),
                "length_param": self.param().name()}


class AngleDimension(Dimension):
    def __init__(self, tool_point: Point, target_point: Point, angle_param: AngleParam or None = None,
                 name: str or None = None):
        name = "AngleDim-1" if name is None else name
        super().__init__(tool=tool_point, target=target_point, param=angle_param, name=name)

    def update_points_from_param(self, updated_objs: typing.List[PymeadObj] = None):
        updated_objs = [] if updated_objs is None else updated_objs

        target_length = self.tool().measure_distance(self.target())
        new_x = self.tool().x().value() + target_length * np.cos(self.param().rad())
        new_y = self.tool().y().value() + target_length * np.sin(self.param().rad())

        if self in updated_objs:
            self.target().force_move(new_x, new_y)
        else:
            updated_objs.append(self)
            self.target().request_move(new_x, new_y, updated_objs=updated_objs)

    def update_param_from_points(self, updated_objs: typing.List[PymeadObj] = None):
        updated_objs = [] if updated_objs is None else updated_objs

        angle = self.tool().measure_angle(self.target())
        if self.param() is None:
            if self.geo_col is None:
                param = AngleParam(value=angle, name="Angle-1")
            else:
                param = self.geo_col.add_param(value=angle, name="Angle-1", unit_type="angle")
        else:
            param = self.param()

        new_value = UNITS.convert_angle_from_base(self.tool().measure_angle(self.target()),
                                                  unit=UNITS.current_angle_unit())

        if self not in updated_objs:
            updated_objs.append(self)

        param.set_value(new_value, updated_objs=updated_objs)

        return param

    def get_dict_rep(self):
        return {"tool_point": self.tool().name(), "target_point": self.target().name(),
                "angle_param": self.param().name()}
