import typing
from abc import abstractmethod

import numpy as np

from pymead.core.parametric_curve2 import ParametricCurve


class GeoCon:
    def __init__(self, tool: ParametricCurve, target: ParametricCurve):
        self._tool = None
        self._target = None
        self.validate()

    def tool(self):
        return self._tool

    def target(self):
        return self._target

    def set_tool(self, tool: ParametricCurve):
        self._tool = tool

    def set_target(self, target: ParametricCurve):
        self._target = target

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def enforce(self, *args, **kwargs):
        pass


class Parallel(GeoCon):
    def __init__(self, tool: ParametricCurve, target: ParametricCurve):
        super().__init__(tool=tool, target=target)
        self._solution = 0
        self._min_solution = 0
        self._max_solution = 1

    def solution(self):
        return self._solution

    def cycle_solution(self):
        if self._solution == self._max_solution:
            self._solution = self._min_solution
        else:
            self._solution += 1
        self.enforce()

    def validate(self):
        if self.tool().curve_type != "LineSegment":
            raise ConstraintValidationError("Found tool in Parallel Constraint with curve_type not equal to "
                                            "'LineSegment'. Only line segments can be constrained to be parallel.")
        if self.target().curve_type != "LineSegment":
            raise ConstraintValidationError("Found target in Parallel Constraint with curve_type not equal to "
                                            "'LineSegment'. Only line segments can be constrained to be parallel.")

    def enforce(self):
        tool_seq = self.tool().point_sequence()
        target_seq = self.target().point_sequence()
        tool_p1 = tool_seq.points()[0]
        tool_p2 = tool_seq.points()[1]
        target_p1 = target_seq.points()[0]
        target_p2 = target_seq.points()[1]
        tool_ang = np.arctan2(tool_p2.y().value() - tool_p1.y().value(), tool_p2.x().value() - tool_p1.y().value())
        target_centroid = 0.5 * (target_p1.as_array() + target_p2.as_array())
        target_length = np.hypot(target_p2.x().value() - target_p1.x().value(),
                                 target_p2.y().value() - target_p1.y().value()
                                 )
        if self.solution() == 0:
            multiplier1 = 1
            multiplier2 = -1
        else:
            multiplier1 = -1
            multiplier2 = 1
        new_p1 = target_centroid + multiplier1 * 0.5 * target_length * np.array([np.cos(tool_ang), np.sin(tool_ang)])
        new_p2 = target_centroid + multiplier2 * 0.5 * target_length * np.array([np.cos(tool_ang), np.sin(tool_ang)])
        target_p1.x().set_value(new_p1[0])
        target_p1.y().set_value(new_p1[1])
        target_p2.x().set_value(new_p2[0])
        target_p2.y().set_value(new_p2[1])


class ConstraintValidationError(Exception):
    pass
