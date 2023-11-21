import typing
from abc import abstractmethod

import numpy as np

from pymead.core.param2 import Param
from pymead.core.point import PointSequence, Point


class GeoCon:
    def __init__(self, tool: PointSequence or Point, target: PointSequence or Point, bidirectional: bool = False):
        self._tool = None
        self._target = None
        self._bidirectional = None
        self.set_tool(tool)
        self.set_target(target)
        self.set_bidirectional(bidirectional)
        self.validate()

    def tool(self):
        return self._tool

    def target(self):
        return self._target

    def bidirectional(self):
        return self._bidirectional

    def set_tool(self, tool: PointSequence or Point):
        self._tool = tool

    def set_target(self, target: PointSequence or Point):
        self._target = target

    def set_bidirectional(self, bidirectional: bool):
        self._bidirectional = bidirectional

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def enforce(self, *args, **kwargs):
        pass


class Parallel(GeoCon):
    def __init__(self, tool: PointSequence, target: PointSequence):
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
        if len(self.tool()) != 2:
            raise ConstraintValidationError(f"Must choose a PointSequence of exactly two points for the parallel"
                                            f"constraint tool. Points chosen: {len(self.tool())}")
        if len(self.target()) != 2:
            raise ConstraintValidationError(f"Must choose a PointSequence of exactly two points for the parallel"
                                            f"constraint target. Points chosen: {len(self.target())}")

    def enforce(self):
        tool_seq = self.tool()
        target_seq = self.target()
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


class PositionConstraint(GeoCon):
    def __init__(self, tool: Point, target: Point, dist: Param, angle: Param,
                 bidirectional: bool = False):
        super().__init__(tool=tool, target=target, bidirectional=bidirectional)
        self.dist = dist
        self.angle = angle
        self.dist.geo_cons.append(self)
        self.angle.geo_cons.append(self)
        self.tool().geo_cons.append(self)
        self.target().geo_cons.append(self)

    def validate(self):
        pass

    def enforce(self, calling_point: Point or str):

        if isinstance(calling_point, str):
            if calling_point == "tool":
                calling_point = self.tool()
            elif calling_point == "target":
                calling_point = self.target()
            else:
                raise ValueError(f"If calling_point is a str, it must be either 'tool' or 'target'. Specified value: "
                                 f"{calling_point}")

        dist = self.dist.value()
        angle = self.angle.value()
        original_tool_x = self.tool().x().value()
        original_tool_y = self.tool().y().value()
        original_target_x = self.target().x().value()
        original_target_y = self.target().y().value()

        if self.bidirectional():
            if calling_point is self.tool():
                self.target().force_move(original_tool_x + dist * np.cos(angle),
                                         original_tool_y + dist * np.sin(angle))
            elif calling_point is self.target():
                self.tool().force_move(original_target_x + dist * np.cos(angle + np.pi),
                                       original_target_y + dist * np.sin(angle + np.pi))
            else:
                raise ValueError("Calling point for the positional constraint was neither tool nor target")
        else:
            if calling_point in [self.tool(), self.target()]:
                self.target().force_move(original_tool_x + dist * np.cos(angle),
                                         original_tool_y + dist * np.sin(angle))
            else:
                raise ValueError("Calling point for the positional constraint was neither tool nor target")


class ConstraintValidationError(Exception):
    pass
