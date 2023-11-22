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


class CollinearConstraint(GeoCon):
    def __init__(self, start_point: Point, middle_point: Point, end_point: Point):
        start_end_seq = PointSequence(points=[start_point, end_point])
        super().__init__(tool=middle_point, target=start_end_seq)
        self.tool().geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)

    def validate(self):
        pass

    def enforce(self, calling_point: Point or str, initial_x: float or None = None, initial_y: float or None = None):
        if isinstance(calling_point, str):
            if calling_point == "middle":
                calling_point = self.tool()
            elif calling_point == "start":
                calling_point = self.target().points()[0]
            elif calling_point == "end":
                calling_point = self.target().points()[1]
            else:
                raise ValueError(f"If calling_point is a str, it must be either 'start', 'middle', or 'end'. "
                                 f"Specified value: {calling_point}")

        if calling_point is self.tool():  # If the middle point called the enforcement
            if initial_x is None:
                raise ValueError("Must set initial_x if calling enforcement of collinear constraint from the "
                                 "middle point")
            if initial_y is None:
                raise ValueError("Must set initial_y if calling enforcement of collinear constraint from the "
                                 "middle point")
            dx = self.tool().x().value() - initial_x
            dy = self.tool().y().value() - initial_y
            for point in self.target().points():
                point.force_move(point.x().value() + dx, point.y().value() + dy)
        elif calling_point is self.target().points()[0]:
            start_point = self.target().points()[0]
            middle_point = self.tool()
            end_point = self.target().points()[1]
            target_angle_minus_pi = middle_point.measure_angle(start_point)
            target_angle = target_angle_minus_pi + np.pi
            length = middle_point.measure_distance(end_point)
            new_x = middle_point.x().value() + length * np.cos(target_angle)
            new_y = middle_point.y().value() + length * np.sin(target_angle)
            end_point.force_move(new_x, new_y)
        elif calling_point is self.target().points()[1]:
            start_point = self.target().points()[0]
            middle_point = self.tool()
            end_point = self.target().points()[1]
            target_angle_minus_pi = middle_point.measure_angle(end_point)
            target_angle = target_angle_minus_pi + np.pi
            length = middle_point.measure_distance(start_point)
            new_x = middle_point.x().value() + length * np.cos(target_angle)
            new_y = middle_point.y().value() + length * np.sin(target_angle)
            start_point.force_move(new_x, new_y)


class ConstraintValidationError(Exception):
    pass
