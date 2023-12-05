from abc import abstractmethod

import numpy as np

from pymead.core.dimensions import LengthDimension, AngleDimension
from pymead.core.param2 import Param
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj


class GeoCon(PymeadObj):
    def __init__(self, tool: PointSequence or Point, target: PointSequence or Point, bidirectional: bool = False,
                 name: str or None = None):
        self._tool = None
        self._target = None
        self._bidirectional = None
        super().__init__(sub_container="geocon")
        self.set_name(name)
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
    def __init__(self, tool: PointSequence, target: PointSequence, name: str or None = None):
        name = "ParallelCon-1" if name is None else name
        super().__init__(tool=tool, target=target, name=name)
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
                 bidirectional: bool = False, name: str or None = None):
        name = "PositionCon-1" if name is None else name
        super().__init__(tool=tool, target=target, bidirectional=bidirectional, name=name)
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
    def __init__(self, start_point: Point, middle_point: Point, end_point: Point, name: str or None = None):
        start_end_seq = PointSequence(points=[start_point, end_point])
        name = "CollinearCon-1" if name is None else name
        super().__init__(tool=middle_point, target=start_end_seq, name=name)
        self.tool().geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)

    def validate(self):
        msg = ("A point in the CollinearConstraint was found to already have an existing CollinearConstraint. To make "
               "these additional points collinear, modify the original collinear constraint.")
        for geo_con in self.tool().geo_cons:
            if isinstance(geo_con, CollinearConstraint):
                raise ValueError(msg)
        for point in self.target().points():
            for geo_con in point.geo_cons:
                if isinstance(geo_con, CollinearConstraint):
                    raise ValueError(msg)

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
            # TODO: if a collinear constraint already exists on a point and another collinear constraint is attempted,
            #  just add the selected points to the existing constraint


class CurvatureConstraint(GeoCon):
    def __init__(self, curve_joint: Point, name: str or None = None):
        if len(curve_joint.curves) != 2:
            raise ConstraintValidationError(f"There must be exactly two curves attached to the curve joint. Found "
                                            f"{len(curve_joint.curves)} curves")
        self.curve_joint = curve_joint
        self.curve_1 = curve_joint.curves[0]
        self.curve_2 = curve_joint.curves[1]
        self.curve_type_1 = self.curve_1.__class__.__name__
        self.curve_type_2 = self.curve_2.__class__.__name__
        curve_joint_index_curve_1 = self.curve_1.point_sequence().points().index(curve_joint)
        curve_joint_index_curve_2 = self.curve_2.point_sequence().points().index(curve_joint)
        self.curve_joint_index_curve_1 = -1 if curve_joint_index_curve_1 != 0 else 0
        self.curve_joint_index_curve_2 = -1 if curve_joint_index_curve_2 != 0 else 0
        self.g2_point_index_curve_1 = 2 if self.curve_joint_index_curve_1 == 0 else -3
        self.g2_point_index_curve_2 = 2 if self.curve_joint_index_curve_2 == 0 else -3
        self.g1_point_index_curve_1 = 1 if self.g2_point_index_curve_1 == 2 else -2
        self.g1_point_index_curve_2 = 1 if self.g2_point_index_curve_2 == 2 else -2
        self.g1_point_curve_1 = self.curve_1.point_sequence().points()[self.g1_point_index_curve_1]
        self.g1_point_curve_2 = self.curve_2.point_sequence().points()[self.g1_point_index_curve_2]

        # G1 parameters
        # self.g1_length_param_curve1 = None
        # self.g1_angle_param_curve1 = None
        # self.g1_length_param_curve2 = None
        # self.g1_angle_param_curve2 = None
        # self.find_g1_dims()

        self.g2_point_curve_1 = self.curve_1.point_sequence().points()[self.g2_point_index_curve_1]
        self.g2_point_curve_2 = self.curve_2.point_sequence().points()[self.g2_point_index_curve_2]
        g1_g2_point_seq = PointSequence(points=[self.g2_point_curve_1, self.g1_point_curve_1,
                                                self.g1_point_curve_2, self.g2_point_curve_2])

        name = "CurvatureCon-1" if name is None else name
        super().__init__(tool=curve_joint, target=g1_g2_point_seq, name=name)
        self.tool().geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)

    # def find_g1_dims(self):
    #     g1_curve1_length_dim = None
    #     g1_curve1_angle_dim = None
    #     g1_curve2_length_dim = None
    #     g1_curve2_angle_dim = None
    #
    #     for dim in self.curve_joint.dims:
    #         if (isinstance(dim, LengthDimension) and dim.tool() == self.curve_joint
    #                 and dim.target() == self.g1_point_curve_1):
    #             g1_curve1_length_dim = dim
    #         if (isinstance(dim, AngleDimension) and dim.tool() == self.curve_joint
    #                 and dim.target() == self.g1_point_curve_1):
    #             g1_curve1_angle_dim = dim
    #         if (isinstance(dim, LengthDimension) and dim.tool() == self.curve_joint
    #                 and dim.target() == self.g1_point_curve_2):
    #             g1_curve2_length_dim = dim
    #         if (isinstance(dim, AngleDimension) and dim.tool() == self.curve_joint
    #                 and dim.target() == self.g1_point_curve_2):
    #             g1_curve2_angle_dim = dim
    #
    #     if any([dim is None for dim in [g1_curve1_length_dim, g1_curve1_angle_dim,
    #                                     g1_curve2_length_dim, g1_curve2_angle_dim]]):
    #         raise ConstraintValidationError("Must have a length and angle dimension for each point immediately "
    #                                         "adjacent to the curve joint")
    #
    #     self.g1_length_param_curve1 = g1_curve1_length_dim.param()
    #     self.g1_angle_param_curve1 = g1_curve2_angle_dim.param()
    #     self.g1_length_param_curve2 = g1_curve1_length_dim.param()
    #     self.g1_angle_param_curve2 = g1_curve2_angle_dim.param()

    def validate(self):
        msg = "A point in the CurvatureConstraint was found to already have an existing CurvatureConstraint"
        for geo_con in self.tool().geo_cons:
            if isinstance(geo_con, CurvatureConstraint):
                raise ConstraintValidationError(msg)
        for point in self.target().points():
            for geo_con in point.geo_cons:
                if isinstance(geo_con, CurvatureConstraint):
                    raise ConstraintValidationError(msg)

    def enforce(self, calling_point: Point, initial_x: float or None = None, initial_y: float or None = None):
        # if isinstance(calling_point, str):
        #     if calling_point == "middle":
        #         calling_point = self.tool()
        #     elif calling_point == "start":
        #         calling_point = self.target().points()[0]
        #     elif calling_point == "end":
        #         calling_point = self.target().points()[1]
        #     else:
        #         raise ValueError(f"If calling_point is a str, it must be either 'start', 'middle', or 'end'. "
        #                          f"Specified value: {calling_point}")

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
        elif calling_point is self.target().points()[0]:  # Curve 1 g2 point modified -> update curve 2 g2 point
            Lt1 = self.g1_point_curve_1.measure_distance(self.curve_joint)
            Lt2 = self.g1_point_curve_2.measure_distance(self.curve_joint)
            Lc1 = self.g1_point_curve_1.measure_distance(self.g2_point_curve_1)
            n1 = self.curve_1.degree
            n2 = self.curve_2.degree
            theta1 = self.g1_point_curve_1.measure_angle(self.curve_joint)
            theta2 = self.g1_point_curve_2.measure_angle(self.curve_joint)
            phi1 = self.g1_point_curve_1.measure_angle(self.g2_point_curve_1)
            phi2 = self.g1_point_curve_2.measure_angle(self.g2_point_curve_2)
            psi1 = theta1 - phi1
            psi2 = theta2 - phi2
            target_Lc2 = (Lt2 * Lt2) / (Lt1 * Lt1) * (
                    (1 - 1 / n1) * Lc1 * np.abs(np.sin(psi1))) / ((1 - 1 / n2) * np.abs(np.sin(psi2)))
            if (0 <= psi1 % (2 * np.pi) < np.pi and 0 <= psi2 % (2 * np.pi) < np.pi) or (
                np.pi <= psi1 % (2 * np.pi) < 2 * np.pi and np.pi <= psi2 % (2 * np.pi) < 2 * np.pi
            ):
                phi2 = theta2 + psi2
            self.target().points()[3].force_move(self.target().points()[2].x().value() + target_Lc2 * np.cos(phi2),
                                                 self.target().points()[2].y().value() + target_Lc2 * np.sin(phi2))
        elif calling_point in self.target().points()[2:]:  # Curve 2 modified -> update curve 1 g2 point

            pass


class ConstraintValidationError(Exception):
    pass
