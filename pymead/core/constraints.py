import typing
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

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
        self.start_point = start_point
        self.middle_point = middle_point
        self.end_point = end_point
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

    def get_dict_rep(self):
        return {"start_point": self.start_point.name(), "middle_point": self.middle_point.name(),
                "end_point": self.end_point.name(), "name": self.name(), "constraint_type": "collinear"}


@dataclass
class CurvatureConstraintData:
    Lt1: float
    Lt2: float
    Lc1: float
    Lc2: float
    n1: int
    n2: int
    theta1: float
    theta2: float
    phi1: float
    phi2: float
    psi1: float
    psi2: float
    R1: float
    R2: float


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
        self.g2_point_curve_1 = self.curve_1.point_sequence().points()[self.g2_point_index_curve_1]
        self.g2_point_curve_2 = self.curve_2.point_sequence().points()[self.g2_point_index_curve_2]
        g1_g2_point_seq = PointSequence(points=[self.g2_point_curve_1, self.g1_point_curve_1,
                                                self.g1_point_curve_2, self.g2_point_curve_2])

        name = "CurvatureCon-1" if name is None else name
        super().__init__(tool=curve_joint, target=g1_g2_point_seq, name=name)
        self.tool().geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)

    def calculate_curvature_data(self):
        Lt1 = self.g1_point_curve_1.measure_distance(self.curve_joint)
        Lt2 = self.g1_point_curve_2.measure_distance(self.curve_joint)
        Lc1 = self.g1_point_curve_1.measure_distance(self.g2_point_curve_1)
        Lc2 = self.g1_point_curve_2.measure_distance(self.g2_point_curve_2)
        n1 = self.curve_1.degree
        n2 = self.curve_2.degree
        phi1 = self.curve_joint.measure_angle(self.g1_point_curve_1)
        phi2 = self.curve_joint.measure_angle(self.g1_point_curve_2)
        theta1 = self.g1_point_curve_1.measure_angle(self.g2_point_curve_1)
        theta2 = self.g1_point_curve_2.measure_angle(self.g2_point_curve_2)
        psi1 = theta1 - phi1
        psi2 = theta2 - phi2
        R1 = np.abs(np.true_divide((Lt1 * Lt1), (Lc1 * (1 - 1 / n1) * np.sin(psi1))))
        R2 = np.abs(np.true_divide((Lt2 * Lt2), (Lc2 * (1 - 1 / n2) * np.sin(psi2))))
        data = CurvatureConstraintData(Lt1=Lt1, Lt2=Lt2, Lc1=Lc1, Lc2=Lc2, n1=n1, n2=n2, theta1=theta1, theta2=theta2,
                                       phi1=phi1, phi2=phi2, psi1=psi1, psi2=psi2, R1=R1, R2=R2)
        return data

    def validate(self):
        msg = "A point in the CurvatureConstraint was found to already have an existing CurvatureConstraint"
        for geo_con in self.tool().geo_cons:
            if isinstance(geo_con, CurvatureConstraint):
                raise ConstraintValidationError(msg)
        for point in self.target().points():
            for geo_con in point.geo_cons:
                if isinstance(geo_con, CurvatureConstraint):
                    raise ConstraintValidationError(msg)

    def enforce(self, calling_point: Point,
                initial_x: float = None, initial_y: float = None,
                initial_psi1: float = None, initial_psi2: float = None, initial_R: float = None):

        if calling_point is self.tool():  # If the middle point called the enforcement
            if initial_x is None:
                raise ValueError("Must set initial_x if calling enforcement of collinear constraint from the "
                                 "middle point")
            if initial_y is None:
                raise ValueError("Must set initial_y if calling enforcement of collinear constraint from the "
                                 "middle point")
            dx = self.tool().x().value() - initial_x
            dy = self.tool().y().value() - initial_y

            collinear_constraint_found = False
            for geo_con in self.tool().geo_cons:
                if isinstance(geo_con, CollinearConstraint):
                    collinear_constraint_found = True
                    break

            points_to_move = [self.target().points()[0], self.target().points()[3]] \
                if collinear_constraint_found else self.target().points()
            for point in points_to_move:
                point.force_move(point.x().value() + dx, point.y().value() + dy)

        elif calling_point is self.target().points()[0]:  # Curve 1 g2 point modified -> update curve 2 g2 point
            # Calculate the new curvature control arm 2 length
            data = self.calculate_curvature_data()
            target_Lc2 = (data.Lt2 * data.Lt2) / (data.Lt1 * data.Lt1) * (
                    (1 - 1 / data.n1) * data.Lc1 * np.abs(np.sin(data.psi1))) / (
                    (1 - 1 / data.n2) * np.abs(np.sin(data.psi2)))

            # Calculate if the radius of curvature needs to be reversed for this step
            if (0 <= data.psi1 % (2 * np.pi) < np.pi and 0 <= data.psi2 % (2 * np.pi) < np.pi) or (
                np.pi <= data.psi1 % (2 * np.pi) < 2 * np.pi and np.pi <= data.psi2 % (2 * np.pi) < 2 * np.pi
            ):
                theta2 = data.phi2 - data.psi2  # Reverse
            else:
                theta2 = data.theta2  # Do not reverse

            # Move the other G2 point to match the curvature
            new_x = self.target().points()[2].x().value() + target_Lc2 * np.cos(theta2)
            new_y = self.target().points()[2].y().value() + target_Lc2 * np.sin(theta2)
            self.target().points()[3].force_move(new_x, new_y)

            data = self.calculate_curvature_data()
            # print(f"{data.psi1 = }, {data.psi2 = }, {data.R1 = }, {data.R2 = }")

        elif calling_point is self.target().points()[3]:  # Curve 2 g2 point modified -> update curve 1 g2 point
            # Calculate the new curvature control arm 1 length
            data = self.calculate_curvature_data()
            target_Lc1 = (data.Lt1 * data.Lt1) / (data.Lt2 * data.Lt2) * (
                    (1 - 1 / data.n2) * data.Lc2 * np.abs(np.sin(data.psi2))) / (
                    (1 - 1 / data.n1) * np.abs(np.sin(data.psi1)))

            # Calculate if the radius of curvature needs to be reversed for this step
            if (0 <= data.psi1 % (2 * np.pi) < np.pi and 0 <= data.psi2 % (2 * np.pi) < np.pi) or (
                    np.pi <= data.psi1 % (2 * np.pi) < 2 * np.pi and np.pi <= data.psi2 % (2 * np.pi) < 2 * np.pi
            ):
                theta1 = data.phi1 - data.psi1  # Reverse
            else:
                theta1 = data.theta1  # Do not reverse

            # Move the other G2 point to match the curvature
            new_x = self.target().points()[1].x().value() + target_Lc1 * np.cos(theta1)
            new_y = self.target().points()[1].y().value() + target_Lc1 * np.sin(theta1)
            self.target().points()[0].force_move(new_x, new_y)

        elif calling_point is self.target().points()[1]:  # Curve 1 g1 point modified -> update curve 2 g1 point and g2 points
            if initial_psi1 is None or initial_psi2 is None or initial_R is None:
                data = self.calculate_curvature_data()
                initial_psi1 = data.psi1
                initial_psi2 = data.psi2
                initial_R = np.abs(data.R1)

            # Calculate the new curvature control arm lengths and absolute angles
            data = self.calculate_curvature_data()
            # print(f"{data.Lt1 = }, {data.Lt2 = }, {initial_R = }, {data.phi1 = }, {initial_psi1 = }, {initial_psi2 = },"
            #       f"{data.R1 = }, {data.R2 = }")

            target_Lc1 = (data.Lt1 * data.Lt1) / (
                    initial_R * (1 - 1 / data.n1) * np.abs(np.sin(initial_psi1)))
            target_theta1 = initial_psi1 + data.phi1

            target_Lc2 = (data.Lt2 * data.Lt2) / (
                    initial_R * (1 - 1 / data.n2) * np.abs(np.sin(initial_psi2)))
            target_theta2 = initial_psi2 + (data.phi1 + np.pi)

            # Move the other G2 points to preserve the radius of curvature and angles
            new_x2 = self.curve_joint.x().value() + data.Lt2 * np.cos(data.phi1 + np.pi)
            new_y2 = self.curve_joint.y().value() + data.Lt2 * np.sin(data.phi1 + np.pi)
            new_x0 = self.target().points()[1].x().value() + target_Lc1 * np.cos(target_theta1)
            new_y0 = self.target().points()[1].y().value() + target_Lc1 * np.sin(target_theta1)
            new_x3 = new_x2 + target_Lc2 * np.cos(target_theta2)
            new_y3 = new_y2 + target_Lc2 * np.sin(target_theta2)
            # print(f"Before, {self.calculate_curvature_data().R1 = }, {self.calculate_curvature_data().R2 = }, {self.calculate_curvature_data().psi1 = }, {self.calculate_curvature_data().psi2 = }")
            self.target().points()[2].force_move(new_x2, new_y2)
            # print(f"After 2, {self.calculate_curvature_data().R1 = }, {self.calculate_curvature_data().R2 = }, {self.calculate_curvature_data().psi1 = }, {self.calculate_curvature_data().psi2 = }")
            self.target().points()[0].force_move(new_x0, new_y0)
            # print(f"After 0, {self.calculate_curvature_data().R1 = }, {self.calculate_curvature_data().R2 = }, {self.calculate_curvature_data().psi1 = }, {self.calculate_curvature_data().psi2 = }")
            self.target().points()[3].force_move(new_x3, new_y3)
            # print(f"After 3, {self.calculate_curvature_data().R1 = }, {self.calculate_curvature_data().R2 = }, {self.calculate_curvature_data().psi1 = }, {self.calculate_curvature_data().psi2 = }")

        elif calling_point is self.target().points()[2]:  # Curve 2 g1 point modified -> update curve 1 g1 point and g2 points
            if initial_psi1 is None or initial_psi2 is None or initial_R is None:
                raise ValueError("Must specify initial_psi1, initial_psi2, and initial_R when enforcing the "
                                 "curvature constraint from a G1 point")

            # Calculate the new curvature control arm lengths and absolute angles
            data = self.calculate_curvature_data()

            target_Lc1 = (data.Lt1 * data.Lt1) / (
                    initial_R * (1 - 1 / data.n1) * np.abs(np.sin(initial_psi1)))
            target_theta1 = initial_psi1 + (data.phi2 + np.pi)

            target_Lc2 = (data.Lt2 * data.Lt2) / (
                    initial_R * (1 - 1 / data.n2) * np.abs(np.sin(initial_psi2)))
            target_theta2 = initial_psi2 + data.phi2

            # Move the other G2 points to preserve the radius of curvature and angles
            new_x1 = self.curve_joint.x().value() + data.Lt1 * np.cos(data.phi2 + np.pi)
            new_y1 = self.curve_joint.y().value() + data.Lt1 * np.sin(data.phi2 + np.pi)
            new_x0 = new_x1 + target_Lc1 * np.cos(target_theta1)
            new_y0 = new_y1 + target_Lc1 * np.sin(target_theta1)
            new_x3 = self.target().points()[2].x().value() + target_Lc2 * np.cos(target_theta2)
            new_y3 = self.target().points()[2].y().value() + target_Lc2 * np.sin(target_theta2)
            self.target().points()[1].force_move(new_x1, new_y1)
            self.target().points()[0].force_move(new_x0, new_y0)
            self.target().points()[3].force_move(new_x3, new_y3)

            # TODO: check this logic. Might be causing a runaway radius of curvature on tangent point rotation

        # elif calling_point == "hold":
        #     print(f"Calling point hold!")
        #     pass

    def get_dict_rep(self):
        return {"curve_joint": self.curve_joint.name(), "name": self.name(), "constraint_type": "curvature"}


class ConstraintValidationError(Exception):
    pass
