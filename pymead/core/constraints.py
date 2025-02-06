import typing
from dataclasses import dataclass
from abc import abstractmethod

import numpy as np

from parametric_curve import ParametricCurveEndpoint
from pymead.core.bezier import Bezier
from pymead.core.constraint_equations import measure_rel_angle3, measure_point_line_distance_unsigned
from pymead.core.line import PolyLine
from pymead.core.param import Param, AngleParam, LengthParam
from pymead.core.point import Point
from pymead.core.pymead_obj import PymeadObj


class GeoCon(PymeadObj):

    default_name: str = ""

    def __init__(self, param: Param or None, child_nodes: list, kind: str,
                 name: str or None = None, secondary_params: typing.List[Param] = None, geo_col=None):
        self._param = None
        self.set_param(param)
        sub_container = "geocon"
        super().__init__(sub_container=sub_container, geo_col=geo_col)
        name = self.default_name if name is None else name
        self.set_name(name)
        if self.param() is not None:
            self.param().geo_cons.append(self)
            if self.param().name() == "unnamed":
                self.param().set_name(f"{self.name()}.par")
        self.child_nodes = child_nodes
        self.kind = kind
        self.secondary_params = [] if secondary_params is None else secondary_params
        self.data = None
        self.add_constraint_to_points()

    def param(self):
        return self._param

    def set_param(self, param: Param):
        self._param = param

    def add_constraint_to_points(self):
        for child_node in self.child_nodes:
            if not isinstance(child_node, Point) or self in child_node.geo_cons:
                return

            child_node.geo_cons.append(self)

    def remove_constraint_from_points(self):
        for child_node in self.child_nodes:
            if not isinstance(child_node, Point) or self not in child_node.geo_cons:
                return

            child_node.geo_cons.remove(self)

    @abstractmethod
    def verify(self) -> bool:
        pass


class DistanceConstraint(GeoCon):

    default_name = "DistCon-1"

    def __init__(self, p1: Point, p2: Point, value: float or LengthParam = None, name: str = None, geo_col=None):
        if p1.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p1 is airfoil-relative.")
        if p2.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p2 is airfoil-relative.")
        self.p1 = p1
        self.p2 = p2
        value = self.p1.measure_distance(self.p2) if value is None else value
        param = value if isinstance(value, Param) else LengthParam(value=value, name="Length-1", geo_col=geo_col)
        self.handle_offset = None
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2], kind="d", geo_col=geo_col)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}
                # "constraint_type": self.__class__.__name__, "handle_offset": self.handle_offset}

    def verify(self) -> bool:
        measured_distance = self.p1.measure_distance(self.p2)
        return np.isclose(measured_distance, self.param().value(), rtol=1e-14)


class AntiParallel3Constraint(GeoCon):

    default_name = "AntiPar3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None, polyline: PolyLine = None,
                 point_on_curve: Point = None, geo_col=None):
        if p1.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p1 is airfoil-relative.")
        if p2.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p2 is airfoil-relative.")
        if p3.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p3 is airfoil-relative.")
        self.p1 = p1
        self.polyline = polyline
        self.point_on_curve = point_on_curve
        if polyline and polyline not in p1.curves and point_on_curve is p1:
            p1.curves.append(polyline)
        self.p2 = p2
        self.p3 = p3
        if polyline and polyline not in p3.curves and point_on_curve is p3:
            p3.curves.append(polyline)
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3", geo_col=geo_col)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__,
                "polyline": self.polyline.name() if self.polyline is not None else None,
                "point_on_curve": self.point_on_curve.name() if self.point_on_curve is not None else None}

    def verify(self) -> bool:
        a1 = self.p2.measure_angle(self.p1)
        a2 = self.p2.measure_angle(self.p3)
        measured_angle = (a1 - a2) % (2 * np.pi)
        return np.isclose(measured_angle, np.pi, rtol=1e-14)


class SymmetryConstraint(GeoCon):

    default_name = "SymCon-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str = None, geo_col=None):
        self.p1 = p1  # Line start point
        self.p2 = p2  # Line end point
        self.p3 = p3  # Tool point
        self.p4 = p4  # Target point
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4|d",
                         geo_col=geo_col)
        for point in self.child_nodes:
            if self not in point.x().geo_cons:
                point.x().geo_cons.append(self)
            if self not in point.y().geo_cons:
                point.y().geo_cons.append(self)

    @staticmethod
    def check_if_point_is_symmetric_target(p: Point):
        for geo_con in p.geo_cons:
            if isinstance(geo_con, SymmetryConstraint) and geo_con.child_nodes[-1] is p:
                return True
        return False

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(), "p4": self.p4.name(),
                "constraint_type": self.__class__.__name__}

    def verify(self) -> bool:
        tool_angle = measure_rel_angle3(self.p3.x().value(), self.p3.y().value(),
                                        self.p2.x().value(), self.p2.y().value(),
                                        self.p1.x().value(), self.p1.y().value())
        target_angle = measure_rel_angle3(self.p1.x().value(), self.p1.y().value(),
                                          self.p2.x().value(), self.p2.y().value(),
                                          self.p4.x().value(), self.p4.y().value())
        if not np.isclose(tool_angle, target_angle, rtol=1e-14):
            return False

        tool_distance_to_line = measure_point_line_distance_unsigned(
            self.p1.x().value(), self.p1.y().value(),
            self.p2.x().value(), self.p2.y().value(),
            self.p3.x().value(), self.p3.y().value()
        )
        target_distance_to_line = measure_point_line_distance_unsigned(
            self.p1.x().value(), self.p1.y().value(),
            self.p2.x().value(), self.p2.y().value(),
            self.p4.x().value(), self.p4.y().value()
        )

        return np.isclose(tool_distance_to_line, target_distance_to_line, rtol=1e-14)


class Perp3Constraint(GeoCon):

    default_name = "Perp3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None, geo_col=None):
        if p1.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p1 is airfoil-relative.")
        if p2.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p2 is airfoil-relative.")
        if p3.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p3 is airfoil-relative.")
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3", geo_col=geo_col)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__}

    def verify(self) -> bool:
        a1 = self.p2.measure_angle(self.p1)
        a2 = self.p2.measure_angle(self.p3)
        measured_angle = (a1 - a2) % (2 * np.pi)
        return np.isclose(measured_angle, 0.5 * np.pi, rtol=1e-14)


class RelAngle3Constraint(GeoCon):

    default_name = "RelAng3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, value: float or AngleParam = None, name: str = None,
                 geo_col=None):
        if p1.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p1 is airfoil-relative.")
        if p2.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p2 is airfoil-relative.")
        if p3.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point p3 is airfoil-relative.")
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        if value is None:
            args = []
            for point in [self.p1, self.p2, self.p3]:
                args.extend([point.x().value(), point.y().value()])
            value = measure_rel_angle3(*args)
        param = value if isinstance(value, Param) else AngleParam(value=value, name="Angle-1", geo_col=geo_col)
        self.handle_offset = None
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3", geo_col=geo_col)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(),
                "p3": self.p3.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}

    def verify(self) -> bool:
        a1 = self.p2.measure_angle(self.p1)
        a2 = self.p2.measure_angle(self.p3)
        measured_angle = (a1 - a2) % (2 * np.pi)
        return np.isclose(measured_angle, self.param().rad() % (2 * np.pi), rtol=1e-14)


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


class ROCurvatureConstraint(GeoCon):

    default_name = "ROCCon-1"

    def __init__(self, curve_joint: Point, value: float or LengthParam = None, name: str = None, geo_col=None,
                 curve_to_modify: Bezier = None):
        if curve_joint.relative_airfoil_name is not None:
            raise InvalidPointError(f"Constraint could not be added because point the curve joint is an "
                                    f"airfoil-relative point.")
        if len(curve_joint.curves) != 2:
            raise ConstraintValidationError(f"There must be exactly two curves attached to the curve joint. Found "
                                            f"{len(curve_joint.curves)} curves")
        self.curve_joint = curve_joint
        self.curve_1 = curve_joint.curves[0]
        self.curve_2 = curve_joint.curves[1]
        self.curve_type_1 = self.curve_1.__class__.__name__
        self.curve_type_2 = self.curve_2.__class__.__name__

        if self.curve_type_1 in ("Bezier", "BSpline"):
            curve_joint_index_curve_1 = self.curve_1.point_sequence().points().index(curve_joint)
            self.curve_joint_index_curve_1 = -1 if curve_joint_index_curve_1 != 0 else 0
            if (self.curve_joint_index_curve_1 == 0 and self.curve_type_1 == "BSpline" and
                    not self.curve_1.is_clamped(ParametricCurveEndpoint.Start)):
                raise ValueError(f"Curve {self.curve_1} is not clamped at the start. "
                                 f"Curvature constraints can only be applied to clamped curves.")
            elif (self.curve_joint_index_curve_1 == 0 and self.curve_type_1 == "BSpline" and
                    not self.curve_1.is_clamped(ParametricCurveEndpoint.Start)):
                raise ValueError(f"Curve {self.curve_1} is not clamped at the start. "
                                 f"Curvature constraints can only be applied to clamped curves.")
            self.g2_point_index_curve_1 = 2 if self.curve_joint_index_curve_1 == 0 else -3
            self.g1_point_index_curve_1 = 1 if self.g2_point_index_curve_1 == 2 else -2
            self.g1_point_curve_1 = self.curve_1.point_sequence().points()[self.g1_point_index_curve_1]
            self.g2_point_curve_1 = self.curve_1.point_sequence().points()[self.g2_point_index_curve_1]
        else:
            (curve_joint_index_curve_1, self.curve_joint_index_curve_1,
             self.g2_point_index_curve_1, self.g1_point_index_curve_1,
             self.g1_point_curve_1, self.g2_point_curve_1) = None, None, None, None, None, None

        if self.curve_type_2 in ("Bezier", "BSpline"):
            curve_joint_index_curve_2 = self.curve_2.point_sequence().points().index(curve_joint)
            self.curve_joint_index_curve_2 = -1 if curve_joint_index_curve_2 != 0 else 0
            if (self.curve_joint_index_curve_2 == 0 and self.curve_type_1 == "BSpline" and
                    not self.curve_1.is_clamped(ParametricCurveEndpoint.Start)):
                raise ValueError(f"Curve {self.curve_1} is not clamped at the start. "
                                 f"Curvature constraints can only be applied to clamped curves.")
            self.g2_point_index_curve_2 = 2 if self.curve_joint_index_curve_2 == 0 else -3
            self.g1_point_index_curve_2 = 1 if self.g2_point_index_curve_2 == 2 else -2
            self.g1_point_curve_2 = self.curve_2.point_sequence().points()[self.g1_point_index_curve_2]
            self.g2_point_curve_2 = self.curve_2.point_sequence().points()[self.g2_point_index_curve_2]
        else:
            (curve_joint_index_curve_2, self.curve_joint_index_curve_2,
             self.g2_point_index_curve_2, self.g1_point_index_curve_2,
             self.g1_point_curve_2, self.g2_point_curve_2) = None, None, None, None, None, None

        points = [self.curve_joint]
        for point in [self.g2_point_curve_1, self.g1_point_curve_1, self.g1_point_curve_2, self.g2_point_curve_2]:
            if point is None:
                continue
            points.append(point)

        secondary_params = []
        if self.curve_type_1 in ("Bezier", "BSpline"):
            secondary_params.append(Param(self.curve_1.degree, "n"))
        if self.curve_type_2 in ("Bezier", "BSpline"):
            secondary_params.append(Param(self.curve_2.degree, "n"))

        if self.curve_type_1 == "PolyLine" and self.curve_type_2 == "PolyLine":
            raise ValueError("Cannot create radius of curvature constraint between two polylines")

        if self.curve_type_1 == "LineSegment" or self.curve_type_2 == "LineSegment":
            param = None
        elif self.curve_type_1 == "PolyLine" or (self.curve_type_1 in ("Bezier", "BSpline") and (
                self.curve_1.t_start is not None or self.curve_1.t_end is not None)):
            data = self.curve_1.evaluate()
            if (np.isclose(data.xy[0, 0], self.curve_joint.x().value()) and
                np.isclose(data.xy[0, 1], self.curve_joint.y().value())):
                R = data.R[0]
            else:
                R = data.R[-1]

            param = LengthParam(value=R, name="ROC-1", enabled=False, geo_col=geo_col) if not isinstance(value, Param) else value
        elif self.curve_type_2 == "PolyLine" or (self.curve_type_2 in ("Bezier", "BSpline") and (
                self.curve_2.t_start is not None or self.curve_2.t_end is not None)):
            data = self.curve_2.evaluate()
            if (np.isclose(data.xy[0, 0], self.curve_joint.x().value()) and
                    np.isclose(data.xy[0, 1], self.curve_joint.y().value())):
                R = data.R[0]
            else:
                R = data.R[-1]
            param = LengthParam(value=R, name="ROC-1", enabled=False, geo_col=geo_col) if not isinstance(value, Param) else value
        else:
            enabled = True
            if value is None:
                curvature_data = self.calculate_curvature_data(self.curve_joint)

                if not self.is_solving_allowed(self.g2_point_curve_1) or curve_to_modify is self.curve_2:
                    value = curvature_data.R1
                    enabled = False
                elif not self.is_solving_allowed(self.g2_point_curve_2) or curve_to_modify is self.curve_1:
                    value = curvature_data.R2
                    enabled = False
                else:
                    value = 0.5 * (curvature_data.R1 + curvature_data.R2)
            param = value if isinstance(value, Param) else LengthParam(value=value, name="ROC-1", geo_col=geo_col)
            param.set_enabled(enabled)

        super().__init__(param=param, child_nodes=points, kind="d", name=name,
                         secondary_params=secondary_params, geo_col=geo_col)

    @staticmethod
    def calculate_curvature_data(curve_joint):

        Lt1, Lc1, n1, theta1, phi1, psi1, R1 = None, None, None, None, None, None, None
        Lt2, Lc2, n2, theta2, phi2, psi2, R2 = None, None, None, None, None, None, None

        curve_1 = curve_joint.curves[0]
        if curve_1.__class__.__name__ in ("Bezier", "BSpline"):
            curve_joint_index_curve_1 = curve_1.point_sequence().points().index(curve_joint)
            curve_joint_index_curve_1 = -1 if curve_joint_index_curve_1 != 0 else 0
            g2_point_index_curve_1 = 2 if curve_joint_index_curve_1 == 0 else -3
            g1_point_index_curve_1 = 1 if g2_point_index_curve_1 == 2 else -2
            g1_point_curve_1 = curve_1.point_sequence().points()[g1_point_index_curve_1]
            g2_point_curve_1 = curve_1.point_sequence().points()[g2_point_index_curve_1]
            Lt1 = g1_point_curve_1.measure_distance(curve_joint)
            Lc1 = g1_point_curve_1.measure_distance(g2_point_curve_1)
            n1 = curve_1.degree
            phi1 = curve_joint.measure_angle(g1_point_curve_1)
            theta1 = g1_point_curve_1.measure_angle(g2_point_curve_1)
            psi1 = theta1 - phi1
            with np.errstate(divide="ignore"):
                R1 = np.abs(np.true_divide((Lt1 * Lt1), (Lc1 * (1 - 1 / n1) * np.sin(psi1))))
        elif curve_1.__class__.__name__ == "LineSegment":
            R1 = np.inf

        curve_2 = curve_joint.curves[1]
        if curve_2.__class__.__name__ in ("Bezier", "BSpline"):
            curve_joint_index_curve_2 = curve_2.point_sequence().points().index(curve_joint)
            curve_joint_index_curve_2 = -1 if curve_joint_index_curve_2 != 0 else 0
            g2_point_index_curve_2 = 2 if curve_joint_index_curve_2 == 0 else -3
            g1_point_index_curve_2 = 1 if g2_point_index_curve_2 == 2 else -2
            g1_point_curve_2 = curve_2.point_sequence().points()[g1_point_index_curve_2]
            g2_point_curve_2 = curve_2.point_sequence().points()[g2_point_index_curve_2]
            Lt2 = g1_point_curve_2.measure_distance(curve_joint)
            Lc2 = g1_point_curve_2.measure_distance(g2_point_curve_2)
            n2 = curve_2.degree
            phi2 = curve_joint.measure_angle(g1_point_curve_2)
            theta2 = g1_point_curve_2.measure_angle(g2_point_curve_2)
            psi2 = theta2 - phi2
            with np.errstate(divide="ignore"):
                R2 = np.abs(np.true_divide((Lt2 * Lt2), (Lc2 * (1 - 1 / n2) * np.sin(psi2))))
        elif curve_2.__class__.__name__ == "LineSegment":
            R2 = np.inf

        if curve_1.__class__.__name__ == "PolyLine":
            data = curve_1.evaluate()
            if (np.isclose(data.xy[0, 0], curve_joint.x().value()) and
                    np.isclose(data.xy[0, 1], curve_joint.y().value())):
                R1 = data.R[0]
            else:
                R1 = data.R[-1]
        if curve_2.__class__.__name__ == "PolyLine":
            data = curve_2.evaluate()
            if (np.isclose(data.xy[0, 0], curve_joint.x().value()) and
                    np.isclose(data.xy[0, 1], curve_joint.y().value())):
                R2 = data.R[0]
            else:
                R2 = data.R[-1]

        data = CurvatureConstraintData(Lt1=Lt1, Lt2=Lt2, Lc1=Lc1, Lc2=Lc2, n1=n1, n2=n2, theta1=theta1, theta2=theta2,
                                       phi1=phi1, phi2=phi2, psi1=psi1, psi2=psi2, R1=R1, R2=R2)
        return data

    @staticmethod
    def is_solving_allowed(g2_point: Point):
        symmetry_constraints = [geo_con for geo_con in g2_point.geo_cons if
                                isinstance(geo_con, SymmetryConstraint)]
        # Check if the point is the symmetry target point
        if any([symmetry_constraint.child_nodes[-1] is g2_point for symmetry_constraint in symmetry_constraints]):
            return False
        return True

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()} <C1={self.curve_1.name()}, C2={self.curve_2.name()}>"

    def get_dict_rep(self):
        value = self.param().name() if self.param() is not None else None
        return {"curve_joint": self.curve_joint.name(), "value": value,
                "constraint_type": self.__class__.__name__}

    def verify(self) -> bool:
        data = self.calculate_curvature_data(self.curve_joint)
        if np.isinf(data.R1) and np.isinf(data.R2):
            return True
        return np.isclose(data.R1, data.R2, rtol=1e-10)


class ConstraintValidationError(Exception):
    pass


class InvalidPointError(Exception):
    pass


class NoSolutionError(Exception):
    pass


class DuplicateConstraintError(Exception):
    pass


class MaxWeakConstraintAttemptsError(Exception):
    pass
