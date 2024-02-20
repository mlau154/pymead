import typing
from dataclasses import dataclass

import numpy as np

from pymead.core.param import Param, AngleParam, LengthParam
from pymead.core.point import Point
from pymead.core.pymead_obj import PymeadObj


class GeoCon(PymeadObj):

    default_name: str = ""

    def __init__(self, param: Param or None, child_nodes: list, kind: str,
                 name: str or None = None, secondary_params: typing.List[Param] = None):
        self._param = None
        self.set_param(param)
        sub_container = "geocon"
        super().__init__(sub_container=sub_container)
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


class DistanceConstraint(GeoCon):

    default_name = "DistCon-1"

    def __init__(self, p1: Point, p2: Point, value: float or LengthParam, name: str = None):
        self.p1 = p1
        self.p2 = p2
        param = value if isinstance(value, Param) else LengthParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2], kind="d")

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class AbsAngleConstraint(GeoCon):

    default_name = "AbsAngleCon-1"

    def __init__(self, p1: Point, p2: Point, value: float or AngleParam, name: str = None):
        self.p1 = p1
        self.p2 = p2
        param = value if isinstance(value, Param) else AngleParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2], kind="a2")

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class AntiParallel3Constraint(GeoCon):

    default_name = "AntiPar3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__}


class SymmetryConstraint(GeoCon):

    default_name = "SymCon-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4|d")

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(), "p4": self.p4.name(),
                "constraint_type": self.__class__.__name__}


class Perp3Constraint(GeoCon):

    default_name = "Perp3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__}


class RelAngle3Constraint(GeoCon):

    default_name = "RelAng3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, value: float or AngleParam, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        param = value if isinstance(value, Param) else AngleParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(),
                "p3": self.p3.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


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

    def __init__(self, curve_joint: Point, value: float or LengthParam, name: str = None):
        if len(curve_joint.curves) != 2:
            raise ConstraintValidationError(f"There must be exactly two curves attached to the curve joint. Found "
                                            f"{len(curve_joint.curves)} curves")
        self.curve_joint = curve_joint
        self.curve_1 = curve_joint.curves[0]
        self.curve_2 = curve_joint.curves[1]
        self.curve_type_1 = self.curve_1.__class__.__name__
        self.curve_type_2 = self.curve_2.__class__.__name__

        if self.curve_type_1 == "Bezier":
            curve_joint_index_curve_1 = self.curve_1.point_sequence().points().index(curve_joint)
            self.curve_joint_index_curve_1 = -1 if curve_joint_index_curve_1 != 0 else 0
            self.g2_point_index_curve_1 = 2 if self.curve_joint_index_curve_1 == 0 else -3
            self.g1_point_index_curve_1 = 1 if self.g2_point_index_curve_1 == 2 else -2
            self.g1_point_curve_1 = self.curve_1.point_sequence().points()[self.g1_point_index_curve_1]
            self.g2_point_curve_1 = self.curve_1.point_sequence().points()[self.g2_point_index_curve_1]
        else:
            (curve_joint_index_curve_1, self.curve_joint_index_curve_1,
             self.g2_point_index_curve_1, self.g1_point_index_curve_1,
             self.g1_point_curve_1, self.g2_point_curve_1) = None, None, None, None, None, None

        if self.curve_type_2 == "Bezier":
            curve_joint_index_curve_2 = self.curve_2.point_sequence().points().index(curve_joint)
            self.curve_joint_index_curve_2 = -1 if curve_joint_index_curve_2 != 0 else 0
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
        if self.curve_type_1 == "Bezier":
            secondary_params.append(Param(self.curve_1.degree, "n"))
        if self.curve_type_2 == "Bezier":
            secondary_params.append(Param(self.curve_2.degree, "n"))

        if self.curve_type_1 == "LineSegment" or self.curve_type_2 == "LineSegment":
            param = None
        elif self.curve_type_1 == "PolyLine":
            data = self.curve_1.evaluate()
            if (np.isclose(data.xy[0, 0], self.curve_joint.x().value()) and
                np.isclose(data.xy[0, 1], self.curve_joint.y().value())):
                R = data.R[0]
            else:
                R = data.R[-1]

            param = LengthParam(value=R, name="ROC-1")
        elif self.curve_type_2 == "PolyLine":
            data = self.curve_2.evaluate()
            if (np.isclose(data.xy[0, 0], self.curve_joint.x().value()) and
                    np.isclose(data.xy[0, 1], self.curve_joint.y().value())):
                R = data.R[0]
            else:
                R = data.R[-1]
            param = LengthParam(value=R, name="ROC-1")
        else:
            param = value if isinstance(value, Param) else LengthParam(value=value, name="ROC-1")

        super().__init__(param=param, child_nodes=points, kind="d", name=name,
                         secondary_params=secondary_params)

    @staticmethod
    def calculate_curvature_data(curve_joint):
        curve_1 = curve_joint.curves[0]
        if curve_1.__class__.__name__ == "Bezier":
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
            R1 = np.abs(np.true_divide((Lt1 * Lt1), (Lc1 * (1 - 1 / n1) * np.sin(psi1))))
        else:
            Lt1, Lc1, n1, theta1, phi1, psi1, R1 = None, None, None, None, None, None, None

        curve_2 = curve_joint.curves[1]
        if curve_2.__class__.__name__ == "Bezier":
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
            R2 = np.abs(np.true_divide((Lt2 * Lt2), (Lc2 * (1 - 1 / n2) * np.sin(psi2))))
        else:
            Lt2, Lc2, n2, theta2, phi2, psi2, R2 = None, None, None, None, None, None, None

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

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()} <C1={self.curve_1.name()}, C2={self.curve_2.name()}>"

    def get_dict_rep(self):
        value = self.param().name() if self.param() is not None else None
        return {"curve_joint": self.curve_joint.name(), "value": value,
                "constraint_type": self.__class__.__name__}


class ConstraintValidationError(Exception):
    pass


class NoSolutionError(Exception):
    pass


class DuplicateConstraintError(Exception):
    pass


class MaxWeakConstraintAttemptsError(Exception):
    pass
