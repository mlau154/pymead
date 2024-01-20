import typing
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from pymead.core import constraint_equations as ceq
from pymead.core.param import Param, AngleParam, LengthParam
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj


class GeoCon(PymeadObj):

    equations: typing.List[typing.Callable] = None
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

    @abstractmethod
    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        pass


class GeoConWeak(PymeadObj):

    equations: typing.List[typing.Callable] = None
    default_name: str = ""

    def __init__(self, name: str or None = None):
        sub_container = "geocon_weak"
        super().__init__(sub_container=sub_container)
        name = self.default_name if name is None else name
        self.set_name(name)

    @abstractmethod
    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        pass


class DistanceConstraint(GeoCon):

    equations = [staticmethod(ceq.distance_constraint)]
    default_name = "DistCon-1"

    def __init__(self, p1: Point, p2: Point, value: float or LengthParam, name: str = None):
        self.p1 = p1
        self.p2 = p2
        param = value if isinstance(value, Param) else LengthParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2], kind="d")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.param()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class DistanceConstraintWeak(GeoConWeak):

    equations = [staticmethod(ceq.distance_constraint_weak)]
    default_name = "DistConWeak-1"

    def __init__(self, p1: Point, p2: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        super().__init__(name=name)

    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p1.x()), int(use_intermediate)],
            [param_list.index(self.p1.y()), int(use_intermediate)],
            [param_list.index(self.p2.x()), int(use_intermediate)],
            [param_list.index(self.p2.y()), int(use_intermediate)],
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {}


class AbsAngleConstraint(GeoCon):

    equations = [staticmethod(ceq.abs_angle_constraint)]
    default_name = "AbsAngleCon-1"

    def __init__(self, p1: Point, p2: Point, value: float or AngleParam, name: str = None):
        self.p1 = p1
        self.p2 = p2
        param = value if isinstance(value, Param) else AngleParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2], kind="a2")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.param()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class AbsAngleConstraintWeak(GeoConWeak):

    equations = [staticmethod(ceq.abs_angle_constraint_weak)]
    default_name = "AbsAngleConWeak-1"

    def __init__(self, p1: Point, p2: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        super().__init__(name)

    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p1.x()), int(use_intermediate)],
            [param_list.index(self.p1.y()), int(use_intermediate)],
            [param_list.index(self.p2.x()), int(use_intermediate)],
            [param_list.index(self.p2.y()), int(use_intermediate)],
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {}


class Parallel3Constraint(GeoCon):

    equations = [staticmethod(ceq.parallel3_constraint)]
    default_name = "Par3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__}


class AntiParallel3Constraint(GeoCon):

    equations = [staticmethod(ceq.antiparallel3_constraint)]
    default_name = "AntiPar3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__}


class Parallel4Constraint(GeoCon):

    equations = [staticmethod(ceq.parallel4_constraint)]
    default_name = "Par4Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2],
            [param_list.index(self.p4.x()), 2],
            [param_list.index(self.p4.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(), "p4": self.p4.name(),
                "constraint_type": self.__class__.__name__}


class AntiParallel4Constraint(GeoCon):

    equations = [staticmethod(ceq.antiparallel4_constraint)]
    default_name = "AntiPar4Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2],
            [param_list.index(self.p4.x()), 2],
            [param_list.index(self.p4.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(), "p4": self.p4.name(),
                "constraint_type": self.__class__.__name__}


class SymmetryConstraint(GeoCon):

    equations = [staticmethod(ceq.perp4_constraint), staticmethod(ceq.points_equidistant_from_line_constraint_signed)]
    default_name = "SymCon-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4|d")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [
            [
                [param_list.index(self.p1.x()), 2],
                [param_list.index(self.p1.y()), 2],
                [param_list.index(self.p2.x()), 2],
                [param_list.index(self.p2.y()), 2],
                [param_list.index(self.p3.x()), 2],
                [param_list.index(self.p3.y()), 2],
                [param_list.index(self.p4.x()), 2],
                [param_list.index(self.p4.y()), 2]
            ],
            [
                [param_list.index(self.p1.x()), 2],
                [param_list.index(self.p1.y()), 2],
                [param_list.index(self.p2.x()), 2],
                [param_list.index(self.p2.y()), 2],
                [param_list.index(self.p3.x()), 2],
                [param_list.index(self.p3.y()), 2],
                [param_list.index(self.p4.x()), 2],
                [param_list.index(self.p4.y()), 2]]
        ]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(), "p4": self.p4.name(),
                "constraint_type": self.__class__.__name__}


class Perp3Constraint(GeoCon):

    equations = [staticmethod(ceq.perp3_constraint)]
    default_name = "Perp3Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(),
                "constraint_type": self.__class__.__name__}


class Perp4Constraint(GeoCon):

    equations = [staticmethod(ceq.perp4_constraint)]
    default_name = "Perp4Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(param=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2],
            [param_list.index(self.p4.x()), 2],
            [param_list.index(self.p4.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(), "p3": self.p3.name(), "p4": self.p4.name(),
                "constraint_type": self.__class__.__name__}


class RelAngle3Constraint(GeoCon):

    equations = [staticmethod(ceq.rel_angle3_constraint)]
    default_name = "RelAng3Con-1"

    def __init__(self, start_point: Point, vertex: Point, end_point: Point, value: float or AngleParam, name: str = None):
        self.start_point = start_point
        self.vertex = vertex
        self.end_point = end_point
        param = value if isinstance(value, Param) else AngleParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.start_point, self.vertex, self.end_point], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.start_point.x()), 2],
            [param_list.index(self.start_point.y()), 2],
            [param_list.index(self.vertex.x()), 2],
            [param_list.index(self.vertex.y()), 2],
            [param_list.index(self.end_point.x()), 2],
            [param_list.index(self.end_point.y()), 2],
            [param_list.index(self.param()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"start_point": self.start_point.name(), "vertex": self.vertex.name(),
                "end_point": self.end_point.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class RelAngle4Constraint(GeoCon):

    equations = [staticmethod(ceq.rel_angle4_constraint)]
    default_name = "RelAng4Con-1"

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, value: float, name: str = None):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        param = value if isinstance(value, Param) else AngleParam(value=value, name="unnamed")
        super().__init__(param=param, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x()), 2],
            [param_list.index(self.p1.y()), 2],
            [param_list.index(self.p2.x()), 2],
            [param_list.index(self.p2.y()), 2],
            [param_list.index(self.p3.x()), 2],
            [param_list.index(self.p3.y()), 2],
            [param_list.index(self.p4.x()), 2],
            [param_list.index(self.p4.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.param().value()}>"

    def get_dict_rep(self) -> dict:
        return {"p1": self.p1.name(), "p2": self.p2.name(),
                "p3": self.p3.name(), "p4": self.p4.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class RelAngle3ConstraintWeak(GeoConWeak):

    equations = [staticmethod(ceq.rel_angle3_constraint_weak)]
    default_name = "RelAng3ConWeak-1"

    def __init__(self, start_point: Point, vertex: Point, end_point: Point, name: str = None):
        self.start_point = start_point
        self.vertex = vertex
        self.end_point = end_point
        super().__init__(name=name)

    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        return [[
            [param_list.index(self.start_point.x()), 2],
            [param_list.index(self.start_point.y()), 2],
            [param_list.index(self.vertex.x()), 2],
            [param_list.index(self.vertex.y()), 2],
            [param_list.index(self.end_point.x()), 2],
            [param_list.index(self.end_point.y()), 2],
            [param_list.index(self.start_point.x()), int(use_intermediate)],
            [param_list.index(self.start_point.y()), int(use_intermediate)],
            [param_list.index(self.vertex.x()), int(use_intermediate)],
            [param_list.index(self.vertex.y()), int(use_intermediate)],
            [param_list.index(self.end_point.x()), int(use_intermediate)],
            [param_list.index(self.end_point.y()), int(use_intermediate)]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {}


class PointOnLineConstraint(GeoCon):

    equations = [staticmethod(ceq.point_on_line_constraint)]
    default_name = "POLCon-1"

    def __init__(self, point: Point, line_start: Point, line_end: Point, name: str = None):
        self.point = point
        self.line_start = line_start
        self.line_end = line_end
        super().__init__(param=None, name=name, child_nodes=[self.point, self.line_start, self.line_end], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.point.x()), 2],
            [param_list.index(self.point.y()), 2],
            [param_list.index(self.line_start.x()), 2],
            [param_list.index(self.line_start.y()), 2],
            [param_list.index(self.line_end.x()), 2],
            [param_list.index(self.line_end.y()), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}"

    def get_dict_rep(self) -> dict:
        return {"point": self.point.name(), "line_start": self.line_start.name(), "line_end": self.line_end.name(),
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

    equations = [staticmethod(ceq.radius_of_curvature_constraint), staticmethod(ceq.radius_of_curvature_constraint)]
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

        points = [self.g2_point_curve_1, self.g1_point_curve_1,
                  self.curve_joint,
                  self.g1_point_curve_2, self.g2_point_curve_2]

        param = value if isinstance(value, Param) else LengthParam(value=value, name="ROC-1")

        super().__init__(param=param, child_nodes=points, kind="d", name=name,
                         secondary_params=[Param(self.curve_1.degree, "n"),
                                           Param(self.curve_2.degree, "n")])

    @staticmethod
    def calculate_curvature_data(curve_joint):
        curve_1 = curve_joint.curves[0]
        curve_2 = curve_joint.curves[1]
        curve_joint_index_curve_1 = curve_1.point_sequence().points().index(curve_joint)
        curve_joint_index_curve_2 = curve_2.point_sequence().points().index(curve_joint)
        curve_joint_index_curve_1 = -1 if curve_joint_index_curve_1 != 0 else 0
        curve_joint_index_curve_2 = -1 if curve_joint_index_curve_2 != 0 else 0
        g2_point_index_curve_1 = 2 if curve_joint_index_curve_1 == 0 else -3
        g2_point_index_curve_2 = 2 if curve_joint_index_curve_2 == 0 else -3
        g1_point_index_curve_1 = 1 if g2_point_index_curve_1 == 2 else -2
        g1_point_index_curve_2 = 1 if g2_point_index_curve_2 == 2 else -2
        g1_point_curve_1 = curve_1.point_sequence().points()[g1_point_index_curve_1]
        g1_point_curve_2 = curve_2.point_sequence().points()[g1_point_index_curve_2]
        g2_point_curve_1 = curve_1.point_sequence().points()[g2_point_index_curve_1]
        g2_point_curve_2 = curve_2.point_sequence().points()[g2_point_index_curve_2]
        Lt1 = g1_point_curve_1.measure_distance(curve_joint)
        Lt2 = g1_point_curve_2.measure_distance(curve_joint)
        Lc1 = g1_point_curve_1.measure_distance(g2_point_curve_1)
        Lc2 = g1_point_curve_2.measure_distance(g2_point_curve_2)
        n1 = curve_1.degree
        n2 = curve_2.degree
        phi1 = curve_joint.measure_angle(g1_point_curve_1)
        phi2 = curve_joint.measure_angle(g1_point_curve_2)
        theta1 = g1_point_curve_1.measure_angle(g2_point_curve_1)
        theta2 = g1_point_curve_2.measure_angle(g2_point_curve_2)
        psi1 = theta1 - phi1
        psi2 = theta2 - phi2
        R1 = np.abs(np.true_divide((Lt1 * Lt1), (Lc1 * (1 - 1 / n1) * np.sin(psi1))))
        R2 = np.abs(np.true_divide((Lt2 * Lt2), (Lc2 * (1 - 1 / n2) * np.sin(psi2))))
        data = CurvatureConstraintData(Lt1=Lt1, Lt2=Lt2, Lc1=Lc1, Lc2=Lc2, n1=n1, n2=n2, theta1=theta1, theta2=theta2,
                                       phi1=phi1, phi2=phi2, psi1=psi1, psi2=psi2, R1=R1, R2=R2)
        return data

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.curve_joint.x()), 2],
            [param_list.index(self.curve_joint.y()), 2],
            [param_list.index(self.g1_point_curve_1.x()), 2],
            [param_list.index(self.g1_point_curve_1.y()), 2],
            [param_list.index(self.g2_point_curve_1.x()), 2],
            [param_list.index(self.g2_point_curve_1.y()), 2],
            [param_list.index(self.param()), 2],
            [param_list.index(self.secondary_params[0]), 2]
        ],
            [
            [param_list.index(self.curve_joint.x()), 2],
            [param_list.index(self.curve_joint.y()), 2],
            [param_list.index(self.g1_point_curve_2.x()), 2],
            [param_list.index(self.g1_point_curve_2.y()), 2],
            [param_list.index(self.g2_point_curve_2.x()), 2],
            [param_list.index(self.g2_point_curve_2.y()), 2],
            [param_list.index(self.param()), 2],
            [param_list.index(self.secondary_params[1]), 2]
        ]]

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()} <C1={self.curve_1.name()}, C2={self.curve_2.name()}>"

    def get_dict_rep(self):
        return {"curve_joint": self.curve_joint.name(), "value": self.param().name(),
                "constraint_type": self.__class__.__name__}


class ConstraintValidationError(Exception):
    pass


class NoSolutionError(Exception):
    pass


class DuplicateConstraintError(Exception):
    pass


class MaxWeakConstraintAttemptsError(Exception):
    pass
