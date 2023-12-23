import typing
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from collections import namedtuple

import numpy as np
from scipy.optimize import root
from jax import jit, jacfwd, debug
from jax import numpy as jnp

from pymead.core import UNITS
from pymead.core.param2 import Param, AngleParam, LengthParam
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj


@jit
def measure_distance(x1: float, y1: float, x2: float, y2: float):
    return jnp.hypot(x1 - x2, y1 - y2)


@jit
def measure_abs_angle(x1: float, y1: float, x2: float, y2: float):
    return (jnp.arctan2(y2 - y1, x2 - x1)) % (2 * jnp.pi)


@jit
def measure_rel_angle3(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return (jnp.arctan2(y1 - y2, x1 - x2) - jnp.arctan2(y3 - y2, x3 - x2)) % (2 * jnp.pi)


@jit
def measure_rel_angle4(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return (jnp.arctan2(y4 - y3, x4 - x3) - jnp.arctan2(y2 - y1, x2 - x1)) % (2 * jnp.pi)


@jit
def measure_radius_of_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return jnp.abs(jnp.true_divide(Lt ** 2, Lc * (1 - 1 / n) * jnp.sin(psi)))


@jit
def measure_curvature_bezier(Lt: float, Lc: float, n: int, psi: float):
    return jnp.abs(jnp.true_divide(Lc * (1 - 1 / n) * jnp.sin(psi), Lt ** 2))


@jit
def measure_data_bezier_curve_joint(xy: np.ndarray, n: np.ndarray):
    phi1 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[1, 0], xy[1, 1])
    phi2 = measure_abs_angle(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    theta1 = measure_abs_angle(xy[1, 0], xy[1, 1], xy[0, 0], xy[0, 1])
    theta2 = measure_abs_angle(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    psi1 = theta1 - phi1
    psi2 = theta2 - phi2
    phi_rel = (phi1 - phi2) % (2 * jnp.pi)
    Lt1 = measure_distance(xy[1, 0], xy[1, 1], xy[2, 0], xy[2, 1])
    Lt2 = measure_distance(xy[2, 0], xy[2, 1], xy[3, 0], xy[3, 1])
    Lc1 = measure_distance(xy[0, 0], xy[0, 1], xy[1, 0], xy[1, 1])
    Lc2 = measure_distance(xy[3, 0], xy[3, 1], xy[4, 0], xy[4, 1])
    kappa1 = measure_curvature_bezier(Lt1, Lc1, n[0], psi1)
    kappa2 = measure_curvature_bezier(Lt2, Lc2, n[1], psi2)
    R1 = jnp.true_divide(1, kappa1)
    R2 = jnp.true_divide(1, kappa2)
    n1 = n[0]
    n2 = n[1]
    field_names = ["phi1", "phi2", "theta1", "theta2", "psi1", "psi2", "phi_rel", "Lt1", "Lt2", "Lc1", "Lc2",
                   "kappa1", "kappa2", "R1", "R2", "n1", "n2"]
    BezierCurveJointData = namedtuple("BezierCurveJointData", field_names=field_names)
    data = BezierCurveJointData(phi1=phi1, phi2=phi2, theta1=theta1, theta2=theta2, psi1=psi1, psi2=psi2,
                                phi_rel=phi_rel, Lt1=Lt1, Lt2=Lt2, Lc1=Lc1, Lc2=Lc2, kappa1=kappa1, kappa2=kappa2,
                                R1=R1, R2=R2, n1=n1, n2=n2)
    return data


@jit
def empty_constraint_weak():
    return 0.0


@jit
def fixed_param_constraint(p_val: float, val: float):
    return p_val - val


@jit
def fixed_param_constraint_weak(new_val: float, old_val: float):
    return new_val - old_val


@jit
def fixed_x_constraint(x: float, val: float):
    return x - val


@jit
def fixed_x_constraint_weak(x_new: float, x_old: float):
    return x_new - x_old


@jit
def fixed_y_constraint(y: float, val: float):
    return y - val


@jit
def fixed_y_constraint_weak(y_new: float, y_old: float):
    return y_new - y_old


@jit
def distance_constraint(x1: float, y1: float, x2: float, y2: float, dist: float):
    return measure_distance(x1, y1, x2, y2) - dist


@jit
def abs_angle_constraint(x1: float, y1: float, x2: float, y2: float, angle: float):
    return measure_abs_angle(x1, y1, x2, y2) - angle


@jit
def abs_angle_constraint_weak(x1_new: float, y1_new: float, x2_new: float, y2_new: float,
                              x1_old: float, y1_old: float, x2_old: float, y2_old: float):
    return (measure_abs_angle(x1_new, y1_new, x2_new, y2_new) -
            measure_abs_angle(x1_old, y1_old, x2_old, y2_old))


@jit
def rel_angle3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, angle: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - angle


@jit
def rel_angle4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float,
                          angle: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - angle


@jit
def perp3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - (jnp.pi / 2)


@jit
def perp4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - (jnp.pi / 2)


@jit
def parallel3_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return measure_rel_angle3(x1, y1, x2, y2, x3, y3) - jnp.pi


@jit
def parallel4_constraint(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float):
    return measure_rel_angle4(x1, y1, x2, y2, x3, y3, x4, y4) - jnp.pi


class DistCon:

    def __init__(self, p1: Point, p2: Point, dist: Param):
        self.p1 = p1
        self.p2 = p2
        self.dist = dist


class GCS:

    # Unconstrained = 0
    # UnderConstrained = 1
    # FullyConstrained = 2
    # OverConstrained = 3

    def __init__(self, parent: Point or Param):
        self.parent = parent
        self.parent.gcs = self
        self.constraint_types = []
        self.equations = []
        self.weak_equations = []
        self.variables = []
        self.weak_arg_indices = []
        self.points = []
        self.params = []
        self.sub_pos = []
        self.original_data = []

    # def check_constraint_state(self):
    #     if len(self.variables) == 0:
    #         return self.Unconstrained
    #     elif len(self.variables)

    @partial(jit, static_argnums=(0,))
    def equation_set(self, x: np.ndarray, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):

        # Evaluate the strong constraints using the updated parameter vector
        constraints = [eq(*x[sub_pos]) for eq, sub_pos in zip(self.equations, self.sub_pos)]

        # Evaluate the weak constraints (functions that are simply used to keep the system fully constrained according
        # to a set of rules and can be overridden by strong constraints)
        weak_constraints = [eq(*[start_param_vec[w[0]] if w[1] == 0
                                 else intermediate_param_vec[w[0]] if w[1] == 1
                                 else x[w[0]] for w in weak_args])
                            for eq, weak_args in zip(self.weak_equations, self.weak_arg_indices)]

        # Combine the lists of strong and weak constraints
        constraints.extend(weak_constraints)

        return jnp.array(constraints)

    @partial(jit, static_argnums=(0,))
    def equation_set2(self, x: np.ndarray, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):

        # Evaluate the strong constraints using the updated parameter vector
        constraints = [eq(*x[sub_pos]) for eq, sub_pos in zip(self.equations, self.sub_pos)]

        # Evaluate the weak constraints (functions that are simply used to keep the system fully constrained according
        # to a set of rules and can be overridden by strong constraints)
        weak_constraints = [eq(*[start_param_vec[w[0]] if w[1] == 0
                                 else intermediate_param_vec[w[0]] if w[1] == 1
                                 else x[w[0]] for w in weak_args])
                            for eq, weak_args in zip(self.weak_equations, self.weak_arg_indices)]

        # Combine the lists of strong and weak constraints
        constraints.extend(weak_constraints)

        return jnp.array(constraints)

    @partial(jit, static_argnums=(0,))
    def jacobian(self, *args):
        """
        Calculates the Jacobian matrix for the non-linear system of equations describing the full set of
        constraints for this point or parameter. According to the Jax docs, forward automatic differentiation
        may have a slight advantage over reverse automatic differentiation for calculating a square Jacobian, so
        ``jacfwd`` is used here.

        Parameters
        ----------
        args

        Returns
        -------
        np.ndarray
            The evaluated (square) Jacobian matrix

        """
        return jacfwd(self.equation_set)(*args)

    @partial(jit, static_argnums=(0,))
    def jacobian2(self, *args):
        """
        Calculates the Jacobian matrix for the non-linear system of equations describing the full set of
        constraints for this point or parameter. According to the Jax docs, forward automatic differentiation
        may have a slight advantage over reverse automatic differentiation for calculating a square Jacobian, so
        ``jacfwd`` is used here.

        Parameters
        ----------
        args

        Returns
        -------
        np.ndarray
            The evaluated (square) Jacobian matrix

        """
        return jacfwd(self.equation_set)(*args)

    def solve(self, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
        return root(self.equation_set, x0=intermediate_param_vec, jac=self.jacobian,
                    args=(start_param_vec, intermediate_param_vec), tol=1e-6)

    def solve2(self, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
        return root(self.equation_set2, x0=intermediate_param_vec, jac=self.jacobian2,
                    args=(start_param_vec, intermediate_param_vec), tol=1e-6)

    def update(self, new_x: np.ndarray):
        """
        Updates the variable points and parameters in the constraint system

        Returns
        -------

        """
        for v, x in zip(self.params, new_x):
            v.set_value(x)

        curves_to_update = []
        for point in self.points:
            if point.canvas_item is not None:
                point.canvas_item.updateCanvasItem(point.x().value(), point.y().value())

            for curve in point.curves:
                if curve not in curves_to_update:
                    curves_to_update.append(curve)

        for curve in curves_to_update:
            curve.update()

    def append_points(self, points: typing.List[Point]):
        for point in points:
            if point not in self.points:
                self.points.append(point)

    def append_params(self, params: typing.List[Param]):
        sub_pos = []
        for param in params:
            if param in self.params:
                sub_pos.append(self.params.index(param))
            else:
                sub_pos.append(len(self.params))
                self.params.append(param)

        self.sub_pos.append(np.array(sub_pos))

    def append_variables(self, variables: typing.List[Param]):
        no_solution = True
        for v in variables:
            if v not in self.variables:
                self.variables.append(v)

                res = self.solve()
                if res.success:
                    no_solution = False
                    self.update(res.x)
                    break
                else:
                    self.variables.remove(v)

        if no_solution:
            raise NoSolutionError("Over-constrained or no solution")

    def add_fixed_x_constraint(self, p: Point, x: LengthParam):
        self.equations.append(fixed_x_constraint)
        self.append_points([p])
        self.append_params([p.x(), x])
        self.constraint_types.append("FixedX")

        # self.append_variables([p.x()])

    def add_fixed_y_constraint(self, p: Point, y: LengthParam):
        self.equations.append(fixed_y_constraint)
        self.append_points([p])
        self.append_params([p.y(), y])
        self.constraint_types.append("FixedY")

        self.append_variables([p.y()])

    def add_distance_constraint(self, p1: Point, p2: Point, dist: LengthParam):
        self.equations.append(distance_constraint)
        self.append_points([p1, p2])
        self.append_params([p1.x(), p1.y(), p2.x(), p2.y(), dist])
        self.constraint_types.append("Distance")

        if self.parent is p1:
            self.weak_equations.append(fixed_x_constraint_weak)
            self.weak_equations.append(fixed_y_constraint_weak)
            self.weak_equations.append(fixed_param_constraint_weak)

            self.weak_arg_indices.append([[self.params.index(p2.x()), 1], [self.params.index(p2.x()), 0]])
            self.weak_arg_indices.append([[self.params.index(p2.y()), 1], [self.params.index(p2.y()), 0]])
            self.weak_arg_indices.append([[self.params.index(dist), 1], [self.params.index(dist), 0]])

        elif self.parent in [p2, dist]:
            self.weak_equations.append(fixed_x_constraint_weak)
            self.weak_equations.append(fixed_y_constraint_weak)
            self.weak_equations.append(fixed_param_constraint_weak)
            self.weak_equations.append(abs_angle_constraint_weak)

            self.weak_arg_indices.append([[self.params.index(p1.x()), 2], [self.params.index(p1.x()), 0]])
            self.weak_arg_indices.append([[self.params.index(p1.y()), 2], [self.params.index(p1.y()), 0]])
            self.weak_arg_indices.append([[self.params.index(dist), 2], [self.params.index(dist), 0]])

            fixed_angle_indices = []
            for old_new_idx in [2, 0]:
                for p in [p2.x(), p2.y(), p1.x(), p1.y()]:
                    p_index = self.params.index(p)
                    fixed_angle_indices.append([p_index, old_new_idx])

            self.weak_arg_indices.append(fixed_angle_indices)

            start_param_vec = np.array([p.value() for p in [p1.x(), p1.y(), p2.x(), p2.y(), dist]])
            intermediate_param_vec = deepcopy(start_param_vec)

            x = deepcopy(start_param_vec)
            weak_constraints = [eq(*[start_param_vec[w[0]] if w[1] == 0
                                     else intermediate_param_vec[w[0]] if w[1] == 1
                                     else x[w[0]] for w in weak_args])
                                for eq, weak_args in zip(self.weak_equations, self.weak_arg_indices)]

            res = self.solve(start_param_vec, intermediate_param_vec)

            self.update(res.x)

            self.weak_arg_indices.pop()

            fixed_angle_indices = []
            for old_new_idx in [2, 1]:
                for p in [p1.x(), p1.y(), p2.x(), p2.y()]:
                    p_index = self.params.index(p)
                    fixed_angle_indices.append([p_index, old_new_idx])

            self.weak_arg_indices.append(fixed_angle_indices)

        #
        # # Try to initialize the constraint by setting point 2 at distance "dist" from point 1
        # # at the original absolute angle
        # if self.parent is p1:
        #     self.equations.append(fixed_x_constraint)
        #
        # if self.parent is p1:
        #     self.append_variables([p1.x(), p1.y(), p2.x(), p2.y()])
        # else:
        #     self.append_variables([p2.x(), p2.y(), p1.x(), p1.y()])

    def add_abs_angle_constraint(self, p1: Point, p2: Point, angle: AngleParam):
        self.equations.append(abs_angle_constraint)
        self.append_points([p1, p2])
        self.append_params([p1.x(), p1.y(), p2.x(), p2.y(), angle])
        self.constraint_types.append("AbsAngle")
        if self.parent is p1:
            self.append_variables([p1.x(), p1.y(), p2.x(), p2.y()])
        else:
            self.append_variables([p2.x(), p2.y(), p1.x(), p1.y()])

    def add_rel_angle3_constraint(self, p1: Point, p2: Point, p3: Point, angle: AngleParam):
        self.equations.append(rel_angle3_constraint)
        self.append_points([p1, p2, p3])
        self.append_params([p1.x(), p1.y(), p2.x(), p2.y(), p3.x(), p3.y(), angle])
        self.constraint_types.append("RelAngle3")

        # Try to initialize the constraint by setting point 1 at relative angle "angle"
        # and retaining the original distance between point 1 and point 2
        target_dist = p1.measure_distance(p2)
        abs_angle_p2p3 = p2.measure_angle(p3)
        if self.parent is p2:
            p1.x().set_value(p2.x().value() + target_dist * np.cos(abs_angle_p2p3 + angle.rad()))
            p1.y().set_value(p2.y().value() + target_dist * np.sin(abs_angle_p2p3 + angle.rad()))

        ang = measure_rel_angle3(p1.x().value(), p1.y().value(), p2.x().value(), p2.y().value(),
                                 p3.x().value(), p3.y().value())

        if self.parent is p1:
            self.append_variables([p1.x(), p1.y(), p3.x(), p3.y(), p2.x(), p2.y()])
        elif self.parent is p2:
            self.append_variables([p2.x(), p2.y(), p1.x(), p1.y(), p3.x(), p3.y()])
        elif self.parent is p3:
            self.append_variables([p3.x(), p3.y(), p1.x(), p1.y(), p2.x(), p2.y()])
        # TODO: revisit. Do we need to p2 here?


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


class RelAngleConstraint(GeoCon):
    def __init__(self, tool: PointSequence, target: PointSequence, angle_param: AngleParam or None = None,
                 name: str or None = None):
        name = "RelAngleCon-1" if name is None else name
        super().__init__(tool=tool, target=target, name=name)
        self._param = None
        self.set_param(angle_param)
        for point in self.tool().points():
            point.geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)
        self.param().geo_cons.append(self)

    def param(self):
        return self._param

    def validate(self):
        if len(self.tool()) != 2:
            raise ConstraintValidationError("RelAngleConstraint tool must contain exactly two points")
        if len(self.target()) != 2:
            raise ConstraintValidationError("RelAngleConstraint target must contain exactly two points")

    def set_param(self, param: Param or None = None):
        if param is None:
            self._param = self.update_param_from_points()
        else:
            self._param = param
            self.update_param_from_points()
        if self not in self._param.dims:
            self._param.dims.append(self)

    def update_points_from_param(self, updated_objs: typing.List[PymeadObj] = None):
        updated_objs = [] if updated_objs is None else updated_objs

        target_length = self.measure_target_length()
        target_angle = self.measure_tool_angle() + self.param().rad()
        new_x = self.tool().x().value() + target_length * np.cos(target_angle)
        new_y = self.tool().y().value() + target_length * np.sin(target_angle)

        if self in updated_objs:
            self.target().force_move(new_x, new_y)
        else:
            updated_objs.append(self)
            self.target().request_move(new_x, new_y, updated_objs=updated_objs)

    def update_param_from_points(self, updated_objs: typing.List[PymeadObj] = None):
        updated_objs = [] if updated_objs is None else updated_objs

        angle = self.measure_target_angle() - self.measure_tool_angle()
        if self.param() is None:
            if self.geo_col is None:
                param = AngleParam(value=angle, name="Angle-1")
            else:
                param = self.geo_col.add_param(value=angle, name="Angle-1", unit_type="angle")
        else:
            param = self.param()

        new_value = UNITS.convert_angle_from_base(angle, unit=UNITS.current_angle_unit())

        if self not in updated_objs:
            updated_objs.append(self)

        param.set_value(new_value, updated_objs=updated_objs)

        return param

    def measure_tool_angle(self) -> float:
        return self.tool().points()[0].measure_angle(self.tool().points()[1])

    def measure_target_angle(self) -> float:
        return self.target().points()[0].measure_angle(self.target().points()[1])

    def measure_target_length(self) -> float:
        return self.target().points()[0].measure_distance(self.target().points()[1])

    def enforce(self, calling_point: Point, updated_objs: typing.List[PymeadObj] = None,
                initial_x: float = None, initial_y: float = None):

        updated_objs = [] if updated_objs is None else updated_objs

        if calling_point in [*self.tool().points(), *self.target().points()]:
            tool_angle = self.measure_tool_angle()
            target_angle = tool_angle + self.param().rad()
            target_length = self.measure_target_length()
            new_x = self.target().points()[0].x().value() + target_length * np.cos(target_angle)
            new_y = self.target().points()[0].y().value() + target_length * np.sin(target_angle)

            if self in updated_objs:
                self.target().points()[1].force_move(new_x, new_y)
            else:
                updated_objs.append(self)
                self.target().points()[1].request_move(new_x, new_y, updated_objs=updated_objs)

    def get_dict_rep(self):
        return {"tool": [pt.name() for pt in self.tool().points()],
                "target": [pt.name() for pt in self.target().points()],
                "angle_param": self.param(),
                "constraint_type": "rel-angle"}


class PerpendicularConstraint(GeoCon):
    def __init__(self, tool: PointSequence, target: PointSequence, name: str or None = None):
        name = "PerpendicularCon-1" if name is None else name
        super().__init__(tool=tool, target=target, name=name)
        for point in self.tool().points():
            point.geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)

    def validate(self):
        if len(self.tool()) != 2:
            raise ConstraintValidationError("PerpendicularConstraint tool must contain exactly two points")
        if len(self.target()) != 2:
            raise ConstraintValidationError("RPerpendicularConstraint target must contain exactly two points")

    def measure_tool_angle(self) -> float:
        return self.tool().points()[0].measure_angle(self.tool().points()[1])

    def measure_target_angle(self) -> float:
        return self.target().points()[0].measure_angle(self.target().points()[1])

    def measure_target_length(self) -> float:
        return self.target().points()[0].measure_distance(self.target().points()[1])

    def enforce(self, calling_point: Point, updated_objs: typing.List[PymeadObj] = None,
                initial_x: float = None, initial_y: float = None):

        updated_objs = [] if updated_objs is None else updated_objs

        if calling_point in [*self.tool().points(), *self.target().points()]:
            tool_angle = self.measure_tool_angle()
            target_angle = tool_angle + np.pi / 2
            target_length = self.measure_target_length()
            new_x = self.target().points()[0].x().value() + target_length * np.cos(target_angle)
            new_y = self.target().points()[0].y().value() + target_length * np.sin(target_angle)

            if self in updated_objs:
                self.target().points()[1].force_move(new_x, new_y)
            else:
                updated_objs.append(self)
                self.target().points()[1].request_move(new_x, new_y, updated_objs=updated_objs)

    def get_dict_rep(self):
        return {"tool": [pt.name() for pt in self.tool().points()],
                "target": [pt.name() for pt in self.target().points()],
                "constraint_type": "perpendicular"}


class ParallelConstraint(GeoCon):
    def __init__(self, tool: PointSequence, target: PointSequence, name: str or None = None):
        name = "ParallelCon-1" if name is None else name
        super().__init__(tool=tool, target=target, name=name)
        for point in self.tool().points():
            point.geo_cons.append(self)
        for point in self.target().points():
            point.geo_cons.append(self)

    def validate(self):
        if len(self.tool()) != 2:
            raise ConstraintValidationError("ParallelConstraint tool must contain exactly two points")
        if len(self.target()) != 2:
            raise ConstraintValidationError("ParallelConstraint target must contain exactly two points")

    def measure_tool_angle(self) -> float:
        return self.tool().points()[0].measure_angle(self.tool().points()[1])

    def measure_target_angle(self) -> float:
        return self.target().points()[0].measure_angle(self.target().points()[1])

    def measure_target_length(self) -> float:
        return self.target().points()[0].measure_distance(self.target().points()[1])

    def enforce(self, calling_point: Point, updated_objs: typing.List[PymeadObj] = None,
                initial_x: float = None, initial_y: float = None):

        updated_objs = [] if updated_objs is None else updated_objs

        if calling_point in [*self.tool().points(), *self.target().points()]:
            tool_angle = self.measure_tool_angle()
            target_angle = tool_angle + np.pi
            target_length = self.measure_target_length()
            new_x = self.target().points()[0].x().value() + target_length * np.cos(target_angle)
            new_y = self.target().points()[0].y().value() + target_length * np.sin(target_angle)

            if self in updated_objs:
                self.target().points()[1].force_move(new_x, new_y)
            else:
                updated_objs.append(self)
                self.target().points()[1].request_move(new_x, new_y, updated_objs=updated_objs)

    def get_dict_rep(self):
        return {"tool": [pt.name() for pt in self.tool().points()],
                "target": [pt.name() for pt in self.target().points()],
                "constraint_type": "parallel"}


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
        # for point in self.target().points():
        #     for geo_con in point.geo_cons:
        #         if isinstance(geo_con, CollinearConstraint):
        #             raise ValueError(msg)

    def enforce(self, calling_point: Point or str, updated_objs: typing.List[PymeadObj] = None,
                initial_x: float or None = None, initial_y: float or None = None):

        updated_objs = [] if updated_objs is None else updated_objs

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

            if self in updated_objs:
                for point in self.target().points():
                    point.force_move(point.x().value() + dx, point.y().value() + dy)
            else:
                updated_objs.append(self)

                # First, force the points to the correct location
                for point in self.target().points():
                    point.force_move(point.x().value() + dx, point.y().value() + dy)

                # Then, call a request move in-place to update the child dimensions/constraints
                for point in self.target().points():
                    point.request_move(point.x().value(), point.y().value(), updated_objs=updated_objs)

        elif calling_point is self.target().points()[0]:
            start_point = self.target().points()[0]
            middle_point = self.tool()
            end_point = self.target().points()[1]
            target_angle_minus_pi = middle_point.measure_angle(start_point)
            target_angle = target_angle_minus_pi + np.pi
            length = middle_point.measure_distance(end_point)
            new_x = middle_point.x().value() + length * np.cos(target_angle)
            new_y = middle_point.y().value() + length * np.sin(target_angle)

            if self in updated_objs:
                end_point.force_move(new_x, new_y)
            else:
                updated_objs.append(self)
                end_point.request_move(new_x, new_y, updated_objs=updated_objs)
        elif calling_point is self.target().points()[1]:
            start_point = self.target().points()[0]
            middle_point = self.tool()
            end_point = self.target().points()[1]
            target_angle_minus_pi = middle_point.measure_angle(end_point)
            target_angle = target_angle_minus_pi + np.pi
            length = middle_point.measure_distance(start_point)
            new_x = middle_point.x().value() + length * np.cos(target_angle)
            new_y = middle_point.y().value() + length * np.sin(target_angle)

            if self in updated_objs:
                start_point.force_move(new_x, new_y)
            else:
                updated_objs.append(self)
                start_point.request_move(new_x, new_y, updated_objs=updated_objs)
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

    def enforce(self, calling_point: Point, updated_objs: typing.List[PymeadObj] = None,
                initial_x: float = None, initial_y: float = None,
                initial_psi1: float = None, initial_psi2: float = None, initial_R: float = None):

        updated_objs = [] if updated_objs is None else updated_objs

        if calling_point is self.tool():  # If the middle point called the enforcement
            if initial_x is None:
                raise ValueError("Must set initial_x if calling enforcement of collinear constraint from the "
                                 "middle point")
            if initial_y is None:
                raise ValueError("Must set initial_y if calling enforcement of collinear constraint from the "
                                 "middle point")
            dx = self.tool().x().value() - initial_x
            dy = self.tool().y().value() - initial_y

            # collinear_constraint_found = False
            # for geo_con in self.tool().geo_cons:
            #     if isinstance(geo_con, CollinearConstraint):
            #         collinear_constraint_found = True
            #         break
            #
            # points_to_move = [self.target().points()[0], self.target().points()[3]] \
            #     if collinear_constraint_found else self.target().points()

            if self in updated_objs:
                for point in self.target().points():
                    point.force_move(point.x().value() + dx, point.y().value() + dy)
            else:
                updated_objs.append(self)
                # First, force the points to the correct location
                for point in self.target().points():
                    point.force_move(point.x().value() + dx, point.y().value() + dy)

                # Then, call a request move in-place to update the child dimensions/constraints
                for point in self.target().points():
                    point.request_move(point.x().value(), point.y().value(), updated_objs=updated_objs)

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

            if self in updated_objs:
                self.target().points()[3].force_move(new_x, new_y)
            else:
                updated_objs.append(self)
                self.target().points()[3].request_move(new_x, new_y, updated_objs=updated_objs)

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

            if self in updated_objs:
                self.target().points()[0].force_move(new_x, new_y)
            else:
                self.target().points()[0].request_move(new_x, new_y, updated_objs=updated_objs)

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

            if self in updated_objs:
                self.target().points()[2].force_move(new_x2, new_y2)
                self.target().points()[0].force_move(new_x0, new_y0)
                self.target().points()[3].force_move(new_x3, new_y3)
            else:
                updated_objs.append(self)
                self.target().points()[2].force_move(new_x2, new_y2)
                self.target().points()[0].force_move(new_x0, new_y0)
                self.target().points()[3].force_move(new_x3, new_y3)

                self.target().points()[2].request_move(self.target().points()[2].x().value(),
                                                       self.target().points()[2].y().value(), updated_objs=updated_objs)
                self.target().points()[0].request_move(self.target().points()[0].x().value(),
                                                       self.target().points()[0].y().value(), updated_objs=updated_objs)
                self.target().points()[3].request_move(self.target().points()[3].x().value(),
                                                       self.target().points()[3].y().value(), updated_objs=updated_objs)

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

            if self in updated_objs:
                self.target().points()[1].force_move(new_x1, new_y1)
                self.target().points()[0].force_move(new_x0, new_y0)
                self.target().points()[3].force_move(new_x3, new_y3)
            else:
                updated_objs.append(self)
                self.target().points()[1].force_move(new_x1, new_y1)
                self.target().points()[0].force_move(new_x0, new_y0)
                self.target().points()[3].force_move(new_x3, new_y3)

                self.target().points()[1].request_move(self.target().points()[1].x().value(),
                                                       self.target().points()[1].y().value(), updated_objs=updated_objs)
                self.target().points()[0].request_move(self.target().points()[0].x().value(),
                                                       self.target().points()[0].y().value(), updated_objs=updated_objs)
                self.target().points()[3].request_move(self.target().points()[3].x().value(),
                                                       self.target().points()[3].y().value(), updated_objs=updated_objs)

            # TODO: check this logic. Might be causing a runaway radius of curvature on tangent point rotation

        # elif calling_point == "hold":
        #     print(f"Calling point hold!")
        #     pass

    def get_dict_rep(self):
        return {"curve_joint": self.curve_joint.name(), "name": self.name(), "constraint_type": "curvature"}


class ConstraintValidationError(Exception):
    pass


class NoSolutionError(Exception):
    pass
