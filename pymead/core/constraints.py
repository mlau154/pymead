import typing
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from jax import jit
from jax import numpy as jnp
import jaxopt

from pymead.core import UNITS
from pymead.core import constraint_equations as ceq
from pymead.core.param2 import Param, AngleParam, LengthParam
from pymead.core.point import PointSequence, Point
from pymead.core.pymead_obj import PymeadObj


class PymeadRootFinder(jaxopt.ScipyRootFinding):
    def __init__(self, equation_system: typing.Callable):
        super().__init__(
            method="hybr",
            jit=True,
            has_aux=False,
            optimality_fun=equation_system,
            tol=1e-6,
            use_jacrev=False,  # Use the forward Jacobian calculation since the matrix is square
        )

    def solve(self, x0: np.ndarray, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
        """
        Solves the compiled non-linear system of equations for this ``Point`` using Jax's wrapper for
        ``scipy.optimize.root``.

        Parameters
        ----------
        x0: np.ndarray
            Initial guess for the solution to the system of equations

        start_param_vec: np.ndarray
            The parameter vector

        intermediate_param_vec: np.ndarray
            The parameter vector

        Returns
        -------
        np.ndarray, dict
            A two-element tuple containing the new parameter vector and information about the solution state
        """
        return self.run(x0, start_param_vec, intermediate_param_vec)


class GeoCon(PymeadObj):

    constraint_equation = None

    def __init__(self, tool: PointSequence or Point or None, target: PointSequence or Point or None,
                 name: str or None = None, weak: bool = False):
        self._tool = None
        self._target = None
        self.weak = weak
        sub_container = "geocon_weak" if weak else "geocon"
        super().__init__(sub_container=sub_container)
        self.set_name(name)
        self.set_tool(tool)
        self.set_target(target)
        self.points = None
        self.params = None
        self.gcs_list = None
        self.possible_weak_constraints = None

    def tool(self):
        return self._tool

    def target(self):
        return self._target

    def set_tool(self, tool: PointSequence or Point or None):
        self._tool = tool

    def set_target(self, target: PointSequence or Point or None):
        self._target = target

    def get_unique_point_list(self) -> typing.List[Point]:
        unique_point_list = []

        if self.tool() is not None:
            # Add the tool point(s) to the list
            if isinstance(self.tool(), Point):
                unique_point_list.append(self.tool())
            elif isinstance(self.tool(), PointSequence):
                unique_point_list.extend(self.tool().points())

        if self.target() is not None:
            # Add the target point(s) to the list
            if isinstance(self.target(), Point):
                unique_point_list.append(self.target())
            elif isinstance(self.target(), PointSequence):
                unique_point_list.extend(self.target().points())

        return unique_point_list

    def compare_point_set(self, other: "GeoCon"):
        return set(self.get_unique_point_list()) == set(other.get_unique_point_list())

    @abstractmethod
    def add_constraint_to_gcs(self):
        pass

    @abstractmethod
    def precompile(self):
        pass

    @abstractmethod
    def solve_and_update(self):
        pass

    @abstractmethod
    def recompile(self):
        pass


class GCS:

    # Unconstrained = 0
    # UnderConstrained = 1
    # FullyConstrained = 2
    # OverConstrained = 3

    def __init__(self, parent: Point or Param):
        self.parent = parent
        self.parent.gcs = self
        self.root_finder = None
        self.strong_constraint_types = []
        self.weak_constraint_types = []
        self.strong_constraints = []
        self.weak_constraints = []
        self.strong_constraint_equations = []
        self.weak_constraint_equations = []
        self.sub_pos = []
        self.weak_arg_indices = []
        self.weak_constraint_generators = []
        self.points = []
        self.params = []
        self.original_data = []

    # def check_constraint_state(self):
    #     if len(self.variables) == 0:
    #         return self.Unconstrained
    #     elif len(self.variables)

    def get_dof(self):
        return len(self.params) - len(self.strong_constraint_equations)

    def get_strong_weak_dof(self):
        return len(self.params) - len(self.strong_constraint_equations) - len(self.weak_constraint_equations)

    def compile_equation_set(self):

        def equation_system(x: np.ndarray, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
            # Evaluate the strong constraints using the updated parameter vector
            constraints = [cnstr(*x[sp]) for cnstr, sp in zip(self.strong_constraint_equations, self.sub_pos)]

            # Evaluate the weak constraints (functions that are simply used to keep the system fully constrained
            # according to a set of rules and can be overridden by strong constraints)
            weak_constraints = [cnstr(*[start_param_vec[w[0]] if w[1] == 0
                                        else intermediate_param_vec[w[0]] if w[1] == 1
                                        else x[w[0]] for w in weak_args])
                                for cnstr, weak_args in zip(self.weak_constraint_equations, self.weak_arg_indices)]

            # Combine the lists of strong and weak constraints
            constraints.extend(weak_constraints)

            return jnp.array(constraints)

        self.root_finder = PymeadRootFinder(jit(equation_system))

    def solve(self, start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
        """
        Solves the compiled non-linear system of equations for this ``Point`` using Jax's wrapper for
        ``scipy.optimize.root``.

        Parameters
        ----------
        start_param_vec: np.ndarray
            The parameter vector

        intermediate_param_vec: np.ndarray
            The parameter vector

        Returns
        -------
        np.ndarray, dict
            A two-element tuple containing the new parameter vector and information about the solution state
        """
        return self.root_finder.run(intermediate_param_vec, start_param_vec, intermediate_param_vec)

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

    # def add_fixed_x_constraint(self, p: Point, x: LengthParam):
    #     self.strong_constraint_equations.append(ceq.fixed_x_constraint)
    #     self.append_points([p])
    #     self.append_params([p.x(), x])
    #     self.constraint_types.append("FixedX")
    #
    # def add_fixed_x_constraint_weak(self, p: Point, x: LengthParam):
    #     self.weak_constraint_equations.append(ceq.fixed_x_constraint_weak)
    #     self.append_points([p])
    #     self.append_params([p.x(), x])
    #     self.constraint_types.append("FixedXWeak")
    #
    # def add_fixed_y_constraint(self, p: Point, y: LengthParam):
    #     self.strong_constraint_equations.append(ceq.fixed_y_constraint)
    #     self.append_points([p])
    #     self.append_params([p.y(), y])
    #     self.constraint_types.append("FixedY")

    def _add_strong_constraint(self, geo_con: GeoCon):

        if geo_con.weak:
            raise ValueError("Tried to add a weak constraint using _add_strong_constraint()")

        # Check if there is already a constraint of the same type present in this GCS that also contains the
        # same point set as the constraint we are trying to add. If so, raise an error.
        for cnstr in self.strong_constraints:
            if geo_con.compare_point_set(cnstr):
                raise DuplicateConstraintError(f"Duplicate constraint detected. Current constraint {geo_con.name()} "
                                               f"has the same point set as an existing constraint ({cnstr.name()}).")

        self.strong_constraints.append(geo_con)
        self.strong_constraint_equations.append(geo_con.constraint_equation)
        self.append_points(geo_con.points)
        self.append_params(geo_con.params)
        self.strong_constraint_types.append(geo_con.__class__.__name__)

    def _remove_strong_constraint(self, geo_con: GeoCon):

        if geo_con.weak:
            raise ValueError("Attempted to remove a weak constraint using _remove_strong_constraint()")

        removal_index = self.strong_constraints.index(geo_con)
        self.strong_constraints.pop(removal_index)
        self.strong_constraint_equations.pop(removal_index)

        # TODO: need logic for removing associated params and points if necessary, as well as the appropriate
        #  weak constraints

        self.strong_constraint_types.pop(removal_index)

    def _add_weak_constraint(self, geo_con: GeoCon):
        if not geo_con.weak:
            raise ValueError("Tried to add a strong constraint using _add_weak_constraint()")

        self.weak_constraints.append(geo_con)
        self.weak_constraint_equations.append(geo_con.constraint_equation)
        self.weak_constraint_types.append(geo_con.__class__.__name__)

    def _remove_weak_constraint(self, geo_con: GeoCon):

        if not geo_con.weak:
            raise ValueError("Attempted to remove a strong constraint using _remove_weak_constraint()")

        removal_index = self.weak_constraints.index(geo_con)
        self.weak_constraints.pop(removal_index)
        self.weak_constraint_equations.pop(removal_index)
        self.weak_constraint_types.pop(removal_index)

    def _add_or_remove_weak_constraints(self, geo_con_strong: GeoCon):

        weak_constraint_counter = 0
        while self.get_strong_weak_dof() != 0 and weak_constraint_counter < 1000:

            weak_constraint_counter += 1

            if self.get_strong_weak_dof() < 0:

                if weak_constraint_counter == 1:
                    priority_constraint_removed = False
                    priority_for_removal = None
                    if "distance" in self.strong_constraint_types[-1].lower():
                        priority_for_removal = "distance"
                    if "angle" in self.strong_constraint_types[-1].lower():
                        priority_for_removal = "angle"

                    if priority_for_removal is not None:
                        for idx, cnstr in enumerate(self.weak_constraints):
                            if (priority_for_removal in self.weak_constraint_types[idx] and
                                    geo_con_strong.compare_point_set(cnstr)):
                                self._remove_weak_constraint(cnstr)
                                priority_constraint_removed = True
                                break

                    if priority_constraint_removed:
                        continue

                    for cnstr in self.weak_constraints[::-1]:
                        if cnstr in geo_con_strong.possible_weak_constraints:
                            self._remove_weak_constraint(cnstr)
                            break
                else:
                    for cnstr in self.weak_constraints[::-1]:
                        if cnstr in geo_con_strong.possible_weak_constraints:
                            self._remove_weak_constraint(cnstr)
                            break

            elif self.get_strong_weak_dof() > 0:
                for cnstr in geo_con_strong.possible_weak_constraints:
                    if cnstr not in self.weak_constraints:
                        # TODO: also need to add a check to see if the fixed param constraint params are the same
                        self._add_weak_constraint(cnstr)
                        break

        if self.get_strong_weak_dof() != 0:
            raise MaxWeakConstraintAttemptsError("Reached maximum number of attempts to add or remove weak constraints "
                                                 "to achieve 0 degrees of freedom")

    def add_constraint(self, geo_con: GeoCon):

        # First, attempt to add the strong constraint to the GCS (early termination is possible here)
        self._add_strong_constraint(geo_con)

        # Then, add or remove weak constraints in a logical order until the sum of strong constraint equations
        # and weak constraint equations is exactly equal to the number of parameters (variables)
        self._add_or_remove_weak_constraints(geo_con)

    @staticmethod
    def compile_constraint(geo_con: GeoCon):

        # Pre-compile each GCS created by the constraint
        print(f"Pre-compiling...")
        geo_con.precompile()

        # Solve the GCS for the constraint and update the points
        print(f"Solving and updating...")
        geo_con.solve_and_update()

        # Re-compile each GCS created by the constraint according to a different set of rules than the initial solve
        geo_con.recompile()

    def add_distance_constraint(self, p1: Point, p2: Point, dist: LengthParam):
        self.strong_constraint_equations.append(ceq.distance_constraint)
        self.append_points([p1, p2])
        self.append_params([p1.x(), p1.y(), p2.x(), p2.y(), dist])
        self.constraint_types.append("Distance")

        if self.parent is p1:
            use_fixed_x = False
            if self.get_strong_weak_dof() >= 3:
                self.weak_constraint_equations.append(ceq.fixed_x_constraint_weak)
                use_fixed_x = True
            use_fixed_y = False
            if (use_fixed_x and self.get_strong_weak_dof() >= 2) or self.get_strong_weak_dof() >= 3:
                self.weak_constraint_equations.append(ceq.fixed_y_constraint_weak)
                use_fixed_y = True
            self.weak_constraint_equations.append(ceq.fixed_param_constraint_weak)

            if use_fixed_x:
                self.weak_arg_indices.append([[self.params.index(p2.x()), 1], [self.params.index(p2.x()), 0]])
            if use_fixed_y:
                self.weak_arg_indices.append([[self.params.index(p2.y()), 1], [self.params.index(p2.y()), 0]])
            self.weak_arg_indices.append([[self.params.index(dist), 1], [self.params.index(dist), 0]])

            self.compile_equation_set()

        elif self.parent in [p2, dist]:
            use_fixed_x = False
            if self.get_strong_weak_dof() >= 3:
                self.weak_constraint_equations.append(ceq.fixed_x_constraint_weak)
                use_fixed_x = True
            use_fixed_y = False
            if (use_fixed_x and self.get_strong_weak_dof() >= 2) or self.get_strong_weak_dof() >= 3:
                self.weak_constraint_equations.append(ceq.fixed_y_constraint_weak)
                use_fixed_y = True
            self.weak_constraint_equations.append(ceq.fixed_param_constraint_weak)
            self.weak_constraint_equations.append(ceq.abs_angle_constraint_weak)

            if use_fixed_x:
                self.weak_arg_indices.append([[self.params.index(p1.x()), 2], [self.params.index(p1.x()), 0]])
            if use_fixed_y:
                self.weak_arg_indices.append([[self.params.index(p1.y()), 2], [self.params.index(p1.y()), 0]])
            self.weak_arg_indices.append([[self.params.index(dist), 2], [self.params.index(dist), 0]])

            fixed_angle_indices = []
            for old_new_idx in [2, 0]:
                for p in [p2.x(), p2.y(), p1.x(), p1.y()]:
                    p_index = self.params.index(p)
                    fixed_angle_indices.append([p_index, old_new_idx])

            self.weak_arg_indices.append(fixed_angle_indices)

            start_param_vec = np.array([p.value() for p in self.params])
            intermediate_param_vec = deepcopy(start_param_vec)

            self.compile_equation_set()

            # Initial solve of the system equations to place the points in the correct starting location
            params, info = self.solve(start_param_vec, intermediate_param_vec)

            self.update(params)

            self.weak_arg_indices.pop()

            fixed_angle_indices = []
            for old_new_idx in [2, 1]:
                for p in [p1.x(), p1.y(), p2.x(), p2.y()]:
                    p_index = self.params.index(p)
                    fixed_angle_indices.append([p_index, old_new_idx])

            self.weak_arg_indices.append(fixed_angle_indices)

            self.compile_equation_set()

    def add_abs_angle_constraint(self, p1: Point, p2: Point, angle: AngleParam):
        self.strong_constraint_equations.append(ceq.abs_angle_constraint)
        self.append_points([p1, p2])
        self.append_params([p1.x(), p1.y(), p2.x(), p2.y(), angle])
        self.constraint_types.append("AbsAngle")

    def add_rel_angle3_constraint(self, p1: Point, p2: Point, p3: Point, angle: AngleParam):
        self.strong_constraint_equations.append(ceq.rel_angle3_constraint)
        self.append_points([p1, p2, p3])
        self.append_params([p1.x(), p1.y(), p2.x(), p2.y(), p3.x(), p3.y(), angle])
        self.constraint_types.append("RelAngle3")


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


# class Perp3Constraint(GeoCon):
#
#     constraint_equation = staticmethod(ceq.perp3_constraint)
#
#     def __init__(self, start_point: Point, vertex: Point, end_point: Point, name: str or None = None):
#         name = "PerpendicularCon-1" if name is None else name
#         super().__init__(tool=tool, target=target, name=name)
#
#     def add_constraint_to_gcs(self):
#         pass
#
#     def precompile(self):
#         pass
#
#     def solve_and_update(self):
#         pass
#
#     def recompile(self):
#         pass
#
#     def get_dict_rep(self):
#         return {"tool": [pt.name() for pt in self.tool().points()],
#                 "target": [pt.name() for pt in self.target().points()],
#                 "constraint_type": "perpendicular"}


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


class FixedParamConstraintWeak(GeoCon):

    constraint_equation = staticmethod(ceq.fixed_param_constraint_weak)

    def __init__(self, param: Param, name: str or None = None):
        name = "FixedParamConWeak-1" if name is None else name
        self.param = param

        super().__init__(tool=None, target=None, name=name, weak=True)

        self.points = []
        self.params = [param]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"param": self.param, "name": self.name(), "constraint_type": self.__class__.__name__}


class FixedXConstraint(GeoCon):

    constraint_equation = staticmethod(ceq.fixed_x_constraint)

    def __init__(self, point: Point, name: str or None = None):
        name = "FixedXCon-1" if name is None else name
        self.point = point

        super().__init__(tool=point, target=None, name=name)

        self.points = [point]
        self.params = [point.x(), point.y()]
        self.possible_weak_constraints = [FixedYConstraintWeak(point=self.point)]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.point.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class FixedXConstraintWeak(GeoCon):

    constraint_equation = staticmethod(ceq.fixed_x_constraint_weak)

    def __init__(self, point: Point, name: str or None = None):
        name = "FixedXConWeak-1" if name is None else name
        self.point = point

        super().__init__(tool=point, target=None, name=name, weak=True)

        self.points = [point]
        self.params = [point.x(), point.y()]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.point.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class FixedYConstraint(GeoCon):

    constraint_equation = staticmethod(ceq.fixed_y_constraint)

    def __init__(self, point: Point, name: str or None = None):
        name = "FixedYCon-1" if name is None else name
        self.point = point

        super().__init__(tool=point, target=None, name=name)

        self.points = [point]
        self.params = [point.x(), point.y()]
        self.possible_weak_constraints = [FixedXConstraintWeak(point=self.point)]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.point.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class FixedYConstraintWeak(GeoCon):

    constraint_equation = staticmethod(ceq.fixed_y_constraint_weak)

    def __init__(self, point: Point, name: str or None = None):
        name = "FixedYConWeak-1" if name is None else name
        self.point = point

        super().__init__(tool=point, target=None, name=name, weak=True)

        self.points = [point]
        self.params = [point.x(), point.y()]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.point.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class AbsAngleConstraint(GeoCon):

    constraint_equation = staticmethod(ceq.abs_angle_constraint)

    def __init__(self, start_point: Point, end_point: Point, angle_param: AngleParam or None = None,
                 name: str or None = None):
        name = "AbsAngleCon-1" if name is None else name
        self.start_point = start_point
        self.end_point = end_point

        super().__init__(tool=start_point, target=end_point, name=name)

        if angle_param is None:
            angle = self.tool().measure_angle(self.target())
            if self.geo_col is None:
                self.angle_param = AngleParam(value=angle, name="AbsAngle-1")
            else:
                self.angle_param = self.geo_col.add_param(value=angle, name="AbsAngle-1", unit_type="angle")
        else:
            self.angle_param = angle_param

        self.points = [start_point, end_point]
        self.params = [start_point.x(), start_point.y(), end_point.x(), end_point.y(), self.angle_param]
        self.possible_weak_constraints = [
            FixedParamConstraintWeak(self.angle_param),
            AbsAngleConstraintWeak(self.tool(), self.target()),
            FixedXConstraintWeak(self.tool()),
            FixedYConstraintWeak(self.tool()),
        ]

    def add_constraint_to_gcs(self):
        gcs1 = GCS(parent=self.start_point) if self.start_point.gcs is None else self.start_point.gcs
        gcs2 = GCS(parent=self.end_point) if self.end_point.gcs is None else self.end_point.gcs
        for gcs in (gcs1, gcs2):
            gcs.add_constraint(self)

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.start_point.name(), "target": self.end_point.name(),
                "angle_param": self.angle_param.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class AbsAngleConstraintWeak(GeoCon):

    constraint_equation = staticmethod(ceq.abs_angle_constraint_weak)

    def __init__(self, start_point: Point, end_point: Point, name: str or None = None):
        name = "AbsAngleConWeak-1" if name is None else name
        self.start_point = start_point
        self.end_point = end_point

        super().__init__(tool=start_point, target=end_point, name=name, weak=True)

        self.points = [start_point, end_point]
        self.params = [start_point.x(), start_point.y(), end_point.x(), end_point.y()]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.start_point.name(), "target": self.end_point.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class DistanceConstraint(GeoCon):

    constraint_equation = staticmethod(ceq.distance_constraint)

    def __init__(self, start_point: Point, end_point: Point, length_param: LengthParam or None = None,
                 name: str or None = None):
        name = "DistanceCon-1" if name is None else name
        self.start_point = start_point
        self.end_point = end_point

        super().__init__(tool=start_point, target=end_point, name=name)

        if length_param is None:
            distance = self.tool().measure_distance(self.target())
            if self.geo_col is None:
                self.length_param = LengthParam(value=distance, name="Length-1")
            else:
                self.length_param = self.geo_col.add_param(value=distance, name="Length-1", unit_type="length")
        else:
            self.length_param = length_param

        self.points = [start_point, end_point]
        self.params = [start_point.x(), start_point.y(), end_point.x(), end_point.y(), self.length_param]
        self.possible_weak_constraints = [
            FixedParamConstraintWeak(self.length_param),
            AbsAngleConstraintWeak(self.tool(), self.target()),
            FixedXConstraintWeak(self.tool()),
            FixedYConstraintWeak(self.tool()),
        ]

    def add_constraint_to_gcs(self):
        gcs1 = GCS(parent=self.start_point) if self.start_point.gcs is None else self.start_point.gcs
        gcs2 = GCS(parent=self.end_point) if self.end_point.gcs is None else self.end_point.gcs
        self.gcs_list = [gcs1, gcs2]
        for gcs in self.gcs_list:
            gcs.add_constraint(self)
        for gcs in self.gcs_list:
            gcs.compile_constraint(self)

    def precompile(self):

        for gcs in self.gcs_list:

            if self.possible_weak_constraints[0] in gcs.weak_constraints[-4:]:
                gcs.weak_arg_indices.append([[gcs.params.index(self.length_param), 2],
                                             [gcs.params.index(self.length_param), 0]])

            if self.possible_weak_constraints[1] in gcs.weak_constraints[-3:]:
                fixed_angle_indices = []
                for old_new_idx in [2, 0]:
                    for p in [self.target().x(), self.target().y(), self.tool().x(), self.tool().y()]:
                        p_index = gcs.params.index(p)
                        fixed_angle_indices.append([p_index, old_new_idx])
                gcs.weak_arg_indices.append(fixed_angle_indices)

            if self.possible_weak_constraints[2] in gcs.weak_constraints[-2:]:
                gcs.weak_arg_indices.append([[gcs.params.index(self.tool().x()), 2],
                                             [gcs.params.index(self.tool().x()), 0]])

            if self.possible_weak_constraints[3] in gcs.weak_constraints[-1:]:
                gcs.weak_arg_indices.append([[gcs.params.index(self.tool().y()), 2],
                                             [gcs.params.index(self.tool().y()), 0]])

            gcs.compile_equation_set()

    def solve_and_update(self):
        # This time use only the GCS for the target, since we only need solve the constraint problem once
        #gcs = self.target().gcs

        for gcs in self.gcs_list:

            start_param_vec = np.array([p.value() for p in gcs.params])
            intermediate_param_vec = deepcopy(start_param_vec)

            # Initial solve of the system equations to place the points in the correct starting location
            params, info = gcs.solve(start_param_vec, intermediate_param_vec)

            print(f"{info = }")

            gcs.update(params)

    def recompile(self):

        for gcs in self.gcs_list:

            if self.possible_weak_constraints[1] in gcs.weak_constraints[-3:]:

                abs_angle_index = gcs.weak_constraints.index(self.possible_weak_constraints[1])

                gcs.weak_arg_indices.pop(abs_angle_index)

                fixed_angle_indices = []
                for old_new_idx in [2, 1]:
                    for p in [self.tool().x(), self.tool().y(), self.target().x(), self.target().y()]:
                        p_index = gcs.params.index(p)
                        fixed_angle_indices.append([p_index, old_new_idx])
                gcs.weak_arg_indices.insert(abs_angle_index, fixed_angle_indices)

                gcs.compile_equation_set()

    def get_dict_rep(self):
        return {"tool": self.start_point.name(), "target": self.end_point.name(),
                "length_param": self.length_param.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class DistanceConstraintWeak(GeoCon):

    constraint_equation = staticmethod(ceq.distance_constraint_weak)

    def __init__(self, start_point: Point, end_point: Point, name: str or None = None):
        name = "DistanceConWeak-1" if name is None else name
        self.start_point = start_point
        self.end_point = end_point

        super().__init__(tool=start_point, target=end_point, name=name, weak=True)

        self.points = [start_point, end_point]
        self.params = [start_point.x(), start_point.y(), end_point.x(), end_point.y()]

    def add_constraint_to_gcs(self):
        pass

    def precompile(self):
        pass

    def solve_and_update(self):
        pass

    def recompile(self):
        pass

    def get_dict_rep(self):
        return {"tool": self.start_point.name(), "target": self.end_point.name(), "name": self.name(),
                "constraint_type": self.__class__.__name__}


class CollinearConstraint(GeoCon):
    def __init__(self, start_point: Point, middle_point: Point, end_point: Point, name: str or None = None):
        start_end_seq = PointSequence(points=[start_point, end_point])
        name = "CollinearCon-1" if name is None else name
        super().__init__(tool=middle_point, target=start_end_seq, name=name)
        self.start_point = start_point
        self.middle_point = middle_point
        self.end_point = end_point

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

    def get_dict_rep(self):
        return {"curve_joint": self.curve_joint.name(), "name": self.name(), "constraint_type": "curvature"}


class ConstraintValidationError(Exception):
    pass


class NoSolutionError(Exception):
    pass


class DuplicateConstraintError(Exception):
    pass


class MaxWeakConstraintAttemptsError(Exception):
    pass
