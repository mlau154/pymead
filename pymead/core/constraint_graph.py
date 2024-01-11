import typing
from abc import abstractmethod
from copy import deepcopy

import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
import networkx

from pymead.core import constraint_equations as ceq
from pymead.core.constraints import PymeadRootFinder


class Param:
    def __init__(self, value: float, name: str):
        self._value = None
        self.set_value(value)
        self.name = name

    def value(self):
        return self._value

    def set_value(self, value: float):
        self._value = value

    def __repr__(self):
        return f"Param {self.name}<v={self.value()}>"


class Entity:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_params(self) -> typing.List[Param]:
        pass


class Point(Entity):
    def __init__(self, x: float, y: float, name: str, fixed: bool = False):
        self.x = Param(x, f"{name}.x")
        self.y = Param(y, f"{name}.y")
        self._fixed = fixed
        self.constraints = []
        super().__init__(name=name)

    def __repr__(self):
        return f"Point {self.name}<x={self.x.value()}, y={self.y.value()}>"

    def get_params(self) -> typing.List[Param]:
        return [self.x, self.y]

    def fixed(self):
        return self._fixed

    def set_fixed(self, fixed: bool):
        self._fixed = fixed


class Constraint:
    equations = None

    def __init__(self, value: float or None, name: str, child_nodes: list, kind: str):
        self.child_nodes = child_nodes
        self.param = Param(value, f"{name}.param") if value is not None else None
        self.name = name
        self.kind = kind
        self.data = None
        self.add_constraint_to_points()

    def add_constraint_to_points(self):
        for child_node in self.child_nodes:
            if not isinstance(child_node, Point) or self in child_node.constraints:
                return

            child_node.constraints.append(self)

    def remove_constraint_from_points(self):
        for child_node in self.child_nodes:
            if not isinstance(child_node, Point) or self not in child_node.constraints:
                return

            child_node.constraints.remove(self)

    @abstractmethod
    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        pass

    # @abstractmethod
    # def pick_starting_point(self) -> Point:
    #     pass


class ConstraintWeak:
    equations = None

    @abstractmethod
    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        pass

    def __init__(self, name: str):
        self.name = name


class DistanceConstraint(Constraint):
    equations = [staticmethod(ceq.distance_constraint)]

    def __init__(self, p1: Point, p2: Point, value: float, name: str):
        self.p1 = p1
        self.p2 = p2
        super().__init__(value=value, name=name, child_nodes=[self.p1, self.p2], kind="d")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.param), 2]
        ]]

    def pick_starting_point(self) -> Point:
        return self.p2

    def __repr__(self):
        return f"DistanceConstraint {self.name}<v={self.param.value()}>"


class AbsAngleConstraint(Constraint):
    equations = [staticmethod(ceq.abs_angle_constraint)]

    def __init__(self, p1: Point, p2: Point, value: float, name: str):
        self.p1 = p1
        self.p2 = p2
        super().__init__(value=value, name=name, child_nodes=[self.p1, self.p2], kind="a2")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.param), 2]
        ]]

    def __repr__(self):
        return f"AbsAngleConstraint {self.name}<v={self.param.value()}>"


class AbsAngleConstraintWeak(ConstraintWeak):
    equations = [staticmethod(ceq.abs_angle_constraint_weak)]

    def __init__(self, p1: Point, p2: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        super().__init__(name)

    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p1.x), int(use_intermediate)],
            [param_list.index(self.p1.y), int(use_intermediate)],
            [param_list.index(self.p2.x), int(use_intermediate)],
            [param_list.index(self.p2.y), int(use_intermediate)],
        ]]

    def __repr__(self):
        return f"AbsAngleConstraintWeak {self.name}"


class Parallel3Constraint(Constraint):
    equations = [staticmethod(ceq.parallel3_constraint)]

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(value=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p3.x), 2],
            [param_list.index(self.p3.y), 2]
        ]]

    def __repr__(self):
        return f"Parallel3Constraint {self.name}"


class AntiParallel3Constraint(Constraint):
    equations = [staticmethod(ceq.antiparallel3_constraint)]

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(value=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p3.x), 2],
            [param_list.index(self.p3.y), 2]
        ]]

    def __repr__(self):
        return f"AntiParallel3Constraint {self.name}"


class Parallel4Constraint(Constraint):
    equations = [staticmethod(ceq.parallel4_constraint)]

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(value=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p3.x), 2],
            [param_list.index(self.p3.y), 2],
            [param_list.index(self.p4.x), 2],
            [param_list.index(self.p4.y), 2]
        ]]

    def __repr__(self):
        return f"Parallel4Constraint {self.name}"


class AntiParallel4Constraint(Constraint):
    equations = [staticmethod(ceq.antiparallel4_constraint)]

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(value=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p3.x), 2],
            [param_list.index(self.p3.y), 2],
            [param_list.index(self.p4.x), 2],
            [param_list.index(self.p4.y), 2]
        ]]

    def __repr__(self):
        return f"AntiParallel4Constraint {self.name}"


class SymmetryConstraint(Constraint):
    equations = [staticmethod(ceq.perp4_constraint), staticmethod(ceq.points_equidistant_from_line_constraint_signed)]

    def __init__(self, p1: Point, p2: Point, p3: Point, p4: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        super().__init__(value=None, name=name, child_nodes=[self.p1, self.p2, self.p3, self.p4], kind="a4|d")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [
            [
                [param_list.index(self.p1.x), 2],
                [param_list.index(self.p1.y), 2],
                [param_list.index(self.p2.x), 2],
                [param_list.index(self.p2.y), 2],
                [param_list.index(self.p3.x), 2],
                [param_list.index(self.p3.y), 2],
                [param_list.index(self.p4.x), 2],
                [param_list.index(self.p4.y), 2]
            ],
            [
                [param_list.index(self.p1.x), 2],
                [param_list.index(self.p1.y), 2],
                [param_list.index(self.p2.x), 2],
                [param_list.index(self.p2.y), 2],
                [param_list.index(self.p3.x), 2],
                [param_list.index(self.p3.y), 2],
                [param_list.index(self.p4.x), 2],
                [param_list.index(self.p4.y), 2]]
        ]

    def __repr__(self):
        return f"ParallelConstraint {self.name}"


class Perp3Constraint(Constraint):
    equations = [staticmethod(ceq.perp3_constraint)]

    def __init__(self, p1: Point, p2: Point, p3: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        super().__init__(value=None, name=name, child_nodes=[self.p1, self.p2, self.p3], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p3.x), 2],
            [param_list.index(self.p3.y), 2]
        ]]

    def __repr__(self):
        return f"Perp3Constraint {self.name}"


class PointOnLineConstraint(Constraint):
    equations = [staticmethod(ceq.point_on_line_constraint)]

    def __init__(self, point: Point, line_start: Point, line_end: Point, name: str):
        self.point = point
        self.line_start = line_start
        self.line_end = line_end
        super().__init__(value=None, name=name, child_nodes=[self.point, self.line_start, self.line_end], kind="a3")

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [[
            [param_list.index(self.point.x), 2],
            [param_list.index(self.point.y), 2],
            [param_list.index(self.line_start.x), 2],
            [param_list.index(self.line_start.y), 2],
            [param_list.index(self.line_end.x), 2],
            [param_list.index(self.line_end.y), 2]
        ]]

    def __repr__(self):
        return f"PointOnLineConstraint {self.name}"


class EquationData:
    def __init__(self):
        self.root_finder = None
        self.constraints = []
        self.equations = []
        self.arg_idx_array = []
        self.variable_pos = []

    def clear(self):
        self.root_finder = None
        self.constraints.clear()
        self.equations.clear()
        self.arg_idx_array.clear()
        self.variable_pos.clear()


class ConstraintGraph(networkx.Graph):
    def __init__(self, gdict: dict = None, cdict: dict = None):
        self.gdict = gdict if gdict is not None else {}
        self.cdict = cdict if cdict is not None else {}
        self.root_finders = {}
        self.constraints = {}
        self.equations = {}
        self.arg_idx_arrays = {}
        self.variable_pos = {}
        self.subgraphs = []
        self.points = []
        super().__init__()

    def add_point(self, point: Point):
        self.add_node(point)
        self.points.append(point)

    def add_constraint(self, constraint: Constraint):
        self.add_node(constraint)
        for child_node in constraint.child_nodes:
            self.add_edge(constraint, child_node)

        # Perform depth-first search on the nodes to find if any connected node has the "data" attr set
        for node in networkx.dfs_preorder_nodes(self, source=constraint):
            if isinstance(node, Constraint) and node.data is not None:
                constraint.data = node.data
                break
        else:  # If not, assign a new instance of the EquationData class to the "data" attr
            constraint.data = EquationData()

        print(f"Adding {constraint}...")

        self.analyze(constraint)

        # self.get_equation_set_for_entity_or_constraint(point, for_initial_solve=True)
        # self.compile_equation_for_entity_or_constraint(point)
        # x, info = self.initial_solve_for_entity_or_constraint(point)
        # self.verify_and_update_entity_or_constraint(point, x, info)

    def get_points_to_fix(self, source: Constraint) -> typing.List[Point]:

        earliest_point = None
        earliest_index = None

        points_to_fix = []

        for node in networkx.dfs_preorder_nodes(self, source=source):
            if not isinstance(node, Point):
                continue

            # If the node has already set by the user to be fixed, append this point to the list and continue
            if node.fixed():
                points_to_fix.append(node)
                continue

            # If this point was added earlier than "earliest_point," then set this point to "earliest_point"
            if earliest_point is None:
                earliest_point = node
                earliest_index = self.points.index(node)
            else:
                if self.points.index(node) > earliest_index:
                    continue
                earliest_point = node
                earliest_index = self.points.index(node)

        # If no fixed points were added by the user, we fix exactly one point (the earliest point) to satisfy
        # the required degrees of freedom for a rigid 2-D body
        if len(points_to_fix) == 0:
            points_to_fix = [earliest_point]

        return points_to_fix

    @staticmethod
    def fix_points(points_to_fix: typing.List[Point]):
        for point in points_to_fix:
            point.set_fixed(True)
        return points_to_fix

    @staticmethod
    def free_points(points_to_free: typing.List[Point]):
        for point in points_to_free:
            point.set_fixed(False)
        return points_to_free

    def analyze(self, constraint: Constraint):

        constraint.data.clear()
        params = self.get_params(source_node=constraint)
        print(f"{len(params) = }")
        points = self.get_points(source_node=constraint)
        points_to_fix = self.get_points_to_fix(source=constraint)
        fixed_points = self.fix_points(points_to_fix)
        strong_constraints = self.get_strong_constraints(source_node=constraint)
        strong_equations = self.get_strong_equations(strong_constraints, source_node=constraint)
        abs_angle_constraints = [cnstr for cnstr in strong_constraints if isinstance(cnstr, AbsAngleConstraint)]
        weak_constraints = []

        dof = 2 * len(points) - len(strong_equations)
        if len(fixed_points) == 0:
            dof -= 2
        else:
            dof -= 2 * len(fixed_points)

        # Only need to add an absolute angle constraint if one does not yet exist and if there is only one fixed point
        if len(abs_angle_constraints) == 0 and len(fixed_points) == 1:
            dof -= 1

        if dof < 0:
            raise OverConstrainedError("System is over-constrained")

        points_may_need_distance = []
        points_may_need_angle = []

        for node in networkx.dfs_preorder_nodes(self, source=constraint):
            if not isinstance(node, Point):
                continue

            if not any(["d" in cnstr.kind for cnstr in self.adj[node]]):
                if node not in points_may_need_distance:
                    points_may_need_distance.append(node)

            if not any(["a" in cnstr.kind for cnstr in self.adj[node]]):
                if node not in points_may_need_angle:
                    points_may_need_angle.append(node)

        # Determine the absolute angle constraint to add to eliminate the rotational degree of freedom, if necessary
        if len(abs_angle_constraints) == 0 and len(fixed_points) == 1:
            # descendants = networkx.descendants_at_distance(self, source=fixed_points[0], distance=2)
            # for descendant in descendants:
            #     # TODO: need to generalize this, possibly using a different technique than descendants_at_distance
            #     end_point = points[1]
            #     abs_angle_constraint = AbsAngleConstraintWeak(fixed_points[0], end_point, "aa1")
            #     weak_constraints.append(abs_angle_constraint)
            #     break
            start_point = fixed_points[0]
            if self.points[0] == start_point:
                end_point = self.points[1]
            else:
                end_point = self.points[0]
            abs_angle_constraint = AbsAngleConstraintWeak(start_point, end_point, "aa1")
            weak_constraints.append(abs_angle_constraint)

        weak_equations = []
        for cnstr in weak_constraints:
            weak_equations.extend(cnstr.equations)

        points_to_vary = [point for point in points if not point.fixed()]
        self.add_points_as_variables(constraint.data, points_to_vary, params=params)

        for eq in strong_equations:
            constraint.data.equations.append(eq)

        for cnstr in strong_constraints:
            # TODO: this means that we need to make the arg_idx_array for each constraint 3-D instead of 2-D
            constraint.data.arg_idx_array.extend(cnstr.get_arg_idx_array(params))
            constraint.data.constraints.append(cnstr)

        for eq in weak_equations:
            constraint.data.equations.append(eq)

        for cnstr in weak_constraints:
            print(f"{cnstr.p1 = }, {cnstr.p2 = }, {cnstr = }, yeet")
            constraint.data.arg_idx_array.extend(cnstr.get_arg_idx_array(params))
            constraint.data.constraints.append(cnstr)

        pass

    def verify_and_update_entity_or_constraint(self, entity_or_constraint: Entity or Constraint, x: np.ndarray,
                                               info):
        if np.any(info.fun_val > 1e-6):
            raise ValueError("Could not solve the constraint problem within tolerance after constraint addition")

        self.update_points(entity_or_constraint, x)

    def _add_constraint(self, vertex_pair: typing.Iterable[Entity], constraint: Constraint):
        # self._add_edge(vertex_pair)

        # Now, analyze the graph to form the equation set, solve the equation set, and update the points
        # point = constraint.pick_starting_point()
        # self.get_equation_set_for_entity_or_constraint(point, for_initial_solve=True)
        # self.compile_equation_for_entity_or_constraint(point)
        # x, info = self.initial_solve_for_entity_or_constraint(point)
        # self.verify_and_update_entity_or_constraint(point, x, info)

        # TODO: Note - might be able to avoid using jit on the initial compile for time savings

        # If the update was successful, compile all the equation sets for point movement
        for point in self.cdict:
            self.get_equation_set_for_constraint(point, for_initial_solve=False)
            self.compile_equation_for_entity_or_constraint(point)

    def move_point(self, point: Point, new_x: float, new_y: float):

        start_param_vec = self.get_param_values()

        point.x.set_value(new_x)
        point.y.set_value(new_y)

        intermediate_param_vec = self.get_param_values()

        x, info = self.final_solve_for_entity_or_constraint(point, start_param_vec, intermediate_param_vec)

        self.verify_and_update_entity_or_constraint(point, x, info)

    def get_strong_constraints(self, source_node: Point or Constraint):

        strong_constraints = []
        for node in networkx.dfs_preorder_nodes(self, source=source_node):
            if isinstance(node, Constraint):
                strong_constraints.append(node)

        return strong_constraints

    def get_strong_equations(self, strong_constraints: typing.List[Constraint] = None,
                             source_node: Point or Constraint = None):
        strong_equations = []

        strong_constraints = self.get_strong_constraints(
            source_node) if strong_constraints is None else strong_constraints

        for constraint in strong_constraints:
            strong_equations.extend(constraint.equations)

        return strong_equations

    def get_points(self, source_node: Point or Constraint):
        points = []

        for node in networkx.dfs_preorder_nodes(self, source=source_node):
            if not isinstance(node, Point):
                continue

            points.append(node)

        return points

    def get_params(self, source_node: Point or Constraint):
        params = []

        for node in networkx.dfs_preorder_nodes(self, source=source_node):
            if isinstance(node, Point):
                params.extend([node.x, node.y])
            elif isinstance(node, Constraint):
                if node.param is None:
                    continue
                params.append(node.param)

        return params

    def get_param_values(self, source_node: Point or Constraint):
        return np.array([p.value() for p in self.get_params(source_node)])

    @staticmethod
    def add_variable(equation_data: EquationData, variable: Param,
                     params: typing.List[Param]):

        var_index = params.index(variable)

        if var_index not in equation_data.variable_pos:
            equation_data.variable_pos.append(var_index)

    def add_points_as_variables(self, equation_data: EquationData, points: typing.List[Point],
                                params: typing.List[Param]):
        param_list = []
        for point in points:
            param_list.extend([point.x, point.y])

        for param in param_list:
            self.add_variable(equation_data, param, params)

    def get_equation_set_for_constraint(self, constraint: Constraint,
                                        params: typing.List[Param] = None,
                                        strong_constraints: typing.List[Constraint] = None,
                                        for_initial_solve: bool = True):

        params = self.get_params(strong_constraints) if params is None else params

        # Clear the equation data
        constraint.data.clear()

        def add_var(start: Point, param: Param, dof: int):
            self.add_variable(start, param, params)
            return dof + 1

        def add_vars(start: Point, param_list: typing.List[Param], dof: int):

            for param in param_list:
                dof = add_var(start, param, dof)

            return dof

        def add_strong_cnstr(cnstr: Constraint, dof: int, strong_added: int):
            constraint.data.constraints.append(cnstr)
            constraint.data.equations.append(cnstr.equations)
            constraint.data.arg_idx_array.append(cnstr.get_arg_idx_array(params))
            return dof - 1, strong_added + 1

        def add_strong_cnstrs(cnstrs: typing.List[Constraint], dof: int, strong_added: int):

            for cnstr in cnstrs:
                dof, strong_added = add_strong_cnstr(cnstr, dof, strong_added)

            return dof, strong_added

        def add_weak_cnstr(cnstr: ConstraintWeak, start: Point, dof: int, use_intermediate: bool = False):
            self.constraints[start].append(cnstr)
            self.equations[start].append(cnstr.equations)
            self.arg_idx_arrays[start].append(cnstr.get_arg_idx_array(params, use_intermediate=use_intermediate))
            return dof - 1

        visited_nodes, visited_edges, queue = [], [], []

        def bfs(starting_node):  # breadth-first search: time complexity O(V+E)
            visited_nodes.append(starting_node)
            queue.append(starting_node)

            dof = 0
            strong_equations_added = 0
            starting_point_added = False

            while queue:  # Creating loop to visit each node
                next_point = queue.pop(0)

                # Add the starting point variables if they have not yet been added
                if not starting_point_added:
                    dof = add_vars(starting_node, [next_point.x, next_point.y], dof)
                    starting_point_added = True

                # Loop through each node adjacent to the current queue node being analyzed
                for neighbor in self.cdict[next_point]:
                    if neighbor not in visited_nodes:
                        visited_nodes.append(neighbor)
                        visited_edges.append({next_point.name, neighbor.name})
                        queue.append(neighbor)

                        # Add the strong constraint equations
                        constraints = self.cdict[next_point][neighbor]
                        dof, strong_equations_added = add_strong_cnstrs(
                            constraints, starting_node, dof, strong_equations_added
                        )

                        if len(visited_nodes) == 2:  # For only the second node analyzed in the entire graph,
                            if len(constraints) == 1 and isinstance(constraints[0], DistanceConstraint):
                                cnstr = AbsAngleConstraintWeak(next_point, neighbor, "a1")
                                dof = add_weak_cnstr(cnstr, starting_node, dof, use_intermediate=not for_initial_solve)

                        elif len(visited_nodes) > 2:  # For all nodes other than the first two,
                            if len(constraints) == 1 and isinstance(constraints[0], DistanceConstraint):
                                cnstr = AbsAngleConstraintWeak(next_point, neighbor, f"a{len(visited_nodes) - 1}")
                                dof = add_weak_cnstr(cnstr, starting_node, dof)
                                dof = add_vars(starting_node, [neighbor.x, neighbor.y], dof)
                    else:  # Even if this node has already been visited, check to see if it forms a cycle
                        if {next_point.name, neighbor.name} not in visited_edges:  # (cycle detected)

                            # TODO: this code currently only works for triangle. Will need to extend this to cycle
                            #  backward along the closed loop for the general polygon case
                            if isinstance(self.constraints[starting_node][-1], ConstraintWeak):
                                # Delete the previous weak constraint
                                self.constraints[starting_node].pop()
                                self.equations[starting_node].pop()
                                self.arg_idx_arrays[starting_node].pop()
                                dof += 1

                                # Add the strong constraint associated with this loop closure edge
                                print(f"{next_point = }, {neighbor = }")
                                constraints = self.cdict[next_point][neighbor]
                                if len(constraints) != 1:
                                    raise ValueError("Overconstrained")
                                dof, strong_equations_added = add_strong_cnstrs(
                                    constraints, starting_node, dof, strong_equations_added
                                )
                            else:
                                raise ValueError("Overconstrained!")
                            visited_edges.append({next_point.name, neighbor.name})

                    if dof == 0 and strong_equations_added == len(self.get_strong_equations()):

                        if (set((frozenset(edge) for edge in visited_edges)) !=
                                set((frozenset(edge) for edge in self.edge_names()))):
                            raise ValueError("Traversed the graph without visiting all the edges")

                        return

            if dof != 0:
                raise ValueError("Traversed the entire graph, but there are still a non-zero number of "
                                 "degrees of freedom")

        # bfs(point)

    @staticmethod
    def compile_equation_for_entity_or_constraint(constraint: Constraint, method: str = "lm"):

        def equation_system(x: np.ndarray,
                            start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
            final_param_vec = jnp.array([v for v in intermediate_param_vec])
            for idx, pos in enumerate(constraint.data.variable_pos):
                final_param_vec = final_param_vec.at[pos].set(x[idx])

            func_outputs = jnp.array([cnstr(*[start_param_vec[w[0]] if w[1] == 0
                                              else intermediate_param_vec[w[0]] if w[1] == 1
                                              else final_param_vec[w[0]] for w in arg_idx_array])
                                      for cnstr, arg_idx_array in zip(constraint.data.equations,
                                                                      constraint.data.arg_idx_array)
                                      ])

            return func_outputs

        constraint.data.root_finder = PymeadRootFinder(jit(equation_system), method=method)

    def initial_solve(self, constraint: Constraint):
        params = self.get_params(constraint)
        x0 = np.array([params[x_pos].value() for x_pos in constraint.data.variable_pos])
        v = np.array([p.value() for p in params])
        w = deepcopy(v)
        x, info = constraint.data.root_finder.solve(x0, v, w)
        return x, info

    def final_solve_for_entity_or_constraint(self, entity_or_constraint: Entity or Constraint,
                                             start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
        params = self.get_params()
        x0 = np.array([params[x_pos].value() for x_pos in self.variable_pos[entity_or_constraint]])
        print(f"{[params[x_pos] for x_pos in self.variable_pos[entity_or_constraint]] = }")
        x, info = self.root_finders[entity_or_constraint].solve(x0, start_param_vec, intermediate_param_vec)
        return x, info

    def update_points(self, constraint: Constraint, new_x: np.ndarray):
        """
        Updates the variable points and parameters in the constraint system

        Returns
        -------

        """
        params = self.get_params(constraint)
        for idx, x in zip(constraint.data.variable_pos, new_x):
            params[idx].set_value(x)


class OverConstrainedError(Exception):
    pass


def main():
    import matplotlib.pyplot as plt

    p1 = Point(0.0, 0.0, "p1")
    p2 = Point(0.3, 0.3, "p2")
    p3 = Point(0.4, 0.6, "p3")
    p4 = Point(0.8, -0.1, "p4")
    p5 = Point(0.2, 0.6, "p5")
    points = [p1, p2, p3, p4, p5]

    original_x = [p.x.value() for p in points]
    original_y = [p.y.value() for p in points]
    plt.plot(original_x, original_y, ls="none", marker="o", color="indianred")
    plt.gca().set_aspect("equal")

    cnstr1 = DistanceConstraint(p1, p2, 1.0, "d1")
    cnstr2 = DistanceConstraint(p2, p3, 0.8, "d2")
    cnstr3 = DistanceConstraint(p2, p4, 0.4, "d3")
    cnstr4 = DistanceConstraint(p1, p5, 1.2, "d4")
    cnstr5 = DistanceConstraint(p3, p1, 0.6, "d5")
    graph = ConstraintGraph()

    graph.add_constraint(cnstr1)
    graph.add_constraint(cnstr2)
    graph.add_constraint(cnstr3)
    graph.add_constraint(cnstr4)
    graph.add_constraint(cnstr5)

    print(f"{np.hypot(p1.x.value() - p2.x.value(), p1.y.value() - p2.y.value())}")
    print(f"{np.hypot(p3.x.value() - p2.x.value(), p3.y.value() - p2.y.value())}")
    print(f"{np.hypot(p1.x.value() - p3.x.value(), p1.y.value() - p3.y.value())}")

    print(f"{p2 = }")

    # new_x = [p.x.value() for p in points]
    # new_y = [p.y.value() for p in points]
    # plt.plot(new_x, new_y, ls="none", marker="s", color="mediumaquamarine", mfc="#aaaaaa88")
    # plt.show()
    #
    # graph.move_point(p2, 0.8, 0.6)
    #
    # print(f"{np.hypot(p1.x.value() - p2.x.value(), p1.y.value() - p2.y.value())}")
    # print(f"{np.hypot(p3.x.value() - p2.x.value(), p3.y.value() - p2.y.value())}")
    # print(f"{np.hypot(p1.x.value() - p3.x.value(), p1.y.value() - p3.y.value())}")

    new_x = [p.x.value() for p in points]
    new_y = [p.y.value() for p in points]
    plt.plot(original_x, original_y, ls="none", marker="o", color="indianred")
    plt.plot(new_x, new_y, ls="none", marker="s", color="steelblue", mfc="#aaaaaa88")
    plt.show()

    pass


def main2():
    import matplotlib.pyplot as plt
    # Initialize the graph
    g = ConstraintGraph()

    # Add the points
    p1 = Point(0.0, 0.0, "p1")
    p2 = Point(0.4, 0.1, "p2")
    p3 = Point(0.6, 0.0, "p3")
    p4 = Point(0.2, 0.8, "p4")
    p5 = Point(1.0, 1.3, "p5")
    points = [p1, p2, p3, p4, p5]
    for point in points:
        g.add_point(point)

    plt.plot([p.x.value() for p in points], [p.y.value() for p in points], ls="none", marker="o", mec="indianred", mfc="indianred", fillstyle="left", markersize=10)

    # Add the constraints
    d1 = DistanceConstraint(p1, p2, 0.2, "d1")
    d2 = DistanceConstraint(p2, p4, 1.5, "d2")
    d3 = DistanceConstraint(p2, p3, 0.5, "d3")
    d4 = DistanceConstraint(p3, p5, 1.4, "d4")
    perp3 = Perp3Constraint(p1, p2, p4, "L1")
    parl = Parallel4Constraint(p2, p4, p3, p5, "//1")
    aparl3 = AntiParallel3Constraint(p1, p2, p3, "ap1")
    constraints = [d1, d2, d3, d4, perp3, parl, aparl3]
    for constraint in constraints:
        g.add_constraint(constraint)

    g.compile_equation_for_entity_or_constraint(aparl3)
    x, info = g.initial_solve(aparl3)
    print(f"{info = }")
    g.update_points(aparl3, new_x=x)

    d4.param.set_value(4.0)
    # g.compile_equation_for_entity_or_constraint(aparl3)
    x, info = g.initial_solve(aparl3)
    print(f"{info = }")
    g.update_points(aparl3, new_x=x)

    d4.param.set_value(5.0)
    # g.compile_equation_for_entity_or_constraint(aparl3)
    x, info = g.initial_solve(aparl3)
    print(f"{info = }")
    g.update_points(aparl3, new_x=x)

    plt.plot([p.x.value() for p in points], [p.y.value() for p in points], ls="none", marker="o", mfc="steelblue", mec="steelblue", fillstyle="right", markersize=10)
    for p in points:
        plt.text(p.x.value() + 0.02, p.y.value() + 0.02, p.name)
    plt.gca().set_aspect("equal")

    plt.show()

    labels = {p: p.name for p in g.nodes}
    subax1 = plt.subplot(121)
    networkx.draw(g, with_labels=True, labels=labels)

    # for node in networkx.dfs_preorder_nodes(g, source=p1):
    #     if isinstance(node, Constraint):
    #         print(f"{node = }")

    # cycle_basis = networkx.cycle_basis(g)
    # # for c in cycle_basis:
    # #     print(f"{c = }")
    # simple_cycles = networkx.chordless_cycles(g)
    # for simple_cycle in simple_cycles:
    #     print(simple_cycle)

    plt.show()
    pass


def main3():
    import matplotlib.pyplot as plt
    # Initialize the graph
    g = ConstraintGraph()

    # Add the points
    p1 = Point(0.0, 0.0, "p1")
    p2 = Point(0.4, 0.1, "p2")
    p3 = Point(0.6, 0.0, "p3")
    points = [p1, p2, p3]
    for point in points:
        g.add_point(point)

    plt.plot([p.x.value() for p in points], [p.y.value() for p in points], ls="none", marker="o", mec="indianred", mfc="indianred", fillstyle="left", markersize=10)

    # Add the constraints
    d1 = DistanceConstraint(p1, p2, 0.6, "d1")
    d2 = DistanceConstraint(p2, p3, 1.0, "d2")
    d3 = DistanceConstraint(p1, p3, 0.8, "d3")
    constraints = [d1, d2, d3]
    for constraint in constraints:
        g.add_constraint(constraint)

    g.compile_equation_for_entity_or_constraint(d3)
    x, info = g.initial_solve(d3)
    print(f"{info = }")
    g.update_points(d3, new_x=x)

    # d4.param.set_value(2.0)
    # # g.compile_equation_for_entity_or_constraint(aparl3)
    # x, info = g.initial_solve(d3)
    # print(f"{info = }")
    # g.update_points(aparl3, new_x=x)

    plt.plot([p.x.value() for p in points], [p.y.value() for p in points], ls="none", marker="o", mfc="steelblue", mec="steelblue", fillstyle="right", markersize=10)
    for p in points:
        plt.text(p.x.value() + 0.02, p.y.value() + 0.02, p.name)
    plt.gca().set_aspect("equal")

    plt.show()


def main4():
    import matplotlib.pyplot as plt
    # Initialize the graph
    g = ConstraintGraph()

    # Add the points
    p1 = Point(0.0, 0.0, "p1")
    p2 = Point(1.0, 0.0, "p2")
    p3 = Point(0.25, 0.05, "p3")
    # p4 = Point(0.5, 0.03, "p4")
    points = [p1, p2, p3]
    for point in points:
        g.add_point(point)

    plt.plot([p.x.value() for p in points], [p.y.value() for p in points], ls="none", marker="o", mec="indianred",
             mfc="indianred", fillstyle="left", markersize=10)

    # Add the constraints
    d1 = DistanceConstraint(p1, p2, 1.0, "d1")
    d2 = DistanceConstraint(p1, p3, 0.2, "d2")
    pol = PointOnLineConstraint(p3, p1, p2, "pol1")
    # d3 = DistanceConstraint(p1, p3, 0.8, "d3")
    constraints = [d1, d2, pol]
    for constraint in constraints:
        g.add_constraint(constraint)

    g.compile_equation_for_entity_or_constraint(pol)
    x, info = g.initial_solve(pol)
    print(f"{info = }")
    g.update_points(pol, new_x=x)

    # d4.param.set_value(2.0)
    # # g.compile_equation_for_entity_or_constraint(aparl3)
    # x, info = g.initial_solve(d3)
    # print(f"{info = }")
    # g.update_points(aparl3, new_x=x)

    plt.plot([p.x.value() for p in points], [p.y.value() for p in points], ls="none", marker="o", mfc="steelblue",
             mec="steelblue", fillstyle="right", markersize=10)
    for p in points:
        plt.text(p.x.value() + 0.02, p.y.value() + 0.02, p.name)
    plt.gca().set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main4()
