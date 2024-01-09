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
    def __init__(self, x: float, y: float, name: str):
        self.x = Param(x, f"{name}.x")
        self.y = Param(y, f"{name}.y")
        super().__init__(name=name)

    def __repr__(self):
        return f"Point {self.name}<x={self.x.value()}, y={self.y.value()}>"

    def get_params(self) -> typing.List[Param]:
        return [self.x, self.y]


class Constraint:

    equation = None

    def __init__(self, value: float, name: str):
        self.param = Param(value, f"{name}.param")
        self.name = name

    @abstractmethod
    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        pass

    @abstractmethod
    def pick_starting_point(self) -> Point:
        pass


class ConstraintWeak:

    equation = None

    @abstractmethod
    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        pass

    def __init__(self, name: str):
        self.name = name


class DistanceConstraint(Constraint):

    equation = staticmethod(ceq.distance_constraint)

    def __init__(self, p1: Point, p2: Point, value: float, name: str):
        self.p1 = p1
        self.p2 = p2
        super().__init__(value=value, name=name)

    def get_arg_idx_array(self, param_list: typing.List[Param]) -> list:
        return [
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.param), 2]
        ]

    def pick_starting_point(self) -> Point:
        return self.p2

    def __repr__(self):
        return f"DistanceConstraint {self.name}<v={self.param.value()}>"


class AbsAngleConstraintWeak(ConstraintWeak):

    equation = staticmethod(ceq.abs_angle_constraint_weak)

    def __init__(self, p1: Point, p2: Point, name: str):
        self.p1 = p1
        self.p2 = p2
        super().__init__(name)

    def get_arg_idx_array(self, param_list: typing.List[Param], use_intermediate: bool = False) -> list:
        return [
            [param_list.index(self.p1.x), 2],
            [param_list.index(self.p1.y), 2],
            [param_list.index(self.p2.x), 2],
            [param_list.index(self.p2.y), 2],
            [param_list.index(self.p1.x), int(use_intermediate)],
            [param_list.index(self.p1.y), int(use_intermediate)],
            [param_list.index(self.p2.x), int(use_intermediate)],
            [param_list.index(self.p2.y), int(use_intermediate)],
        ]

    def __repr__(self):
        return f"AbsAngleConstraintWeak {self.name}"


class ConstraintGraph(networkx.Graph):
    def __init__(self, gdict: dict = None, cdict: dict = None):
        self.gdict = gdict if gdict is not None else {}
        self.cdict = cdict if cdict is not None else {}
        self.root_finders = {}
        self.constraints = {}
        self.equations = {}
        self.arg_idx_arrays = {}
        self.variable_pos = {}
        super().__init__()

    def edge_names(self):
        return self._find_edge_names()

    def _add_edge(self, vertex_pair: typing.Iterable[Entity]):
        edge = set(vertex_pair)
        (v1, v2) = tuple(edge)

        if v1 in self.gdict and v2 in self.gdict[v1]:
            raise ValueError("Attempted to add a duplicate edge")

        if v1 in self.gdict:
            self.gdict[v1].append(v2)
            self.cdict[v1][v2] = []
        else:
            self.gdict[v1] = [v2]
            self.cdict[v1] = {v2: []}

        if v2 in self.gdict:
            self.gdict[v2].append(v1)
            self.cdict[v2][v1] = []
        else:
            self.gdict[v2] = [v1]
            self.cdict[v2] = {v1: []}

        for v in self.gdict:
            if v not in self.root_finders:
                self.root_finders[v] = None

    def _add_constraint_to_edge(self, vertex_pair: typing.Iterable[Entity], constraint: Constraint):
        edge = set(vertex_pair)
        (v1, v2) = tuple(edge)

        if v1 in self.cdict and v2 in self.cdict[v1] and any(
                [constraint.__class__.__name__ == cnstr.__class__.__name__ for cnstr in self.cdict[v1][v2]]):
            raise ValueError(f"Attempted to add a duplicate constraint ({constraint.name}) to edge connecting "
                             f"{v1.name} and {v2.name}")

        self.cdict[v1][v2].append(constraint)
        self.cdict[v2][v1].append(constraint)

    def verify_and_update_entity_or_constraint(self, entity_or_constraint: Entity or Constraint, x: np.ndarray,
                                               info):
        if np.any(info.fun_val > 1e-6):
            raise ValueError("Could not solve the constraint problem within tolerance after constraint addition")

        self.update_from_entity_or_constraint(entity_or_constraint, x)

    def add_constraint(self, vertex_pair: typing.Iterable[Entity], constraint: Constraint):
        self._add_edge(vertex_pair)
        self._add_constraint_to_edge(vertex_pair, constraint)

        # Now, analyze the graph to form the equation set, solve the equation set, and update the points
        point = constraint.pick_starting_point()
        self.get_equation_set_for_entity_or_constraint(point, for_initial_solve=True)
        self.compile_equation_for_entity_or_constraint(point)
        x, info = self.initial_solve_for_entity_or_constraint(point)
        self.verify_and_update_entity_or_constraint(point, x, info)

        # TODO: Note - might be able to avoid using jit on the initial compile for time savings

        # If the update was successful, compile all the equation sets for point movement
        for point in self.cdict:
            self.get_equation_set_for_entity_or_constraint(point, for_initial_solve=False)
            self.compile_equation_for_entity_or_constraint(point)

    def move_point(self, point: Point, new_x: float, new_y: float):

        start_param_vec = self.get_param_values()

        point.x.set_value(new_x)
        point.y.set_value(new_y)

        intermediate_param_vec = self.get_param_values()

        x, info = self.final_solve_for_entity_or_constraint(point, start_param_vec, intermediate_param_vec)

        print(f"{[k for k in self.cdict.keys()] = }")

        self.verify_and_update_entity_or_constraint(point, x, info)

    def _find_edges(self):
        edges = []
        for vertex in self.gdict:
            for next_vertex in self.gdict[vertex]:
                if {next_vertex, vertex} not in edges:
                    edges.append({vertex, next_vertex})

        return edges

    def _find_edge_names(self):
        edge_names = []
        for vertex in self.gdict:
            for next_vertex in self.gdict[vertex]:
                if {next_vertex.name, vertex.name} not in edge_names:
                    edge_names.append({vertex.name, next_vertex.name})

        return edge_names

    def get_strong_constraints(self):

        strong_constraints = []
        for edge in self.edges():
            (v1, v2) = tuple(edge)
            strong_constraints.extend(self.cdict[v1][v2])

        return strong_constraints

    def get_strong_equations(self, strong_constraints: typing.List[Constraint] = None):
        strong_constraints = self.get_strong_constraints() if strong_constraints is None else strong_constraints
        return [cnstr.equation for cnstr in strong_constraints]

    def get_params(self, strong_constraints: typing.List[Constraint] = None):
        params = []
        for entity in self.gdict:
            params.extend(entity.get_params())

        # Now add the parameters associated with the strong constraints to the list
        strong_constraints = self.get_strong_constraints() if strong_constraints is None else strong_constraints
        for cnstr in strong_constraints:
            params.append(cnstr.param)

        return params

    def get_param_values(self):
        return np.array([p.value() for p in self.get_params()])

    def add_variable(self, entity_or_constraint: Entity or Constraint, variable: Param,
                     params: typing.List[Param]):

        var_index = params.index(variable)

        if var_index not in self.variable_pos[entity_or_constraint]:
            self.variable_pos[entity_or_constraint].append(var_index)

    def get_equation_set_for_entity_or_constraint(self, entity_or_constraint: Entity or Constraint,
                                                  params: typing.List[Param] = None,
                                                  strong_constraints: typing.List[Constraint] = None,
                                                  for_initial_solve: bool = True):

        params = self.get_params(strong_constraints) if params is None else params

        # Clear the equation sub-container for this entity if it exists, otherwise create the sub-container
        if entity_or_constraint in self.equations:
            self.equations[entity_or_constraint].clear()
        else:
            self.equations[entity_or_constraint] = []

        # Repeat for the argument index array attribute
        if entity_or_constraint in self.arg_idx_arrays:
            self.arg_idx_arrays[entity_or_constraint].clear()
        else:
            self.arg_idx_arrays[entity_or_constraint] = []

        # Repeat for the variable position array attribute
        if entity_or_constraint in self.variable_pos:
            self.variable_pos[entity_or_constraint].clear()
        else:
            self.variable_pos[entity_or_constraint] = []

        # Repeat for constraint container
        if entity_or_constraint in self.constraints:
            self.constraints[entity_or_constraint].clear()
        else:
            self.constraints[entity_or_constraint] = []

        # Traverse the graph starting with the specified entity_or_constraint and continue until DOF = 0
        if isinstance(entity_or_constraint, Constraint):
            point = entity_or_constraint.pick_starting_point()
        else:
            point = entity_or_constraint

        def add_var(start: Point, param: Param, dof: int):
            self.add_variable(start, param, params)
            return dof + 1

        def add_vars(start: Point, param_list: typing.List[Param], dof: int):

            for param in param_list:
                dof = add_var(start, param, dof)

            return dof

        def add_strong_cnstr(cnstr: Constraint, start: Point, dof: int, strong_added: int):
            self.constraints[start].append(cnstr)
            self.equations[start].append(cnstr.equation)
            self.arg_idx_arrays[start].append(cnstr.get_arg_idx_array(params))
            return dof - 1, strong_added + 1

        def add_strong_cnstrs(cnstrs: typing.List[Constraint], start: Point, dof: int, strong_added: int):

            for cnstr in cnstrs:
                dof, strong_added = add_strong_cnstr(cnstr, start, dof, strong_added)

            return dof, strong_added

        def add_weak_cnstr(cnstr: ConstraintWeak, start: Point, dof: int, use_intermediate: bool = False):
            self.constraints[start].append(cnstr)
            self.equations[start].append(cnstr.equation)
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
                        print(f"{starting_node = }, {visited_edges = }")
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

        bfs(point)

    def compile_equation_for_entity_or_constraint(self, entity_or_constraint: Entity or Constraint):

        def equation_system(x: np.ndarray,
                            start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):

            final_param_vec = jnp.array([v for v in intermediate_param_vec])
            for idx, pos in enumerate(self.variable_pos[entity_or_constraint]):
                final_param_vec = final_param_vec.at[pos].set(x[idx])

            func_outputs = jnp.array([cnstr(*[start_param_vec[w[0]] if w[1] == 0
                                      else intermediate_param_vec[w[0]] if w[1] == 1
                                      else final_param_vec[w[0]] for w in arg_idx_array])
                                      for cnstr, arg_idx_array in zip(self.equations[entity_or_constraint],
                                                                      self.arg_idx_arrays[entity_or_constraint])
                                      ])

            return func_outputs

        self.root_finders[entity_or_constraint] = PymeadRootFinder(jit(equation_system))

    def initial_solve_for_entity_or_constraint(self, entity_or_constraint: Entity or Constraint):
        params = self.get_params()
        x0 = np.array([params[x_pos].value() for x_pos in self.variable_pos[entity_or_constraint]])
        v = np.array([p.value() for p in params])
        w = deepcopy(v)
        x, info = self.root_finders[entity_or_constraint].solve(x0, v, w)
        return x, info

    def final_solve_for_entity_or_constraint(self, entity_or_constraint: Entity or Constraint,
                                             start_param_vec: np.ndarray, intermediate_param_vec: np.ndarray):
        params = self.get_params()
        x0 = np.array([params[x_pos].value() for x_pos in self.variable_pos[entity_or_constraint]])
        print(f"{[params[x_pos] for x_pos in self.variable_pos[entity_or_constraint]] = }")
        x, info = self.root_finders[entity_or_constraint].solve(x0, start_param_vec, intermediate_param_vec)
        return x, info

    def update_from_entity_or_constraint(self, entity_or_constraint: Entity or Constraint, new_x: np.ndarray):
        """
        Updates the variable points and parameters in the constraint system

        Returns
        -------

        """
        params = self.get_params()
        for idx, x in zip(self.variable_pos[entity_or_constraint], new_x):
            params[idx].set_value(x)


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

    graph.add_constraint([p1, p2], cnstr1)
    graph.add_constraint([p2, p3], cnstr2)
    graph.add_constraint([p2, p4], cnstr3)
    graph.add_constraint([p1, p5], cnstr4)
    graph.add_constraint([p3, p1], cnstr5)

    print(f"{np.hypot(p1.x.value() - p2.x.value(), p1.y.value() - p2.y.value())}")
    print(f"{np.hypot(p3.x.value() - p2.x.value(), p3.y.value() - p2.y.value())}")
    print(f"{np.hypot(p1.x.value() - p3.x.value(), p1.y.value() - p3.y.value())}")

    print(f"{p2 = }")

    new_x = [p.x.value() for p in points]
    new_y = [p.y.value() for p in points]
    plt.plot(new_x, new_y, ls="none", marker="s", color="mediumaquamarine", mfc="#aaaaaa88")
    plt.show()

    graph.move_point(p2, 0.8, 0.6)

    print(f"{np.hypot(p1.x.value() - p2.x.value(), p1.y.value() - p2.y.value())}")
    print(f"{np.hypot(p3.x.value() - p2.x.value(), p3.y.value() - p2.y.value())}")
    print(f"{np.hypot(p1.x.value() - p3.x.value(), p1.y.value() - p3.y.value())}")

    new_x = [p.x.value() for p in points]
    new_y = [p.y.value() for p in points]
    plt.plot(original_x, original_y, ls="none", marker="o", color="indianred")
    plt.plot(new_x, new_y, ls="none", marker="s", color="steelblue", mfc="#aaaaaa88")
    plt.show()

    pass


if __name__ == "__main__":
    main()
