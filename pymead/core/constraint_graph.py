from copy import deepcopy

import jaxopt
from jax import numpy as jnp
from jax import jit
import numpy as np
import networkx

from pymead.core.point import Point
from pymead.core.constraints import *


class PymeadRootFinder(jaxopt.ScipyRootFinding):
    def __init__(self, equation_system: typing.Callable, method: str = "lm"):
        super().__init__(
            method=method,
            jit=True,
            has_aux=False,
            optimality_fun=equation_system,
            tol=1e-10,
            use_jacrev=True,  # Use the forward Jacobian calculation since the matrix is square
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


class EquationData:

    def __init__(self):
        self.root_finders = {}
        self.geo_cons = []
        self.equations = []
        self.arg_idx_array = []
        self.variable_pos = []

    def clear(self):
        self.root_finders.clear()
        self.geo_cons.clear()
        self.equations.clear()
        self.arg_idx_array.clear()
        self.variable_pos.clear()


class ConstraintGraph(networkx.Graph):
    def __init__(self):
        self.points = []
        self.constraint_params = []
        super().__init__()

    def add_point(self, point: Point):
        """
        Adds a ``Point`` as a new (unconnected) node in the graph, and to the graph instance's list of points.

        Parameters
        ----------
        point: Point
            Point to add to the graph

        Returns
        -------

        """
        point.gcs = self
        self.add_node(point)
        self.points.append(point)

    def remove_point(self, point: Point):
        connected_constraints = point.geo_cons.copy()
        self.remove_node(point)
        self.points.remove(point)

        for constraint in connected_constraints:
            self.remove_constraint(constraint)

    def add_constraint(self, constraint: GeoCon):
        """
        Adds the specified constraint to the graph, then analyzes the entire graph of connected points and constraints
        and adds any weak constraints needed to make the problem well-constrained. The non-linear system of equations
        representing the constraints is then solved to tolerance, and all points updated.

        Parameters
        ----------
        constraint: GeoCon
            GeoCon to add to the graph

        Returns
        -------

        """
        if constraint.param() is not None:
            constraint.param().gcs = self
            if constraint.param() not in self.constraint_params:
                self.constraint_params.append(constraint.param())

        self.add_node(constraint)
        for child_node in constraint.child_nodes:
            self.add_edge(constraint, child_node)

        # Perform depth-first search on the nodes to find if any connected node has the "data" attr set
        for node in networkx.dfs_preorder_nodes(self, source=constraint):
            if isinstance(node, GeoCon) and node.data is not None:
                constraint.data = node.data
                break
        else:  # If not, assign a new instance of the EquationData class to the "data" attr
            constraint.data = EquationData()

        print(f"Adding {constraint}...")

        # Analyze the constraint to add the strong constraints and any weak constraints necessary to make the problem
        # well-constrained
        self.analyze(constraint)

        # Compile two separate root finders: one using Levenberg-Marquardt damped non-linear least squares algorithm,
        # another using MINPACK's hybrd/hybrj routines
        self.compile_equation_for_entity_or_constraint(constraint, method="lm")
        self.compile_equation_for_entity_or_constraint(constraint, method="hybr")

        print(f"{constraint.data.geo_cons = }")

        # Solve using first the least-squares method and then MINPACK if necessary. Update the points if the solution
        # falls within the tolerance specified in the PymeadRootFinder class
        self.multisolve_and_update(constraint)

    def remove_constraint(self, constraint: GeoCon):

        adj_points = [nbr for nbr in self.adj[constraint]].copy()  # Retain a copy of the constraint's neighbors

        self.remove_node(constraint)

        for point in adj_points:
            point.geo_cons.remove(constraint)
            connected_components = networkx.node_connected_component(self, point)
            # TODO: need to re-compile the equations for potentially all the points that were originally a member
            #  of this constraint

    def get_points_to_fix(self, source: GeoCon) -> typing.List[Point]:

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

    def analyze(self, constraint: GeoCon):

        constraint.data.clear()
        params = self.get_params()
        print(f"{params = }")
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
        a4_constraints = []

        for node in networkx.dfs_preorder_nodes(self, source=constraint):

            if isinstance(node, Point) and not any(["d" in cnstr.kind for cnstr in self.adj[node]]):
                if node not in points_may_need_distance:
                    points_may_need_distance.append(node)

            if isinstance(node, Point) and not any(["a" in cnstr.kind for cnstr in self.adj[node]]):
                if node not in points_may_need_angle:
                    points_may_need_angle.append(node)

            if isinstance(node, GeoCon) and node.kind == "a4":
                a4_constraints.append(node)

        for point in points_may_need_distance:

            # Weak distance constraint algorithm

            if dof == 0:
                break

            # If this was the last point added, use the second-to-last point as p2. Otherwise, use the next point.
            if self.points.index(point) == len(self.points) - 1:
                p2 = self.points[-2]
            else:
                p2 = self.points[self.points.index(point) + 1]

            # Add a weak distance constraint between this point and the next candidate point, chosen somewhat
            # arbitrarily for now
            dist_constraint_weak = DistanceConstraintWeak(point, p2, "dcw1")
            weak_constraints.append(dist_constraint_weak)
            dof -= 1

        for cnstr in a4_constraints:

            # A4 constraint algorithm

            if dof == 0:
                break

            # 2. For each "a4," check that each of the subsets {{1,2},{3,4}} and {{2,3},{1,4}} is constrained w/ "a3"
            # a3_constrained_points = []
            # for point in [cnstr.p1, cnstr.p2, cnstr.p3, cnstr.p4]:
            #     if any([c.kind == "a3" for c in point.geo_cons]):
            #         a3_constrained_points.append(point)

            add_a3 = True
            analyze_angle_loop = True

            # 2 (cont). If less than 3 points in the a4 constraint have an a3 constraint, then we know for sure that an
            # a3 constraint connecting two adjacent sides of the a4 constraint will be required
            # if len(a3_constrained_points) < 3:
            #     analyze_angle_loop = False

            point_pairs = {
                0: [cnstr.p1, cnstr.p2],
                1: [cnstr.p3, cnstr.p4],
                2: [cnstr.p2, cnstr.p3],
                3: [cnstr.p1, cnstr.p4]
            }

            common_neighbors = []

            for idx, pair in point_pairs.items():
                common_neighbors.append(
                    [n for n in networkx.common_neighbors(self, pair[0], pair[1]) if n.kind == "a3"]
                )

            point_pair_n_constraints = {idx: len(neighbors) for idx, neighbors in enumerate(common_neighbors)}

            pair_combos = [[0, 2], [0, 3], [1, 2], [1, 3]]
            found_combo = False

            for pair_combo in pair_combos:
                if point_pair_n_constraints[pair_combo[0]] > 0 and point_pair_n_constraints[pair_combo[1]] > 0:
                    if len(set(common_neighbors[pair_combo[0]]).intersection(common_neighbors[pair_combo[1]])) > 0:
                        add_a3 = False
                        break
                    found_combo = True

            if not found_combo:
                analyze_angle_loop = False

            # This next block should all go inside the next if statement
            cycles = [cycle for cycle in networkx.simple_cycles(self) if all([isinstance(node, Point) or (
                    isinstance(node, GeoCon) and node.kind in ["a3", "a4"]) for node in cycle])]
            for cycle in cycles:
                print(f"{cycle = }")

            # 4. Angle loop closure
            if add_a3 and analyze_angle_loop:
                visited_nodes = []

            if not add_a3:
                continue

            # Add the weak a3 constraint
            candidate_point_lists = [[cnstr.p2, cnstr.p1, cnstr.p3], [cnstr.p4, cnstr.p2, cnstr.p1]]
            check_dist_point_lists = [[cnstr.p1, cnstr.p3], [cnstr.p2, cnstr.p4]]
            for candidate_point_list, check_dist_point_list in zip(candidate_point_lists, check_dist_point_lists):
                common_dist_neighbors = (
                    [n for n in networkx.common_neighbors(
                        self, check_dist_point_list[0], check_dist_point_list[1]) if "d" in n.kind]
                )
                if len(common_dist_neighbors) == 0:
                    continue

                weak_a3_constraint = RelAngle3ConstraintWeak(*candidate_point_list, name="ra3w")
                weak_constraints.append(weak_a3_constraint)
                break

        for point in points_may_need_angle:

            # Weak relative angle constraint algorithm

            if dof == 0:
                break

            for cnstr in point.geo_cons:
                if isinstance(cnstr, DistanceConstraint) or isinstance(cnstr, DistanceConstraintWeak):
                    start_point = point
                    vertex = cnstr.p2 if cnstr.p2 is not point else cnstr.p1
                    end_point = None

                    for sub_cnstr in vertex.geo_cons:
                        if sub_cnstr is cnstr:
                            continue
                        if isinstance(sub_cnstr, RelAngle3Constraint) or isinstance(sub_cnstr, RelAngle3ConstraintWeak):
                            end_point = sub_cnstr.start_point
                        elif (isinstance(sub_cnstr, Perp3Constraint) or isinstance(sub_cnstr, Parallel3Constraint) or
                              isinstance(sub_cnstr, AntiParallel3Constraint)):
                            end_point = sub_cnstr.p1
                    if end_point is not None:
                        rel_angle3_constraint_weak = RelAngle3ConstraintWeak(start_point, vertex, end_point, "ra3")
                        weak_constraints.append(rel_angle3_constraint_weak)
                        dof -= 1
                        break

                    for sub_cnstr in vertex.geo_cons:
                        if sub_cnstr is cnstr:
                            continue

                        if isinstance(sub_cnstr, DistanceConstraint) or isinstance(sub_cnstr, DistanceConstraintWeak):
                            end_point = sub_cnstr.p1 if sub_cnstr.p1 not in [start_point, vertex] else sub_cnstr.p2

                    if end_point is not None:
                        rel_angle3_constraint_weak = RelAngle3ConstraintWeak(start_point, vertex, end_point, "ra3")
                        weak_constraints.append(rel_angle3_constraint_weak)
                        dof -= 1
                        break

        # Determine the absolute angle constraint to add to eliminate the rotational degree of freedom, if necessary
        if len(abs_angle_constraints) == 0 and len(fixed_points) == 1:
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
            constraint.data.arg_idx_array.extend(cnstr.get_arg_idx_array(params))
            constraint.data.geo_cons.append(cnstr)

        for eq in weak_equations:
            constraint.data.equations.append(eq)

        for cnstr in weak_constraints:
            constraint.data.arg_idx_array.extend(cnstr.get_arg_idx_array(params))
            constraint.data.geo_cons.append(cnstr)

        pass

    @staticmethod
    def verify_constraint_addition(info):
        if np.any(info.fun_val > 1e-6):
            raise ValueError("Could not solve the constraint problem within tolerance after constraint addition")

    def get_strong_constraints(self, source_node: Point or GeoCon):

        strong_constraints = []
        for node in networkx.dfs_preorder_nodes(self, source=source_node):
            if isinstance(node, GeoCon):
                strong_constraints.append(node)

        return strong_constraints

    def get_strong_equations(self, strong_constraints: typing.List[GeoCon] = None,
                             source_node: Point or GeoCon = None):
        strong_equations = []

        strong_constraints = self.get_strong_constraints(
            source_node) if strong_constraints is None else strong_constraints

        for constraint in strong_constraints:
            strong_equations.extend(constraint.equations)

        return strong_equations

    def get_points(self, source_node: Point or GeoCon):
        points = []

        for node in networkx.dfs_preorder_nodes(self, source=source_node):
            if not isinstance(node, Point):
                continue

            points.append(node)

        return points

    def get_params(self):
        params = []

        for point in self.points:
            params.extend([point.x(), point.y()])

        # for node in networkx.dfs_preorder_nodes(self, source=source_node):
        #     # if isinstance(node, Point):
        #     #     params.extend([node.x(), node.y()])
        #     if isinstance(node, GeoCon):
        #         if node.param() is None:
        #             continue
        #         params.append(node.param())

        for constraint_param in self.constraint_params:
            params.append(constraint_param)

        return params

    def get_param_values(self):
        return np.array([p.value() for p in self.get_params()])

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
            param_list.extend([point.x(), point.y()])

        for param in param_list:
            self.add_variable(equation_data, param, params)

    @staticmethod
    def compile_equation_for_entity_or_constraint(constraint: GeoCon, method: str = "lm"):

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

        constraint.data.root_finders[method] = PymeadRootFinder(jit(equation_system), method=method)

    def solve(self, constraint: GeoCon, method: str):
        params = self.get_params()
        x0 = np.array([params[x_pos].value() for x_pos in constraint.data.variable_pos])
        v = np.array([p.value() for p in params])
        w = deepcopy(v)
        x, info = constraint.data.root_finders[method].solve(x0, v, w)
        return x, info

    def multisolve_and_update(self, constraint: GeoCon):
        x, info = self.solve(constraint, method="lm")

        try:
            self.verify_constraint_addition(info)
        except ValueError:
            # Update the points anyway and try to solve using the other root-finding method
            self.update_points(constraint, new_x=x)
            x, info = self.solve(constraint, method="hybr")
            try:
                self.verify_constraint_addition(info)
            except ValueError:
                raise ValueError("Could not converge the solution within tolerance using both root-finding methods")

        self.update_points(constraint, new_x=x)

    def update_points(self, constraint: GeoCon, new_x: np.ndarray):
        """
        Updates the variable points and parameters in the constraint system

        Returns
        -------

        """
        params = self.get_params()
        for idx, x in zip(constraint.data.variable_pos, new_x):
            params[idx].set_value(x)

        curves_to_update = []
        for point in self.points:
            if point.canvas_item is not None:
                point.canvas_item.updateCanvasItem(point.x().value(), point.y().value())

            for curve in point.curves:
                if curve not in curves_to_update:
                    curves_to_update.append(curve)

        airfoils_to_update = []
        for curve in curves_to_update:
            if curve.airfoil is not None and curve.airfoil not in airfoils_to_update:
                airfoils_to_update.append(curve.airfoil)
            curve.update()

        for airfoil in airfoils_to_update:
            airfoil.update_coords()
            airfoil.canvas_item.generatePicture()

        for node in networkx.dfs_preorder_nodes(self, source=constraint):
            if isinstance(node, GeoCon):
                node.canvas_item.update()


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

    original_x = [p.x().value() for p in points]
    original_y = [p.y().value() for p in points]
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

    print(f"{np.hypot(p1.x().value() - p2.x().value(), p1.y().value() - p2.y().value())}")
    print(f"{np.hypot(p3.x().value() - p2.x().value(), p3.y().value() - p2.y().value())}")
    print(f"{np.hypot(p1.x().value() - p3.x().value(), p1.y().value() - p3.y().value())}")

    print(f"{p2 = }")

    # new_x = [p.x().value() for p in points]
    # new_y = [p.y().value() for p in points]
    # plt.plot(new_x, new_y, ls="none", marker="s", color="mediumaquamarine", mfc="#aaaaaa88")
    # plt.show()
    #
    # graph.move_point(p2, 0.8, 0.6)
    #
    # print(f"{np.hypot(p1.x().value() - p2.x().value(), p1.y().value() - p2.y().value())}")
    # print(f"{np.hypot(p3.x().value() - p2.x().value(), p3.y().value() - p2.y().value())}")
    # print(f"{np.hypot(p1.x().value() - p3.x().value(), p1.y().value() - p3.y().value())}")

    new_x = [p.x().value() for p in points]
    new_y = [p.y().value() for p in points]
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

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o", mec="indianred", mfc="indianred", fillstyle="left", markersize=10)

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
        plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o",
                 mec="indianred", mfc="indianred", fillstyle="left", markersize=10)
        plt.show()

    d4.param().set_value(4.0)
    d4.param().set_value(5.0)

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o",
             mfc="steelblue", mec="steelblue", fillstyle="right", markersize=10)
    for p in points:
        plt.text(p.x().value() + 0.02, p.y().value() + 0.02, p.name())
    plt.gca().set_aspect("equal")

    plt.show()

    labels = {p: p.name() for p in g.nodes}
    subax1 = plt.subplot(121)
    networkx.draw(g, with_labels=True, labels=labels)

    # for node in networkx.dfs_preorder_nodes(g, source=p1):
    #     if isinstance(node, GeoCon):
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

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o", mec="indianred", mfc="indianred", fillstyle="left", markersize=10)

    # Add the constraints
    d1 = DistanceConstraint(p1, p2, 0.6, "d1")
    d2 = DistanceConstraint(p2, p3, 1.0, "d2")
    d3 = DistanceConstraint(p1, p3, 0.8, "d3")
    constraints = [d1, d2, d3]
    for constraint in constraints:
        g.add_constraint(constraint)

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o", mfc="steelblue", mec="steelblue", fillstyle="right", markersize=10)
    for p in points:
        plt.text(p.x().value() + 0.02, p.y().value() + 0.02, p.name())
    plt.gca().set_aspect("equal")

    plt.show()


def main4():
    import matplotlib.pyplot as plt
    # Initialize the graph
    g = ConstraintGraph()

    # Add the points
    p1 = Point(0.0, 0.0, "p1")
    p2 = Point(1.0, 0.0, "p2")
    p3 = Point(-0.05, 0.05, "p3")
    # p4 = Point(0.5, 0.03, "p4")
    points = [p1, p2, p3]
    for point in points:
        g.add_point(point)

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o", mec="indianred",
             mfc="indianred", fillstyle="left", markersize=10)

    # Add the constraints
    d1 = DistanceConstraint(p1, p2, 1.0, "d1")
    d2 = DistanceConstraint(p1, p3, 0.2, "d2")
    pol = PointOnLineConstraint(p3, p1, p2, "pol1")
    # d3 = DistanceConstraint(p1, p3, 0.8, "d3")
    constraints = [d1, d2, pol]
    for constraint in constraints:
        g.add_constraint(constraint)

    # d4.param.set_value(2.0)
    # # g.compile_equation_for_entity_or_constraint(aparl3)
    # x, info = g.initial_solve(d3)
    # print(f"{info = }")
    # g.update_points(aparl3, new_x=x)

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o", mfc="steelblue",
             mec="steelblue", fillstyle="right", markersize=10)
    for p in points:
        plt.text(p.x().value() + 0.02, p.y().value() + 0.02, p.name())
    plt.gca().set_aspect("equal")

    plt.show()


def main5():
    import matplotlib.pyplot as plt
    g = ConstraintGraph()

    # Add the points
    p1 = Point(0.0, 0.0, "p1")
    p2 = Point(0.4, 0.1, "p2")
    p3 = Point(0.1, -0.1, "p3")
    p4 = Point(0.5, -0.15, "p4")
    points = [p1, p2, p3, p4]
    for point in points:
        g.add_point(point)

    d1 = DistanceConstraint(p1, p2, 0.5, "d1")
    d2 = DistanceConstraint(p3, p4, 0.55, "d2")
    par = Parallel4Constraint(p1, p2, p3, p4, "par")
    for constraint in [d1, d2, par]:
        g.add_constraint(constraint)

    plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o", mfc="indianred")
    plt.show()

    g.remove_constraint(par)


if __name__ == "__main__":
    main2()
