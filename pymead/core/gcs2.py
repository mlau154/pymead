import networkx

from pymead.core.constraints import *
from pymead.core.constraint_equations import *
from pymead.core.point import Point


class GCS2(networkx.DiGraph):
    def __init__(self):
        super().__init__()
        self.points = {}
        self.roots = []

    def add_point(self, point: Point):
        point.gcs = self
        self.add_node(point)
        self.points[point.name()] = point

    def remove_point(self, point: Point):
        self.remove_node(point)
        self.points.pop(point.name())

    def _check_if_constraint_creates_new_cluster(self, constraint: GeoCon):
        for point in constraint.child_nodes:
            # If there is already an edge attached to any of the points in this constraint, do not create a new root
            if len(self.in_edges(nbunch=point)) > 0 or len(self.out_edges(nbunch=point)) > 0:
                return False
        return True

    def _set_distance_constraint_as_root(self, constraint: DistanceConstraint):
        constraint.child_nodes[0].root = True
        constraint.child_nodes[1].rotation_handle = True
        if constraint.child_nodes[0] not in [r[0] for r in self.roots]:
            self.roots.append((constraint.child_nodes[0], constraint.child_nodes[1], constraint))

    def _delete_root_status(self, root_node: Point):
        for edge in self.out_edges(nbunch=root_node, data=True):
            if "distance" not in edge[2].keys():
                continue
            else:
                constraint = edge[2]["distance"]

            if not (constraint.child_nodes[0].rotation_handle or constraint.child_nodes[1].rotation_handle):
                continue

            for node in constraint.child_nodes:
                node.root = False
                node.rotation_handle = False

            root_idx = [r[0] for r in self.roots].index(root_node)
            self.roots.pop(root_idx)
            return

        raise ValueError("Could not detect the distance constraint or rotation handle associated with this root")

    def _check_if_constraint_addition_requires_cluster_merge(self, constraint: GeoCon):
        """
        Check if the addition of this constraint requires a cluster merge by analyzing how many nodes have incident
        edges. Any constraint with more than one incident edge requires a cluster merge.

        Parameters
        ----------
        constraint: GeoCon
            Constraint to test for cluster merge

        Returns
        -------
        bool
            ``True`` if there are at least two nodes with incident edges, otherwise ``False``
        """
        unique_roots = []
        for point in constraint.child_nodes:
            root = self._discover_root_from_node(point)
            if root and root not in unique_roots:
                unique_roots.append(root)
        return True if len(unique_roots) > 1 else False

    def _identify_cluster_roots_for_constraint_addition(self, constraint: GeoCon):
        pass

    def _check_if_node_has_incident_edge(self, node: Point):
        in_edges = [edge for edge in self.in_edges(nbunch=node)]
        return True if in_edges else False

    def _check_if_node_reaches_root(self, source_node: Point):
        for node in networkx.dfs_preorder_nodes(self, source=source_node):
            if node.root:
                return True
        return False

    def _discover_root_from_node(self, source_node: Point):
        """
        Traverse backward along the constraint graph from the ``source_node`` until the root node is reached.
        Returns ``None`` if the root was not found.

        Parameters
        ----------
        source_node: Point
            Starting point

        Returns
        -------
        Point or None
            The root node/point if found, otherwise ``None``
        """
        for node in networkx.bfs_tree(self, source=source_node, reverse=True):
            if node.root:
                return node
        return None

    def _add_distance_constraint_to_directed_edge(self, constraint: DistanceConstraint,
                                                  first_constraint_in_cluster: bool):

        def _add_edge_or_append_data(node1: Point, node2: Point):
            edge_data = self.get_edge_data(node1, node2)
            if edge_data:
                if "distance" in edge_data.keys():
                    raise ValueError("Cannot add a second distance constraint between the same pair of points")
                else:
                    networkx.set_edge_attributes(self, {(node1, node2): constraint}, name="distance")
            else:
                self.add_edge(node1, node2, distance=constraint)

        p1_edges = [edge for edge in self.in_edges(nbunch=constraint.p1, data=True)]
        p2_edges = [edge for edge in self.in_edges(nbunch=constraint.p2, data=True)]
        edge_lists = [p1_edges, p2_edges]
        point_pairs = [(constraint.p1, constraint.p2), (constraint.p2, constraint.p1)]

        if first_constraint_in_cluster:
            _add_edge_or_append_data(constraint.p1, constraint.p2)
            return

        for edge_list, point_pair in zip(edge_lists, point_pairs):

            if len(edge_list) > 1:
                raise ValueError("Detected multiple edges for the same pair of points when adding distance constraint")

            if len(edge_list) <= 1:
                if (self._check_if_node_has_incident_edge(point_pair[1])
                        or self._check_if_node_reaches_root(point_pair[1])):
                    continue

                _add_edge_or_append_data(point_pair[0], point_pair[1])
                return

        raise ValueError("Failed to add distance constraint")

    def _add_angle_constraint_to_directed_edge(
            self, constraint: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint,
            first_constraint_in_cluster: bool):

        if first_constraint_in_cluster:
            raise ValueError("First constraint in a cluster must be a distance constraint")

        in_edges_p1 = tuple([edge for edge in self.in_edges(nbunch=constraint.p1, data=True)])
        in_edges_p2 = tuple([edge for edge in self.in_edges(nbunch=constraint.p2, data=True)])
        in_edges_p3 = tuple([edge for edge in self.in_edges(nbunch=constraint.p3, data=True)])

        edge_data_p21 = self.get_edge_data(constraint.p2, constraint.p1)
        edge_data_p23 = self.get_edge_data(constraint.p2, constraint.p3)

        angle_in_p21 = False if not edge_data_p21 else "angle" in edge_data_p21.keys()
        angle_in_p23 = False if not edge_data_p23 else "angle" in edge_data_p23.keys()

        if angle_in_p21 and angle_in_p23:
            raise ConstraintValidationError(f"{constraint} already has angle constraints associated with both"
                                            f" pairs of points")

        if edge_data_p21 and not angle_in_p21 and not (
                constraint.p2.root and "distance" in edge_data_p21.keys() and constraint.p1.rotation_handle):
            networkx.set_edge_attributes(self, {(constraint.p2, constraint.p1): constraint}, name="angle")
            return
        if edge_data_p23 and not angle_in_p23 and not (
                constraint.p2.root and "distance" in edge_data_p23.keys() and constraint.p3.rotation_handle):
            networkx.set_edge_attributes(self, {(constraint.p2, constraint.p3): constraint}, name="angle")
            return

        if len(in_edges_p1) > 0:
            # if angle_in_p12 or angle_in_p21:
            if angle_in_p21 and not constraint.p3.rotation_handle:
                self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                return
            # if angle_in_p23 or angle_in_p32:
            if angle_in_p23:
                raise ValueError("Cannot create a valid angle constraint from this case")
            if constraint.p2 not in [nbr for nbr in self.neighbors(constraint.p3)]:
                self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                return
        if len(in_edges_p2) > 0:
            # if angle_in_p12 or angle_in_p21:
            if angle_in_p21 and not constraint.p3.rotation_handle:
                self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                return
            # if angle_in_p23 or angle_in_p32:
            if angle_in_p23 and not constraint.p1.rotation_handle:
                self.add_edge(constraint.p2, constraint.p1, angle=constraint)
                return
            if constraint.p2 not in [nbr for nbr in self.neighbors(constraint.p3)]:
                self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                return
        if len(in_edges_p3) > 0:
            # if angle_in_p12 or angle_in_p21:
            if angle_in_p21:
                raise ValueError("Cannot create a valid angle constraint from this case")
            # if angle_in_p23 or angle_in_p32:
            if angle_in_p23 and not constraint.p1.rotation_handle:
                self.add_edge(constraint.p2, constraint.p1, angle=constraint)
                return
            if constraint.p2 not in [nbr for nbr in self.neighbors(constraint.p1)]:
                self.add_edge(constraint.p2, constraint.p1, angle=constraint)
                return

        if not constraint.p3.rotation_handle and constraint.p2 not in [nbr for nbr in self.neighbors(constraint.p3)]:
            self.add_edge(constraint.p2, constraint.p3, angle=constraint)
            return
        if not constraint.p1.rotation_handle and constraint.p2 not in [nbr for nbr in self.neighbors(constraint.p1)]:
            self.add_edge(constraint.p2, constraint.p1, angle=constraint)
            return

        raise ValueError("Relative angle constraint could not be created")

    def add_constraint(self, constraint: GeoCon):

        # Check if this constraint creates a new cluster
        first_constraint_in_cluster = self._check_if_constraint_creates_new_cluster(constraint)

        # If it does, make sure it is a distance constraint and set it as the root constraint
        if first_constraint_in_cluster:
            if not isinstance(constraint, DistanceConstraint):
                raise ValueError("The first constraint in a constraint cluster must be a distance constraint")
            self._set_distance_constraint_as_root(constraint)

        # If the constraint has a Param associated with it, pass the GCS reference to this parameter
        if constraint.param() is not None:
            constraint.param().gcs = self

        needs_cluster_merge = self._check_if_constraint_addition_requires_cluster_merge(constraint)

        if isinstance(constraint, DistanceConstraint):
            self._add_distance_constraint_to_directed_edge(constraint, first_constraint_in_cluster)
        elif isinstance(constraint, RelAngle3Constraint) or isinstance(
                constraint, AntiParallel3Constraint) or isinstance(constraint, Perp3Constraint):
            self._add_angle_constraint_to_directed_edge(constraint, first_constraint_in_cluster)
        elif isinstance(constraint, SymmetryConstraint):
            points_solved = self.solve_symmetry_constraint(constraint)
            self.update_canvas_items(points_solved)
        elif isinstance(constraint, ROCurvatureConstraint):
            points_solved = self.solve_roc_constraint(constraint)
            self.update_canvas_items(points_solved)

    def remove_constraint(self, constraint: GeoCon):
        raise NotImplementedError("Constraint removal not yet implemented")

    def make_point_copies(self):
        return {point: Point(x=point.x().value(), y=point.y().value()) for point in self.points.values()}

    def _solve_distance_constraint(self, source: DistanceConstraint):
        points_solved = []
        edge_data_p12 = self.get_edge_data(source.p1, source.p2)
        if edge_data_p12 and edge_data_p12["distance"] is source:
            angle = source.p1.measure_angle(source.p2)
            start = source.p2
        else:
            angle = source.p2.measure_angle(source.p1)
            start = source.p1
        old_distance = source.p1.measure_distance(source.p2)
        new_distance = source.param().value()
        dx = (new_distance - old_distance) * np.cos(angle)
        dy = (new_distance - old_distance) * np.sin(angle)

        for point in networkx.bfs_tree(self, source=start):
            point.x().set_value(point.x().value() + dx)
            point.y().set_value(point.y().value() + dy)
            if point not in points_solved:
                points_solved.append(point)

        return points_solved

    def _solve_perp_parallel_constraint(self, source: AntiParallel3Constraint or Perp3Constraint):
        points_solved = []
        edge_data_p21 = self.get_edge_data(source.p2, source.p1)
        edge_data_p23 = self.get_edge_data(source.p2, source.p3)

        old_angle = (source.p2.measure_angle(source.p1) - source.p2.measure_angle(source.p3)) % (2 * np.pi)
        new_angle = np.pi if isinstance(source, AntiParallel3Constraint) else np.pi / 2
        d_angle = new_angle - old_angle
        rotation_point = source.p2

        if edge_data_p21 and "angle" in edge_data_p21 and edge_data_p21["angle"] is source:
            start = source.p1
            # d_angle *= -1
        elif edge_data_p23 and "angle" in edge_data_p23 and edge_data_p23["angle"] is source:
            start = source.p3
            d_angle *= -1
        else:
            raise ValueError("Somehow no angle constraint found between the three points")

        rotation_mat = np.array([[np.cos(d_angle), -np.sin(d_angle)], [np.sin(d_angle), np.cos(d_angle)]])
        rotation_point_mat = np.array([[rotation_point.x().value()], [rotation_point.y().value()]])

        for point in networkx.bfs_tree(self, source=start):
            dx_dy = np.array([[point.x().value() - rotation_point.x().value()],
                              [point.y().value() - rotation_point.y().value()]])
            new_xy = (rotation_mat @ dx_dy + rotation_point_mat).flatten()
            point.x().set_value(new_xy[0])
            point.y().set_value(new_xy[1])
            if point not in points_solved:
                points_solved.append(point)
        return points_solved

    def _solve_rel_angle3_constraint(self, source: RelAngle3Constraint):
        points_solved = []
        edge_data_p21 = self.get_edge_data(source.p2, source.p1)
        edge_data_p23 = self.get_edge_data(source.p2, source.p3)

        old_angle = (source.p2.measure_angle(source.p1) -
                     source.p2.measure_angle(source.p3)) % (2 * np.pi)
        new_angle = source.param().rad()
        d_angle = new_angle - old_angle
        rotation_point = source.p2

        if edge_data_p21 and "angle" in edge_data_p21 and edge_data_p21["angle"] is source:
            start = source.p1 if not source.p2.root else source.p2
        elif edge_data_p23 and "angle" in edge_data_p23 and edge_data_p23["angle"] is source:
            start = source.p3 if not source.p2.root else source.p2
            d_angle *= -1
        else:
            raise ValueError("Somehow no angle constraint found between the three points")

        rotation_mat = np.array([[np.cos(d_angle), -np.sin(d_angle)], [np.sin(d_angle), np.cos(d_angle)]])
        rotation_point_mat = np.array([[rotation_point.x().value()], [rotation_point.y().value()]])

        # Get all the points that might need to rotate
        all_downstream_points = []
        rotation_handle = None
        for point in networkx.bfs_tree(self, source=start):
            all_downstream_points.append(point)
            if point.rotation_handle:
                rotation_handle = point

        # Get the branch to cut, if there is one
        root_rotation_branch = []
        if source.p2.root and rotation_handle is not None:
            root_rotation_branch = [point for point in networkx.bfs_tree(self, source=rotation_handle)]

        for point in all_downstream_points:
            if point is source.p2 or point in root_rotation_branch:
                continue
            dx_dy = np.array([[point.x().value() - rotation_point.x().value()],
                              [point.y().value() - rotation_point.y().value()]])
            new_xy = (rotation_mat @ dx_dy + rotation_point_mat).flatten()
            point.x().set_value(new_xy[0])
            point.y().set_value(new_xy[1])
            if point not in points_solved:
                points_solved.append(point)
        return points_solved

    def solve(self, source: GeoCon):
        points_solved = []
        symmetry_points_solved = []
        roc_points_solved = []
        if isinstance(source, DistanceConstraint):
            points_solved.extend(self._solve_distance_constraint(source))
        elif isinstance(source, AntiParallel3Constraint) or isinstance(source, Perp3Constraint):
            points_solved.extend(self._solve_perp_parallel_constraint(source))
        elif isinstance(source, RelAngle3Constraint):
            points_solved.extend(self._solve_rel_angle3_constraint(source))
        elif isinstance(source, SymmetryConstraint):
            symmetry_points_solved = self.solve_symmetry_constraint(source)
        elif isinstance(source, ROCurvatureConstraint):
            roc_points_solved = self.solve_roc_constraint(source)

        other_points_solved = self.solve_other_constraints(points_solved)
        other_points_solved = list(
            set(other_points_solved).union(set(symmetry_points_solved)).union(set(roc_points_solved)))

        points_solved = list(set(points_solved).union(set(other_points_solved)))

        networkx.draw_circular(self, labels={point: point.name() for point in self.nodes})
        from matplotlib import pyplot as plt
        plt.show()

        return points_solved

    def move_root(self, root: Point, dx: float, dy: float):
        if not root.root:
            raise ValueError("Cannot move a point that is not a root of a constraint cluster")
        points_solved = []
        for point in networkx.bfs_tree(self, source=root):
            point.x().set_value(point.x().value() + dx)
            point.y().set_value(point.y().value() + dy)
            if point not in points_solved:
                points_solved.append(point)
        self.solve_other_constraints(points_solved)
        return points_solved

    @staticmethod
    def solve_symmetry_constraint(constraint: SymmetryConstraint):
        x1, y1 = constraint.p1.x().value(), constraint.p1.y().value()
        x2, y2 = constraint.p2.x().value(), constraint.p2.y().value()
        x3, y3 = constraint.p3.x().value(), constraint.p3.y().value()
        line_angle = measure_abs_angle(x1, y1, x2, y2)
        tool_angle = measure_rel_angle3(x1, y1, x2, y2, x3, y3)
        if tool_angle < np.pi:
            mirror_angle = line_angle - np.pi / 2
        elif tool_angle > np.pi:
            mirror_angle = line_angle + np.pi / 2
        else:
            # Rare case where the point is coincident with the line: just make p4 = p3
            constraint.p4.request_move(constraint.p3.x().value(), constraint.p3.y().value(), force=True)
            return

        mirror_distance = 2 * measure_point_line_distance_unsigned(x1, y1, x2, y2, x3, y3)
        constraint.p4.request_move(
            x3 + mirror_distance * np.cos(mirror_angle),
            y3 + mirror_distance * np.sin(mirror_angle), force=True
        )

        return constraint.child_nodes

    @staticmethod
    def solve_roc_constraint(constraint: ROCurvatureConstraint):

        def solve_for_single_curve(p_g1: Point, p_g2: Point, n: int):
            Lc = measure_curvature_length_bezier(
                constraint.curve_joint.x().value(), constraint.curve_joint.y().value(),
                p_g1.x().value(), p_g1.y().value(),
                p_g2.x().value(), p_g2.y().value(), constraint.param().value(), n
            )
            angle = p_g1.measure_angle(p_g2)
            p_g2.request_move(p_g1.x().value() + Lc * np.cos(angle),
                              p_g1.y().value() + Lc * np.sin(angle), force=True)

        solve_for_single_curve(constraint.g1_point_curve_1, constraint.g2_point_curve_1, constraint.curve_1.degree)
        solve_for_single_curve(constraint.g1_point_curve_2, constraint.g2_point_curve_2, constraint.curve_2.degree)

        return constraint.child_nodes

    def solve_symmetry_constraints(self, points: typing.List[Point]):
        points_solved = []
        symmetry_constraints_solved = []
        for point in points:
            symmetry_constraints = [geo_con for geo_con in point.geo_cons if isinstance(geo_con, SymmetryConstraint)]
            for symmetry_constraint in symmetry_constraints:
                if symmetry_constraint in symmetry_constraints_solved:
                    continue
                symmetry_points_solved = self.solve_symmetry_constraint(symmetry_constraint)
                symmetry_constraints_solved.append(symmetry_constraint)
                for symmetry_point_solved in symmetry_points_solved:
                    if symmetry_point_solved in points_solved:
                        continue

                    points_solved.append(symmetry_point_solved)

        return points_solved

    def solve_roc_constraints(self, points: typing.List[Point]):
        points_solved = []
        roc_constraints_solved = []
        for point in points:
            roc_constraints = [geo_con for geo_con in point.geo_cons if isinstance(geo_con, ROCurvatureConstraint)]
            for roc_constraint in roc_constraints:
                if roc_constraint in roc_constraints_solved:
                    continue
                roc_points_solved = self.solve_roc_constraint(roc_constraint)
                roc_constraints_solved.append(roc_constraint)
                for roc_point_solved in roc_points_solved:
                    if roc_point_solved in points_solved:
                        continue

                    points_solved.append(roc_point_solved)

        return points_solved

    def solve_other_constraints(self, points: typing.List[Point]):
        symmetry_points_solved = self.solve_symmetry_constraints(points)
        roc_points_solved = self.solve_roc_constraints(points)
        return list(set(symmetry_points_solved).union(roc_points_solved))

    @staticmethod
    def update_canvas_items(points_solved: typing.List[Point]):

        curves_to_update = []
        for point in points_solved:
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
            if airfoil.canvas_item is not None:
                airfoil.canvas_item.generatePicture()

        constraints_to_update = []
        for point in points_solved:
            for geo_con in point.geo_cons:
                if geo_con not in constraints_to_update:
                    constraints_to_update.append(geo_con)

        for geo_con in constraints_to_update:
            if isinstance(geo_con, GeoCon) and geo_con.canvas_item is not None:
                geo_con.canvas_item.update()


class ConstraintValidationError(Exception):
    pass
