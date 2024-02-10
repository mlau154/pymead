import networkx

from pymead.core.constraints import *
from pymead.core.constraint_equations import *
from pymead.core.point import Point


class GCS2(networkx.DiGraph):
    def __init__(self):
        super().__init__()
        self.points = {}
        self.roots = []

    def _add_point(self, point: Point):
        point.gcs = self
        self.add_node(point)
        self.points[point.name()] = point

    def _remove_point(self, point: Point):
        self.remove_node(point)
        self.points.pop(point.name())

    def _check_if_constraint_creates_new_cluster(self, constraint: GeoCon):
        for point in constraint.child_nodes:
            # If there is already an edge attached to any of the points in this constraint, do not create a new root
            if len(self.in_edges(nbunch=point)) > 0 or len(self.out_edges(nbunch=point)) > 0:
                return False
        return True

    def _set_edge_as_root(self, u: Point, v: Point):
        u.root = True
        v.rotation_handle = True
        if u not in [edge[0] for edge in self.roots]:
            self.roots.append((u, v))

    def _delete_root_status(self, root_node: Point, rotation_handle_node: Point):
        print(f"{root_node = }")
        root_node.root = False
        rotation_handle_node.rotation_handle = False
        root_idx = [r[0] for r in self.roots].index(root_node)
        self.roots.pop(root_idx)

    def _get_unique_roots_from_constraint(self, constraint: GeoCon):
        unique_roots = []
        for point in constraint.child_nodes:
            root = self._discover_root_from_node(point)
            if root and root not in unique_roots:
                unique_roots.append(root)

        if len(unique_roots) > 2:
            raise ValueError("Found more than two unique roots connected to the constraint being added")

        return unique_roots

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
        # for node in networkx.bfs_tree(self, source=source_node, reverse=False):
        #     if node.root:
        #         return node
        return None

    def _orient_flow_away_from_root(self, root: Point):
        constraints_needing_reassign = []
        while True:
            constraints_to_flip = []
            points_from_root = [point for point in networkx.dfs_preorder_nodes(self, source=root)]
            for point in points_from_root:
                in_edges = self.in_edges(nbunch=point)
                u_values = [in_edge[0] for in_edge in in_edges]
                v_values = [in_edge[1] for in_edge in in_edges]
                set_difference = list(set(u_values) - set(points_from_root))
                for u_value in set_difference:
                    constraints_to_flip.extend(u_value.geo_cons)
                    in_edge_idx = u_values.index(u_value)
                    v = v_values[in_edge_idx]
                    self.remove_edge(u_value, v)
                    self.add_edge(v, u_value)
                    print(f"Removing edge {u_value}, {v} and adding edge {v}, {u_value}")
            constraints_needing_reassign.extend(constraints_to_flip)
            if len(constraints_to_flip) == 0:
                break
        return list(set(constraints_needing_reassign))

    def _reassign_distance_constraint(self, dist_con: DistanceConstraint):
        edge_data_12 = self.get_edge_data(dist_con.p1, dist_con.p2)
        if edge_data_12 is not None:
            networkx.set_edge_attributes(self, {(dist_con.p1, dist_con.p2): dist_con}, name="distance")
            return
        edge_data_21 = self.get_edge_data(dist_con.p2, dist_con.p1)
        if edge_data_21 is not None:
            networkx.set_edge_attributes(self, {(dist_con.p2, dist_con.p1): dist_con}, name="distance")
            return
        raise ValueError(f"Could not reassign distance constraint {dist_con}")

    def _reassign_angle_constraint(self, angle_con: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint):
        edge_data_21 = self.get_edge_data(angle_con.p2, angle_con.p1)
        if edge_data_21 is not None:
            networkx.set_edge_attributes(self, {(angle_con.p2, angle_con.p1): angle_con}, name="angle")
            return
        edge_data_23 = self.get_edge_data(angle_con.p2, angle_con.p3)
        if edge_data_23 is not None:
            networkx.set_edge_attributes(self, {(angle_con.p2, angle_con.p3): angle_con}, name="angle")
            return
        raise ValueError(f"Could not reassign angle constraint {angle_con}")

    def _reassign_constraint(self, constraint: GeoCon):
        if isinstance(constraint, DistanceConstraint):
            self._reassign_distance_constraint(constraint)
        elif isinstance(constraint, RelAngle3Constraint) or isinstance(constraint, Perp3Constraint) or isinstance(
                constraint, AntiParallel3Constraint):
            self._reassign_angle_constraint(constraint)
        else:
            raise ValueError(f"Constraint reassignment for constraints of type {constraint} are not implemented")

    def _reassign_constraints(self, constraints: typing.List[GeoCon]):
        for constraint in constraints:
            self._reassign_constraint(constraint)

    def _assign_distance_constraint(self, dist_con: DistanceConstraint):
        self.add_edge(dist_con.p1, dist_con.p2)
        self._reassign_distance_constraint(dist_con)

    def _assign_angle_constraint(self, angle_con: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint):
        if not (self.get_edge_data(angle_con.p1, angle_con.p2)
                or self.get_edge_data(angle_con.p2, angle_con.p1)):
            self.add_edge(angle_con.p2, angle_con.p1)
        if not (self.get_edge_data(angle_con.p2, angle_con.p3) or
                self.get_edge_data(angle_con.p3, angle_con.p2)):
            self.add_edge(angle_con.p2, angle_con.p3)
        self._reassign_angle_constraint(angle_con)

    def _add_ghost_edges_to_angle_constraint(self, constraint: RelAngle3Constraint or AntiParallel3Constraint or
                                                               Perp3Constraint):
        # Only add ghost edges if there is not a concrete edge present (even if facing the wrong direction)
        if not (self.get_edge_data(constraint.p1, constraint.p2)
                or self.get_edge_data(constraint.p2, constraint.p1)):
            self.add_edge(constraint.p2, constraint.p1)
        if not (self.get_edge_data(constraint.p2, constraint.p3) or
                self.get_edge_data(constraint.p3, constraint.p2)):
            self.add_edge(constraint.p2, constraint.p3)

    def _test_if_cluster_is_branching(self, root_node: Point):
        subgraph = self.subgraph([node for node in networkx.dfs_preorder_nodes(self, source=root_node)])
        return networkx.is_branching(subgraph)

    def _merge_clusters_with_constraint(self, constraint: GeoCon, unique_roots: typing.List[Point]):

        def _determine_merged_cluster_root():
            current_roots = [r[0] for r in self.roots]
            if current_roots.index(unique_roots[0]) < current_roots.index(unique_roots[1]):
                return unique_roots[0]
            else:
                return unique_roots[1]

        def _delete_root_status_of_other_roots(new_root: Point):
            for root in unique_roots:
                if root is new_root:
                    continue
                self._identify_and_delete_root(root)
                break

        merged_cluster_root = _determine_merged_cluster_root()
        _delete_root_status_of_other_roots(merged_cluster_root)
        constraints_needing_reassign = self._orient_flow_away_from_root(merged_cluster_root)
        self._reassign_constraints(constraints_needing_reassign)
        return merged_cluster_root

    def add_constraint(self, constraint: GeoCon):

        # Check if this constraint creates a new cluster
        first_constraint_in_cluster = self._check_if_constraint_creates_new_cluster(constraint)

        # If it does, set it as the root constraint
        if first_constraint_in_cluster and (isinstance(constraint, DistanceConstraint) or isinstance(
                constraint, RelAngle3Constraint) or isinstance(
            constraint, Perp3Constraint) or isinstance(
            constraint, AntiParallel3Constraint)):
            self._set_edge_as_root(constraint.p1, constraint.p2)

        # If the constraint has a Param associated with it, pass the GCS reference to this parameter
        if constraint.param() is not None:
            constraint.param().gcs = self

        if isinstance(constraint, DistanceConstraint):
            self._assign_distance_constraint(constraint)
        elif isinstance(constraint, RelAngle3Constraint) or isinstance(
                constraint, AntiParallel3Constraint) or isinstance(constraint, Perp3Constraint):
            self._assign_angle_constraint(constraint)
            self._add_ghost_edges_to_angle_constraint(constraint)

        if isinstance(constraint, DistanceConstraint) or isinstance(constraint, AntiParallel3Constraint) or isinstance(
                constraint, Perp3Constraint) or isinstance(constraint, RelAngle3Constraint):
            unique_roots = self._get_unique_roots_from_constraint(constraint)
            merge_clusters = False if len(unique_roots) < 2 else True
            if merge_clusters:
                root = self._merge_clusters_with_constraint(constraint, unique_roots)
            else:
                constraints_to_reassign = self._orient_flow_away_from_root(unique_roots[0])
                self._reassign_constraints(constraints_to_reassign)
                root = self._discover_root_from_node(constraint.p1)

            print(f"{root = }")

            # Check if the addition of this constraint creates a closed loop
            is_branching = self._test_if_cluster_is_branching(root)
            print(f"{is_branching = }")
            if not is_branching:
                raise ValueError("Detected a closed loop in the constraint graph. Closed loop sets of constraints "
                                 "are currently not supported in pymead")

        elif isinstance(constraint, SymmetryConstraint):
            points_solved = self.solve_symmetry_constraint(constraint)
            self.update_canvas_items(points_solved)
        elif isinstance(constraint, ROCurvatureConstraint):
            points_solved = self.solve_roc_constraint(constraint)
            self.update_canvas_items(points_solved)

    def _identify_and_delete_root(self, root_node: Point):
        for edge in self.out_edges(nbunch=root_node):
            if edge[1].rotation_handle:
                self._delete_root_status(root_node, edge[1])
                return
        raise ValueError("Could not identify root to remove")

    def _remove_distance_constraint_from_directed_edge(self, constraint: DistanceConstraint):
        edges_removed = None
        edge_data_12 = self.get_edge_data(constraint.p1, constraint.p2)
        if edge_data_12 is not None and "distance" in edge_data_12.keys():
            angle_constraint_present = False
            for geo_con in constraint.p2.geo_cons:
                if isinstance(
                        geo_con, RelAngle3Constraint) or isinstance(
                    geo_con, AntiParallel3Constraint) or isinstance(
                    geo_con, Perp3Constraint) and geo_con.p2 is constraint.p2:
                    angle_constraint_present = True
                    break
            if angle_constraint_present:
                edge_data_12.pop("distance")
            else:
                if constraint.p1.root:
                    self._identify_and_delete_root(constraint.p1)
                self.remove_edge(constraint.p1, constraint.p2)
                edges_removed = [(constraint.p1, constraint.p2)]

            return edges_removed

        edge_data_21 = self.get_edge_data(constraint.p2, constraint.p1)
        if edge_data_21 is not None and "distance" in edge_data_21.keys():
            angle_constraint_present = False
            for geo_con in constraint.p1.geo_cons:
                if (isinstance(geo_con, RelAngle3Constraint) or
                        isinstance(geo_con, AntiParallel3Constraint) or
                        isinstance(geo_con, Perp3Constraint) and geo_con.p2 is constraint.p1):
                    angle_constraint_present = True
                    break
            if angle_constraint_present:
                edge_data_21.pop("distance")
            else:
                if constraint.p2.root:
                    self._identify_and_delete_root(constraint.p2)
                self.remove_edge(constraint.p2, constraint.p1)
                edges_removed = [(constraint.p2, constraint.p1)]

            return edges_removed

        raise ValueError(f"Failed to remove distance constraint {constraint}")

    def _remove_angle_constraint_from_directed_edge(self, constraint: RelAngle3Constraint or
                                                                      AntiParallel3Constraint or
                                                                      Perp3Constraint):
        edges_removed = None
        edge_data_21 = self.get_edge_data(constraint.p2, constraint.p1)
        if edge_data_21 is not None and "angle" in edge_data_21.keys():
            if "distance" in edge_data_21.keys():
                edge_data_21.pop("angle")
            else:
                self.remove_edge(constraint.p2, constraint.p1)
                edges_removed = [(constraint.p2, constraint.p1)]

            # Remove the ghost edge if there is one
            edge_data_32 = self.get_edge_data(constraint.p3, constraint.p2)
            if edge_data_32 is not None and len(edge_data_32) == 0:
                if constraint.p3.root:
                    self._identify_and_delete_root(constraint.p3)
                self.remove_edge(constraint.p3, constraint.p2)
                edges_removed.append((constraint.p3, constraint.p2))

            return edges_removed

        edge_data_23 = self.get_edge_data(constraint.p2, constraint.p3)
        if edge_data_23 is not None and "angle" in edge_data_23.keys():
            if "distance" in edge_data_23.keys():
                edge_data_23.pop("angle")
            else:
                self.remove_edge(constraint.p2, constraint.p3)
                edges_removed = [(constraint.p2, constraint.p3)]

            # Remove the ghost edge if there is one
            edge_data_12 = self.get_edge_data(constraint.p1, constraint.p2)
            if edge_data_12 is not None and len(edge_data_12) == 0:
                if constraint.p1.root:
                    self._identify_and_delete_root(constraint.p1)
                self.remove_edge(constraint.p1, constraint.p2)
                edges_removed.append((constraint.p1, constraint.p2))

            return edges_removed

        raise ValueError(f"Failed to remove angle constraint {constraint}")

    def _assign_new_root_if_required(self, v_of_edge_removed: Point):
        neighbors_of_v = [nbr for nbr in self.adj[v_of_edge_removed]]
        if len(neighbors_of_v) > 0:
            self._set_edge_as_root(v_of_edge_removed, neighbors_of_v[0])
        else:
            pass  # This means we trimmed the end of the branch

    def _update_roots_based_on_constraint_removal(self, edges_removed: typing.List[tuple]):
        for edge_removed in edges_removed:
            self._assign_new_root_if_required(edge_removed[1])

    def remove_constraint(self, constraint: GeoCon):
        edges_removed = None
        if isinstance(constraint, DistanceConstraint):
            edges_removed = self._remove_distance_constraint_from_directed_edge(constraint)
        elif isinstance(constraint, RelAngle3Constraint) or isinstance(
                constraint, AntiParallel3Constraint) or isinstance(constraint, Perp3Constraint):
            edges_removed = self._remove_angle_constraint_from_directed_edge(constraint)

        if edges_removed is not None:
            self._update_roots_based_on_constraint_removal(edges_removed)

        for child_node in constraint.child_nodes:
            child_node.geo_cons.remove(constraint)

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
