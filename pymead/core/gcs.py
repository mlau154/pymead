import networkx

from pymead.core.constraints import *
from pymead.core.constraint_equations import *
from pymead.core.line import PolyLine
from pymead.core.point import Point


class GCS(networkx.DiGraph):
    """
    A Geometric Constraint Solver (GCS) used to maintain constraints between points in pymead, implemented using
    the directed graph class in ``networkx``. Instances of this
    class should not normally be created directly; the geometry collection creates objects of this class when
    ``GeometryCollection.add_constraint`` is called.

    This constraint solver, while not powerful enough to handle arbitrary systems of equations, is nevertheless
    sufficient for handling many common types of constraints needed for airfoil systems. The constraints are solved
    using a graph-constructive approach and maintained using simple rigid body transformations. These transformations
    occur for a given set of points when the parameter value of a constraint "upstream" of this set of points is
    modified. All the points downstream of this constraint are handled as a rigid body (even if they are not
    fully constrained as such) to preserve each of the relative constraints.

    The primary constraints in pymead are distance constraints and relative angle constraints (with specialized options
    for perpendicular and antiparallel constraints). The angle constraints are formed between three points
    (start, vertex, and end). Two other special kinds of constraints, symmetry constraints and radius of curvature
    constraints, are solved following the rigid body transformation.
    """
    def __init__(self):
        """
        Constructor for the geometric constraint solver.
        """
        super().__init__()
        self.points = {}
        self.roots = []
        self.cluster_angle_params = {}
        self.geo_col = None
        self.partial_symmetry_solves = []

    def add_point(self, point: Point):
        """
        Adds a point (node) to the graph (and to the ``"points"`` attribute). Also assigns this graph to the point.

        Parameters
        ----------
        point: Point
            Point to add

        Returns
        -------

        """
        point.gcs = self
        self.add_node(point)
        self.points[point.name()] = point

    def remove_point(self, point: Point):
        """
        Removes a point (node) from the graph (and from the ``"points"`` attribute).

        Parameters
        ----------
        point: Point
            Point to remove

        Returns
        -------

        """
        self.remove_node(point)
        self.points.pop(point.name())

    def _check_if_constraint_creates_new_cluster(self, constraint: GeoCon):
        """
        Determines if a new cluster is formed by adding this constraint by checking if there are any edges connected
        in either direction to any of the points associated with the constraint. Only if no edges are found is a new
        cluster created.

        Parameters
        ----------
        constraint: GeoCon
            Constraint to analyze

        Returns
        -------
        bool
            ``True`` if adding this constraint forms a cluster, ``False`` otherwise

        """
        for point in constraint.child_nodes:
            # If there is already an edge attached to any of the points in this constraint, do not create a new root
            if len(self.in_edges(nbunch=point)) > 0 or len(self.out_edges(nbunch=point)) > 0:
                return False
        return True

    def _set_edge_as_root(self, u: Point, v: Point):
        """
        Sets the given edge as a root by setting the ``u`` value as the root and the ``v`` value as the rotation
        handle of the constraint cluster.

        Parameters
        ----------
        u: Point
            Starting node of the edge

        v: Point
            Terminating node of the edge

        Returns
        -------

        """
        u.root = True
        v.rotation_handle = True
        if u not in [edge[0] for edge in self.roots]:
            self.roots.append((u, v))

        cluster_angle_exceptions = [
            # Do not add a cluster angle parameter if the root was created as part of an antiparallel constraint between
            # a polyline and another curve
            any([isinstance(curve, PolyLine) for curve in u.curves]) and not all(
                [u in curve.point_sequence().points() for curve in u.curves]),

            # Do not add a cluster angle parameter if the root was created as part of an antiparallel constraint between
            # a curve defined by symmetric constraints and a normal Bézier curve
            (SymmetryConstraint.check_if_point_is_symmetric_target(u) and
                SymmetryConstraint.check_if_point_is_symmetric_target(v))
        ]
        if any(cluster_angle_exceptions):
            v.rotation_handle = False  # TODO: check if this change works for other cases
            return

        if v.rotation_param is not None:
            param = v.rotation_param
        else:
            param = self.geo_col.add_param(value=u.measure_angle(v), name="ClusterAngle-1", unit_type="angle", root=u,
                                           rotation_handle=v)
        self.cluster_angle_params[u] = param
        param.gcs = self

    def _delete_root_status(self, root_node: Point, rotation_handle_node: Point = None):
        """
        Removes root status from the given edge.

        Parameters
        ----------
        root_node: Point
            ``u``-value of the edge

        rotation_handle_node: Point
            ``v``-value of the edge

        Returns
        -------

        """
        root_node.root = False
        if rotation_handle_node is not None:
            rotation_handle_node.rotation_handle = False
            rotation_handle_node.rotation_param = None
        if root_node in self.cluster_angle_params:
            cluster_angle_param = self.cluster_angle_params.pop(root_node)
            if cluster_angle_param is not None:
                self.geo_col.remove_pymead_obj(cluster_angle_param, constraint_removal=True)
        root_idx = [r[0] for r in self.roots].index(root_node)
        self.roots.pop(root_idx)

    def _identify_root_from_rotation_handle(self, rotation_handle: Point) -> Point:
        """
        Computes the root of the cluster given the rotation handle as the ``u``-value of the edge incident to the
        rotation handle.

        Parameters
        ----------
        rotation_handle: Point
            Rotation handle of the constraint cluster

        Returns
        -------
        Point
            Root of the constraint cluster

        """
        if not isinstance(rotation_handle, Point):
            raise ValueError(f"Detected rotation handle that is not a point ({rotation_handle = })")
        in_edges = [edge for edge in self.in_edges(nbunch=rotation_handle)]
        if not len(in_edges) == 1:
            raise ValueError("Invalid rotation handle. Rotation should have exactly one incident edge "
                             "(the cluster root)")
        return in_edges[0][0]

    def _identify_rotation_handle(self, root_node: Point):
        """
        Identifies the rotation handle by starting at the root node and testing each of the root node's neighbors
        until the rotation handle is found.

        Parameters
        ----------
        root_node: Point
            Root of the constraint cluster

        Returns
        -------
        Point
            Rotation handle

        """
        for edge in self.out_edges(nbunch=root_node):
            if edge[1].rotation_handle:
                return edge[1]
        raise ValueError("Could not identify rotation handle")

    def _identify_and_delete_root(self, root_node: Point):
        """
        Identifies the root edge by starting at the root node and testing each of the root node's neighbors
        until the rotation handle is found (the ``v``-value of the root edge). Then, ``self._delete_root_status`` is
        applied to demote the edge.

        Parameters
        ----------
        root_node: Point
            Root of the constraint cluster

        Returns
        -------

        """
        for edge in self.out_edges(nbunch=root_node):
            if edge[1].rotation_handle:
                self._delete_root_status(root_node, edge[1])
                return
        self._delete_root_status(root_node=root_node)

    def _get_unique_roots_from_constraint(self, constraint: GeoCon) -> typing.List[Point]:
        """
        Gets the unique roots associated with the addition of the input constraint. Used to determine whether a cluster
        merge should occur.

        Parameters
        ----------
        constraint: GeoCon
            Constraint from which to determine the unique, connected set of cluster roots

        Returns
        -------
        typing.List[Point]
            List of points representing the unique roots found. An error is raised if more than two are found, since
            this should not be possible when adding a constraint.

        """
        unique_roots = []
        for point in constraint.child_nodes:
            root = self._discover_root_from_node(point)
            if root and root not in unique_roots:
                unique_roots.append(root)

        if len(unique_roots) > 2:
            raise ValueError("Found more than two unique roots connected to the constraint being added")

        return unique_roots

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
        for node in networkx.bfs_tree(self, source=source_node, reverse=False):
            if node.root:
                return node
        return None

    def _orient_flow_away_from_root(self, root: Point) -> typing.List[GeoCon]:
        """
        Orients the flow of the edges away from the given root of a constraint cluster. Critical method that ensures
        the rigid body transformation triggered by a given constraint value change applies to the correct set of points
        to preserve the constraint state.

        Parameters
        ----------
        root: Point
            Root of the constraint cluster

        Returns
        -------
        typing.List[GeoCon]
            A list of constraints associated with the edges that were affected by this reorientation. These constraints
            need re-assigning to apply the constraint data to the flipped edges.
        """
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
            constraints_needing_reassign.extend(constraints_to_flip)
            if len(constraints_to_flip) == 0:
                break
        return list(set(constraints_needing_reassign))

    def _check_if_root_flows_into_polyline(self, root_node: Point) -> Point or bool:
        for point in networkx.dfs_preorder_nodes(self, source=root_node):
            if any([isinstance(curve, PolyLine) and point not in curve.point_sequence().points()
                    for curve in point.curves]):
                # This is the case where the root is the newly created tangent point on the polyline
                return False
            if point is root_node:
                continue
            if any([isinstance(curve, PolyLine) for curve in point.curves]):
                return point
        return False

    def move_root(self, new_root: Point):
        old_root = self._discover_root_from_node(new_root)
        self._identify_and_delete_root(old_root)
        needs_set_edge = False
        for nbr in self.adj[new_root]:
            self._set_edge_as_root(new_root, nbr)
            break
        else:
            needs_set_edge = True
        constraints_needing_reassign = self._orient_flow_away_from_root(new_root)
        self._reassign_constraints(constraints_needing_reassign)
        if not needs_set_edge:
            return

        for nbr in self.adj[new_root]:
            self._set_edge_as_root(new_root, nbr)
            break
        else:
            raise ValueError("Failed to move root")

    def _reassign_distance_constraint(self, dist_con: DistanceConstraint):
        """
        Re-assigns distance constraint data to a flipped edge.

        Parameters
        ----------
        dist_con: DistanceConstraint
            Constraint to re-assign

        Returns
        -------

        """
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
        """
        Re-assigns angle constraint data to a swapped edge.

        Parameters
        ----------
        angle_con: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint
            Constraint to re-assign

        Returns
        -------

        """
        edge_data_21 = self.get_edge_data(angle_con.p2, angle_con.p1)
        if edge_data_21 is not None and "angle" not in edge_data_21.keys():
            networkx.set_edge_attributes(self, {(angle_con.p2, angle_con.p1): angle_con}, name="angle")
            return
        edge_data_23 = self.get_edge_data(angle_con.p2, angle_con.p3)
        if edge_data_23 is not None and "angle" not in edge_data_23.keys():
            networkx.set_edge_attributes(self, {(angle_con.p2, angle_con.p3): angle_con}, name="angle")
            return

    def _reassign_constraint(self, constraint: GeoCon):
        """
        Convenience method that applies either ``self._reassign_distance_constraint`` or
        ``self._reassign_angle_constraint`` depending on the type of the input constraint.

        Parameters
        ----------
        constraint: GeoCon
            Constraint to re-assign

        Returns
        -------

        """
        if isinstance(constraint, DistanceConstraint):
            self._reassign_distance_constraint(constraint)
        elif isinstance(constraint, RelAngle3Constraint) or isinstance(constraint, Perp3Constraint) or isinstance(
                constraint, AntiParallel3Constraint):
            self._reassign_angle_constraint(constraint)
        else:
            raise ValueError(f"Constraint reassignment for constraints of type {constraint} are not implemented")

    def _reassign_constraints(self, constraints: typing.List[GeoCon]):
        """
        Convenience method that applies ``self._reassign_constraint`` to a list of constraints

        Parameters
        ----------
        constraints: typing.List[GeoCon]
            List of constraints to re-assign

        Returns
        -------

        """
        for constraint in constraints:
            if (isinstance(constraint, DistanceConstraint) or isinstance(constraint, RelAngle3Constraint) or
                    isinstance(constraint, Perp3Constraint) or isinstance(constraint, AntiParallel3Constraint)):
                self._reassign_constraint(constraint)

    def _check_distance_constraint_for_duplicates(self, dist_con: DistanceConstraint):
        for gc in self.geo_col.container()["geocon"].values():
            if gc is dist_con:
                continue
            if not isinstance(gc, DistanceConstraint):
                continue
            if (dist_con.p1 is gc.p1 and dist_con.p2 is gc.p2) or (dist_con.p1 is gc.p2 and dist_con.p2 is gc.p1):
                raise ValueError("A distance constraint already exists between these two points")

    def _check_angle_constraint_for_duplicates(self, angle_con: RelAngle3Constraint or AntiParallel3Constraint or
                                                                Perp3Constraint):
        for gc in self.geo_col.container()["geocon"].values():
            if gc is angle_con:
                continue
            if not isinstance(gc, RelAngle3Constraint) and not isinstance(
                    gc, AntiParallel3Constraint) and not isinstance(gc, Perp3Constraint):
                continue
            if (gc.p1 is angle_con.p1 and gc.p3 is angle_con.p3) or (
                    gc.p1 is angle_con.p3 and gc.p3 is angle_con.p1):
                raise ValueError("An angle constraint already exists between these three points")

    def check_constraint_for_duplicates(self, geo_con: GeoCon):
        if isinstance(geo_con, DistanceConstraint):
            self._check_distance_constraint_for_duplicates(geo_con)
        elif isinstance(geo_con, RelAngle3Constraint) or isinstance(
                geo_con, AntiParallel3Constraint) or isinstance(geo_con, Perp3Constraint):
            self._check_angle_constraint_for_duplicates(geo_con)

    def _assign_distance_constraint(self, dist_con: DistanceConstraint):
        """
        Assigns a distance constraint by first adding an edge from ``p1`` to ``p2`` with no data and subsequently
        applying the constraint data.

        Parameters
        ----------
        dist_con: DistanceConstraint
            Constraint to assign

        Returns
        -------

        """
        if (SymmetryConstraint.check_if_point_is_symmetric_target(dist_con.p1) and
            SymmetryConstraint.check_if_point_is_symmetric_target(dist_con.p2)):
            raise ValueError("Could not assign distance constraint")
        if SymmetryConstraint.check_if_point_is_symmetric_target(dist_con.p1):
            self.add_edge(dist_con.p1, dist_con.p2)
        elif SymmetryConstraint.check_if_point_is_symmetric_target(dist_con.p2):
            self.add_edge(dist_con.p2, dist_con.p1)
        else:
            self.add_edge(dist_con.p1, dist_con.p2)
        self._reassign_distance_constraint(dist_con)

    def _assign_angle_constraint(self, angle_con: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint):
        """
        Assigns an angle constraint by first adding an edge from either ``p2`` to ``p1`` or ``p2`` to ``p3``
        (depending on which edge already has data) with no data and subsequently applying the constraint data.


        Parameters
        ----------
        angle_con: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint
            Constraint to assign

        Returns
        -------

        """
        if (SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p1) and
                SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p2) and
                SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p3)):
            raise ValueError("Could not add angle constraint")
        if (not (self.get_edge_data(angle_con.p1, angle_con.p2)
                or self.get_edge_data(angle_con.p2, angle_con.p1)) and
                not (SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p1) and
                     SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p2))):
            self.add_edge(angle_con.p2, angle_con.p1)
        if (not (self.get_edge_data(angle_con.p2, angle_con.p3) or
                 self.get_edge_data(angle_con.p3, angle_con.p2)) and
                  not (SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p2) and
                       SymmetryConstraint.check_if_point_is_symmetric_target(angle_con.p3))):
            self.add_edge(angle_con.p2, angle_con.p3)
        self._reassign_angle_constraint(angle_con)

    def _add_ghost_edges_to_angle_constraint(
            self, constraint: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint):
        """
        Adds any required "ghost" edges (edges with no data) to an angle constraint to ensure that there are edges
        connecting both (``p1`` and ``p2``) and (``p2`` and ``p3``). Performing this action ensures that any upstream
        changes in constraint parameter value properly flow down to this angle constraint.

        Parameters
        ----------
        constraint: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint
            Angle constraint to add ghost edges to (if necessary)

        Returns
        -------

        """
        # Only add ghost edges if there is not a concrete edge present (even if facing the wrong direction)
        if not (self.get_edge_data(constraint.p1, constraint.p2)
                or self.get_edge_data(constraint.p2, constraint.p1)):
            if (SymmetryConstraint.check_if_point_is_symmetric_target(constraint.p1) and
                SymmetryConstraint.check_if_point_is_symmetric_target(constraint.p2)):
                self.add_edge(constraint.p1, constraint.p2)
                if constraint.p2.root:
                    self.move_root(constraint.p1)
            else:
                self.add_edge(constraint.p2, constraint.p1)
        if not (self.get_edge_data(constraint.p2, constraint.p3) or
                self.get_edge_data(constraint.p3, constraint.p2)):
            if (SymmetryConstraint.check_if_point_is_symmetric_target(constraint.p2) and
                    SymmetryConstraint.check_if_point_is_symmetric_target(constraint.p3)):
                self.add_edge(constraint.p3, constraint.p2)
                if constraint.p2.root:
                    self.move_root(constraint.p3)
            else:
                self.add_edge(constraint.p2, constraint.p3)

    # def _assign_symmetry_constraint(self, constraint: SymmetryConstraint):
    #     self.add_edge(constraint.p1, constraint.p4)
    #     self.add_edge(constraint.p2, constraint.p4)
    #     self.add_edge(constraint.p3, constraint.p4)

    def _test_if_cluster_is_branching(self, root_node: Point) -> bool:
        """
        Computes the subgraph corresponding to the constraint cluster given by the input root node, and checks if
        this subgraph is branching: each node must have exactly one parent.

        Parameters
        ----------
        root_node: Point
            Root of the cluster to analyze

        Returns
        -------
        bool
            Whether this constraint cluster is branching. If not, an error should be thrown (since closed loops
            of constraints are not possible in pymead)

        """
        subgraph = self.subgraph([node for node in networkx.dfs_preorder_nodes(self, source=root_node)])
        return networkx.is_branching(subgraph)

    def _merge_clusters_with_constraint(self, unique_roots: typing.List[Point]):

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
        if first_constraint_in_cluster and (
                isinstance(constraint, DistanceConstraint) or
                isinstance(constraint, RelAngle3Constraint) or
                isinstance(constraint, Perp3Constraint) or
                isinstance(constraint, AntiParallel3Constraint)):
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
        # elif isinstance(constraint, SymmetryConstraint):
        #     self._assign_symmetry_constraint(constraint)

        if isinstance(constraint, DistanceConstraint) or isinstance(constraint, AntiParallel3Constraint) or isinstance(
                constraint, Perp3Constraint) or isinstance(constraint, RelAngle3Constraint):
            unique_roots = self._get_unique_roots_from_constraint(constraint)
            merge_clusters = False if len(unique_roots) < 2 else True
            if merge_clusters:
                root = self._merge_clusters_with_constraint(unique_roots)
            else:
                constraints_to_reassign = self._orient_flow_away_from_root(unique_roots[0])
                self._reassign_constraints(constraints_to_reassign)
                root = self._discover_root_from_node(constraint.p1)

            # Check if the addition of this constraint creates a closed loop
            is_branching = self._test_if_cluster_is_branching(root)
            if not is_branching:
                raise ValueError("Detected a closed loop in the constraint graph. Closed loop sets of constraints "
                                 "are currently not supported in pymead")

            new_root = self._check_if_root_flows_into_polyline(root)
            if new_root:
                self.move_root(new_root)

        elif isinstance(constraint, SymmetryConstraint):
            points_solved = self.solve_symmetry_constraint(constraint)
            self.update_canvas_items(points_solved)
        elif isinstance(constraint, ROCurvatureConstraint):
            points_solved = self.solve_roc_constraint(constraint)
            self.update_canvas_items(points_solved)

    def _remove_distance_constraint_from_directed_edge(self, constraint: DistanceConstraint):
        edges_removed = None
        edge_data_12 = self.get_edge_data(constraint.p1, constraint.p2)
        if edge_data_12 is not None and "distance" in edge_data_12.keys() and edge_data_12["distance"] is constraint:
            angle_constraint_present = False
            for geo_con in constraint.p2.geo_cons:
                if (isinstance(geo_con, RelAngle3Constraint) or
                        isinstance(geo_con, AntiParallel3Constraint) or
                        isinstance(geo_con, Perp3Constraint) and geo_con.p2 is constraint.p2):
                    angle_constraint_present = True
                    break
            if angle_constraint_present:
                edge_data_12.pop("distance")
            else:
                self.remove_edge(constraint.p1, constraint.p2)
                edges_removed = [(constraint.p1, constraint.p2)]

            return edges_removed

        edge_data_21 = self.get_edge_data(constraint.p2, constraint.p1)
        if edge_data_21 is not None and "distance" in edge_data_21.keys() and edge_data_21["distance"] is constraint:
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
                self.remove_edge(constraint.p2, constraint.p1)
                edges_removed = [(constraint.p2, constraint.p1)]

            return edges_removed

    def _remove_angle_constraint_from_directed_edge(
            self, constraint: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint):
        edges_removed = []
        edge_data_21 = self.get_edge_data(constraint.p2, constraint.p1)
        if edge_data_21 is not None and "angle" in edge_data_21.keys() and edge_data_21["angle"] is constraint:
            if "distance" in edge_data_21.keys():
                edge_data_21.pop("angle")
            else:
                self.remove_edge(constraint.p2, constraint.p1)
                edges_removed = [(constraint.p2, constraint.p1)]

            # Remove the ghost edge if there is one
            edge_data_32 = self.get_edge_data(constraint.p3, constraint.p2)
            if edge_data_32 is not None and len(edge_data_32) == 0:
                self.remove_edge(constraint.p3, constraint.p2)
                edges_removed.append((constraint.p3, constraint.p2))

            return edges_removed

        edge_data_23 = self.get_edge_data(constraint.p2, constraint.p3)
        if edge_data_23 is not None and "angle" in edge_data_23.keys() and edge_data_23["angle"] is constraint:
            if "distance" in edge_data_23.keys():
                edge_data_23.pop("angle")
            else:
                self.remove_edge(constraint.p2, constraint.p3)
                edges_removed = [(constraint.p2, constraint.p3)]

            # Remove the ghost edge if there is one
            edge_data_12 = self.get_edge_data(constraint.p1, constraint.p2)
            if edge_data_12 is not None and len(edge_data_12) == 0:
                self.remove_edge(constraint.p1, constraint.p2)
                edges_removed.append((constraint.p1, constraint.p2))

            return edges_removed

    def _assign_new_root_if_required(self, v_of_edge_removed: Point):
        neighbors_of_v = [nbr for nbr in self.adj[v_of_edge_removed]]
        if len(neighbors_of_v) > 0:
            self._set_edge_as_root(v_of_edge_removed, neighbors_of_v[0])
        else:
            pass  # This means we trimmed the end of the branch

    def _update_roots_based_on_constraint_removal(self, edges_removed: typing.List[tuple]):
        for edge_removed in edges_removed:
            if edge_removed[1].rotation_handle:
                self._delete_root_status(edge_removed[0], edge_removed[1])
                for out_edge in self.out_edges(nbunch=edge_removed[0]):
                    self._set_edge_as_root(out_edge[0], out_edge[1])
                    break
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
            if constraint in child_node.x().geo_cons:
                child_node.x().geo_cons.remove(constraint)
            if constraint in child_node.y().geo_cons:
                child_node.y().geo_cons.remove(constraint)

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
            point.x().set_value(point.x().value() + dx, direct_user_request=False)
            point.y().set_value(point.y().value() + dy, direct_user_request=False)
            if point not in points_solved:
                points_solved.append(point)

        return points_solved

    def _solve_angle_constraint(self, source: RelAngle3Constraint or AntiParallel3Constraint or Perp3Constraint):

        points_solved = []
        edge_data_p21 = self.get_edge_data(source.p2, source.p1)
        edge_data_p23 = self.get_edge_data(source.p2, source.p3)

        old_angle = (source.p2.measure_angle(source.p1) -
                     source.p2.measure_angle(source.p3)) % (2 * np.pi)
        if isinstance(source, RelAngle3Constraint):
            new_angle = source.param().rad()
        elif isinstance(source, AntiParallel3Constraint):
            new_angle = np.pi
        elif isinstance(source, Perp3Constraint):
            new_angle = np.pi / 2
        else:
            raise ValueError(f"{type(source)} is not a valid type for angle constraint")
        d_angle = new_angle - old_angle
        rotation_point = source.p2

        if edge_data_p21 and "angle" in edge_data_p21 and edge_data_p21["angle"] is source:
            if source.p1.rotation_handle and source.p2.root:
                start = source.p3
                d_angle *= -1
            else:
                start = source.p1
        elif edge_data_p23 and "angle" in edge_data_p23 and edge_data_p23["angle"] is source:
            if source.p3.rotation_handle and source.p2.root:
                start = source.p1
            else:
                start = source.p3
                d_angle *= -1
        else:
            raise ValueError("Somehow no angle constraint found between the three points")

        rotation_mat = np.array([[np.cos(d_angle), -np.sin(d_angle)], [np.sin(d_angle), np.cos(d_angle)]])
        rotation_point_mat = np.array([[rotation_point.x().value()], [rotation_point.y().value()]])

        # Get all the points that might need to rotate
        additional_branch_starting_points = []
        if start is not source.p2:
            for geo_con in start.geo_cons:
                if geo_con is source:
                    continue
                if not (isinstance(geo_con, AntiParallel3Constraint) or isinstance(
                        geo_con, RelAngle3Constraint) or isinstance(geo_con, Perp3Constraint)):
                    continue
                if geo_con.p2 is not source.p2:
                    continue
                # if not geo_con.p1 in self.adj[source.p2] or not geo_con.p3 in self.adj[source.p2]:
                #     continue
                if start is source.p3 and geo_con.p3 is source.p3:
                    additional_branch_starting_points.append(geo_con.p1)
                elif start is source.p3 and geo_con.p1 is source.p3:
                    additional_branch_starting_points.append(geo_con.p3)
                elif start is source.p1 and geo_con.p3 is source.p1:
                    additional_branch_starting_points.append(geo_con.p1)
                elif start is source.p1 and geo_con.p1 is source.p1:
                    additional_branch_starting_points.append(geo_con.p3)

        all_downstream_points = []
        rotation_handle = None
        for source_point in [start, *additional_branch_starting_points]:
            for point in networkx.bfs_tree(self, source=source_point):
                # if point not in points_to_exclude:
                all_downstream_points.append(point)
                if not rotation_handle and point.rotation_handle:
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
            point.x().set_value(new_xy[0], direct_user_request=False)
            point.y().set_value(new_xy[1], direct_user_request=False)
            if point not in points_solved:
                points_solved.append(point)
        return points_solved

    def solve(self, source: GeoCon):

        points_solved = []
        symmetry_points_solved = []
        roc_points_solved = []
        if isinstance(source, DistanceConstraint):
            points_solved.extend(self._solve_distance_constraint(source))
        elif (isinstance(source, RelAngle3Constraint) or isinstance(source, AntiParallel3Constraint) or
              isinstance(source, Perp3Constraint)):
            points_solved.extend(self._solve_angle_constraint(source))
        elif isinstance(source, SymmetryConstraint):
            points_solved.extend(self.solve_symmetry_constraint(source))
        elif isinstance(source, ROCurvatureConstraint):
            points_solved.extend(self.solve_roc_constraint(source))

        other_points_solved = self.solve_other_constraints(points_solved)
        other_points_solved = list(
            set(other_points_solved).union(
                set(symmetry_points_solved)).union(
                set(roc_points_solved))
        )

        points_solved = list(set(points_solved).union(set(other_points_solved)))

        return points_solved

    def translate_cluster(self, root: Point, dx: float, dy: float):
        if not root.root:
            raise ValueError("Cannot move a point that is not a root of a constraint cluster")
        points_solved = []
        for point in networkx.bfs_tree(self, source=root):
            point.x().set_value(point.x().value() + dx, direct_user_request=False)
            point.y().set_value(point.y().value() + dy, direct_user_request=False)
            if point not in points_solved:
                points_solved.append(point)
        self.solve_other_constraints(points_solved)
        return points_solved

    def rotate_cluster(self, rotation_handle: Point, new_rotation_handle_x: float = None,
                       new_rotation_handle_y: float = None,
                       new_rotation_angle: float = None):
        root = self._identify_root_from_rotation_handle(rotation_handle)
        if not root.root:
            raise ValueError("Cannot move a point that is not a root of a constraint cluster")
        old_rotation_handle_angle = root.measure_angle(rotation_handle)
        if new_rotation_angle is None:
            new_rotation_handle_angle = root.measure_angle(Point(new_rotation_handle_x, new_rotation_handle_y))
        else:
            new_rotation_handle_angle = new_rotation_angle
        delta_angle = new_rotation_handle_angle - old_rotation_handle_angle
        root_x = root.x().value()
        root_y = root.y().value()

        points_solved = []
        for point in networkx.bfs_tree(self, source=root):
            if point is root:
                continue
            old_x = point.x().value()
            old_y = point.y().value()
            new_x = (old_x - root_x) * np.cos(delta_angle) - (old_y - root_y) * np.sin(delta_angle) + root_x
            new_y = (old_x - root_x) * np.sin(delta_angle) + (old_y - root_y) * np.cos(delta_angle) + root_y
            point.x().set_value(new_x, direct_user_request=False)
            point.y().set_value(new_y, direct_user_request=False)
            if point not in points_solved:
                points_solved.append(point)
        self.solve_other_constraints(points_solved)
        return points_solved, root

    def solve_symmetry_constraint(self, constraint: SymmetryConstraint):
        # This code prevents recursion errors for the case where a new cluster is created from a symmetry constraint
        # target. The symmetry constraint only gets solved if it has not yet been added to the list of partial solves
        if constraint in self.partial_symmetry_solves:
            return []
        else:
            self.partial_symmetry_solves.append(constraint)

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
            return constraint.child_nodes

        mirror_distance = 2 * measure_point_line_distance_unsigned(x1, y1, x2, y2, x3, y3)
        constraint.p4.request_move(
            x3 + mirror_distance * np.cos(mirror_angle),
            y3 + mirror_distance * np.sin(mirror_angle), force=True
        )

        self.partial_symmetry_solves.remove(constraint)

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

        def solve_for_single_curve_zero_curvature(p_g1: Point, p_g2: Point):
            p_g2.request_move(p_g1.x().value(), p_g1.y().value(), force=True)

        if constraint.curve_type_1 == "Bezier" and constraint.curve_type_2 == "LineSegment":
            solve_for_single_curve_zero_curvature(constraint.g1_point_curve_1, constraint.g2_point_curve_1)
            return constraint.child_nodes

        if constraint.curve_type_2 == "Bezier" and constraint.curve_type_1 == "LineSegment":
            solve_for_single_curve_zero_curvature(constraint.g1_point_curve_2, constraint.g2_point_curve_2)
            return constraint.child_nodes

        if constraint.curve_type_1 == "Bezier":
            if constraint.is_solving_allowed(constraint.g2_point_curve_1):
                solve_for_single_curve(constraint.g1_point_curve_1, constraint.g2_point_curve_1, constraint.curve_1.degree)
            else:
                R1 = ROCurvatureConstraint.calculate_curvature_data(constraint.curve_joint).R1
                constraint.param().set_value(R1, force=True)
        if constraint.curve_type_2 == "Bezier":
            if constraint.is_solving_allowed(constraint.g2_point_curve_2):
                solve_for_single_curve(constraint.g1_point_curve_2, constraint.g2_point_curve_2, constraint.curve_2.degree)
            else:
                R2 = ROCurvatureConstraint.calculate_curvature_data(constraint.curve_joint).R2
                constraint.param().set_value(R2, force=True)

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
        roc_points_solved = self.solve_roc_constraints(points)
        symmetry_points_solved = self.solve_symmetry_constraints(points)
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
