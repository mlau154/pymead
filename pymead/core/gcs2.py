import networkx

from pymead.core.constraints import *
from pymead.core.constraint_equations import *
from pymead.core.point import Point


class GCS2(networkx.DiGraph):
    def __init__(self):
        super().__init__()
        self.points = {}
        self.clusters = {}

    def add_point(self, point: Point):
        point.gcs = self
        self.add_node(point)
        self.points[point.name()] = point

    def remove_point(self, point: Point):
        self.remove_node(point)
        self.points.pop(point.name())

    def add_constraint(self, constraint: GeoCon):

        for point in constraint.child_nodes:
            # If there is already an edge attached to any of the points in this constraint, do not create a new root
            if len(self.in_edges(nbunch=point)) > 0 or len(self.out_edges(nbunch=point)) > 0:
                first_constraint_in_cluster = False
                break
        else:
            first_constraint_in_cluster = True

        if first_constraint_in_cluster:
            constraint.child_nodes[0].root = True

        if constraint.param() is not None:
            constraint.param().gcs = self

        if isinstance(constraint, DistanceConstraint):
            in_edges_p1 = tuple([edge for edge in self.in_edges(nbunch=constraint.p1, data=True) if "distance" in edge[2].keys()])
            in_edges_p2 = tuple([edge for edge in self.in_edges(nbunch=constraint.p2, data=True) if "distance" in edge[2].keys()])
            if len(in_edges_p1) > 0 and len(in_edges_p2) > 0:
                raise ConstraintValidationError(f"Failed to draw distance constraint {constraint} due to identified"
                                                f" incident distance constraints on both start and end points")

            if len(in_edges_p1) > 0:
                edge_data = self.get_edge_data(constraint.p1, constraint.p2)
                if edge_data:
                    if "distance" in self.get_edge_data(constraint.p1, constraint.p2).keys():
                        raise ConstraintValidationError(f"Cannot add a duplicate distance constraint between"
                                                        f" {constraint.p1} and {constraint.p2}")
                    networkx.set_edge_attributes(self, {(constraint.p1, constraint.p2): constraint},
                                                 name="distance")
                    return
                self.add_edge(constraint.p1, constraint.p2, distance=constraint)
                return
            if len(in_edges_p2) > 0:
                edge_data = self.get_edge_data(constraint.p2, constraint.p1)
                if edge_data:
                    if "distance" in self.get_edge_data(constraint.p2, constraint.p1).keys():
                        raise ConstraintValidationError(f"Cannot add a duplicate distance constraint between"
                                                        f" {constraint.p2} and {constraint.p1}")
                    networkx.set_edge_attributes(self, {(constraint.p2, constraint.p1): constraint},
                                                 name="distance")
                    return
                self.add_edge(constraint.p2, constraint.p1, distance=constraint)
                return

            # Default case
            self.add_edge(constraint.p1, constraint.p2, distance=constraint)
            return

        elif isinstance(constraint, RelAngle3Constraint):
            in_edges_p1 = tuple([edge for edge in self.in_edges(nbunch=constraint.start_point, data=True)])
            in_edges_p2 = tuple([edge for edge in self.in_edges(nbunch=constraint.vertex, data=True)])
            in_edges_p3 = tuple([edge for edge in self.in_edges(nbunch=constraint.end_point, data=True)])

            # edge_data_p12 = self.get_edge_data(constraint.start_point, constraint.vertex)
            edge_data_p21 = self.get_edge_data(constraint.vertex, constraint.start_point)
            edge_data_p23 = self.get_edge_data(constraint.vertex, constraint.end_point)
            # edge_data_p32 = self.get_edge_data(constraint.end_point, constraint.vertex)

            # angle_in_p12 = False if not edge_data_p12 else "angle" in edge_data_p12.keys()
            angle_in_p21 = False if not edge_data_p21 else "angle" in edge_data_p21.keys()
            angle_in_p23 = False if not edge_data_p23 else "angle" in edge_data_p23.keys()
            # angle_in_p32 = False if not edge_data_p32 else "angle" in edge_data_p32.keys()

            # if (angle_in_p12 or angle_in_p21) and (angle_in_p23 or angle_in_p32):
            if angle_in_p21 and angle_in_p23:
                raise ConstraintValidationError(f"{constraint} already has angle constraints associated with both"
                                                f" pairs of points")

            # if edge_data_p12:
            #     networkx.set_edge_attributes(self, {(constraint.start_point, constraint.vertex): constraint}, name="angle")
            #     return
            if edge_data_p21:
                networkx.set_edge_attributes(self, {(constraint.vertex, constraint.start_point): constraint}, name="angle")
                return
            if edge_data_p23:
                networkx.set_edge_attributes(self, {(constraint.vertex, constraint.end_point): constraint}, name="angle")
                return
            # if edge_data_p32:
            #     networkx.set_edge_attributes(self, {(constraint.end_point, constraint.vertex): constraint}, name="angle")
            #     return

            if len(in_edges_p1) > 0:
                # if angle_in_p12 or angle_in_p21:
                if angle_in_p21:
                    self.add_edge(constraint.vertex, constraint.end_point, angle=constraint)
                    return
                # if angle_in_p23 or angle_in_p32:
                if angle_in_p23:
                    raise ValueError("Cannot create a valid angle constraint from this case")
                self.add_edge(constraint.vertex, constraint.end_point, angle=constraint)
                return
            if len(in_edges_p2) > 0:
                # if angle_in_p12 or angle_in_p21:
                if angle_in_p21:
                    self.add_edge(constraint.vertex, constraint.end_point, angle=constraint)
                    return
                # if angle_in_p23 or angle_in_p32:
                if angle_in_p23:
                    self.add_edge(constraint.vertex, constraint.start_point, angle=constraint)
                    return
                self.add_edge(constraint.vertex, constraint.end_point, angle=constraint)
                return
            if len(in_edges_p3) > 0:
                # if angle_in_p12 or angle_in_p21:
                if angle_in_p21:
                    raise ValueError("Cannot create a valid angle constraint from this case")
                # if angle_in_p23 or angle_in_p32:
                if angle_in_p23:
                    self.add_edge(constraint.vertex, constraint.start_point, angle=constraint)
                    return
                self.add_edge(constraint.vertex, constraint.start_point, angle=constraint)
                return

            self.add_edge(constraint.vertex, constraint.end_point, angle=constraint)
            return
        elif isinstance(constraint, AntiParallel3Constraint) or isinstance(constraint, Perp3Constraint):
            in_edges_p1 = tuple([edge for edge in self.in_edges(nbunch=constraint.p1, data=True)])
            in_edges_p2 = tuple([edge for edge in self.in_edges(nbunch=constraint.p2, data=True)])
            in_edges_p3 = tuple([edge for edge in self.in_edges(nbunch=constraint.p3, data=True)])

            # edge_data_p12 = self.get_edge_data(constraint.p1, constraint.p2)
            edge_data_p21 = self.get_edge_data(constraint.p2, constraint.p1)
            edge_data_p23 = self.get_edge_data(constraint.p2, constraint.p3)
            # edge_data_p32 = self.get_edge_data(constraint.p3, constraint.p2)

            # angle_in_p12 = False if not edge_data_p12 else "angle" in edge_data_p12.keys()
            angle_in_p21 = False if not edge_data_p21 else "angle" in edge_data_p21.keys()
            angle_in_p23 = False if not edge_data_p23 else "angle" in edge_data_p23.keys()
            # angle_in_p32 = False if not edge_data_p32 else "angle" in edge_data_p32.keys()

            # if (angle_in_p12 or angle_in_p21) and (angle_in_p23 or angle_in_p32):
            if angle_in_p21 and angle_in_p23:
                raise ConstraintValidationError(f"{constraint} already has angle constraints associated with both"
                                                f" pairs of points")

            # if edge_data_p12:
            #     networkx.set_edge_attributes(self, {(constraint.p1, constraint.p2): constraint}, name="angle")
            #     return
            if edge_data_p21:
                networkx.set_edge_attributes(self, {(constraint.p2, constraint.p1): constraint}, name="angle")
                return
            if edge_data_p23:
                networkx.set_edge_attributes(self, {(constraint.p2, constraint.p3): constraint}, name="angle")
                return
            # if edge_data_p32:
            #     networkx.set_edge_attributes(self, {(constraint.p3, constraint.p2): constraint}, name="angle")
            #     return

            if len(in_edges_p1) > 0:
                # if angle_in_p12 or angle_in_p21:
                if angle_in_p21:
                    self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                    return
                # if angle_in_p23 or angle_in_p32:
                if angle_in_p23:
                    raise ValueError("Cannot create a valid angle constraint from this case")
                self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                return
            if len(in_edges_p2) > 0:
                # if angle_in_p12 or angle_in_p21:
                if angle_in_p21:
                    self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                    return
                # if angle_in_p23 or angle_in_p32:
                if angle_in_p23:
                    self.add_edge(constraint.p2, constraint.p1, angle=constraint)
                    return
                self.add_edge(constraint.p2, constraint.p3, angle=constraint)
                return
            if len(in_edges_p3) > 0:
                # if angle_in_p12 or angle_in_p21:
                if angle_in_p21:
                    raise ValueError("Cannot create a valid angle constraint from this case")
                # if angle_in_p23 or angle_in_p32:
                if angle_in_p23:
                    self.add_edge(constraint.p2, constraint.p1, angle=constraint)
                    return
                self.add_edge(constraint.p2, constraint.p1, angle=constraint)
                return

            self.add_edge(constraint.p2, constraint.p3, angle=constraint)
            return
        elif isinstance(constraint, SymmetryConstraint):
            self.solve_symmetry_constraint(constraint)
            self.update_canvas_items(constraint)
        elif isinstance(constraint, ROCurvatureConstraint):
            self.solve_roc_constraint(constraint)
            self.update_canvas_items(constraint)

    def remove_constraint(self, constraint: GeoCon):
        raise NotImplementedError("Constraint removal not yet implemented")

    def make_point_copies(self):
        return {point: Point(x=point.x().value(), y=point.y().value()) for point in self.points.values()}

    def solve(self, source: GeoCon):
        points_solved = []
        if isinstance(source, DistanceConstraint):
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
        elif isinstance(source, AntiParallel3Constraint) or isinstance(source, Perp3Constraint):
            edge_data_p21 = self.get_edge_data(source.p2, source.p1)
            edge_data_p23 = self.get_edge_data(source.p2, source.p3)

            old_angle = (source.p2.measure_angle(source.p1) - source.p2.measure_angle(source.p3)) % (2 * np.pi)
            new_angle = np.pi if isinstance(source, AntiParallel3Constraint) else np.pi / 2
            d_angle = new_angle - old_angle
            rotation_point = source.p2

            if edge_data_p21 and edge_data_p21["angle"] is source:
                start = source.p1
                # d_angle *= -1
            elif edge_data_p23 and edge_data_p23["angle"] is source:
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

        elif isinstance(source, RelAngle3Constraint):
            edge_data_p21 = self.get_edge_data(source.vertex, source.start_point)
            edge_data_p23 = self.get_edge_data(source.vertex, source.end_point)

            old_angle = (source.vertex.measure_angle(source.start_point) -
                         source.vertex.measure_angle(source.end_point)) % (2 * np.pi)
            new_angle = source.param().rad()
            d_angle = new_angle - old_angle
            rotation_point = source.vertex

            if edge_data_p21 and edge_data_p21["angle"] is source:
                start = source.start_point
            elif edge_data_p23 and edge_data_p23["angle"] is source:
                start = source.end_point
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

        elif isinstance(source, SymmetryConstraint):
            self.solve_symmetry_constraint(source)
        elif isinstance(source, ROCurvatureConstraint):
            self.solve_roc_constraint(source)

        self.solve_other_constraints(points_solved)

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
            constraint.p4.request_move(constraint.p3.x().value(), constraint.p3.y().value(), force=True)
            return

        mirror_distance = 2 * measure_point_line_distance_unsigned(x1, y1, x2, y2, x3, y3)
        constraint.p4.request_move(
            x3 + mirror_distance * np.cos(mirror_angle),
            y3 + mirror_distance * np.sin(mirror_angle), force=True
        )

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

    def solve_symmetry_constraints(self, points: typing.List[Point]):
        symmetry_constraints_solved = []
        for point in points:
            symmetry_constraints = [geo_con for geo_con in point.geo_cons if isinstance(geo_con, SymmetryConstraint)]
            for symmetry_constraint in symmetry_constraints:
                if symmetry_constraint in symmetry_constraints_solved:
                    continue
                self.solve_symmetry_constraint(symmetry_constraint)
                symmetry_constraints_solved.append(symmetry_constraint)

    def solve_roc_constraints(self, points: typing.List[Point]):
        roc_constraints_solved =[]
        for point in points:
            roc_constraints = [geo_con for geo_con in point.geo_cons if isinstance(geo_con, ROCurvatureConstraint)]
            for roc_constraint in roc_constraints:
                if roc_constraint in roc_constraints_solved:
                    continue
                self.solve_roc_constraint(roc_constraint)
                roc_constraints_solved.append(roc_constraint)

    def solve_other_constraints(self, points: typing.List[Point]):
        self.solve_symmetry_constraints(points)
        self.solve_roc_constraints(points)

    def update_canvas_items(self, source: GeoCon or Point):

        points_to_update = []
        if isinstance(source, Point):
            starting_points = [source]
        elif isinstance(source, GeoCon):
            starting_points = source.child_nodes
        else:
            raise ValueError("source must be of type GeoCon or of type Point")
        for starting_point in starting_points:
            for point in networkx.dfs_preorder_nodes(self, source=starting_point):
                if point in points_to_update:
                    continue
                points_to_update.append(point)

        curves_to_update = []
        for point in points_to_update:
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
        for point in networkx.dfs_preorder_nodes(self, source=source.child_nodes[0]):
            for geo_con in point.geo_cons:
                if geo_con not in constraints_to_update:
                    constraints_to_update.append(geo_con)

        for geo_con in constraints_to_update:
            if isinstance(geo_con, GeoCon) and geo_con.canvas_item is not None:
                geo_con.canvas_item.update()


class ConstraintValidationError(Exception):
    pass



def main():
    pass
    # geo_col = GeometryCollection()
    # gcs2 = GCS2()
    # points = [
    #     geo_col.add_point(0.0, 0.0),
    #     geo_col.add_point(1.1, 0.0),
    #     geo_col.add_point(0.1, -0.1),
    #     geo_col.add_point(0.5, 0.2),
    #     geo_col.add_point(0.2, 0.7),
    #     geo_col.add_point(0.4, 0.4)
    # ]
    # for point in points:
    #     gcs2.add_point(point)
    # constraints = [
    #     DistanceConstraint(points[0], points[1], 0.25),
    #     DistanceConstraint(points[1], points[2], 0.35),
    #     DistanceConstraint(points[2], points[3], 0.4),
    #     DistanceConstraint(points[1], points[4], 0.15),
    #     DistanceConstraint(points[2], points[5], 0.2),
    #     AntiParallel3Constraint(points[0], points[1], points[2]),
    #     AntiParallel3Constraint(points[1], points[2], points[3]),
    #     Perp3Constraint(points[4], points[1], points[0]),
    #     Perp3Constraint(points[5], points[2], points[1])
    # ]
    # for constraint in constraints:
    #     gcs2.add_constraint(constraint)
    #     gcs2.solve(constraint)
    #     print(f"{points[0].measure_distance(points[1]) = }")
    #     print(f"{points[1].measure_distance(points[2]) = }")
    #     print(f"{points[2].measure_distance(points[3]) = }")
    #     print(f"{points[2].measure_distance(points[4]) = }")
    #     print(f"{points[3].measure_distance(points[5]) = }")
    #     # plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o")
    #     # plt.show()
    #
    # constraints[0].param().set_value(0.35)
    # gcs2.solve(constraints[0])
    # plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o")
    # plt.show()
    #
    # networkx.draw_networkx(gcs2)
    # plt.show()
    #
    # for point in points:
    #     tree = networkx.bfs_tree(gcs2, source=point)
    #     networkx.draw_networkx(tree)
    #     plt.show()
    #
    # plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o")
    # plt.gca().set_aspect("equal")
    # plt.show()
    # networkx.draw_networkx(gcs2)
    # plt.show()


if __name__ == "__main__":
    main()
