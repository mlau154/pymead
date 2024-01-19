from unittest import TestCase

import numpy as np

from constraint_graph import ConstraintGraph
from pymead.core.constraints import *
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.point import Point
from pymead.core.param import LengthParam, AngleParam


class GCSTests(TestCase):

    def test_fixed_x_constraint(self):
        p = Point(0.5, 0.3)
        x = LengthParam(0.4, "L1")

        gcs = ConstraintGraph()
        gcs.add_fixed_x_constraint(p, x)

        self.assertAlmostEqual(p.x().value(), 0.4, places=6)
        self.assertAlmostEqual(p.y().value(), 0.3, places=6)

        # Try moving the point
        p.x().set_value(0.8)
        p.y().set_value(2.0)

        res = gcs.solve()

        # Make sure the solver completed successfully
        self.assertTrue(res.success)

        gcs.update(res.x)

        # Make sure that y moved but x stayed the same
        self.assertAlmostEqual(p.x().value(), 0.4, places=6)
        self.assertAlmostEqual(p.y().value(), 2.0, places=6)

    def test_fixed_y_constraint(self):
        p = Point(0.5, 0.3)
        y = LengthParam(0.4, "L1")

        gcs = ConstraintGraph()
        gcs.add_fixed_y_constraint(p, y)

        self.assertAlmostEqual(p.x().value(), 0.5, places=6)
        self.assertAlmostEqual(p.y().value(), 0.4, places=6)

        # Try moving the point
        p.x().set_value(0.8)
        p.y().set_value(2.0)

        res = gcs.solve()

        # Make sure the solver completed successfully
        self.assertTrue(res.success)

        gcs.update(res.x)

        # Make sure that x moved but y stayed the same
        self.assertAlmostEqual(p.x().value(), 0.8, places=6)
        self.assertAlmostEqual(p.y().value(), 0.4, places=6)

    def test_fixed_x_and_y_constraint_combo(self):
        p = Point(0.5, 0.3)
        x = LengthParam(0.9, "x")
        y = LengthParam(0.4, "y")

        gcs = ConstraintGraph()
        gcs.add_fixed_x_constraint(p, x)
        gcs.add_fixed_y_constraint(p, y)

        self.assertAlmostEqual(p.x().value(), 0.9, places=6)
        self.assertAlmostEqual(p.y().value(), 0.4, places=6)

        # Try moving the point
        p.x().set_value(0.8)
        p.y().set_value(2.0)

        res = gcs.solve()

        # Make sure the solver completed successfully
        self.assertTrue(res.success)

        gcs.update(res.x)

        # Make sure that x moved but y stayed the same
        self.assertAlmostEqual(p.x().value(), 0.9, places=6)
        self.assertAlmostEqual(p.y().value(), 0.4, places=6)

    def test_distance_constraint(self):

        geo_col = GeometryCollection()

        target_distance_value = 0.6

        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.5, 0.2)
        starting_angle = p1.measure_angle(p2)

        dist = geo_col.add_param(target_distance_value, unit_type="length")

        cnstr = geo_col.add_distance_constraint(p1, p2, dist)

        cnstr.add_constraint_to_gcs()

        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_angle(p2), starting_angle, places=6)

        print("Moving the target point to 0.7, 0.6")
        p2.request_move(0.7, 0.6)
        p_test = Point(0.7, 0.6)
        new_angle = p1.measure_angle(p_test)

        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_angle(p2), new_angle, places=6)

    def test_two_distance_constraints_different_distance(self):
        geo_col = GeometryCollection()

        target_distance_value_1 = 0.6
        length_param_1 = geo_col.add_param(target_distance_value_1, unit_type="length")

        target_distance_value_2 = 0.75
        length_param_2 = geo_col.add_param(target_distance_value_2, unit_type="length")

        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.5, 0.8)
        p3 = geo_col.add_point(-0.3, -0.4)

        # Add the first distance constraint
        cnstr1 = DistanceConstraint(p1, p2, value=length_param_1)
        geo_col.add_constraint(cnstr1)

        # Add the second distance constraint
        cnstr2 = DistanceConstraint(p2, p3, value=length_param_2)
        geo_col.add_constraint(cnstr2)

        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value_1, places=6)
        self.assertAlmostEqual(p2.measure_distance(p3), target_distance_value_2, places=6)

        cnstr1.param().set_value(0.7)
        self.assertAlmostEqual(p1.measure_distance(p2), 0.7, places=6)
        self.assertAlmostEqual(p2.measure_distance(p3), target_distance_value_2, places=6)

        cnstr2.param().set_value(0.8)
        self.assertAlmostEqual(p1.measure_distance(p2), 0.7, places=6)
        self.assertAlmostEqual(p2.measure_distance(p3), 0.8, places=6)

    def test_two_distance_constraints_same_distance(self):
        geo_col = GeometryCollection()

        target_distance_value = 0.6
        length_param = geo_col.add_param(target_distance_value, name="dist", unit_type="length")

        p1 = Point(0.0, 0.0)
        p2 = Point(0.5, 0.8)
        p3 = Point(-0.3, -0.4)
        starting_angle_12 = p1.measure_angle(p2)
        starting_angle_13 = p1.measure_angle(p3)

        # Add the first distance constraint
        cnstr1 = geo_col.add_distance_constraint(p1, p2, length_param)
        cnstr1.add_constraint_to_gcs()

        # Add the second distance constraint
        cnstr2 = geo_col.add_distance_constraint(p2, p3, length_param)
        cnstr2.add_constraint_to_gcs()

        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_distance(p3), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_angle(p2), starting_angle_12, places=6)
        self.assertAlmostEqual(p1.measure_angle(p3), starting_angle_13, places=6)

        p1.request_move(-0.1, -0.1)
        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_distance(p3), target_distance_value, places=6)

        p2.request_move(0.3, 0.6)
        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_distance(p3), target_distance_value, places=6)

        p3.request_move(-0.4, 1.0)
        self.assertAlmostEqual(p1.measure_distance(p2), target_distance_value, places=6)
        self.assertAlmostEqual(p1.measure_distance(p3), target_distance_value, places=6)

    def test_abs_angle_constraint(self):
        target_angle_value = 0.8

        p1 = Point(0.1, 0.063)
        p2 = Point(0.5, 0.2)
        angle = AngleParam(target_angle_value, "angle")
        gcs = ConstraintGraph()
        gcs.add_abs_angle_constraint(p1, p2, angle)
        res = gcs.solve()

        # Make sure the solver completed successfully
        self.assertTrue(res.success)

        gcs.update(res.x)

        self.assertAlmostEqual(p1.measure_angle(p2), target_angle_value, places=6)

    def test_length_abs_angle_combo(self):
        target_length_value = 0.5
        target_angle_value = 0.8

        p1 = Point(0.1, 0.063)
        p2 = Point(0.5, 0.2)
        dist = LengthParam(target_length_value, "dist")
        angle = AngleParam(target_angle_value, "angle")
        gcs = ConstraintGraph()
        gcs.add_distance_constraint(p1, p2, dist)
        gcs.add_abs_angle_constraint(p1, p2, angle)

        res = gcs.solve()

        # Make sure the solver completed successfully
        self.assertTrue(res.success)

        # Update the point values
        gcs.update(res.x)

        self.assertAlmostEqual(p1.measure_distance(p2), target_length_value, places=5)
        self.assertAlmostEqual(p1.measure_angle(p2), target_angle_value, places=5)

    def test_rel_angle3_constraint(self):
        target_angle_value = np.pi / 4

        p1 = Point(0.5, 0.2)
        p2 = Point(0.0, 0.0)
        p3 = Point(1.0, 0.0)
        angle = AngleParam(target_angle_value, "angle")
        gcs = ConstraintGraph()
        gcs.add_rel_angle3_constraint(p1, p2, p3, angle)

        res = gcs.solve()

        # Make sure the solver completed successfully
        self.assertTrue(res.success)

        # Update the point values
        gcs.update(res.x)

        self.assertAlmostEqual(p2.measure_angle(p1) - p2.measure_angle(p3), target_angle_value, places=6)

    def test_rel_angle3_double_length_combo(self):
        target_angle_values = [np.pi / 4, 0.2, 3 * np.pi / 2, 0.64]
        target_length1_values = [0.5, 0.1, 0.7, 10.0, 0.01]
        target_length2_values = [0.6, 0.21, 0.08, 20.5, 1.6]

        idx = 0
        for target_angle_value, target_length1, target_length2 in zip(
                target_angle_values, target_length1_values, target_length2_values):

            p1 = Point(0.1, 0.2)
            p2 = Point(-0.3, 0.0)
            p3 = Point(1.0, 0.0)
            angle = AngleParam(target_angle_value, "angle")
            length1 = LengthParam(target_length1, "length1")
            length2 = LengthParam(target_length2, "length2")
            gcs = ConstraintGraph()

            gcs.add_rel_angle3_constraint(p1, p2, p3, angle)
            gcs.add_distance_constraint(p2, p1, length1)
            gcs.add_distance_constraint(p2, p3, length2)

            res = gcs.solve()

            # Make sure the solver completed successfully
            self.assertTrue(res.success)

            # Update the point values
            gcs.update(res.x)

            print(f"{p1.as_array() = }, {p2.as_array() = }, {p3.as_array() = }")

            self.assertAlmostEqual((p2.measure_angle(p1) - p2.measure_angle(p3)) % (2 * np.pi), target_angle_value,
                                   places=6)
            self.assertAlmostEqual(p1.measure_distance(p2), target_length1, places=6)
            self.assertAlmostEqual(p2.measure_distance(p3), target_length2, places=6)

            p2.x().set_value(0.4)
            p2.y().set_value(8.8)

            res = gcs.solve()

            # Make sure the solver completed successfully
            self.assertTrue(res.success)

            # Update the point values
            gcs.update(res.x)

            self.assertAlmostEqual((p2.measure_angle(p1) - p2.measure_angle(p3)) % (2 * np.pi), target_angle_value,
                                   places=6)
            self.assertAlmostEqual(p1.measure_distance(p2), target_length1, places=6)
            self.assertAlmostEqual(p2.measure_distance(p3), target_length2, places=6)

            print(f"{p1.as_array() = }, {p2.as_array() = }, {p3.as_array() = }")

            idx += 1

    def test_symmetry_constraint(self):
        import matplotlib.pyplot as plt
        geo_col = GeometryCollection()

        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(1.0, 0.05)
        p3 = geo_col.add_point(0.3, 0.0)
        p4 = geo_col.add_point(0.3, 0.15)
        p5 = geo_col.add_point(0.3, -0.15)
        points = [p1, p2, p3, p4, p5]

        d1_par = geo_col.add_param(1.0, unit_type="length")
        d1 = DistanceConstraint(p1, p2, value=d1_par)
        d2_par = geo_col.add_param(0.35, unit_type="length")
        d2 = DistanceConstraint(p1, p3, value=d2_par)
        d3_par = geo_col.add_param(0.15, unit_type="length")
        d3 = DistanceConstraint(p3, p4, value=d3_par)
        perp3 = Perp3Constraint(p1, p3, p4)
        pol = PointOnLineConstraint(p3, p1, p2)
        sym = SymmetryConstraint(p1, p2, p5, p4)

        constraints = [d1, d2, d3, perp3, pol, sym]
        for constraint in constraints:
            geo_col.add_constraint(constraint)

        plt.plot([p.x().value() for p in points], [p.y().value() for p in points], ls="none", marker="o",
                 mfc="indianred")
        plt.show()
