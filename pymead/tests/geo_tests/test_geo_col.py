import unittest

import numpy as np

from pymead.core.constraints import PositionConstraint, CollinearConstraint
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.param2 import Param, DesVar
from pymead.core.point import Point


class GeoColTests(unittest.TestCase):
    geo_col = GeometryCollection(geo_ui=None)

    def test_add_remove_param(self):
        self.geo_col.add_param(0.1, "LC1")
        self.geo_col.add_param(0.2, "LC2")
        self.geo_col.add_param(0.15, "LC2")
        self.geo_col.add_param(0.5, "LC2")
        param_container = self.geo_col.container()["params"]
        self.assertIn("LC1", param_container)
        self.assertIn("LC2", param_container)
        self.assertIn("LC2-2", param_container)
        self.assertIn("LC2-3", param_container)

        self.geo_col.remove_param("LC2-2")
        self.assertNotIn("LC2-2", param_container)

    def test_add_remove_point(self):
        point_container = self.geo_col.container()["points"]
        self.geo_col.add_point(0.5, 0.1)
        self.geo_col.add_point(0.3, 0.7)
        self.geo_col.add_point(-0.1, -0.2)

        self.geo_col.remove_point("Point-2")

        self.assertIn("Point", point_container)
        self.assertIn("Point-3", point_container)
        self.assertNotIn("Point-2", point_container)

        param_container = self.geo_col.container()["params"]
        self.assertIn("Point.x", param_container)
        self.assertIn("Point.y", param_container)
        self.assertIn("Point-3.x", param_container)
        self.assertIn("Point-3.y", param_container)
        self.assertNotIn("Point-2.x", param_container)
        self.assertNotIn("Point-2.y", param_container)

    def test_add_remove_desvar(self):
        desvar_container = self.geo_col.container()["desvar"]

        self.geo_col.add_desvar(0.5, "DV")
        self.geo_col.add_desvar(0.7, "DV", lower=0.4, upper=1.0)

        self.assertIn("DV", desvar_container)
        self.assertIn("DV-2", desvar_container)

        # Make sure a ValueError is raised when we try to assign a lower bound greater than the value, or an upper
        # bound lower than the value
        self.assertRaises(ValueError, self.geo_col.add_desvar, value=0.4, name="DV", lower=0.5, upper=0.8)
        self.assertRaises(ValueError, self.geo_col.add_desvar, value=0.6, name="DV", lower=0.5, upper=0.1)

        # Make sure a ValueError is raised when we try to assign an upper bound lower than the lower bound,
        # or upper and lower bounds that are too close together (this could raise a divide by zero error when extracting
        # the bounds-normalized value)
        self.assertRaises(ValueError, self.geo_col.add_desvar, value=0.4, name="DV", lower=0.3, upper=0.1)
        self.assertRaises(ValueError, self.geo_col.add_desvar, value=0.5, name="DV", lower=0.5-1e-20, upper=0.5+1e-20)

        self.geo_col.remove_desvar("DV")
        self.assertNotIn("DV", desvar_container)

        self.geo_col.add_desvar(0.4, "DV", 0.2, 0.8)
        self.assertIn("DV-3", desvar_container)

        self.geo_col.remove_desvar("DV-2")
        self.assertNotIn("DV-2", desvar_container)

    def test_bounded_point_movement(self):
        point = self.geo_col.add_point(4.0, 3.0)
        point2 = self.geo_col.add_point(1.5, 2.5)

        self.geo_col.promote_param_to_desvar(point2.x(), lower=0.0, upper=10.0)
        self.geo_col.promote_param_to_desvar(point2.y(), lower=0.0, upper=10.0)

        point2.x().set_value(-1.0)
        point2.y().set_value(-2.0)

        dist = point2.measure_distance(point)
        self.assertAlmostEqual(dist, 5.0)

    def test_bounds_normalization(self):
        desvar = self.geo_col.add_desvar(0.4, "DV", lower=0.2, upper=0.6)

        bn_value = desvar.value(bounds_normalized=True)
        self.assertAlmostEqual(bn_value, 0.5)

        desvar.set_value(0.25, bounds_normalized=True)
        self.assertAlmostEqual(desvar.value(), 0.3)

    def test_alphabetical_sub_container_list(self):
        for _ in range(12):
            self.geo_col.add_param(0.5, "Spin")
        for _ in range(5):
            self.geo_col.add_param(0.1, "myParam")
        self.geo_col.remove_param("myParam-3")
        for _ in range(11):
            self.geo_col.add_point(0.3, 0.1)

        alphabetical_list = self.geo_col.alphabetical_sub_container_key_list("params")

        self.assertGreater(alphabetical_list.index("myParam-5"), alphabetical_list.index("myParam"))
        self.assertGreater(alphabetical_list.index("Spin-10"), alphabetical_list.index("Spin-9"))
        self.assertGreater(alphabetical_list.index("Point-10.x"), alphabetical_list.index("Point-9.x"))
        self.assertGreater(alphabetical_list.index("Spin-5"), alphabetical_list.index("Point-4.y"))
        self.assertGreater(alphabetical_list.index("Spin-6"), alphabetical_list.index("myParam"))

    def test_extract_assign_design_variable_values(self):
        geo_col = GeometryCollection(geo_ui=None)
        dv1 = geo_col.add_desvar(0.3, "dv", lower=0.1, upper=0.5)
        dv2 = geo_col.add_desvar(0.8, "dv", lower=0.0, upper=2.0)
        dv3 = geo_col.add_desvar(-4.0, "dv", lower=-10.0, upper=10.0)
        dv_values = geo_col.extract_design_variable_values(bounds_normalized=False)
        self.assertAlmostEqual(dv_values[0], 0.3)
        self.assertAlmostEqual(dv_values[1], 0.8)
        self.assertAlmostEqual(dv_values[2], -4.0)
        dv_values_bn = geo_col.extract_design_variable_values(bounds_normalized=True)
        self.assertAlmostEqual(dv_values_bn[0], 0.5)
        self.assertAlmostEqual(dv_values_bn[1], 0.4)
        self.assertAlmostEqual(dv_values_bn[2], 0.3)

        # Variable assignment
        dv1.set_value(0.2)
        self.assertEqual(0.2, dv1.value())
        dv1.set_value(0.75, bounds_normalized=True)
        self.assertEqual(0.4, dv1.value())
        dv1.set_value(-10.0)
        self.assertEqual(0.1, dv1.value())

        # List of variables assignment
        dv_vals = [0.5, 1.0, 0.5]
        geo_col.assign_design_variable_values(dv_vals, bounds_normalized=True)
        self.assertAlmostEqual(0.3, dv1.value())
        self.assertAlmostEqual(2.0, dv2.value())
        self.assertAlmostEqual(0.0, dv3.value())

        # Make sure that an error is raised if we try to assign a list of variables different in length than the
        # number of design variables
        self.assertRaises(ValueError, geo_col.assign_design_variable_values,
                          dv_values=[0.6, 0.4, 0.2, 0.8], bounds_normalized=True)


class ParamTests(unittest.TestCase):
    def test_dict_gen(self):
        param = Param.set_from_dict_rep({"name": "LC1", "value": 0.5})
        self.assertEqual(param.value(), 0.5)
        self.assertEqual(param.name(), "LC1")

        desvar = DesVar.set_from_dict_rep({"name": "myDV", "value": 0.3, "lower": 0.1, "upper": 0.9})
        self.assertEqual(desvar.value(), 0.3)
        self.assertAlmostEqual(desvar.value(bounds_normalized=True), 0.25)
        desvar.set_value(0.5)
        desvar_dict = desvar.get_dict_rep()
        self.assertEqual(desvar_dict["value"], 0.5)
        self.assertEqual(desvar_dict["lower"], 0.1)
        self.assertEqual(desvar_dict["upper"], 0.9)


class ConstraintTests(unittest.TestCase):
    def test_pos_constraint(self):
        dv_length = DesVar(0.2, "length", lower=0.1, upper=0.3)
        dv_angle = DesVar(0.5, "angle", lower=0.1, upper=0.9)
        point1 = Point(0.2, 0.1, name="tool_point", setting_from_geo_col=True)
        point2 = Point(-0.1, 0.3, name="target_point", setting_from_geo_col=True)
        constraint = PositionConstraint(tool=point1, target=point2, dist=dv_length, angle=dv_angle, bidirectional=False)
        constraint.enforce("tool")
        dv_length.set_value(0.25)
        new_x1 = point1.x().value()
        new_y1 = point1.y().value()
        new_x2 = point2.x().value()
        new_y2 = point2.y().value()
        self.assertAlmostEqual(0.25, np.hypot(new_y2 - new_y1, new_x2 - new_x1))
        self.assertAlmostEqual(0.5, np.arctan2(new_y2 - new_y1, new_x2 - new_x1))

        point1.request_move(0.3, 0.0)
        new_x1 = point1.x().value()
        new_y1 = point1.y().value()
        new_x2 = point2.x().value()
        new_y2 = point2.y().value()
        self.assertAlmostEqual(0.3, new_x1)
        self.assertAlmostEqual(0.0, new_y1)
        self.assertAlmostEqual(0.25, np.hypot(new_y2 - new_y1, new_x2 - new_x1))
        self.assertAlmostEqual(0.5, np.arctan2(new_y2 - new_y1, new_x2 - new_x1))

        point2.request_move(0.8, 0.4)
        new_x1 = point1.x().value()
        new_y1 = point1.y().value()
        newer_x2 = point2.x().value()
        newer_y2 = point2.y().value()
        self.assertAlmostEqual(0.3, new_x1)
        self.assertAlmostEqual(0.0, new_y1)
        self.assertAlmostEqual(0.25, np.hypot(newer_y2 - new_y1, newer_x2 - new_x1))
        self.assertAlmostEqual(0.5, np.arctan2(newer_y2 - new_y1, newer_x2 - new_x1))
        # The x and y location of point2 should not have changed, because point2 has no degrees of freedom
        self.assertAlmostEqual(new_x2, newer_x2)
        self.assertAlmostEqual(new_y2, newer_y2)

    def test_pos_constraint_bidirectional(self):
        dv_length = DesVar(0.2, "length", lower=0.1, upper=0.3)
        dv_angle = DesVar(0.5, "angle", lower=0.1, upper=0.9)
        point1 = Point(0.2, 0.1, name="tool_point", setting_from_geo_col=True)
        point2 = Point(-0.1, 0.3, name="target_point", setting_from_geo_col=True)
        constraint = PositionConstraint(tool=point1, target=point2, dist=dv_length, angle=dv_angle, bidirectional=True)
        constraint.enforce("tool")
        dv_length.set_value(0.25)
        new_x1 = point1.x().value()
        new_y1 = point1.y().value()
        new_x2 = point2.x().value()
        new_y2 = point2.y().value()
        self.assertAlmostEqual(0.25, np.hypot(new_y2 - new_y1, new_x2 - new_x1))
        self.assertAlmostEqual(0.5, np.arctan2(new_y2 - new_y1, new_x2 - new_x1))

        point1.request_move(0.3, 0.0)
        new_x1 = point1.x().value()
        new_y1 = point1.y().value()
        new_x2 = point2.x().value()
        new_y2 = point2.y().value()
        self.assertAlmostEqual(0.3, new_x1)
        self.assertAlmostEqual(0.0, new_y1)
        self.assertAlmostEqual(0.25, np.hypot(new_y2 - new_y1, new_x2 - new_x1))
        self.assertAlmostEqual(0.5, np.arctan2(new_y2 - new_y1, new_x2 - new_x1))

        point2.request_move(0.8, 0.4)
        new_x1 = point1.x().value()
        new_y1 = point1.y().value()
        new_x2 = point2.x().value()
        new_y2 = point2.y().value()
        self.assertAlmostEqual(0.8, new_x2)
        self.assertAlmostEqual(0.4, new_y2)
        self.assertAlmostEqual(0.25, np.hypot(new_y2 - new_y1, new_x2 - new_x1))
        self.assertAlmostEqual(0.5, np.arctan2(new_y2 - new_y1, new_x2 - new_x1))

    def test_collinear_constraint(self):
        start_point = Point(-1.0, -0.5, name="start_point", setting_from_geo_col=True)
        middle_point = Point(0.5, 1.0, name="middle_point", setting_from_geo_col=True)
        end_point = Point(3.0, 2.0, name="end_point", setting_from_geo_col=True)

        original_end_middle_distance = middle_point.measure_distance(end_point)
        original_start_end_distance = start_point.measure_distance(end_point)
        constraint = CollinearConstraint(start_point=start_point, middle_point=middle_point, end_point=end_point)
        constraint.enforce("start")

        # Ensure that the three points are now collinear after initial enforcement
        angle_minus_pi = middle_point.measure_angle(start_point)
        self.assertAlmostEqual(middle_point.measure_angle(end_point), angle_minus_pi + np.pi)

        # Ensure that the distance between the end and middle points remains the same as before enforcement
        self.assertAlmostEqual(middle_point.measure_distance(end_point), original_end_middle_distance)

        # Move the middle point and ensure that the three points are still collinear
        middle_point.request_move(0.0, 0.0)
        self.assertAlmostEqual(start_point.measure_angle(middle_point), middle_point.measure_angle(end_point))

        # Move the start point and ensure that the three points are still collinear
        start_point.request_move(-5.0, 10.0)
        self.assertAlmostEqual(start_point.measure_angle(middle_point), middle_point.measure_angle(end_point))

        # Move the start point and ensure that the three points are still collinear
        start_point.request_move(5.0, -2.0)
        self.assertAlmostEqual(start_point.measure_angle(middle_point), middle_point.measure_angle(end_point))
