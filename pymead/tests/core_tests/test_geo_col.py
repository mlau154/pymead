import os
import unittest
import tempfile

import numpy as np

from pymead.core import UNITS
from pymead.core.bezier import Bezier
from pymead.core.dimensions import LengthDimension, AngleDimension
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.line import LineSegment
from pymead.core.mea import MEA
from pymead.core.param import Param, DesVar, LengthParam, AngleParam
from pymead.core.point import Point, PointSequence
from pymead.core.airfoil import Airfoil, ClosureError, BranchError


temp_dir = tempfile.gettempdir()


class GeoColTests(unittest.TestCase):
    geo_col = GeometryCollection()

    def test_add_remove_param(self):
        param1 = self.geo_col.add_param(0.1, "LC1")
        param2 = self.geo_col.add_param(0.2, "LC2")
        param3 = self.geo_col.add_param(0.15, "LC2")
        param4 = self.geo_col.add_param(0.5, "LC2")
        param_container = self.geo_col.container()["params"]
        self.assertIn("LC1", param_container)
        self.assertIn("LC2", param_container)
        self.assertIn("LC2-2", param_container)
        self.assertIn("LC2-3", param_container)

        self.geo_col.remove_pymead_obj(pymead_obj=param3)
        self.assertNotIn("LC2-2", param_container)

    def test_add_remove_point(self):
        point_container = self.geo_col.container()["points"]
        point1 = self.geo_col.add_point(0.5, 0.1)
        point2 = self.geo_col.add_point(0.3, 0.7)
        point3 = self.geo_col.add_point(-0.1, -0.2)

        self.geo_col.remove_pymead_obj(point2)

        self.assertIn("Point-1", point_container)
        self.assertIn("Point-3", point_container)
        self.assertNotIn("Point-2", point_container)

    def test_add_remove_desvar(self):
        desvar_container = self.geo_col.container()["desvar"]

        dv1 = self.geo_col.add_desvar(0.5, "DV")
        dv2 = self.geo_col.add_desvar(0.7, "DV", lower=0.4, upper=1.0)

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

        self.geo_col.remove_pymead_obj(dv1)
        self.assertNotIn("DV", desvar_container)

        dv3 = self.geo_col.add_desvar(0.4, "DV", 0.2, 0.8)
        self.assertIn("DV-3", desvar_container)

        self.geo_col.remove_pymead_obj(dv2)
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
        geo_col = GeometryCollection()
        for _ in range(12):
            geo_col.add_param(0.5, "Spin")
        myParams = [geo_col.add_param(0.1, "myParam") for _ in range(5)]
        geo_col.remove_pymead_obj(myParams[2])
        for _ in range(11):
            geo_col.add_point(0.3, 0.1)

        alphabetical_list = geo_col.alphabetical_sub_container_key_list("params")

        self.assertGreater(alphabetical_list.index("myParam-5"), alphabetical_list.index("myParam"))
        self.assertGreater(alphabetical_list.index("Spin-10"), alphabetical_list.index("Spin-9"))
        self.assertGreater(alphabetical_list.index("Spin-6"), alphabetical_list.index("myParam"))

    def test_extract_assign_design_variable_values(self):
        geo_col = GeometryCollection()
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

    def test_complex_airfoil_geo_col(self):
        geo_col = GeometryCollection()
        le = geo_col.add_point(0.0, 0.0)
        upper1 = geo_col.add_point(0.0, 0.05)
        upper2 = geo_col.add_point(0.05, 0.05)
        upper3 = geo_col.add_point(0.6, 0.03)
        upper5 = geo_col.add_point(0.8, 0.04)
        upper4 = geo_col.add_point(1.0, 0.005)
        te = geo_col.add_point(1.0, 0.0)
        lower1 = geo_col.add_point(0.0, -0.03)
        lower2 = geo_col.add_point(0.03, -0.03)
        lower3 = geo_col.add_point(0.7, 0.03)
        lower4 = geo_col.add_point(1.0, -0.005)

        upper = geo_col.add_bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper5]))
        upper_line = geo_col.add_line(point_sequence=PointSequence(points=[upper5, upper4]))
        lower = geo_col.add_bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]))
        geo_col.add_line(point_sequence=PointSequence(points=[upper4, te]))
        geo_col.add_line(point_sequence=PointSequence(points=[te, lower4]))
        airfoil = geo_col.add_airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=upper4, lower_surf_end=lower4)

        self.assertEqual(3, len(airfoil.curves))
        self.assertEqual(2, len(airfoil.curves_to_reverse))

        self.assertEqual(airfoil.curves, [upper_line, upper, lower])
        self.assertEqual(airfoil.curves_to_reverse, [upper_line, upper])


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

    def test_curvature_constraint(self):
        geo_col = GeometryCollection()
        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.05, 0.12)
        p3 = geo_col.add_point(0.12, 0.36)
        p4 = geo_col.add_point(0.05, -0.12)
        p5 = geo_col.add_point(0.35, -0.52)
        b1 = geo_col.add_bezier(point_sequence=PointSequence(points=[p1, p2, p3]))
        b2 = geo_col.add_bezier(point_sequence=PointSequence(points=[p1, p4, p5]))
        curvature_con = geo_col.add_curvature_constraint(curve_joint=p1)

        # Test the curvature data method
        data = curvature_con.calculate_curvature_data()
        self.assertIs(p2, curvature_con.g1_point_curve_1)
        self.assertIs(p3, curvature_con.g2_point_curve_1)
        self.assertIs(p4, curvature_con.g1_point_curve_2)
        self.assertIs(p5, curvature_con.g2_point_curve_2)
        self.assertAlmostEqual(0.13, data.Lt1)
        self.assertAlmostEqual(0.13, data.Lt2)
        self.assertAlmostEqual(0.25, data.Lc1)
        self.assertAlmostEqual(0.5, data.Lc2)
        self.assertAlmostEqual(np.arctan2(0.12, 0.05), data.phi1)
        self.assertAlmostEqual(np.arctan2(-0.12, 0.05), data.phi2)
        self.assertAlmostEqual(np.arctan2(0.24, 0.07), data.theta1)
        self.assertAlmostEqual(np.arctan2(-0.4, 0.3), data.theta2)

        curvature_con.enforce(p2)
        data = curvature_con.calculate_curvature_data()
        self.assertAlmostEqual(data.phi1 % (2 * np.pi), (data.phi2 + np.pi) % (2 * np.pi))
        self.assertAlmostEqual(data.R1, data.R2)

        old_psi1 = data.psi1
        old_psi2 = data.psi2
        p2.request_move(0.06, 0.13)
        data = curvature_con.calculate_curvature_data()
        self.assertAlmostEqual(data.psi1, old_psi1)
        self.assertAlmostEqual(data.psi2, old_psi2)
        self.assertAlmostEqual(data.phi1 % (2 * np.pi), (data.phi2 + np.pi) % (2 * np.pi))
        self.assertAlmostEqual(data.R1, data.R2)

        # Check that the evaluated curvature at the curve joint matches exactly between both curves
        b1_data = b1.evaluate()
        b2_data = b2.evaluate()
        self.assertAlmostEqual(b1_data.R[0], b2_data.R[0])
        self.assertAlmostEqual(b1_data.k[0], b2_data.k[0])

    def test_length_dimension(self):
        geo_col = GeometryCollection()
        # First, test the case where a param is directly specified
        param = geo_col.add_param(0.25, "LengthParam")
        start_point = geo_col.add_point(0.0, 0.0)
        end_point = geo_col.add_point(0.3, 0.4)
        geo_col.add_length_dimension(tool_point=start_point, target_point=end_point, length_param=param)

        # Make sure that the param length value gets changed to sqrt(0.3**2 + 0.4**2)
        self.assertAlmostEqual(0.5, param.value())

        # Now, test the case where a param is not directly specified
        dimension = geo_col.add_length_dimension(tool_point=start_point, target_point=end_point)
        end_point.request_move(8.0, 6.0)

        # Make sure that the param length value gets changed to sqrt(8**2 + 6**2)
        self.assertAlmostEqual(10.0, dimension.param().value())

        # Change the length parameter value and make sure that the point is set to the appropriate position
        print("Setting value to 5.0")
        param.set_value(5.0)
        new_length = start_point.measure_distance(end_point)
        self.assertAlmostEqual(5.0, new_length)
        self.assertAlmostEqual(0.0, start_point.x().value())
        self.assertAlmostEqual(0.0, start_point.y().value())

    def test_chained_length_dimension(self):
        geo_col = GeometryCollection()
        # First, test the case where a param is directly specified
        param = geo_col.add_param(0.25, "LengthParam")
        start_point = geo_col.add_point(0.0, 0.0)
        end_point = geo_col.add_point(0.3, 0.4)
        geo_col.add_length_dimension(tool_point=start_point, target_point=end_point, length_param=param)

        # Make sure that the param length value gets changed to sqrt(0.3**2 + 0.4**2)
        self.assertAlmostEqual(0.5, param.value())

        # Now, test the case where a param is not directly specified
        dimension = geo_col.add_length_dimension(tool_point=start_point, target_point=end_point)
        end_point.request_move(8.0, 6.0)

        # Make sure that the param length value gets changed to sqrt(8**2 + 6**2)
        self.assertAlmostEqual(10.0, dimension.param().value())

    def test_angle_dimension(self):
        # Set the units to degrees
        UNITS.set_current_angle_unit("deg")

        # First, test the case where a param is directly specified
        param = AngleParam(0.25, "AngleDim")
        start_point = Point(0.0, 0.0, "p1", setting_from_geo_col=True)
        end_point = Point(0.4, 0.4, "p3", setting_from_geo_col=True)
        AngleDimension(tool_point=start_point, target_point=end_point, angle_param=param)

        # Make sure that the param angle value gets changed to 45 degrees
        self.assertAlmostEqual(45.0, param.value())

        # Now, test the case where a param is not directly specified
        dimension = AngleDimension(tool_point=start_point, target_point=end_point)
        end_point.request_move(-4.0, 4.0)

        # Make sure that the param length value gets changed to sqrt(8**2 + 6**2)
        self.assertAlmostEqual(135.0, dimension.param().value())

    def test_units(self):
        start_point = Point(0.0, 0.0, "p1", setting_from_geo_col=True)
        end_point = Point(0.3, 0.4, "p3", setting_from_geo_col=True)

        dimension = LengthDimension(tool_point=start_point, target_point=end_point)

        UNITS.set_current_length_unit("mm")
        start_point.x().set_unit()
        start_point.y().set_unit()
        end_point.x().set_unit()
        end_point.y().set_unit()
        dimension.param().set_unit()

        self.assertAlmostEqual(300.0, end_point.x().value())
        self.assertAlmostEqual(400.0, end_point.y().value())
        self.assertAlmostEqual(dimension.param().value(), 500.0)

        end_point.request_move(8000.0, 6000.0)

        UNITS.set_current_length_unit("m")
        start_point.x().set_unit()
        start_point.y().set_unit()
        end_point.x().set_unit()
        end_point.y().set_unit()
        dimension.param().set_unit()

        # Make sure that the points get moved to 8, 6 and param length value gets changed to sqrt(8**2 + 6**2)
        self.assertAlmostEqual(8.0, end_point.x().value())
        self.assertAlmostEqual(6.0, end_point.y().value())
        self.assertAlmostEqual(10.0, dimension.param().value())

        end_point.request_move(0.0762, 0.1016)

        UNITS.set_current_length_unit("in")
        start_point.x().set_unit()
        start_point.y().set_unit()
        end_point.x().set_unit()
        end_point.y().set_unit()
        dimension.param().set_unit()

        # Make sure that the points get moved to 8, 6 and param length value gets changed to sqrt(8**2 + 6**2)
        self.assertAlmostEqual(3.0, end_point.x().value())
        self.assertAlmostEqual(4.0, end_point.y().value())
        self.assertAlmostEqual(5.0, dimension.param().value())


class AirfoilTests(unittest.TestCase):

    def test_correct_airfoil_generation_sharp_te(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)

        upper = Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, te]), name="UpperSurf")
        lower = Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, te]), name="LowerSurf")
        airfoil = Airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=te, lower_surf_end=te)

        self.assertEqual(2, len(airfoil.curves))
        self.assertEqual(1, len(airfoil.curves_to_reverse))

        self.assertEqual(airfoil.curves, [upper, lower])
        self.assertEqual(airfoil.curves_to_reverse, [upper])

    def test_correct_airfoil_generation_blunt_te(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        upper = Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]), name="UpperSurf")
        lower = Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        LineSegment(point_sequence=PointSequence(points=[te, upper4]))
        LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        airfoil = Airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=upper4, lower_surf_end=lower4)

        self.assertEqual(2, len(airfoil.curves))
        self.assertEqual(1, len(airfoil.curves_to_reverse))

        self.assertEqual(airfoil.curves, [upper, lower])
        self.assertEqual(airfoil.curves_to_reverse, [upper])

    def test_no_te_lines_airfoil_generation_blunt_te(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]), name="UpperSurf")
        Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        # LineSegment(point_sequence=PointSequence(points=[te, upper4]))
        # LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        self.assertRaises(ClosureError, Airfoil, leading_edge=le, trailing_edge=te, upper_surf_end=upper4,
                          lower_surf_end=lower4)

    def test_no_upper_te_line_airfoil_generation_blunt_te(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]), name="UpperSurf")
        Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        # LineSegment(point_sequence=PointSequence(points=[te, upper4]))
        LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        self.assertRaises(ClosureError, Airfoil, leading_edge=le, trailing_edge=te, upper_surf_end=upper4,
                          lower_surf_end=lower4)

    def test_no_lower_te_line_airfoil_generation_blunt_te(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]), name="UpperSurf")
        Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        LineSegment(point_sequence=PointSequence(points=[te, upper4]))
        # LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        self.assertRaises(ClosureError, Airfoil, leading_edge=le, trailing_edge=te, upper_surf_end=upper4,
                          lower_surf_end=lower4)

    def test_extra_branch_airfoil_generation_blunt_te_1(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper4]), name="UpperSurf")
        Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        Bezier(point_sequence=PointSequence(points=[le, upper2, upper3]))  # Extra curve branching from LE
        LineSegment(point_sequence=PointSequence(points=[te, upper4]))
        LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        self.assertRaises(BranchError, Airfoil, leading_edge=le, trailing_edge=te, upper_surf_end=upper4,
                          lower_surf_end=lower4)

    def test_complex_airfoil_1(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper5 = Point(0.8, 0.04)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        upper = Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper5]), name="UpperSurf")
        upper_line = LineSegment(point_sequence=PointSequence(points=[upper5, upper4]))
        lower = Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        LineSegment(point_sequence=PointSequence(points=[upper4, te]))
        LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        airfoil = Airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=upper4, lower_surf_end=lower4)

        self.assertEqual(3, len(airfoil.curves))
        self.assertEqual(2, len(airfoil.curves_to_reverse))

        self.assertEqual(airfoil.curves, [upper_line, upper, lower])
        self.assertEqual(airfoil.curves_to_reverse, [upper_line, upper])

    def test_coords(self):
        le = Point(0.0, 0.0)
        upper1 = Point(0.0, 0.05)
        upper2 = Point(0.05, 0.05)
        upper3 = Point(0.6, 0.03)
        upper5 = Point(0.8, 0.04)
        upper4 = Point(1.0, 0.005)
        te = Point(1.0, 0.0)
        lower1 = Point(0.0, -0.03)
        lower2 = Point(0.03, -0.03)
        lower3 = Point(0.7, 0.03)
        lower4 = Point(1.0, -0.005)

        Bezier(point_sequence=PointSequence(points=[le, upper1, upper2, upper3, upper5]), name="UpperSurf")
        LineSegment(point_sequence=PointSequence(points=[upper5, upper4]))
        Bezier(point_sequence=PointSequence(points=[le, lower1, lower2, lower3, lower4]), name="LowerSurf")
        LineSegment(point_sequence=PointSequence(points=[upper4, te]))
        LineSegment(point_sequence=PointSequence(points=[te, lower4]))
        airfoil = Airfoil(leading_edge=le, trailing_edge=te, upper_surf_end=upper4, lower_surf_end=lower4)

        coords = airfoil.get_coords_selig_format()

        # Make sure no points repeat
        repeat = False
        prev_xy = None
        for row in coords:
            if prev_xy is not None:
                if all([el_old == el_new for el_old, el_new in zip(prev_xy, row)]):
                    repeat = True
                    break
            prev_xy = row

        self.assertFalse(repeat)


class MEATests(unittest.TestCase):

    def test_blade_file_output(self):
        # Make the Bezier curve arrays
        b1_array = np.array([[0, 0], [0.0, 0.1], [0.3, 0.08], [0.7, 0.07], [1.0, 0.0]])
        b2_array = np.array([[0.0, -0.1], [0.3, -0.08], [0.7, -0.07]])
        b3_array = np.array([[0, 0], [0.0, 0.1], [0.3, 0.08], [0.7, 0.07], [1.0, 0.0]])
        b4_array = np.array([[0.0, -0.1], [0.3, -0.08], [0.7, -0.07]])
        b5_array = np.array([[0, 0], [0.0, 0.1], [0.3, 0.08], [0.7, 0.07], [1.0, 0.0]])
        b6_array = np.array([[0.0, -0.1], [0.3, -0.08], [0.7, -0.07]])

        b3_array += np.column_stack((np.zeros(shape=(b1_array.shape[0])), 0.2 * np.ones(shape=(b1_array.shape[0]))))
        b4_array += np.column_stack((np.zeros(shape=(b2_array.shape[0])), 0.2 * np.ones(shape=(b2_array.shape[0]))))
        b5_array += np.column_stack((np.zeros(shape=(b1_array.shape[0])), -0.2 * np.ones(shape=(b1_array.shape[0]))))
        b6_array += np.column_stack((np.zeros(shape=(b2_array.shape[0])), -0.2 * np.ones(shape=(b2_array.shape[0]))))

        # Generate the Points
        b1_points = [Point(xy[0], xy[1]) for xy in b1_array]
        b2_points = [b1_points[0]]
        b2_points.extend([Point(xy[0], xy[1]) for xy in b2_array])
        b2_points.append(b1_points[-1])
        b3_points = [Point(xy[0], xy[1]) for xy in b3_array]
        b4_points = [b3_points[0]]
        b4_points.extend([Point(xy[0], xy[1]) for xy in b4_array])
        b4_points.append(b3_points[-1])
        b5_points = [Point(xy[0], xy[1]) for xy in b5_array]
        b6_points = [b5_points[0]]
        b6_points.extend([Point(xy[0], xy[1]) for xy in b6_array])
        b6_points.append(b5_points[-1])

        # Generate the curves
        bez_curves = [Bezier(point_sequence=PointSequence(points=points)) for points in
                      [b1_points, b2_points, b3_points, b4_points, b5_points, b6_points]]

        # Generate the airfoils
        a1 = Airfoil(leading_edge=b1_points[0], trailing_edge=b1_points[-1], upper_surf_end=b1_points[-1],
                     lower_surf_end=b1_points[-1])
        a2 = Airfoil(leading_edge=b3_points[0], trailing_edge=b3_points[-1], upper_surf_end=b3_points[-1],
                     lower_surf_end=b3_points[-1])
        a3 = Airfoil(leading_edge=b5_points[0], trailing_edge=b5_points[-1], upper_surf_end=b5_points[-1],
                     lower_surf_end=b5_points[-1])

        # Generate the MEA
        mea = MEA(airfoils=[a1, a2, a3])

        # Output the blade file
        blade_file_dir = temp_dir
        airfoil_sys_name = "testAirfoilSys"
        blade_file = mea.write_mses_blade_file(airfoil_sys_name=airfoil_sys_name, blade_file_dir=blade_file_dir)

        # Load the coordinates from the created file
        blade = np.loadtxt(blade_file, skiprows=2)

        # Make sure that the 999.0 delimiter shows up exactly twice in the created blade file in each column
        # This means that there are 3 airfoils.
        self.assertEqual(2, len(np.where(blade[:, 0] == 999.0)[0]))
        self.assertEqual(2, len(np.where(blade[:, 1] == 999.0)[0]))

        # Remove the created blade file
        if os.path.exists(blade_file):
            os.remove(blade_file)
