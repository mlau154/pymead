import os
from unittest import TestCase

from pymead import TEST_DIR
from pymead.utils.read_write_files import load_data
from pymead.core.constraints import *
from pymead.core.geometry_collection import GeometryCollection


class GCSTests(TestCase):

    def test_rel_angle3_constraint(self):

        geo_col = GeometryCollection()
        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.4, -0.1)
        p3 = geo_col.add_point(0.5, 0.5)
        d1 = geo_col.add_constraint("RelAngle3Constraint", p1, p2, p3, value=3*np.pi/4)
        a1 = p2.measure_angle(p1)
        a2 = p2.measure_angle(p3)
        ra1 = (a1 - a2) % (2 * np.pi)
        self.assertAlmostEqual(ra1, d1.param().value())

    def test_rel_angle3_then_two_distance(self):
        geo_col = GeometryCollection()
        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.4, -0.1)
        p3 = geo_col.add_point(0.5, 0.5)
        ra1 = geo_col.add_constraint("RelAngle3Constraint", p1, p2, p3, value=3 * np.pi / 4)
        d1 = geo_col.add_constraint("DistanceConstraint", p1, p2, value=0.5)
        d2 = geo_col.add_constraint("DistanceConstraint", p2, p3, value=0.6)
        a1 = p2.measure_angle(p1)
        a2 = p2.measure_angle(p3)
        ra_val = (a1 - a2) % (2 * np.pi)
        self.assertAlmostEqual(ra_val, ra1.param().value())
        self.assertAlmostEqual(p1.measure_distance(p2), d1.param().value())
        self.assertAlmostEqual(p2.measure_distance(p3), d2.param().value())

    def test_symmetry_constraint(self):
        geo_col = GeometryCollection()
        p1 = geo_col.add_point(0.4, -0.03)
        p2 = geo_col.add_point(1.2, -0.03)
        p3 = geo_col.add_point(0.5, 0.22)
        p4 = geo_col.add_point(0.8, 0.9)
        geo_col.add_constraint("SymmetryConstraint", p1, p2, p3, p4)
        self.assertAlmostEqual(p4.x().value(), 0.5)
        self.assertAlmostEqual(p4.y().value(), -0.28)

    def test_mirror_ap3_constraint(self):
        geo_col_file = os.path.join(TEST_DIR, "core_tests", "mirror_ap3_constraint.jmea")
        geo_col = GeometryCollection.set_from_dict_rep(load_data(geo_col_file))
        self.assertTrue(geo_col.verify_all())

    def test_double_angle_constraint(self):
        geo_col_file = os.path.join(TEST_DIR, "core_tests", "double_angle.jmea")
        geo_col = GeometryCollection.set_from_dict_rep(load_data(geo_col_file))
        self.assertTrue(geo_col.verify_all())

    def test_connected_angles_constraint(self):
        geo_col_file = os.path.join(TEST_DIR, "core_tests", "connected_angles.jmea")
        geo_col = GeometryCollection.set_from_dict_rep(load_data(geo_col_file))
        self.assertTrue(geo_col.verify_all())

    def test_alt_fp_parametrization(self):
        geo_col_file = os.path.join(TEST_DIR, "core_tests", "alt_fp_parametrization_no_angle7.jmea")
        geo_col = GeometryCollection.set_from_dict_rep(load_data(geo_col_file))
        self.assertTrue(geo_col.verify_all())

    def test_alt_fp_parametrization_p7_first(self):
        # Test adding the lower free point angle in the order Point-7 --> LE --> TE
        geo_col_file = os.path.join(TEST_DIR, "core_tests", "alt_fp_parametrization_p7_first.jmea")
        geo_col = GeometryCollection.set_from_dict_rep(load_data(geo_col_file))
        self.assertTrue(geo_col.verify_all())

    def test_alt_fp_parametrization_p7_last(self):
        # Test adding the lower free point angle in the order TE --> LE --> Point-7
        geo_col_file = os.path.join(TEST_DIR, "core_tests", "alt_fp_parametrization_p7_last.jmea")
        geo_col = GeometryCollection.set_from_dict_rep(load_data(geo_col_file))
        self.assertTrue(geo_col.verify_all())
