from unittest import TestCase

import numpy as np

from pymead.core.gcs2 import GCS2
from pymead.core.constraints import *
from pymead.core.geometry_collection import GeometryCollection
from pymead.core.point import Point
from pymead.core.param import LengthParam, AngleParam


class GCSTests(TestCase):

    def test_rel_angle3_constraint(self):

        geo_col = GeometryCollection()
        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.4, -0.1)
        p3 = geo_col.add_point(0.5, 0.5)
        d1 = RelAngle3Constraint(p1, p2, p3, value=3*np.pi/4)
        geo_col.add_constraint(d1)
        geo_col.gcs.solve(d1)
        geo_col.gcs.update_from_points(d1)
        a1 = p2.measure_angle(p1)
        a2 = p2.measure_angle(p3)
        ra1 = (a1 - a2) % (2 * np.pi)
        self.assertAlmostEqual(ra1, d1.param().value())

    def test_rel_angle3_then_two_distance(self):
        geo_col = GeometryCollection()
        p1 = geo_col.add_point(0.0, 0.0)
        p2 = geo_col.add_point(0.4, -0.1)
        p3 = geo_col.add_point(0.5, 0.5)
        ra1 = RelAngle3Constraint(p1, p2, p3, value=3 * np.pi / 4)
        d1 = DistanceConstraint(p1, p2, value=0.5)
        d2 = DistanceConstraint(p2, p3, value=0.6)
        for cnstr in [ra1, d1, d2]:
            geo_col.add_constraint(cnstr)
            geo_col.gcs.solve(cnstr)
            geo_col.gcs.update_from_points(cnstr)
        a1 = p2.measure_angle(p1)
        a2 = p2.measure_angle(p3)
        ra_val = (a1 - a2) % (2 * np.pi)
        self.assertAlmostEqual(ra_val, ra1.param().value())
        self.assertAlmostEqual(p1.measure_distance(p2), d1.param().value())
        self.assertAlmostEqual(p2.measure_distance(p3), d2.param().value())
