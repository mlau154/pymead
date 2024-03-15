import unittest

from pymead.core.geometry_collection import GeometryCollection


class ParamEquationsTest(unittest.TestCase):

    def test_add(self):
        geo_col = GeometryCollection()
        geo_col.add_param(1.2, name="A")
        geo_col.add_param(1.5, name="B")
        C = geo_col.add_param(0.0, name="C")
        C.update_equation("$A + $B")
        self.assertAlmostEqual(2.7, C.value())
