import unittest

from pymead.utils.get_airfoil import extract_data_from_airfoiltools
class GetAirfoil(unittest.TestCase):

    def test_extract_data_from_airfoiltools(self):

        get_airfoil=extract_data_from_airfoiltools("n0012-il", None)
        self.assertEqual(get_airfoil.ndim, 2)