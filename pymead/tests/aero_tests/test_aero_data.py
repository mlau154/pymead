import os
import unittest

from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.analysis.calc_aero_data import run_xfoil, XFOILSettings
from pymead import TEST_DIR


class CalcAeroData(unittest.TestCase):

    def test_run_xfoil(self):

        xfoil_settings = XFOILSettings(
            base_dir=os.path.join(TEST_DIR, "aero_tests"),
            airfoil_name="xfoil_test",
            Re=2213243.6863567195832729,
            mode=0,
            alfa=3.0
        )

        aero_data, xfoil_log = run_xfoil(
            xfoil_settings=xfoil_settings,
            coords=extract_data_from_airfoiltools("n0012-il"),
        )

        self.assertAlmostEqual(aero_data["Cl"], 0.3325, places=4)
        self.assertAlmostEqual(aero_data["Cd"], 0.00582, places=7)
        self.assertAlmostEqual(aero_data["Cm"], 0.0015, places=4)
        self.assertAlmostEqual(aero_data["L/D"], 57.1306, places=4)

    def test_run_xfoil_inviscid(self):
        xfoil_settings = XFOILSettings(
            base_dir=os.path.join(TEST_DIR, "aero_tests"),
            airfoil_name="xfoil_test",
            mode=0,
            alfa=3.0,
            visc=False
        )

        aero_data, xfoil_log = run_xfoil(
            xfoil_settings=xfoil_settings,
            coords=extract_data_from_airfoiltools("n0012-il"),
        )

        self.assertAlmostEqual(aero_data["Cl"], 0.3649, places=4)
        self.assertAlmostEqual(aero_data["Cm"], -0.0043, places=4)
