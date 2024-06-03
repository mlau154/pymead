import os
import unittest

from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.analysis.calc_aero_data import run_xfoil
from pymead import TEST_DIR


class CalcAeroData(unittest.TestCase):

    def test_run_xfoil(self):
        xfoil_settings = {
            "Re": 2213243.6863567195832729,
            "Ma": 0.1,
            "prescribe": "Angle of Attack (deg)",
            "timeout": 8.0,
            "iter": 150,
            "xtr": [1.0, 1.0],
            "N": 9.0,
            "airfoil_analysis_dir": os.path.join(TEST_DIR, "aero_tests"),
            "airfoil_coord_file_name": "xfoil_test",
            "visc": True,
            "alfa": 3.0
        }

        aero_data, xfoil_log = run_xfoil(
            airfoil_name=xfoil_settings["airfoil_coord_file_name"],
            base_dir=xfoil_settings["airfoil_analysis_dir"],
            xfoil_settings=xfoil_settings,
            coords=extract_data_from_airfoiltools("n0012-il"),
        )

        self.assertAlmostEqual(aero_data["Cl"], 0.3325, places=4)
        self.assertAlmostEqual(aero_data["Cd"], 0.00582, places=7)
        self.assertAlmostEqual(aero_data["Cm"], 0.0015, places=4)
        self.assertAlmostEqual(aero_data["L/D"], 57.1306, places=4)


    def test_run_xfoil_inviscid(self):
        xfoil_settings = {
            "Re": 0,
            "Ma": 0.1,
            "prescribe": "Angle of Attack (deg)",
            "timeout": 8.0,
            "iter": 150,
            "xtr": [1.0, 1.0],
            "N": 9.0,
            "airfoil_analysis_dir": os.path.join(TEST_DIR, "aero_tests"),
            "airfoil_coord_file_name": "xfoil_test",
            "visc": False,
            "alfa": 3.0
        }

        aero_data, xfoil_log = run_xfoil(
            airfoil_name=xfoil_settings["airfoil_coord_file_name"],
            base_dir=xfoil_settings["airfoil_analysis_dir"],
            xfoil_settings=xfoil_settings,
            coords=extract_data_from_airfoiltools("n0012-il"),
        )

        self.assertAlmostEqual(aero_data["Cl"], 0.3649, places=4)
        self.assertAlmostEqual(aero_data["Cm"], -0.0043, places=4)
