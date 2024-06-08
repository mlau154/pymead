import os
import unittest
import numpy as np

from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.analysis.calc_aero_data import run_xfoil, XFOILSettings
from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead import TEST_DIR
from pymead.utils.read_write_files import load_data
from pymead.analysis.calc_aero_data import (run_xfoil, XFOILSettings, MSETSettings, MSESSettings, MPOLARSettings,
                                            AirfoilMSETMeshingParameters, calculate_aero_data)
from pymead.core.geometry_collection import GeometryCollection
from pymead import TEST_DIR, EXAMPLES_DIR, DependencyNotFoundError


class CalcAeroData(unittest.TestCase):

    def test_run_xfoil(self):

        xfoil_settings = XFOILSettings(
            base_dir=os.path.join(TEST_DIR, "aero_tests"),
            airfoil_name="xfoil_test",
            Re=2213243.6863567195832729,
            mode=0,
            alfa=3.0
        )

        try:
            aero_data, xfoil_log = run_xfoil(
                xfoil_settings=xfoil_settings,
                coords=extract_data_from_airfoiltools("n0012-il"),
            )

            self.assertAlmostEqual(aero_data["Cl"], 0.3325, places=4)
            self.assertAlmostEqual(aero_data["Cd"], 0.00582, places=7)
            self.assertAlmostEqual(aero_data["Cm"], 0.0015, places=4)
            self.assertAlmostEqual(aero_data["L/D"], 57.1306, places=4)

        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")

    def test_run_xfoil_inviscid(self):
        xfoil_settings = XFOILSettings(
            base_dir=os.path.join(TEST_DIR, "aero_tests"),
            airfoil_name="xfoil_test",
            mode=0,
            alfa=3.0,
            visc=False
        )

        try:
            aero_data, xfoil_log = run_xfoil(
                xfoil_settings=xfoil_settings,
                coords=extract_data_from_airfoiltools("n0012-il"),
            )

            self.assertAlmostEqual(aero_data["Cl"], 0.3649, places=4)
            self.assertAlmostEqual(aero_data["Cm"], -0.0043, places=4)

        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")

    def test_calculate_aero_data_viscous(self):
        xfoil_settings = XFOILSettings(
            base_dir=os.path.join(TEST_DIR, "aero_tests"),
            airfoil_name="calculate_aero_data_viscous_test",
            Re=2213243.6863567195832729,
            mode=0,
            alfa=3.0
        )

        aero_data, xfoil_log = calculate_aero_data(
            conn=None,
            airfoil_name="calculate_aero_data_viscous_test",
            xfoil_settings=xfoil_settings,
            airfoil_coord_dir=os.path.join(TEST_DIR, "aero_tests")
        )

        self.assertAlmostEqual(aero_data["Cl"], 0.3325, places=4)
        self.assertAlmostEqual(aero_data["Cd"], 0.00582, places=7)
        self.assertAlmostEqual(aero_data["Cm"], 0.0015, places=4)
        self.assertAlmostEqual(aero_data["L/D"], 57.1306, places=4)

    def test_calculate_aero_data_inviscid(self):
        xfoil_settings = XFOILSettings(
            base_dir=os.path.join(TEST_DIR, "aero_tests"),
            airfoil_name="xfoil_test",
            mode=0,
            alfa=3.0,
            visc=False
        )

        aero_data, xfoil_log = calculate_aero_data(
            conn=None,
            airfoil_name="calculate_aero_data_viscous_test",
            xfoil_settings=xfoil_settings,
            airfoil_coord_dir=os.path.join(TEST_DIR, "aero_tests")
        )

        self.assertAlmostEqual(aero_data["Cl"], 0.3649, places=4)
        self.assertAlmostEqual(aero_data["Cm"], -0.0043, places=4)



    def test_run_mpolar(self):
        geo_col = GeometryCollection.set_from_dict_rep(
            load_data(os.path.join(EXAMPLES_DIR, "basic_airfoil_sharp.jmea"))
        )
        mea = geo_col.add_mea([geo_col.container()["airfoils"]["Airfoil-1"]])
        mset_settings = MSETSettings(multi_airfoil_grid={"Airfoil-1": AirfoilMSETMeshingParameters()})
        mses_settings = MSESSettings({"Airfoil-1": [0.1, 0.1]}, Re=5.0e6, Ma=0.3)
        mpolar_settings = MPOLARSettings()

        try:
            aero_data, logs = calculate_aero_data(
                conn=None,
                airfoil_coord_dir=os.path.join(TEST_DIR, "aero_tests"),
                airfoil_name="mpolar_test",
                mea=mea,
                mea_airfoil_names=["Airfoil-1"],
                tool="MSES",
                mset_settings=mset_settings,
                mses_settings=mses_settings,
                mpolar_settings=mpolar_settings,
                alfa_array=np.linspace(-1.0, 1.0, 11)
            )

            self.assertTrue(len(aero_data["Cd"]) == 11)

        except DependencyNotFoundError as e:
            print(f"Warning: {str(e)}")
