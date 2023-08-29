import unittest
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import LineCollection
import numpy as np

# from pymead.analysis.calc_aero_data import calculate_CPV_mses
from pymead.analysis.read_aero_data import convert_blade_file_to_3d_array, read_streamline_grid_from_mses
from pymead.post.mses_field import generate_field_matplotlib
from pymead.utils.read_write_files import load_data


class CPKTest(unittest.TestCase):
    def test_CPK_calculation(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_41\analysis\analysis_17"

        # CPV = calculate_CPV_mses(analysis_subdir=analysis_dir)
        # print(f"{CPV = }")

    def test_CPV_calculation_opt_pai(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_61\analysis\analysis_284_w0-100"

        # CPV = calculate_CPV_mses(analysis_subdir=analysis_dir)
        # print(f"{CPV = }")
