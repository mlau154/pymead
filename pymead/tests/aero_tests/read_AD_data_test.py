import unittest
import os

from pymead.analysis.read_aero_data import read_actuator_disk_data_mses, read_grid_stats_from_mses


class ReadADDataTest(unittest.TestCase):
    def test_read(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_61\analysis\analysis_284_w0-100"
        grid_stats = read_grid_stats_from_mses(os.path.join(analysis_dir, "mplot_grid_stats.log"))
        read_actuator_disk_data_mses(os.path.join(analysis_dir, "mses.log"), grid_stats)
