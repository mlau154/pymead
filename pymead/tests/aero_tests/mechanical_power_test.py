import unittest
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import LineCollection
import numpy as np

from pymead.analysis.calc_aero_data import calculate_CPK_mses
from pymead.analysis.read_aero_data import convert_blade_file_to_3d_array, read_streamline_grid_from_mses
from pymead.post.mses_field import generate_field_matplotlib
from pymead.utils.read_write_files import load_data


class PKTest(unittest.TestCase):
    def test_PK_calculation(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_41\analysis\analysis_17"
        # fig, ax = plt.subplots()
        #
        # plt.scatter(x_grid, y_grid)
        #
        # segs1 = np.stack((x_grid, y_grid), axis=2)
        # segs2 = segs1.transpose(1, 0, 2)
        # plt.gca().add_collection(LineCollection(segs1))
        # plt.gca().add_collection(LineCollection(segs2))
        # start = 0
        # end = start + 362
        # for i in range(66):
        #     plt.gca().plot(data[start:end, 0], data[start:end, 1], color="black")
        #     start += 362
        #     end += 362
        #
        # ax.scatter(field[0], field[1], marker="s", color="red")

        CPK = calculate_CPK_mses(analysis_subdir=analysis_dir)
        print(f"{CPK = }")

        # quad = generate_field_matplotlib(var="p", axs=ax, analysis_subdir=analysis_dir,
        #                                  cmap_field="Spectral_r",
        #                                       cmap_airfoil=mpl_colors.ListedColormap("gray"),
        #                                       vmin=-1.5, vmax=0.5)
        #
        # coords = convert_blade_file_to_3d_array(os.path.join(analysis_dir, "blade.analysis_17"))
        # for airfoil in coords:
        #     ax.plot(airfoil[:, 0], airfoil[:, 1], color="black", zorder=500)
        #     polygon = Polygon(airfoil, closed=False, color="#000000AA")
        #     ax.add_patch(polygon)

        # x_nac_le_field = field[0, grid_stats["ILE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # y_nac_le_field = field[1, grid_stats["ILE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # x_nac_te_field = field[0, grid_stats["ITE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # y_nac_te_field = field[1, grid_stats["ITE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # x_main_te_field = field[0, grid_stats["ITE1"][0] - 2, grid_stats["Jside2"][1] - 2:grid_stats["Jside1"][0] - 4]
        # y_main_te_field = field[1, grid_stats["ITE1"][0] - 2, grid_stats["Jside2"][1] - 2:grid_stats["Jside1"][0] - 4]
        # x_main_te_field2 = field[0, grid_stats["ITE1"][0] - 1, :]
        # y_main_te_field2 = field[1, grid_stats["ITE1"][0] - 1, :]
        # print(f"{grid_stats['Jside1'][0] - 3 = }")
        # print(f"{grid_stats['Jside2'][1] - 2 = }")
        # print(f"{y_main_te_field = }")
        # x_nac_te_coord = coords[2][0, 0]
        # y_nac_te_coord = coords[2][0, 1]
        # x_main_te_coord = coords[0][-1, 0]
        # y_main_te_coord = coords[0][-1, 1]
        # alf_hub_90 = np.arctan2(y_main_te_coord - y_nac_te_coord, x_main_te_coord - x_nac_te_coord)
        # print(f"{np.rad2deg(alf_hub_90) = }")
        # ax.plot(x_nac_le_field, y_nac_le_field, marker="o", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_nac_te_coord, y_nac_te_coord, marker="o", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_nac_te_field, y_nac_te_field, marker="x", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_main_te_coord, y_main_te_coord, marker="o", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_main_te_field, y_main_te_field, marker="x", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_main_te_field2, y_main_te_field2, marker="*", ls="none", mfc="none", mec="black", markersize=12)
        plt.show()
        pass

    def test_CPK_calculation_baseline(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_70\analysis\analysis_0"
        calculate_CPK_mses(analysis_dir)

    def test_PK_calculation_opt_pai(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_61\analysis\analysis_284_w0-100"
        # fig, ax = plt.subplots()
        #
        # plt.scatter(x_grid, y_grid)
        #
        # segs1 = np.stack((x_grid, y_grid), axis=2)
        # segs2 = segs1.transpose(1, 0, 2)
        # plt.gca().add_collection(LineCollection(segs1))
        # plt.gca().add_collection(LineCollection(segs2))
        # start = 0
        # end = start + 362
        # for i in range(66):
        #     plt.gca().plot(data[start:end, 0], data[start:end, 1], color="black")
        #     start += 362
        #     end += 362
        #
        # ax.scatter(field[0], field[1], marker="s", color="red")

        CPK = calculate_CPK_mses(analysis_subdir=analysis_dir)
        print(f"{CPK = }")

        # quad = generate_field_matplotlib(var="p", axs=ax, analysis_subdir=analysis_dir,
        #                                  cmap_field="Spectral_r",
        #                                       cmap_airfoil=mpl_colors.ListedColormap("gray"),
        #                                       vmin=-1.5, vmax=0.5)
        #
        # coords = convert_blade_file_to_3d_array(os.path.join(analysis_dir, "blade.analysis_17"))
        # for airfoil in coords:
        #     ax.plot(airfoil[:, 0], airfoil[:, 1], color="black", zorder=500)
        #     polygon = Polygon(airfoil, closed=False, color="#000000AA")
        #     ax.add_patch(polygon)
        #
        # x_nac_le_field = field[0, grid_stats["ILE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # y_nac_le_field = field[1, grid_stats["ILE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # x_nac_te_field = field[0, grid_stats["ITE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # y_nac_te_field = field[1, grid_stats["ITE2"][2] - 1, grid_stats["Jside2"][2] - 1]
        # x_main_te_field = field[0, grid_stats["ITE1"][0] - 2, grid_stats["Jside2"][1] - 2:grid_stats["Jside1"][0] - 4]
        # y_main_te_field = field[1, grid_stats["ITE1"][0] - 2, grid_stats["Jside2"][1] - 2:grid_stats["Jside1"][0] - 4]
        # x_main_te_field2 = field[0, grid_stats["ITE1"][0] - 1, :]
        # y_main_te_field2 = field[1, grid_stats["ITE1"][0] - 1, :]
        # print(f"{grid_stats['Jside1'][0] - 3 = }")
        # print(f"{grid_stats['Jside2'][1] - 2 = }")
        # print(f"{y_main_te_field = }")
        # x_nac_te_coord = coords[2][0, 0]
        # y_nac_te_coord = coords[2][0, 1]
        # x_main_te_coord = coords[0][-1, 0]
        # y_main_te_coord = coords[0][-1, 1]
        # alf_hub_90 = np.arctan2(y_main_te_coord - y_nac_te_coord, x_main_te_coord - x_nac_te_coord)
        # print(f"{np.rad2deg(alf_hub_90) = }")
        # ax.plot(x_nac_le_field, y_nac_le_field, marker="o", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_nac_te_coord, y_nac_te_coord, marker="o", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_nac_te_field, y_nac_te_field, marker="x", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_main_te_coord, y_main_te_coord, marker="o", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_main_te_field, y_main_te_field, marker="x", ls="none", mfc="none", mec="black", markersize=12)
        # ax.plot(x_main_te_field2, y_main_te_field2, marker="*", ls="none", mfc="none", mec="black", markersize=12)
        plt.show()
        pass

    def test_PK_calculation_opt_pai2(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_68\analysis\analysis_158_w0-100"
        CPK = calculate_CPK_mses(analysis_subdir=analysis_dir)
        print(f"{CPK = }")
