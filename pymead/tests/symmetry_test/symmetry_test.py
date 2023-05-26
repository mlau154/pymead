from pymead.utils.read_write_files import load_data
from pymead.core.line import InfiniteLine
from pymead.core.bezier import Bezier
import numpy as np
import unittest
from copy import deepcopy


class SymmetryTest(unittest.TestCase):
    def test_symmetry_overwing(self):
        control_points = load_data('control_points2.json')
        hub_control_point_array = []
        main_control_point_array = []
        nacelle_control_point_array = []
        for curve in control_points['A1']:
            hub_control_point_array.append(np.array(curve))
        for curve in control_points['A0']:
            main_control_point_array.append(np.array(curve))
        for curve in control_points['A2']:
            nacelle_control_point_array.append(np.array(curve))
        bezier_1_main = Bezier(main_control_point_array[0])
        bezier_2_main = Bezier(main_control_point_array[1])
        bezier_1_nac = Bezier(nacelle_control_point_array[-1])
        bezier_2_nac = Bezier(nacelle_control_point_array[-2])
        main_ctrlpt_0 = main_control_point_array[0][0, :]
        main_ctrlpt_1 = main_control_point_array[0][1, :]
        nac_ctrlpt_m1 = nacelle_control_point_array[-1][-1, :]
        nac_ctrlpt_m2 = nacelle_control_point_array[-1][-2, :]
        main_prob_pt_2 = main_control_point_array[0][2, :]
        main_prob_pt_1 = main_control_point_array[0][3, :]
        nac_prob_pt_2 = nacelle_control_point_array[-1][2, :]
        nac_prob_pt_1 = nacelle_control_point_array[-1][1, :]
        main_len = np.hypot(main_prob_pt_2[0] - main_prob_pt_1[0], main_prob_pt_2[1] - main_prob_pt_1[1])
        nac_len = np.hypot(nac_prob_pt_2[0] - nac_prob_pt_1[0], nac_prob_pt_2[1] - nac_prob_pt_1[1])
        main_theta_te_1_abs = np.arctan2(main_ctrlpt_0[1] - main_ctrlpt_1[1], main_ctrlpt_0[0] - main_ctrlpt_1[0])
        nac_theta_te_2_abs = np.arctan2(nac_ctrlpt_m1[1] - nac_ctrlpt_m2[1], nac_ctrlpt_m1[0] - nac_ctrlpt_m2[0])
        coords = load_data('coords2.json')
        te_xy = (hub_control_point_array[0][0, :] + hub_control_point_array[-1][-1, :]) / 2
        le_xy = hub_control_point_array[2][-1, :]
        hub_alf_abs = np.arctan2(te_xy[1] - le_xy[1],
                                 te_xy[0] - le_xy[0])
        nacelle_coords = np.array(coords['A2'])
        main_coords = np.array(coords['A0'])[::-1]
        inf_line = InfiniteLine(x1=le_xy[0], y1=le_xy[1], theta_rad=hub_alf_abs)
        std_coeffs = inf_line.get_standard_form_coeffs()
        nacelle_distance = np.array([])
        main_distance = np.array([])
        nacelle_curve_3 = np.array(control_points['A2'][2])
        nacelle_curve_4 = np.array(control_points['A2'][3])
        main_curve_1 = np.array(control_points['A0'][0])[::-1]
        main_curve_2 = np.array(control_points['A0'][1])[::-1]
        hub_curve_1 = np.array(control_points['A1'][0])
        hub_curve_2 = np.array(control_points['A1'][1])
        hub_curve_3 = np.array(control_points['A1'][2])
        hub_curve_4 = np.array(control_points['A1'][3])[::-1]
        hub_curve_5 = np.array(control_points['A1'][4])[::-1]
        hub_curve_6 = np.array(control_points['A1'][5])[::-1]
        nacelle_curve_4_dist = []
        nacelle_curve_3_dist = []
        main_curve_1_dist = []
        main_curve_2_dist = []
        hub_curve_1_dist = []
        hub_curve_2_dist = []
        hub_curve_3_dist = []
        hub_curve_4_dist = []
        hub_curve_5_dist = []
        hub_curve_6_dist = []
        for coord_idx in range(len(nacelle_coords)):
            nacelle_distance = np.append(nacelle_distance, (std_coeffs['A'] * nacelle_coords[coord_idx, 0] +
                                                            std_coeffs['B'] * nacelle_coords[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_distance = np.append(main_distance, (std_coeffs['A'] * main_coords[coord_idx, 0] +
                                                      std_coeffs['B'] * main_coords[coord_idx, 1] + std_coeffs['C']
                                                      ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(nacelle_curve_4)):
            nacelle_curve_4_dist = np.append(nacelle_curve_4_dist, (std_coeffs['A'] * nacelle_curve_4[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_4[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_1_dist = np.append(main_curve_1_dist, (std_coeffs['A'] * main_curve_1[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_1[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        for coord_idx in range(len(nacelle_curve_3)):
            nacelle_curve_3_dist = np.append(nacelle_curve_3_dist, (std_coeffs['A'] * nacelle_curve_3[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_3[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_2_dist = np.append(main_curve_2_dist, (std_coeffs['A'] * main_curve_2[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_2[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_1)):
            hub_curve_1_dist = np.append(hub_curve_1_dist, (std_coeffs['A'] * hub_curve_1[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_1[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_6_dist = np.append(hub_curve_6_dist, (std_coeffs['A'] * hub_curve_6[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_6[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_2)):
            hub_curve_2_dist = np.append(hub_curve_2_dist, (std_coeffs['A'] * hub_curve_2[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_2[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_5_dist = np.append(hub_curve_5_dist, (std_coeffs['A'] * hub_curve_5[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_5[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_3)):
            hub_curve_3_dist = np.append(hub_curve_3_dist, (std_coeffs['A'] * hub_curve_3[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_3[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_4_dist = np.append(hub_curve_4_dist, (std_coeffs['A'] * hub_curve_4[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_4[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        diff14 = np.abs(nacelle_curve_4_dist) - np.abs(main_curve_1_dist)
        diff23 = np.abs(nacelle_curve_3_dist) - np.abs(main_curve_2_dist)
        diff16 = np.abs(hub_curve_1_dist) - np.abs(hub_curve_6_dist)
        diff25 = np.abs(hub_curve_2_dist) - np.abs(hub_curve_5_dist)
        diff34 = np.abs(hub_curve_3_dist) - np.abs(hub_curve_4_dist)
        diff = np.abs(nacelle_distance) - np.abs(main_distance)
        test_zero_tol = 1e-12

        # Check 1: ensure nacelle and main element are symmetric from the fan tip to the trailing edge:
        self.assertTrue(np.all(np.isclose(diff[298:], 0.0, rtol=test_zero_tol)))

        # Control point checks: make sure all control points from fan to trailing edge are symmetric
        for diff_ in [diff14, diff23, diff16, diff25, diff34]:
            self.assertTrue(np.all(np.isclose(diff_, 0.0, rtol=test_zero_tol)))
        pass

    def test_symmetry_overwing_updated(self):
        numbered_control_points = deepcopy(load_data('control_points_230_overwing.json'))
        control_points = {f'A{idx}': v for idx, v in enumerate(numbered_control_points)}
        hub_control_point_array = []
        main_control_point_array = []
        nacelle_control_point_array = []
        for curve in control_points['A1']:
            hub_control_point_array.append(np.array(curve))
        for curve in control_points['A0']:
            main_control_point_array.append(np.array(curve))
        for curve in control_points['A2']:
            nacelle_control_point_array.append(np.array(curve))
        bezier_1_main = Bezier(main_control_point_array[0])
        bezier_2_main = Bezier(main_control_point_array[1])
        bezier_1_nac = Bezier(nacelle_control_point_array[-1])
        bezier_2_nac = Bezier(nacelle_control_point_array[-2])
        main_ctrlpt_0 = main_control_point_array[0][0, :]
        main_ctrlpt_1 = main_control_point_array[0][1, :]
        nac_ctrlpt_m1 = nacelle_control_point_array[-1][-1, :]
        nac_ctrlpt_m2 = nacelle_control_point_array[-1][-2, :]
        main_prob_pt_2 = main_control_point_array[0][2, :]
        main_prob_pt_1 = main_control_point_array[0][3, :]
        nac_prob_pt_2 = nacelle_control_point_array[-1][2, :]
        nac_prob_pt_1 = nacelle_control_point_array[-1][1, :]
        main_len = np.hypot(main_prob_pt_2[0] - main_prob_pt_1[0], main_prob_pt_2[1] - main_prob_pt_1[1])
        nac_len = np.hypot(nac_prob_pt_2[0] - nac_prob_pt_1[0], nac_prob_pt_2[1] - nac_prob_pt_1[1])
        main_theta_te_1_abs = np.arctan2(main_ctrlpt_0[1] - main_ctrlpt_1[1], main_ctrlpt_0[0] - main_ctrlpt_1[0])
        nac_theta_te_2_abs = np.arctan2(nac_ctrlpt_m1[1] - nac_ctrlpt_m2[1], nac_ctrlpt_m1[0] - nac_ctrlpt_m2[0])
        coords = load_data('coords2.json')
        te_xy = (hub_control_point_array[0][0, :] + hub_control_point_array[-1][-1, :]) / 2
        le_xy = hub_control_point_array[2][-1, :]
        hub_alf_abs = np.arctan2(te_xy[1] - le_xy[1],
                                 te_xy[0] - le_xy[0])
        nacelle_coords = np.array(coords['A2'])
        main_coords = np.array(coords['A0'])[::-1]
        inf_line = InfiniteLine(x1=le_xy[0], y1=le_xy[1], theta_rad=hub_alf_abs)
        std_coeffs = inf_line.get_standard_form_coeffs()
        nacelle_distance = np.array([])
        main_distance = np.array([])
        nacelle_curve_3 = np.array(control_points['A2'][2])
        nacelle_curve_4 = np.array(control_points['A2'][3])
        main_curve_1 = np.array(control_points['A0'][0])[::-1]
        main_curve_2 = np.array(control_points['A0'][1])[::-1]
        hub_curve_1 = np.array(control_points['A1'][0])
        hub_curve_2 = np.array(control_points['A1'][1])
        hub_curve_3 = np.array(control_points['A1'][2])
        hub_curve_4 = np.array(control_points['A1'][3])[::-1]
        hub_curve_5 = np.array(control_points['A1'][4])[::-1]
        hub_curve_6 = np.array(control_points['A1'][5])[::-1]
        nacelle_curve_4_dist = []
        nacelle_curve_3_dist = []
        main_curve_1_dist = []
        main_curve_2_dist = []
        hub_curve_1_dist = []
        hub_curve_2_dist = []
        hub_curve_3_dist = []
        hub_curve_4_dist = []
        hub_curve_5_dist = []
        hub_curve_6_dist = []
        for coord_idx in range(len(nacelle_coords)):
            nacelle_distance = np.append(nacelle_distance, (std_coeffs['A'] * nacelle_coords[coord_idx, 0] +
                                                            std_coeffs['B'] * nacelle_coords[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_distance = np.append(main_distance, (std_coeffs['A'] * main_coords[coord_idx, 0] +
                                                      std_coeffs['B'] * main_coords[coord_idx, 1] + std_coeffs['C']
                                                      ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(nacelle_curve_4)):
            nacelle_curve_4_dist = np.append(nacelle_curve_4_dist, (std_coeffs['A'] * nacelle_curve_4[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_4[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_1_dist = np.append(main_curve_1_dist, (std_coeffs['A'] * main_curve_1[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_1[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        for coord_idx in range(len(nacelle_curve_3)):
            nacelle_curve_3_dist = np.append(nacelle_curve_3_dist, (std_coeffs['A'] * nacelle_curve_3[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_3[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_2_dist = np.append(main_curve_2_dist, (std_coeffs['A'] * main_curve_2[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_2[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_1)):
            hub_curve_1_dist = np.append(hub_curve_1_dist, (std_coeffs['A'] * hub_curve_1[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_1[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_6_dist = np.append(hub_curve_6_dist, (std_coeffs['A'] * hub_curve_6[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_6[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_2)):
            hub_curve_2_dist = np.append(hub_curve_2_dist, (std_coeffs['A'] * hub_curve_2[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_2[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_5_dist = np.append(hub_curve_5_dist, (std_coeffs['A'] * hub_curve_5[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_5[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_3)):
            hub_curve_3_dist = np.append(hub_curve_3_dist, (std_coeffs['A'] * hub_curve_3[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_3[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_4_dist = np.append(hub_curve_4_dist, (std_coeffs['A'] * hub_curve_4[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_4[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        diff14 = np.abs(nacelle_curve_4_dist) - np.abs(main_curve_1_dist)
        diff23 = np.abs(nacelle_curve_3_dist) - np.abs(main_curve_2_dist)
        diff16 = np.abs(hub_curve_1_dist) - np.abs(hub_curve_6_dist)
        diff25 = np.abs(hub_curve_2_dist) - np.abs(hub_curve_5_dist)
        diff34 = np.abs(hub_curve_3_dist) - np.abs(hub_curve_4_dist)
        diff = np.abs(nacelle_distance) - np.abs(main_distance)
        test_zero_tol = 1e-12

        # Control point checks: make sure all control points from fan to trailing edge are symmetric
        for diff_ in [diff14, diff23, diff16, diff25, diff34]:
            self.assertTrue(np.all(np.isclose(diff_, 0.0, rtol=test_zero_tol)))
        pass

    def test_symmetry_underwing(self):
        control_points = load_data('control_points_upper1.json')
        hub_control_point_array = []
        main_control_point_array = []
        nacelle_control_point_array = []
        for curve in control_points['A1']:
            hub_control_point_array.append(np.array(curve))
        for curve in control_points['A0']:
            main_control_point_array.append(np.array(curve))
        for curve in control_points['A2']:
            nacelle_control_point_array.append(np.array(curve))
        coords = load_data('coords2.json')
        te_xy = (hub_control_point_array[0][0, :] + hub_control_point_array[-1][-1, :]) / 2
        le_xy = hub_control_point_array[2][-1, :]
        hub_alf_abs = np.arctan2(te_xy[1] - le_xy[1],
                                 te_xy[0] - le_xy[0])
        nacelle_coords = np.array(coords['A2'])
        main_coords = np.array(coords['A0'])[::-1]
        inf_line = InfiniteLine(x1=le_xy[0], y1=le_xy[1], theta_rad=hub_alf_abs)
        std_coeffs = inf_line.get_standard_form_coeffs()
        nacelle_distance = np.array([])
        main_distance = np.array([])
        nacelle_curve_1 = np.array(control_points['A2'][0])
        nacelle_curve_2 = np.array(control_points['A2'][1])
        main_curve_3 = np.array(control_points['A0'][2])[::-1]
        main_curve_4 = np.array(control_points['A0'][3])[::-1]
        hub_curve_1 = np.array(control_points['A1'][0])
        hub_curve_2 = np.array(control_points['A1'][1])
        hub_curve_3 = np.array(control_points['A1'][2])
        hub_curve_4 = np.array(control_points['A1'][3])[::-1]
        hub_curve_5 = np.array(control_points['A1'][4])[::-1]
        hub_curve_6 = np.array(control_points['A1'][5])[::-1]
        nacelle_curve_4_dist = []
        nacelle_curve_3_dist = []
        main_curve_1_dist = []
        main_curve_2_dist = []
        hub_curve_1_dist = []
        hub_curve_2_dist = []
        hub_curve_3_dist = []
        hub_curve_4_dist = []
        hub_curve_5_dist = []
        hub_curve_6_dist = []
        for coord_idx in range(len(nacelle_coords)):
            nacelle_distance = np.append(nacelle_distance, (std_coeffs['A'] * nacelle_coords[coord_idx, 0] +
                                                            std_coeffs['B'] * nacelle_coords[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_distance = np.append(main_distance, (std_coeffs['A'] * main_coords[coord_idx, 0] +
                                                      std_coeffs['B'] * main_coords[coord_idx, 1] + std_coeffs['C']
                                                      ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(nacelle_curve_2)):
            nacelle_curve_4_dist = np.append(nacelle_curve_4_dist, (std_coeffs['A'] * nacelle_curve_2[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_2[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_1_dist = np.append(main_curve_1_dist, (std_coeffs['A'] * main_curve_3[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_3[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        for coord_idx in range(len(nacelle_curve_1)):
            nacelle_curve_3_dist = np.append(nacelle_curve_3_dist, (std_coeffs['A'] * nacelle_curve_1[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_1[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_2_dist = np.append(main_curve_2_dist, (std_coeffs['A'] * main_curve_4[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_4[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_1)):
            hub_curve_1_dist = np.append(hub_curve_1_dist, (std_coeffs['A'] * hub_curve_1[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_1[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_6_dist = np.append(hub_curve_6_dist, (std_coeffs['A'] * hub_curve_6[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_6[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_2)):
            hub_curve_2_dist = np.append(hub_curve_2_dist, (std_coeffs['A'] * hub_curve_2[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_2[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_5_dist = np.append(hub_curve_5_dist, (std_coeffs['A'] * hub_curve_5[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_5[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_3)):
            hub_curve_3_dist = np.append(hub_curve_3_dist, (std_coeffs['A'] * hub_curve_3[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_3[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_4_dist = np.append(hub_curve_4_dist, (std_coeffs['A'] * hub_curve_4[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_4[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        diff14 = np.abs(nacelle_curve_4_dist) - np.abs(main_curve_1_dist)
        diff23 = np.abs(nacelle_curve_3_dist) - np.abs(main_curve_2_dist)
        diff16 = np.abs(hub_curve_1_dist) - np.abs(hub_curve_6_dist)
        diff25 = np.abs(hub_curve_2_dist) - np.abs(hub_curve_5_dist)
        diff34 = np.abs(hub_curve_3_dist) - np.abs(hub_curve_4_dist)

        test_zero_tol = 1e-15

        # Control point checks: make sure all control points from fan to trailing edge are symmetric
        for diff_ in [diff14, diff23, diff16, diff25, diff34]:
            self.assertTrue(np.all(np.isclose(diff_, 0.0, rtol=test_zero_tol)))
        pass

    def test_symmetry_underwing_updated(self):
        numbered_control_points = deepcopy(load_data('control_points_180_underwing.json'))
        control_points = {f'A{idx}': v for idx, v in enumerate(numbered_control_points)}
        hub_control_point_array = []
        main_control_point_array = []
        nacelle_control_point_array = []
        for curve in control_points['A1']:
            hub_control_point_array.append(np.array(curve))
        for curve in control_points['A0']:
            main_control_point_array.append(np.array(curve))
        for curve in control_points['A2']:
            nacelle_control_point_array.append(np.array(curve))
        coords = load_data('coords2.json')
        te_xy = (hub_control_point_array[0][0, :] + hub_control_point_array[-1][-1, :]) / 2
        le_xy = hub_control_point_array[2][-1, :]
        hub_alf_abs = np.arctan2(te_xy[1] - le_xy[1],
                                 te_xy[0] - le_xy[0])
        nacelle_coords = np.array(coords['A2'])
        main_coords = np.array(coords['A0'])[::-1]
        inf_line = InfiniteLine(x1=le_xy[0], y1=le_xy[1], theta_rad=hub_alf_abs)
        std_coeffs = inf_line.get_standard_form_coeffs()
        nacelle_distance = np.array([])
        main_distance = np.array([])
        nacelle_curve_1 = np.array(control_points['A2'][0])
        nacelle_curve_2 = np.array(control_points['A2'][1])
        main_curve_3 = np.array(control_points['A0'][2])[::-1]
        main_curve_4 = np.array(control_points['A0'][3])[::-1]
        hub_curve_1 = np.array(control_points['A1'][0])
        hub_curve_2 = np.array(control_points['A1'][1])
        hub_curve_3 = np.array(control_points['A1'][2])
        hub_curve_4 = np.array(control_points['A1'][3])[::-1]
        hub_curve_5 = np.array(control_points['A1'][4])[::-1]
        hub_curve_6 = np.array(control_points['A1'][5])[::-1]
        nacelle_curve_4_dist = []
        nacelle_curve_3_dist = []
        main_curve_1_dist = []
        main_curve_2_dist = []
        hub_curve_1_dist = []
        hub_curve_2_dist = []
        hub_curve_3_dist = []
        hub_curve_4_dist = []
        hub_curve_5_dist = []
        hub_curve_6_dist = []
        for coord_idx in range(len(nacelle_coords)):
            nacelle_distance = np.append(nacelle_distance, (std_coeffs['A'] * nacelle_coords[coord_idx, 0] +
                                                            std_coeffs['B'] * nacelle_coords[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_distance = np.append(main_distance, (std_coeffs['A'] * main_coords[coord_idx, 0] +
                                                      std_coeffs['B'] * main_coords[coord_idx, 1] + std_coeffs['C']
                                                      ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(nacelle_curve_2)):
            nacelle_curve_4_dist = np.append(nacelle_curve_4_dist, (std_coeffs['A'] * nacelle_curve_2[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_2[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_1_dist = np.append(main_curve_1_dist, (std_coeffs['A'] * main_curve_3[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_3[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        for coord_idx in range(len(nacelle_curve_1)):
            nacelle_curve_3_dist = np.append(nacelle_curve_3_dist, (std_coeffs['A'] * nacelle_curve_1[coord_idx, 0] +
                                                                    std_coeffs['B'] * nacelle_curve_1[coord_idx, 1] + std_coeffs['C']
                                                                    ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            main_curve_2_dist = np.append(main_curve_2_dist, (std_coeffs['A'] * main_curve_4[coord_idx, 0] +
                                                              std_coeffs['B'] * main_curve_4[coord_idx, 1] + std_coeffs['C']
                                                              ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_1)):
            hub_curve_1_dist = np.append(hub_curve_1_dist, (std_coeffs['A'] * hub_curve_1[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_1[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_6_dist = np.append(hub_curve_6_dist, (std_coeffs['A'] * hub_curve_6[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_6[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_2)):
            hub_curve_2_dist = np.append(hub_curve_2_dist, (std_coeffs['A'] * hub_curve_2[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_2[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_5_dist = np.append(hub_curve_5_dist, (std_coeffs['A'] * hub_curve_5[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_5[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
        for coord_idx in range(len(hub_curve_3)):
            hub_curve_3_dist = np.append(hub_curve_3_dist, (std_coeffs['A'] * hub_curve_3[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_3[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))
            hub_curve_4_dist = np.append(hub_curve_4_dist, (std_coeffs['A'] * hub_curve_4[coord_idx, 0] +
                                                            std_coeffs['B'] * hub_curve_4[coord_idx, 1] + std_coeffs['C']
                                                            ) / np.hypot(std_coeffs['A'], std_coeffs['B']))

        diff14 = np.abs(nacelle_curve_4_dist) - np.abs(main_curve_1_dist)
        diff23 = np.abs(nacelle_curve_3_dist) - np.abs(main_curve_2_dist)
        diff16 = np.abs(hub_curve_1_dist) - np.abs(hub_curve_6_dist)
        diff25 = np.abs(hub_curve_2_dist) - np.abs(hub_curve_5_dist)
        diff34 = np.abs(hub_curve_3_dist) - np.abs(hub_curve_4_dist)

        test_zero_tol = 1e-15

        # Control point checks: make sure all control points from fan to trailing edge are symmetric
        for diff_ in [diff14, diff23, diff16, diff25, diff34]:
            self.assertTrue(np.all(np.isclose(diff_, 0.0, rtol=test_zero_tol)))
        pass
