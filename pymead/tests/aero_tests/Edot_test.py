import unittest

from pymead.analysis.calc_aero_data import calculate_CPK_power_consumption


class EdotTest(unittest.TestCase):
    def test_Edot_baseline(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_70\analysis\analysis_0_w50-50"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"Baseline: {Edot = }")

    def test_Edot_opt(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_70\analysis\analysis_500_w50-50"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"Opt: {Edot = }")

    def test_Edot_opt_oblique_shock_1(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_82\analysis\analysis_127_w0-100"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"Opt: {Edot = }")

    def test_Edot_opt_oblique_shock_2(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_82\analysis\analysis_127_w50-50"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"Opt: {Edot = }")

    def test_Edot_opt_oblique_shock_3(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_82\analysis\analysis_127_w60-40"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"Opt: {Edot = }")

    def test_Edot_opt_oblique_shock_4(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_82\analysis\analysis_127_w70-30"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"Opt: {Edot = }")
    #
    # def test_Edot_opt_oblique_shock_5(self):
    #     analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\ga_opt_82\analysis\analysis_127_w100-0"
    #     Edot = calculate_CPK_power_consumption(analysis_dir)
    #     print(f"Opt: {Edot = }")

    def test_Edot_opt_6(self):
        analysis_dir = r"C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_05_03_A\analysis_temp\ga_airfoil_7"
        Edot = calculate_CPK_power_consumption(analysis_dir)
        print(f"{Edot = }")
