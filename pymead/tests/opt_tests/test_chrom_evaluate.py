import os
import shutil
import warnings
from copy import deepcopy

from pymead.core.geometry_collection import GeometryCollection
from pymead.gui.dialogs import convert_opt_settings_to_param_dict
from pymead.optimization.pop_chrom import Chromosome, Population
from pymead.utils.read_write_files import load_data
from pymead import TEST_DIR, EXAMPLES_DIR


def test_isolated_propulsor_evaluate():

    if shutil.which("mset") is None:
        warnings.warn("MSES suite executable 'mset' not found on system path. Skipping this test.")
        return
    if shutil.which("mses") is None:
        warnings.warn("MSES suite executable 'mses' not found on system path. Skipping this test.")
        return
    if shutil.which("mplot") is None:
        warnings.warn("MPLOT suite executable 'mplot' not found on system path. Skipping this test.")
        return

    jmea_file = os.path.join(EXAMPLES_DIR, "isolated_propulsor.jmea")
    geo_col_dict = load_data(jmea_file)
    settings_file = os.path.join(TEST_DIR, "opt_tests", "iso_prop_test_fpr104.json")
    test_opt_dir = os.path.join(TEST_DIR, "opt_tests", "test_opt")
    if not os.path.exists(test_opt_dir):
        os.mkdir(test_opt_dir)
    settings_data = load_data(settings_file)
    settings_data["General Settings"]["mea_file"] = jmea_file
    settings_data["Genetic Algorithm"]["root_dir"] = test_opt_dir
    settings_data["Genetic Algorithm"]["num_processors"] = 1

    try:
        param_dict = convert_opt_settings_to_param_dict(settings_data, len(list(geo_col_dict["desvar"].keys())))

        chromosome = Chromosome(
            geo_col_dict=deepcopy(geo_col_dict),
            param_dict=param_dict,
            generation=0,
            population_idx=0,
            mea_name="MEA-1"
        )
        population = Population(
            param_dict=deepcopy(param_dict),
            generation=0,
            parents=[chromosome],
            verbose=True,
            skip_parent_assignment=False
        )
        population.eval_pop_fitness()

    finally:
        opt_dir = os.path.join(test_opt_dir, settings_data["Genetic Algorithm"]["opt_dir_name"] + "_0")
        if os.path.exists(opt_dir):
            shutil.rmtree(opt_dir)


def test_isolated_propulsor_opt_evaluate():

    if shutil.which("mset") is None:
        warnings.warn("MSES suite executable 'mset' not found on system path. Skipping this test.")
        return
    if shutil.which("mses") is None:
        warnings.warn("MSES suite executable 'mses' not found on system path. Skipping this test.")
        return
    if shutil.which("mplot") is None:
        warnings.warn("MPLOT suite executable 'mplot' not found on system path. Skipping this test.")
        return

    jmea_file = os.path.join(EXAMPLES_DIR, "isolated_propulsor.jmea")
    geo_col_dict = load_data(jmea_file)
    geo_col = GeometryCollection.set_from_dict_rep(geo_col_dict)
    alg_file = os.path.join(TEST_DIR, "opt_tests", "algorithm_gen_25.pkl")
    alg = load_data(alg_file)
    X = alg.opt.get("X")[0].tolist()
    geo_col.assign_design_variable_values(X, bounds_normalized=True)
    geo_col_dict = geo_col.get_dict_rep()
    settings_file = os.path.join(TEST_DIR, "opt_tests", "iso_prop_test_fpr104.json")
    test_opt_dir = os.path.join(TEST_DIR, "opt_tests", "test_opt")
    if not os.path.exists(test_opt_dir):
        os.mkdir(test_opt_dir)
    settings_data = load_data(settings_file)
    settings_data["General Settings"]["mea_file"] = jmea_file
    settings_data["Genetic Algorithm"]["root_dir"] = test_opt_dir
    settings_data["Genetic Algorithm"]["num_processors"] = 1

    try:
        param_dict = convert_opt_settings_to_param_dict(settings_data, len(list(geo_col_dict["desvar"].keys())))

        chromosome = Chromosome(
            geo_col_dict=deepcopy(geo_col_dict),
            param_dict=param_dict,
            generation=0,
            population_idx=0,
            mea_name="MEA-1"
        )
        population = Population(
            param_dict=deepcopy(param_dict),
            generation=0,
            parents=[chromosome],
            verbose=True,
            skip_parent_assignment=False
        )
        population.eval_pop_fitness()

    finally:
        opt_dir = os.path.join(test_opt_dir, settings_data["Genetic Algorithm"]["opt_dir_name"] + "_0")
        if os.path.exists(opt_dir):
            shutil.rmtree(opt_dir)
