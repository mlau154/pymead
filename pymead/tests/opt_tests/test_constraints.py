import os
from copy import deepcopy

import numpy as np

from pymead.core.geometry_collection import GeometryCollection
from pymead.gui.dialogs import convert_opt_settings_to_param_dict

from pymead.optimization.pop_chrom import Chromosome, Population
from pymead import EXAMPLES_DIR, TEST_DIR
from pymead.utils.read_write_files import load_data


def test_thickness_check():
    geo_col_dict = load_data(os.path.join(EXAMPLES_DIR, "basic_airfoil_sharp_dv.jmea"))
    param_dict = convert_opt_settings_to_param_dict(load_data(os.path.join(
        TEST_DIR, "opt_tests", "test_airfoil_thickness_settings.json")))
    chromosome = Chromosome(
        geo_col_dict=deepcopy(geo_col_dict),
        param_dict=param_dict,
        generation=0,
        population_idx=0,
        airfoil_name="Airfoil-1"
    )
    chromosome.generate()

    # Geometry check should pass since the min val of max thickness is set to 0.11, and the evaluated max thickness
    # should be ~0.1114
    assert chromosome.valid_geometry

    # Change the minimum value of maximum thickness to 0.12, regenerate the chromosome, and make sure the thickness
    # test fails
    chromosome.geo_col_dict = deepcopy(geo_col_dict)
    chromosome.param_dict["constraints"]["Airfoil-1"]["min_val_of_max_thickness"][0] = 0.12
    chromosome.generate()
    assert not chromosome.valid_geometry


def test_thickness_at_points_check():
    geo_col_dict = load_data(os.path.join(TEST_DIR, "opt_tests", "thickness_at_points.jmea"))
    param_dict = convert_opt_settings_to_param_dict(load_data(os.path.join(
        TEST_DIR, "opt_tests", "test_airfoil_thickness_settings.json")))
    chromosome = Chromosome(
        geo_col_dict=deepcopy(geo_col_dict),
        param_dict=param_dict,
        generation=0,
        population_idx=0,
        airfoil_name="Airfoil-1"
    )

    thickness_points = 20
    thickness_data_to_pass = np.column_stack(
        (np.linspace(0.4, 0.6, thickness_points),
         0.04 * np.ones(thickness_points))
    ).tolist()
    thickness_data_to_fail = np.column_stack(
        (np.linspace(0.4, 0.6, thickness_points),
         0.10 * np.ones(thickness_points))
    ).tolist()

    chromosome.param_dict["constraints"]["Airfoil-1"]["check_thickness_at_points"] = True
    chromosome.param_dict["constraints"]["Airfoil-1"]["thickness_at_points"] = thickness_data_to_pass
    chromosome.generate()
    assert chromosome.valid_geometry

    chromosome.geo_col_dict = deepcopy(geo_col_dict)
    chromosome.param_dict["constraints"]["Airfoil-1"]["thickness_at_points"] = thickness_data_to_fail
    chromosome.generate()
    assert not chromosome.valid_geometry


def test_chromosome_eval_fitness_with_invalid_max_thickness():
    geo_col_dict = load_data(os.path.join(EXAMPLES_DIR, "basic_airfoil_sharp_dv.jmea"))
    param_dict = convert_opt_settings_to_param_dict(load_data(os.path.join(
        TEST_DIR, "opt_tests", "test_airfoil_thickness_settings.json")))
    chromosome = Chromosome(
        geo_col_dict=deepcopy(geo_col_dict),
        param_dict=param_dict,
        generation=0,
        population_idx=0,
        airfoil_name="Airfoil-1"
    )
    chromosome.param_dict["constraints"]["Airfoil-1"]["min_val_of_max_thickness"][0] = 0.12
    population = Population(param_dict=param_dict, generation=0, parents=[chromosome])
    chromosome = population.eval_chromosome_fitness(chromosome)
    assert not chromosome.valid_geometry
    assert chromosome.fitness is None


def test_eval_pop_fitness_with_invalid_max_thickness():
    geo_col_dict = load_data(os.path.join(EXAMPLES_DIR, "basic_airfoil_sharp_dv.jmea"))
    param_dict = convert_opt_settings_to_param_dict(load_data(os.path.join(
        TEST_DIR, "opt_tests", "test_airfoil_thickness_settings.json")))
    chromosome = Chromosome(
        geo_col_dict=deepcopy(geo_col_dict),
        param_dict=param_dict,
        generation=0,
        population_idx=0,
        airfoil_name="Airfoil-1"
    )
    chromosome.param_dict["constraints"]["Airfoil-1"]["min_val_of_max_thickness"][0] = 0.12
    chromosome.param_dict['num_processors'] = 1
    population = Population(param_dict=param_dict, generation=0, parents=[chromosome])
    population.eval_pop_fitness()
    assert len(population.converged_chromosomes) == 0


def test_chromosome_generate_with_invalid_max_thickness_scaled_airfoil():
    scale_factor = 50
    geo_col_dict = load_data(os.path.join(TEST_DIR, "opt_tests", "thickness_at_points.jmea"))
    geo_col = GeometryCollection.set_from_dict_rep(geo_col_dict)
    for point in geo_col.container()["points"].values():
        x, y = point.x().value(), point.y().value()
        point.request_move(x * scale_factor, y * scale_factor)
    geo_col_dict = geo_col.get_dict_rep()
    param_dict = convert_opt_settings_to_param_dict(load_data(os.path.join(
        TEST_DIR, "opt_tests", "test_airfoil_thickness_settings.json")))
    chromosome = Chromosome(
        geo_col_dict=deepcopy(geo_col_dict),
        param_dict=param_dict,
        generation=0,
        population_idx=0,
        airfoil_name="Airfoil-1"
    )
    chromosome.param_dict["constraints"]["Airfoil-1"]["min_val_of_max_thickness"][0] = 0.17
    chromosome.generate()
    assert not chromosome.valid_geometry
