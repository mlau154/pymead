import typing
from copy import deepcopy
from multiprocessing import Pool
import multiprocessing.connection

import numpy as np

from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead.core.geometry_collection import GeometryCollection
from pymead.utils.pymead_mp import kill_xfoil_mses_processes, pool_terminate_multi_tiered


class CustomGASettings:
    pass


class Chromosome:
    def __init__(self, geo_col_dict: dict, param_dict: dict, generation: int,
                 population_idx: int, mea_name: str = None, airfoil_name: str = None,
                 genes: list or None = None, verbose: bool = True, evaluate_geometric_constraints: bool = True):
        """
        Chromosome class constructor. Each Chromosome is the member of a particular Population.
        """
        # Keyword argument validation
        if mea_name is None and airfoil_name is None:
            raise ValueError("Must specify either mea_name (for MSES) or airfoil_name (for XFOIL) for the Chromosome")
        elif mea_name is not None and airfoil_name is not None:
            raise ValueError("Must specify only one of mea_name (for MSES) or airfoil_name (for XFOIL) for the "
                             "Chromosome")

        self.geo_col_dict = geo_col_dict
        self.geo_col = None
        self.mea_name = mea_name
        self.airfoil_name = airfoil_name
        self.mea = None
        self.airfoil = None
        self.airfoil_list = None
        self.mea_airfoil_names = None

        self.param_dict = deepcopy(param_dict)
        self.genes = deepcopy(genes)
        self.generation = generation
        self.population_idx = population_idx
        self.evaluate_geometric_constraints = evaluate_geometric_constraints

        # Might be able to remove a number of these attributes
        self.coords = None
        self.control_points = None
        self.airfoil_state = None
        self.verbose = verbose
        self.airfoil_sys_generated = False
        self.intersection_checked = False
        self.valid_geometry = False
        self.mutated = False
        self.fitness = None
        self.forces = None

    def generate(self):
        """
        Chromosome generation flow
        :return:
        """
        self.geo_col = GeometryCollection.set_from_dict_rep(deepcopy(self.geo_col_dict))
        self.mea = None if self.mea_name is None else self.geo_col.container()["mea"][self.mea_name]
        self.airfoil = None if self.airfoil_name is None else self.geo_col.container()["airfoils"][self.airfoil_name]
        self.airfoil_list = [self.airfoil] if self.airfoil is not None else self.mea.airfoils
        if self.mea is not None:
            self.mea_airfoil_names = [airfoil.name() for airfoil in self.mea.airfoils]
        if self.verbose:
            print(f'Generating chromosome idx = {self.population_idx}, gen = {self.generation}')
        self.generate_airfoil_sys_from_genes()

        if self.evaluate_geometric_constraints:
            self.chk_self_intersection()
            for airfoil in self.airfoil_list:
                airfoil_name = airfoil.name()
                if self.valid_geometry:
                    if self.param_dict["constraints"][airfoil_name]['min_radius_curvature'][1]:
                        self.chk_min_radius(airfoil_name=airfoil_name)
                if self.valid_geometry:
                    if self.param_dict["constraints"][airfoil_name]['min_val_of_max_thickness'][1]:
                        self.chk_max_thickness(airfoil_name=airfoil_name)
                if self.valid_geometry:
                    cnstr_spec = self.param_dict["constraints"][airfoil_name]["thickness_at_points"]
                    if cnstr_spec is not None and cnstr_spec[1]:
                        self.check_thickness_at_points(airfoil_name=airfoil_name)
                if self.valid_geometry:
                    if self.param_dict["constraints"][airfoil_name]["min_area"][1]:
                        self.check_min_area(airfoil_name=airfoil_name)
                if self.valid_geometry:
                    cnstr_spec = self.param_dict["constraints"][airfoil_name]["internal_geometry"]
                    if cnstr_spec is not None and cnstr_spec[1]:
                        if "Before" in cnstr_spec[2]:
                            self.check_contains_points(airfoil_name=airfoil_name)
                        else:
                            raise ValueError("Internal geometry timing after "
                                             "aerodynamic evaluation not yet implemented")

        # Set the equation_dict to None because it contains the unpicklable __builtins__ module
        for param in self.geo_col.container()["params"].values():
            param.equation_dict = None
        for desvar in self.geo_col.container()["desvar"].values():
            desvar.equation_dict = None

    def get_coords(self):
        if self.airfoil is not None:
            coords = self.airfoil.get_scaled_coords()
        else:
            coords, transformation_kwargs = self.mea.get_coords_list_chord_relative(
                max_airfoil_points=self.param_dict['mset_settings']["downsampling_max_pts"] if bool(
                    self.param_dict['mset_settings']["use_downsampling"]) else None,
                curvature_exp=self.param_dict['mset_settings']["downsampling_curve_exp"]
            )
        return coords

    def generate_airfoil_sys_from_genes(self) -> dict:
        """
        Converts Chromosome's gene list into a set of discrete airfoil system coordinates
        :return:
        """
        if self.genes is not None:
            self.geo_col.assign_design_variable_values(dv_values=self.genes, bounds_normalized=True)
        self.update_param_dict()  # updates the MSES settings from the geometry (just for XCDELH right now)
        self.coords = self.get_coords()
        self.airfoil_sys_generated = True
        return self.param_dict

    def update_param_dict(self):
        if self.param_dict["tool"] != "MSES":
            return

        # Update the x/c-location of the actuator disks from the geometry collection value if necessary
        if "XCDELH-Param" in self.param_dict["mses_settings"]:
            xcdelh_params = self.param_dict["mses_settings"]["XCDELH-Param"]
            for idx, xcdelh_param in enumerate(xcdelh_params):
                if xcdelh_param:
                    if xcdelh_param in self.geo_col.container()["params"]:
                        self.param_dict["mses_settings"]["XCDELH"][idx] = self.geo_col.container()["params"][xcdelh_param].value()
                    elif xcdelh_param in self.geo_col.container()["desvar"]:
                        self.param_dict["mses_settings"]["XCDELH"][idx] = self.geo_col.container()["desvar"][xcdelh_param].value()
                    else:
                        raise ValueError(f"Could not find XCDELH parameter {xcdelh_param}")

        # Update the fan pressure ratio from the updated design variable values if necessary
        if "PTRHIN-DesVar" in self.param_dict["mses_settings"]:
            for ad_idx, fpr_dv_list in enumerate(self.param_dict["mses_settings"]["PTRHIN-DesVar"]):
                for fpr_idx, fpr_dv in enumerate(fpr_dv_list):
                    self.param_dict["mses_settings"]["PTRHIN-DesVar"][ad_idx][fpr_idx] = self.geo_col.container()["desvar"][fpr_dv].value()

    def chk_self_intersection(self) -> bool:
        """
        Checks if airfoil geometry is self-intersecting
        :return: Boolean flag
        """
        try:
            for airfoil in self.airfoil_list:  # For each airfoil,
                self_intersecting = airfoil.check_self_intersection()
                if self_intersecting:  # If the intersection array is not empty (& thus there is a
                    # self-intersection somewhere),
                    self_intersection_flag = True  # Set self-intersection flag to True
                    self.valid_geometry = False
                    if self.verbose:
                        print('Failed self-intersection test')
                    return self_intersection_flag  # Return the self-intersection flag and break out of the method
            self_intersection_flag = False  # If the whole method runs with no self-intersections, set the
            # flag to False
            self.valid_geometry = True
            return self_intersection_flag  # Return the false self-intersection flag, meaning that this geometry is
            # good to go
        finally:
            self.intersection_checked = True

    def chk_min_radius(self, airfoil_name: str) -> bool:
        chord_relative = False
        cnstr_spec = self.param_dict["constraints"][airfoil_name]["min_radius_curvature"]
        if len(cnstr_spec) > 2:
            if cnstr_spec[2] == "Non-Dimensional":
                chord_relative = True
        min_radius = self.geo_col.container()["airfoils"][airfoil_name].compute_min_radius(
            chord_relative=chord_relative)
        min_radius_too_small = min_radius < cnstr_spec[0]
        self.valid_geometry = not min_radius_too_small
        if self.verbose:
            print(f'Min radius of curvature too small? {min_radius_too_small}. Min radius is {min_radius}')
        return min_radius_too_small

    def chk_max_thickness(self, airfoil_name: str) -> bool:
        airfoil_frame_relative = True
        cnstr_spec = self.param_dict["constraints"][airfoil_name]["min_val_of_max_thickness"]
        if len(cnstr_spec) > 2:
            if cnstr_spec[2] == "Dimensional":
                airfoil_frame_relative = False
        thickness_data = self.geo_col.container()["airfoils"][airfoil_name].compute_thickness(
            airfoil_frame_relative=airfoil_frame_relative)
        max_thickness = thickness_data["t_max"] if "t_max" in thickness_data else thickness_data["t/c_max"]
        if max_thickness < cnstr_spec[0]:
            max_thickness_too_small = True
        else:
            max_thickness_too_small = False
        if max_thickness_too_small:
            max_thickness_too_small_flag = True
            self.valid_geometry = False
            if self.verbose:
                print(f'Max thickness is {max_thickness}. '
                      f'Failed thickness test in chk_max_thickness, trying again...')
            for cnstr in self.geo_col.container()["geocon"].values():
                cnstr.data = None
            return max_thickness_too_small_flag
        else:
            max_thickness_too_small_flag = False
            self.valid_geometry = True
            if self.verbose:
                print(f'Max thickness is {max_thickness}. Passed thickness test. Continuing...')
            return max_thickness_too_small_flag

    def check_thickness_at_points(self, airfoil_name: str):
        airfoil_frame_relative = True
        vertical_check = False
        cnstr_spec = self.param_dict["constraints"][airfoil_name]["thickness_at_points"]
        if len(cnstr_spec) > 1:
            airfoil_frame_relative = "Non-Dimensional" in cnstr_spec[2]
            vertical_check = "Vertical" in cnstr_spec[2]

        thickness_array = np.array(cnstr_spec[0])
        x_array = thickness_array[:, 0]
        t_array = thickness_array[:, 1]
        airfoil = self.geo_col.container()["airfoils"][airfoil_name]
        thickness = airfoil.compute_thickness_at_points(
            x_array, airfoil_frame_relative=airfoil_frame_relative, vertical_check=vertical_check)
        if np.any(thickness < t_array):
            if self.verbose:
                print(f"Minimum required thickness condition not met at some point. Trying again")
            self.valid_geometry = False
        else:
            if self.verbose:
                print(f"Minimum required thickness condition met everywhere [success]")
            self.valid_geometry = True

    def check_min_area(self, airfoil_name: str):
        airfoil_frame_relative = True
        cnstr_spec = self.param_dict["constraints"][airfoil_name]["min_area"]
        if len(cnstr_spec) > 2:
            airfoil_frame_relative = cnstr_spec[2] == "Non-Dimensional"
        area = self.geo_col.container()["airfoils"][airfoil_name].compute_area(
            airfoil_frame_relative=airfoil_frame_relative)
        required_min_area = cnstr_spec[0]
        if area < required_min_area:
            if self.verbose:
                print(f'Area is {area} < required min. area ({required_min_area}). Trying again...')
            self.valid_geometry = False
        else:
            if self.verbose:
                print(f'Area is {area} >= minimum req. area ({required_min_area}) [success]. '
                      f'Continuing...')
            self.valid_geometry = True

    def check_contains_points(self, airfoil_name: str) -> bool:
        airfoil_frame_relative, rotate_with_airfoil, translate_with_airfoil, scale_with_airfoil = True, True, True, True
        cnstr_spec = self.param_dict["constraints"][airfoil_name]["internal_geometry"]
        if len(cnstr_spec) > 1:
            if "Rotate/Translate w/ Airfoil" in cnstr_spec[2]:
                scale_with_airfoil = False
                airfoil_frame_relative = False
            elif "Absolute" in cnstr_spec[2]:
                airfoil_frame_relative = False
                rotate_with_airfoil = False
                translate_with_airfoil = False
                scale_with_airfoil = False

        contains_points = self.geo_col.container()["airfoils"][airfoil_name].contains_line_string(
            points=cnstr_spec[0],
            airfoil_frame_relative=airfoil_frame_relative,
            rotate_with_airfoil=rotate_with_airfoil,
            translate_with_airfoil=translate_with_airfoil,
            scale_with_airfoil=scale_with_airfoil
        )
        if not contains_points:
            self.valid_geometry = False
            if self.verbose:
                print(f"Failed internal geometry test.")
            return self.valid_geometry
        self.valid_geometry = True
        if self.verbose:
            print(f"Passed internal geometry test. Continuing...")
        return self.valid_geometry


class Population:
    def __init__(self, param_dict: dict, generation: int,
                 parents: typing.List[Chromosome] or None, verbose: bool = True,
                 skip_parent_assignment: bool = False):
        self.param_dict = deepcopy(param_dict)
        self.generation = generation
        self.verbose = verbose
        self.population = []
        self.parent_indices = []
        self.converged_chromosomes = []
        if not skip_parent_assignment:
            for population_idx, chromosome in enumerate(parents):
                chromosome.population_idx = population_idx
                self.population.append(chromosome)
                # self.parents.append(chromosome)
                self.parent_indices.append(population_idx)
            self.num_parents = len(self.parent_indices)

    def eval_chromosome_fitness(self, chromosome: Chromosome):
        """
        Evaluates the fitness of a particular chromosome
        """
        # print(f"Generating {chromosome.population_idx}...")
        chromosome.generate()
        if chromosome.evaluate_geometric_constraints:
            if not chromosome.valid_geometry:
                print(f"Geometry invalid for chromosome {chromosome.population_idx} "
                      f"(either self-intersecting or fails to meet a constraint)")
                return chromosome
            else:
                print(f"Geometry {chromosome.population_idx} passed all the tests. Continuing on to evaluation...")
        else:
            print("Skipped evaluating geometric constraints. Continuing...")
        if chromosome.fitness is None:
            if self.verbose:
                print(f'Chromosome {chromosome.population_idx} '
                      f'(generation: {chromosome.generation}): Evaluating fitness...')
            xfoil_settings, mset_settings, mses_settings, mplot_settings = None, None, None, None
            if chromosome.param_dict['tool'] == 'XFOIL':
                tool = 'XFOIL'
                xfoil_settings = chromosome.param_dict['xfoil_settings']
                xfoil_settings["base_dir"] = chromosome.param_dict["base_folder"]
                xfoil_settings["airfoil_name"] = chromosome.param_dict["name"][chromosome.population_idx]
            elif chromosome.param_dict['tool'] == 'MSES':
                tool = 'MSES'
                mset_settings = chromosome.param_dict['mset_settings']
                mses_settings = chromosome.param_dict['mses_settings']
                mplot_settings = chromosome.param_dict['mplot_settings']
            else:
                raise ValueError('Only XFOIL and MSES are supported as tools in the optimization framework')

            # Multipoint Tags
            multipoint_tags = chromosome.param_dict["multipoint_tags"] \
                if "multipoint_tags" in chromosome.param_dict else None

            chromosome.forces, _ = calculate_aero_data(None,
                                                       chromosome.param_dict['base_folder'],
                                                       chromosome.param_dict['name'][chromosome.population_idx],
                                                       coords=chromosome.coords, tool=tool,
                                                       mea_airfoil_names=chromosome.mea_airfoil_names,
                                                       xfoil_settings=xfoil_settings,
                                                       mset_settings=mset_settings,
                                                       mses_settings=mses_settings,
                                                       mplot_settings=mplot_settings,
                                                       export_Cp=True,
                                                       multipoint_tags=multipoint_tags)

            if (xfoil_settings is not None and xfoil_settings["multi_point_stencil"] is None) or (
                    mses_settings is not None and mses_settings['multi_point_stencil'] is None):
                if chromosome.forces['converged'] and not chromosome.forces['errored_out'] \
                        and not chromosome.forces['timed_out']:
                    chromosome.fitness = 1  # Set to any value that is not False and not None
            else:
                if all(chromosome.forces['converged']) and not any(chromosome.forces['errored_out']) and not any(chromosome.forces['timed_out']):
                    chromosome.fitness = 1  # Set to any value that is not False and not None

            if self.verbose and chromosome.fitness is not None:
                if "CPK" in chromosome.forces.keys():
                    print(f"Fitness evaluated successfully for chromosome {chromosome.population_idx} with "
                          f"generation: {chromosome.generation} | CPK = {chromosome.forces['CPK']} | C_d = {chromosome.forces['Cd']} | C_l = "
                          f"{chromosome.forces['Cl']} | C_m = {chromosome.forces['Cm']}")
                else:
                    print(f"Fitness evaluated successfully for chromosome {chromosome.population_idx} with "
                          f"generation: {chromosome.generation} | C_d = {chromosome.forces['Cd']} | C_l = "
                          f"{chromosome.forces['Cl']} | C_m = {chromosome.forces['Cm']}")
        return chromosome

    @staticmethod
    def generate_chromosome(chromosome: Chromosome):
        chromosome.generate()
        return chromosome

    def generate_chromosomes_parallel(self):
        print("Generating chromosomes in parallel...")
        with Pool(processes=self.param_dict['num_processors']) as pool:
            result = pool.map(self.generate_chromosome, self.population)
        for chromosome in result:
            self.population = [chromosome if c.population_idx == chromosome.population_idx
                               else c for c in self.population]

    def eval_pop_fitness(self, sig: multiprocessing.connection.Connection = None):
        """
        Evaluates the fitness of the population using parallel processing
        """

        def _end_pool(chr_pool: multiprocessing.Pool):
            pool_terminate_multi_tiered(chr_pool)
            kill_xfoil_mses_processes()

        n_eval = 0
        n_converged_chromosomes = 0
        pool = Pool(processes=self.param_dict['num_processors'])

        for chromosome in pool.imap_unordered(self.eval_chromosome_fitness, self.population):

            if chromosome.fitness is not None:
                if chromosome.evaluate_geometric_constraints:
                    assert chromosome.valid_geometry
                self.converged_chromosomes.append(chromosome)
                n_converged_chromosomes += 1

            n_eval += 1

            gen = 1 if self.generation == 0 else self.generation
            status_bar_message = f"Generation {gen} of {self.param_dict['n_max_gen']}: Converged " \
                                 f"{n_converged_chromosomes} of {self.param_dict['population_size']} chromosomes. " \
                                 f"Total evaluations: {n_eval}\n"

            if sig is not None:
                try:
                    sig.send(("message", status_bar_message))
                except BrokenPipeError:
                    _end_pool(pool)
                    break
            else:
                print(status_bar_message)

            if n_converged_chromosomes >= self.param_dict["population_size"]:
                _end_pool(pool)
                break

        if n_converged_chromosomes < self.param_dict["population_size"]:
            message_to_display = ("Ran out of chromosomes to analyze before population size was reached. "
                                  "Increase the number of offspring in the optimization settings to allow "
                                  "more chromosomes to be generated.")
            if sig is not None:
                try:
                    sig.send(("disp_message_box", message_to_display))
                    _end_pool(pool)
                except BrokenPipeError:
                    _end_pool(pool)
                    return n_eval
            else:
                print(message_to_display)
                _end_pool(pool)

        for chromosome in self.converged_chromosomes:
            # Re-write the population such that the order of the results from the multiprocessing.Pool does not
            # matter
            self.population = [chromosome if c.population_idx == chromosome.population_idx
                               else c for c in self.population]
        return n_eval

    def all_chromosomes_fitness_converged(self):
        for chromosome in self.population:
            if chromosome.fitness is None:
                return False
        return True
