import typing
from copy import deepcopy
from multiprocessing import Pool, active_children
from multiprocessing.connection import Connection

import numpy as np

from pymead.analysis.calc_aero_data import calculate_aero_data
from pymead.core.airfoil import Airfoil
from pymead.core.geometry_collection import GeometryCollection
from pymead.utils.pymead_mp import kill_child_processes


class CustomGASettings:
    pass


class Chromosome:
    def __init__(self, geo_col_dict: dict, param_dict: dict, generation: int,
                 population_idx: int, mea_name: str = None, airfoil_name: str = None,
                 genes: list or None = None, verbose: bool = True):
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
        self.geo_col = GeometryCollection.set_from_dict_rep(self.geo_col_dict)
        self.mea = None if self.mea_name is None else self.geo_col.container()["mea"][self.mea_name]
        self.airfoil = None if self.airfoil_name is None else self.geo_col.container()["airfoils"][self.airfoil_name]
        self.airfoil_list = [self.airfoil] if self.airfoil is not None else self.mea.airfoils
        if self.mea is not None:
            self.mea_airfoil_names = [airfoil.name() for airfoil in self.mea.airfoils]
        # self.mea_object = MEA.generate_from_param_dict(self.mea)
        # if deactivate_airfoil_graphs:
        #     self.mea_object.remove_airfoil_graphs()
        if self.verbose:
            print(f'Generating chromosome idx = {self.population_idx}, gen = {self.generation}')
        self.generate_airfoil_sys_from_genes()
        line_strings = self.get_airfoil_line_strings(airfoil_frame_relative=False)
        relative_line_strings = self.get_airfoil_line_strings(airfoil_frame_relative=True)
        airfoil_polygons = self.get_airfoil_polygons(line_strings)
        self.chk_self_intersection(line_strings)
        for airfoil in self.airfoil_list:
            airfoil_name = airfoil.name()
            if self.valid_geometry:
                if self.param_dict['constraints'][airfoil_name]['min_radius_curvature'][1]:
                    self.chk_min_radius(airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_dict['constraints'][airfoil_name]['min_val_of_max_thickness'][1]:
                    self.chk_max_thickness(relative_line_strings[airfoil_name], airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_dict['constraints'][airfoil_name]['thickness_at_points'] is not None:
                    self.check_thickness_at_points(relative_line_strings[airfoil_name], airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_dict['constraints'][airfoil_name]['min_area'][1]:
                    self.check_min_area(airfoil_polygons[airfoil_name], airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_dict['constraints'][airfoil_name]['internal_geometry'] is not None:
                    if self.param_dict['constraints'][airfoil_name]['internal_geometry_timing'] == 'Before Aerodynamic Evaluation':
                        self.check_contains_points(airfoil_polygons[airfoil_name], airfoil_name=airfoil_name)
                    else:
                        raise ValueError('Internal geometry timing after aerodynamic evaluation not yet implemented')
            if self.valid_geometry:
                if self.param_dict['constraints'][airfoil_name]['external_geometry'] is not None:
                    if self.param_dict['constraints'][airfoil_name]['external_geometry_timing'] == 'Before Aerodynamic Evaluation':
                        self.check_if_inside_points(airfoil_name=airfoil_name)
                    else:
                        raise ValueError('External geometry timing after aerodynamic evaluation not yet implemented')

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
        xcdelh_params = self.param_dict["mses_settings"]["XCDELH-Param"]
        for idx, xcdelh_param in enumerate(xcdelh_params):
            if xcdelh_param:
                if xcdelh_param in self.geo_col.container()["params"]:
                    self.param_dict["mses_settings"]["XCDELH"][idx] = self.geo_col.container()["params"][xcdelh_param].value()
                elif xcdelh_param in self.geo_col.container()["desvar"]:
                    self.param_dict["mses_settings"]["XCDELH"][idx] = self.geo_col.container()["desvar"][xcdelh_param].value()
                else:
                    raise ValueError(f"Could not find XCDELH parameter {xcdelh_param}")

    def get_airfoil_line_strings(self, airfoil_frame_relative: bool):
        line_strings = {}
        for airfoil in self.airfoil_list:
            if airfoil_frame_relative:
                coords = airfoil.get_chord_relative_coords(airfoil.coords)
            else:
                coords = airfoil.coords
            line_strings[airfoil.name()] = Airfoil.create_line_string(Airfoil.convert_coords_to_shapely_format(coords))
        return line_strings

    @staticmethod
    def get_airfoil_polygons(line_strings: dict):
        return {k: Airfoil.create_shapely_polygon(v) for k, v in line_strings.items()}

    def chk_self_intersection(self, line_strings: dict) -> bool:
        """
        Checks if airfoil geometry is self-intersecting
        :return: Boolean flag
        """
        try:
            if self.airfoil_sys_generated:
                for airfoil in self.airfoil_list:  # For each airfoil,
                    self_intersecting = airfoil.check_self_intersection(line_strings[airfoil.name()])
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
            else:
                raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')
        finally:
            self.intersection_checked = True

    def chk_min_radius(self, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            min_radius = self.geo_col.container()["airfoils"][airfoil_name].compute_min_radius()
            min_radius_too_small = min_radius < self.param_dict['constraints'][airfoil_name]['min_radius_curvature'][0]
            self.valid_geometry = not min_radius_too_small
            if self.verbose:
                print(f'Min radius of curvature too small? {min_radius_too_small}. Min radius is {min_radius}')
            return min_radius_too_small

    def chk_max_thickness(self, line_string, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            thickness_data = self.geo_col.container()["airfoils"][airfoil_name].compute_thickness(
                line_string)
            max_thickness = thickness_data["t/c_max"]
            if max_thickness < self.param_dict['constraints'][airfoil_name]['min_val_of_max_thickness'][0]:
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
        else:
            raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')

    def check_thickness_at_points(self, line_string, airfoil_name: str):
        if self.airfoil_sys_generated:
            thickness_array = np.array(self.param_dict['constraints'][airfoil_name]['thickness_at_points'])
            x_over_c_array = thickness_array[:, 0]
            t_over_c_array = thickness_array[:, 1]
            airfoil = self.geo_col.container()["airfoils"][airfoil_name]
            thickness = airfoil.compute_thickness_at_points(x_over_c_array, line_string)
            if np.any(thickness < t_over_c_array):
                if self.verbose:
                    print(f"Minimum required thickness condition not met at some point. Trying again")
                self.valid_geometry = False
            else:
                if self.verbose:
                    print(f"Minimum required thickness condition met everywhere [success]")
                self.valid_geometry = True
        else:
            raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')

    def check_min_area(self, airfoil_polygon, airfoil_name: str):
        if self.airfoil_sys_generated:
            area = self.geo_col.container()["airfoils"][airfoil_name].compute_area(airfoil_polygon)
            required_min_area = self.param_dict['constraints'][airfoil_name]['min_area'][0]
            if area < required_min_area:
                if self.verbose:
                    print(f'Area is {area} < required min. area ({required_min_area}). Trying again...')
                self.valid_geometry = False
            else:
                if self.verbose:
                    print(f'Area is {area} >= minimum req. area ({required_min_area}) [success]. '
                          f'Continuing...')
                self.valid_geometry = True
        else:
            raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')

    def check_contains_points(self, airfoil_polygon, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            if not self.geo_col.container()["airfoils"][airfoil_name].contains_line_string(
                    airfoil_polygon, self.param_dict['constraints'][airfoil_name]['internal_geometry']):
                self.valid_geometry = False
                return self.valid_geometry
        self.valid_geometry = True
        return self.valid_geometry

    def check_if_inside_points(self, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            if not self.geo_col.container()["airfoils"][airfoil_name].within_line_string_until_point(
                    self.param_dict['constraints'][airfoil_name]['external_geometry'],
                    self.param_dict['cutoff_point'],
                    self.param_dict['ext_transform_kwargs']
            ):
                self.valid_geometry = False
                return self.valid_geometry
        self.valid_geometry = True
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
        if not chromosome.valid_geometry:
            print(f"Geometry invalid for chromosome {chromosome.population_idx} "
                  f"(either self-intersecting or fails to meet a constraint)")
            return chromosome
        else:
            print(f"Geometry {chromosome.population_idx} passed all the tests. Continuing on to evaluation...")
        if chromosome.fitness is None:
            if self.verbose:
                print(f'Chromosome {chromosome.population_idx + 1} '
                      f'(generation: {chromosome.generation}): Evaluating fitness...')
            xfoil_settings, mset_settings, mses_settings, mplot_settings = None, None, None, None
            if chromosome.param_dict['tool'] == 'XFOIL':
                tool = 'XFOIL'
                xfoil_settings = chromosome.param_dict['xfoil_settings']
            elif chromosome.param_dict['tool'] == 'MSES':
                tool = 'MSES'
                mset_settings = chromosome.param_dict['mset_settings']
                mses_settings = chromosome.param_dict['mses_settings']
                mplot_settings = chromosome.param_dict['mplot_settings']
            else:
                raise ValueError('Only XFOIL and MSES are supported as tools in the optimization framework')

            chromosome.forces, _ = calculate_aero_data(chromosome.param_dict['base_folder'],
                                                       chromosome.param_dict['name'][chromosome.population_idx],
                                                       coords=chromosome.coords, tool=tool,
                                                       mea_airfoil_names=chromosome.mea_airfoil_names,
                                                       xfoil_settings=xfoil_settings,
                                                       mset_settings=mset_settings,
                                                       mses_settings=mses_settings,
                                                       mplot_settings=mplot_settings,
                                                       export_Cp=True)

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
                    print(f"Fitness evaluated successfully for chromosome {chromosome.population_idx + 1} with "
                          f"generation: {chromosome.generation} | CPK = {chromosome.forces['CPK']} | C_d = {chromosome.forces['Cd']} | C_l = "
                          f"{chromosome.forces['Cl']} | C_m = {chromosome.forces['Cm']}")
                else:
                    print(f"Fitness evaluated successfully for chromosome {chromosome.population_idx + 1} with "
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

    def eval_pop_fitness(self, sig: Connection = None):
        """
        Evaluates the fitness of the population using parallel processing
        """
        n_eval = 0
        with Pool(processes=self.param_dict['num_processors']) as pool:
            result = pool.imap_unordered(self.eval_chromosome_fitness, self.population)
            # if self.verbose:
            #     print(f'result = {result}')
            for chromosome in result:

                if chromosome.fitness is not None:
                    self.converged_chromosomes.append(chromosome)
                n_converged_chromosomes = len(self.converged_chromosomes)
                n_eval += 1

                gen = 1 if self.generation == 0 else self.generation
                status_bar_message = f"Generation {gen} of {self.param_dict['n_max_gen']}: Converged " \
                                     f"{n_converged_chromosomes} of {self.param_dict['population_size']} chromosomes. " \
                                     f"Total evaluations: {n_eval}\n"

                if sig is not None:
                    try:
                        sig.send(("message", status_bar_message))
                    except BrokenPipeError:
                        pool.terminate()
                else:
                    print(status_bar_message)

                if n_converged_chromosomes >= self.param_dict["population_size"]:
                    for child in active_children():
                        kill_child_processes(child.pid)
                    pool.terminate()
                    pool.join()
                    break

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
