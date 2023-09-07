import os
import time
import typing
from multiprocessing import Pool
from multiprocessing.connection import Connection
from copy import deepcopy

import numpy as np
from benedict import benedict

from pymead.core.mea import MEA
from pymead.analysis.calc_aero_data import calculate_aero_data


class CustomGASettings:
    pass
# class CustomGASettings:
#     def __init__(self,
#                  population_size: int = os.cpu_count() - 1,
#                  mutation_bounds: list or typing.Tuple[list] = ([-0.01, 0.01], [-0.05, 0.05], [-0.1, 0.1]),
#                  mutation_methods: str or typing.Tuple[str] or None = ('random-reset', 'random-perturb'),
#                  max_genes_to_mutate: int = 1,
#                  mutation_probability: float = 0.05,
#                  max_mutation_attempts_per_chromosome: int = 500,
#                  ):
#
#         if population_size >= 1:
#             self.population_size = population_size
#         else:
#             raise ValueError('population_size must be a positive integer')
#
#         if type(mutation_bounds) == list:
#             self.mutation_bounds = (mutation_bounds,)
#         else:
#             self.mutation_bounds = mutation_bounds
#
#         if mutation_methods is not None:
#             if type(mutation_methods) == str:
#                 self.mutation_methods = (mutation_methods,)
#             else:
#                 self.mutation_methods = mutation_methods
#             mutation_method_list = ['random-reset', 'random-perturb']
#             if any(mutation_method not in mutation_method_list for mutation_method in self.mutation_methods):
#                 raise ValueError(f'mutation_methods must be one of {mutation_method_list}')
#         else:
#             self.mutation_methods = mutation_methods
#
#         if max_genes_to_mutate >= 0:
#             self.max_genes_to_mutate = max_genes_to_mutate
#         else:
#             raise ValueError('max_genes_to_mutate must be a non-negative integer')
#
#         if 0 <= mutation_probability <= 1:
#             self.mutation_probability = mutation_probability
#         else:
#             raise ValueError('mutation_probability must be a valid probability (between 0 and 1, inclusive)')
#
#         if max_mutation_attempts_per_chromosome >= 0:
#             self.max_mutation_attempts_per_chromosome = max_mutation_attempts_per_chromosome
#         else:
#             raise ValueError('max_mutation_attempts_per_chromosome must be greater than or equal to 0')


class Chromosome:
    def __init__(self, param_dict: dict, generation: int,
                 population_idx: int, mea: dict, genes: list = None, verbose: bool = True):
        """
        Chromosome class constructor. Each Chromosome is the member of a particular Population.
        """
        self.genes = deepcopy(genes)
        self.generation = generation
        self.population_idx = population_idx
        self.mea = deepcopy(mea)
        self.mea_object = None
        self.param_set = deepcopy(param_dict)
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
        print(f"Generating {self.population_idx} with {os.getpid() = } from param_dict...")
        self.mea_object = MEA.generate_from_param_dict(self.mea)
        if self.verbose:
            print(f'Generating chromosome idx = {self.population_idx}, gen = {self.generation}')
        self.generate_airfoil_sys_from_genes()
        self.chk_self_intersection()
        for airfoil_name in self.mea_object.airfoils.keys():
            if self.valid_geometry:
                if self.param_set['constraints'][airfoil_name]['min_radius_curvature'][1]:
                    self.chk_min_radius(airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_set['constraints'][airfoil_name]['min_val_of_max_thickness'][1]:
                    self.chk_max_thickness(airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_set['constraints'][airfoil_name]['thickness_at_points'] is not None:
                    self.check_thickness_at_points(airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_set['constraints'][airfoil_name]['min_area'][1]:
                    self.check_min_area(airfoil_name=airfoil_name)
            if self.valid_geometry:
                if self.param_set['constraints'][airfoil_name]['internal_geometry'] is not None:
                    if self.param_set['constraints'][airfoil_name]['internal_geometry_timing'] == 'Before Aerodynamic Evaluation':
                        self.check_contains_points(airfoil_name=airfoil_name)
                    else:
                        raise ValueError('Internal geometry timing after aerodynamic evaluation not yet implemented')
            if self.valid_geometry:
                if self.param_set['constraints'][airfoil_name]['external_geometry'] is not None:
                    if self.param_set['constraints'][airfoil_name]['external_geometry_timing'] == 'Before Aerodynamic Evaluation':
                        self.check_if_inside_points(airfoil_name=airfoil_name)
                    else:
                        raise ValueError('External geometry timing after aerodynamic evaluation not yet implemented')
        self.coords = tuple([self.mea_object.airfoils[k].get_coords(
            body_fixed_csys=False, as_tuple=True) for k in self.param_set['mset_settings']['airfoil_order']])
        self.control_points = [[c.P.tolist() for c in self.mea_object.airfoils[k].curve_list]
                               for k in self.mea_object.airfoils.keys()]
        self.airfoil_state = {k: {p: getattr(a, p).value for p in ['c', 'alf', 'dx', 'dy']} for k, a in self.mea_object.airfoils.items()}
        self.mea_object = None

    def generate_airfoil_sys_from_genes(self) -> dict:
        """
        Converts Chromosome's gene list into a set of discrete airfoil system coordinates
        :return:
        """
        if self.genes is not None:
            self.mea_object.update_parameters(self.genes)
            self.update_param_dict()  # updates the MSES settings from the geometry (just for XCDELH right now)
        else:
            self.coords = tuple([self.mea_object.airfoils[k].get_coords(
                body_fixed_csys=False, as_tuple=True) for k in self.param_set['mset_settings']['airfoil_order']])
        self.airfoil_sys_generated = True
        return self.param_set

    def update_param_dict(self):
        if self.param_set["tool"] != "MSES":
            return
        dben = benedict(self.mea_object.param_dict)
        for idx, from_geometry in enumerate(self.param_set['mses_settings']['from_geometry']):
            for k, v in from_geometry.items():
                self.param_set['mses_settings'][k][idx] = dben[v.replace('$', '')].value

    def chk_self_intersection(self) -> bool:
        """
        Checks if airfoil geometry is self-intersecting
        :return: Boolean flag
        """
        try:
            if self.airfoil_sys_generated:
                for airfoil in self.mea_object.airfoils.values():  # For each airfoil,
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
            else:
                raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')
        finally:
            self.intersection_checked = True

    def chk_min_radius(self, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            min_radius = self.mea_object.airfoils[airfoil_name].compute_min_radius()
            min_radius_too_small = min_radius < self.param_set['constraints'][airfoil_name]['min_radius_curvature'][0]
            self.valid_geometry = not min_radius_too_small
            # if self.verbose:
            #     print(f'Min radius of curvature too small? {min_radius_too_small}. Min radius is {min_radius}')
            return min_radius_too_small

    def chk_max_thickness(self, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            _, _, max_thickness = self.mea_object.airfoils[airfoil_name].compute_thickness()
            if max_thickness < self.param_set['constraints'][airfoil_name]['min_val_of_max_thickness'][0]:
                max_thickness_too_small = True
            else:
                max_thickness_too_small = False
            if max_thickness_too_small:
                max_thickness_too_small_flag = True
                self.valid_geometry = False
                if self.verbose:
                    print(f'Max thickness is {max_thickness}. '
                          f'Failed thickness test in chk_max_thickness, trying again...')
                return max_thickness_too_small_flag
            else:
                max_thickness_too_small_flag = False
                self.valid_geometry = True
                if self.verbose:
                    print(f'Max thickness is {max_thickness}. Passed thickness test. Continuing...')
                return max_thickness_too_small_flag
        else:
            raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')

    def check_thickness_at_points(self, airfoil_name: str):
        if self.airfoil_sys_generated:
            thickness_array = np.array(self.param_set['constraints'][airfoil_name]['thickness_at_points'])
            x_over_c_array = thickness_array[:, 0]
            t_over_c_array = thickness_array[:, 1]
            thickness = self.mea_object.airfoils[airfoil_name].compute_thickness_at_points(x_over_c_array)
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

    def check_min_area(self, airfoil_name: str):
        if self.airfoil_sys_generated:
            area = self.mea_object.airfoils[airfoil_name].compute_area()
            if area < self.param_set['constraints'][airfoil_name]['min_area'][0]:
                if self.verbose:
                    print(f'Area is {area} < required min. area ({self.param_set["min_area"]}). Trying again...')
                self.valid_geometry = False
            else:
                if self.verbose:
                    print(f'Area is {area} >= minimum req. area ({self.param_set["min_area"]}) [success]. '
                          f'Continuing...')
                self.valid_geometry = True
        else:
            raise Exception('Airfoil system has not yet been generated. Aborting self-intersection check.')

    def check_contains_points(self, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            if not self.mea_object.airfoils[airfoil_name].contains_line_string(self.param_set['constraints'][airfoil_name]['internal_geometry']):
                self.valid_geometry = False
                return self.valid_geometry
        self.valid_geometry = True
        return self.valid_geometry

    def check_if_inside_points(self, airfoil_name: str) -> bool:
        if self.airfoil_sys_generated:
            if not self.mea_object.airfoils[airfoil_name].within_line_string_until_point(self.param_set['constraints'][airfoil_name]['external_geometry'],
                                                                          self.param_set['cutoff_point'],
                                                                          self.param_set['ext_transform_kwargs']):
                self.valid_geometry = False
                return self.valid_geometry
        self.valid_geometry = True
        return self.valid_geometry

    # def mutate_robust(self):
    #     """
    #     Tries to mutate a particular Chromosome until geometry is valid or max_mutation_attempts reached
    #     :return:
    #     """
    #     self.mutated = True
    #     mutation_attempts = 0
    #     max_mutation_attempts = self.ga_settings.max_mutation_attempts_per_chromosome
    #     while True:
    #         if mutation_attempts < max_mutation_attempts:
    #             temp_saved_genes = deepcopy(self.genes)
    #             mutation_attempts += 1
    #             num_genes_to_mutate = np.random.randint(low=1, high=1 + self.ga_settings.max_genes_to_mutate)
    #             genes_to_mutate = random.sample([*range(0, len(self.genes))], num_genes_to_mutate)
    #             for gene_idx in genes_to_mutate:
    #                 mutation_method = random.sample(self.ga_settings.mutation_methods, 1)[0]
    #                 if mutation_method == 'random-reset':
    #                     if self.verbose:
    #                         print('Mutating random-reset...')
    #                     self.genes[gene_idx] = np.random.uniform(low=0, high=1)
    #                 elif mutation_method == 'random-perturb':
    #                     if self.verbose:
    #                         print(f'Mutating random-perturb with bounds {self.mutation_bounds}...')
    #                     self.genes[gene_idx] = self.genes[gene_idx] + np.random.uniform(
    #                         low=self.mutation_bounds[0],
    #                         high=self.mutation_bounds[1])
    #                     if self.genes[gene_idx] < 0:
    #                         self.genes[gene_idx] = 0
    #                     elif self.genes[gene_idx] > 1:
    #                         self.genes[gene_idx] = 1
    #                 else:
    #                     raise Exception('Invalid mutation_method designation in ga_settings. Aborting GA.')
    #             self.generate()
    #             if self.valid_geometry:
    #                 break
    #             else:
    #                 self.genes = deepcopy(temp_saved_genes)
    #         else:
    #             raise Exception(f'Exceeded maximum number of mutation attempts ({max_mutation_attempts}). Aborting GA.')


class Population:
    def __init__(self, param_dict: dict, generation: int,
                 parents: typing.List[Chromosome] or None, mea: dict, verbose: bool = True,
                 skip_parent_assignment: bool = False):
        self.param_set = deepcopy(param_dict)
        self.mea = deepcopy(mea)
        self.generation = generation
        self.verbose = verbose
        self.population = []
        self.parent_indices = []
        self.converged_chromosomes = []
        self.parents = deepcopy(parents)
        if not skip_parent_assignment:
            for population_idx, chromosome in enumerate(self.parents):
                chromosome.population_idx = population_idx
                self.population.append(chromosome)
                # self.parents.append(chromosome)
                self.parent_indices.append(population_idx)
            self.num_parents = len(self.parent_indices)

    # def generate(self):
    #     self.random_perturb_best()
    #     self.mutation()
    #
    # def regenerate(self, attempt: int):
    #     """Regenerates the population after a Chromosome fitness evaluation fails"""
    #     if self.verbose:
    #         print(f"Performing random perturbation...")
    #     self.random_perturb_best()
    #     if self.verbose:
    #         print(f"Performing mutation...")
    #     self.mutation()
    #
    # def random_perturb_best(self):
    #     best_parent = self.parents[0]
    #     for population_idx in range(self.ga_settings.population_size):
    #         chromosome = Chromosome(self.param_set, population_idx=population_idx, generation=self.generation,
    #                                 mea=self.mea)
    #         chromosome = self.random_perturb(chromosome=chromosome, best_parent=best_parent)
    #         self.population.append(chromosome)
    #     self.population = self.population[1:]
    #
    # def random_perturb(self, chromosome: Chromosome, best_parent: Chromosome):
    #     random_perturb_attempts = 0
    #     while True:
    #         if random_perturb_attempts >= 10:
    #             chromosome.mutation_bounds = random.sample(self.ga_settings.mutation_bounds, 1)[0]
    #             random_perturb_attempts = 0
    #         random_perturb_attempts += 1
    #         chromosome.genes = deepcopy(best_parent.genes)
    #         for gene_idx, gene in enumerate(chromosome.genes):
    #             chromosome.genes[gene_idx] = chromosome.genes[gene_idx] + np.random.uniform(
    #                 low=chromosome.mutation_bounds[0],
    #                 high=chromosome.mutation_bounds[1])
    #             if chromosome.genes[gene_idx] < 0:
    #                 chromosome.genes[gene_idx] = 0
    #             elif chromosome.genes[gene_idx] > 1:
    #                 chromosome.genes[gene_idx] = 1
    #         chromosome.generate()
    #         if chromosome.valid_geometry:
    #             break
    #     return chromosome
    #
    # def mutation(self):
    #     """
    #     Mutates all non-parent chromosomes in population at a low probability with a biased coin flip
    #     """
    #     mutation_possible_list = [chromosome for chromosome in self.population if chromosome.category != 'parent'
    #                               and chromosome.fitness is None]
    #     for chromosome in mutation_possible_list:  # For each chromosome that is allowed to mutate,
    #         # Toss a coin that is heavily weighted toward the outcome of no mutation:
    #         biased_coin_flip = np.random.binomial(n=1, p=self.ga_settings.mutation_probability)
    #         if biased_coin_flip:  # If the coin flip results in a mutation,
    #             chromosome.mutate_robust()  # Mutate the chromosome

    def eval_chromosome_fitness(self, chromosome: Chromosome):
        """
        Evaluates the fitness of a particular chromosome
        """
        print(f"Generating {chromosome.population_idx}...")
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
            if chromosome.param_set['tool'] == 'XFOIL':
                tool = 'XFOIL'
                xfoil_settings = chromosome.param_set['xfoil_settings']
            elif chromosome.param_set['tool'] == 'MSES':
                tool = 'MSES'
                mset_settings = chromosome.param_set['mset_settings']
                mses_settings = chromosome.param_set['mses_settings']
                mplot_settings = chromosome.param_set['mplot_settings']
            else:
                raise ValueError('Only XFOIL and MSES are supported as tools in the optimization framework')

            chromosome.forces, _ = calculate_aero_data(chromosome.param_set['base_folder'],
                                                       chromosome.param_set['name'][chromosome.population_idx],
                                                       coords=chromosome.coords, tool=tool,
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
        with Pool(processes=self.param_set['num_processors']) as pool:
            result = pool.map(self.generate_chromosome, self.population)
        for chromosome in result:
            self.population = [chromosome if c.population_idx == chromosome.population_idx
                               else c for c in self.population]

    def eval_pop_fitness(self, sig: Connection = None):
        """
        Evaluates the fitness of the population using parallel processing
        """
        n_eval = 0
        # print("Evaluating chromosomes in parallel using multiprocessing.Pool.map_async()...")
        with Pool(processes=self.param_set['num_processors']) as pool:
            # pool_sig.emit(pool)
            # sig.send(("pool", pool))
            result = pool.imap_unordered(self.eval_chromosome_fitness, self.population)
            if self.verbose:
                print(f'result = {result}')
            # print(f"{pool = }")
            for chromosome in result:
                # TODO: need to be able to receive signal here to stop the Pool rather than just the parent Process
                # if sig is not None:
                #     status, data = sig.recv()
                #     print(f"{status = }")
                if chromosome.fitness is not None:
                    self.converged_chromosomes.append(chromosome)
                n_converged_chromosomes = len(self.converged_chromosomes)
                n_eval += 1
                print(f"{n_converged_chromosomes = }")

                gen = 1 if self.generation == 0 else self.generation
                status_bar_message = f"Generation {gen} of {self.param_set['n_max_gen']}: Converged " \
                                     f"{n_converged_chromosomes} of {self.param_set['population_size']} chromosomes. " \
                                     f"Total evaluations: {n_eval}\n"

                if sig is not None:
                    # sig.emit(status_bar_message)
                    sig.send(("message", status_bar_message))
                else:
                    print(status_bar_message)

                # if self.conn is not None:
                #     self.conn.send(status_bar_message)

                if n_converged_chromosomes >= self.param_set["population_size"]:
                    # print(f"Converged enough chromosomes (at least population_size = {self.param_set['population_size']}. "
                    #       f"Closing pool. {len(self.converged_chromosomes) = }")
                    break
                # else:
                #     print(f"Haven't yet converged enough chromosomes. {len(self.converged_chromosomes) = }.")
        for chromosome in self.converged_chromosomes:
            # print(f"{chromosome.population_idx = }")
            # print(f"Chromosome index {chromosome.population_idx} is converged. Setting it to the population")

            # Re-write the population such that the order of the results from the multiprocessing.Pool does not
            # matter
            self.population = [chromosome if c.population_idx == chromosome.population_idx
                               else c for c in self.population]
        # print("Done writing converged chromosomes to the population.")
        return n_eval

    def all_chromosomes_fitness_converged(self):
        for chromosome in self.population:
            if chromosome.fitness is None:
                return False
        return True

    # def xfoil_timeout_callback(self, chromosome: Chromosome):
    #     """
    #     Simple print statement for MSES timeout callback
    #     """
    #     if self.verbose:
    #         print(f"Fitness evaluation failed for chromosome {chromosome.population_idx + 1} of "
    #               f"{self.ga_settings.population_size} with "
    #               f"generation: {chromosome.generation} | category: {chromosome.category} | "
    #               f"mutated: {chromosome.mutated}")
