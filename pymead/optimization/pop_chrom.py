import numpy as np
import os
import typing
import random
from multiprocessing import Pool
from copy import deepcopy
from pymead.core.mea import MEA
from pymead.analysis.calc_aero_data import calculate_aero_data


class CustomGASettings:
    def __init__(self,
                 population_size: int = os.cpu_count() - 1,
                 mutation_bounds: list or typing.Tuple[list] = ([-0.01, 0.01], [-0.05, 0.05], [-0.1, 0.1]),
                 mutation_methods: str or typing.Tuple[str] or None = ('random-reset', 'random-perturb'),
                 max_genes_to_mutate: int = 1,
                 mutation_probability: float = 0.05,
                 max_mutation_attempts_per_chromosome: int = 500,
                 ):

        if population_size >= 1:
            self.population_size = population_size
        else:
            raise ValueError('population_size must be a positive integer')

        if type(mutation_bounds) == list:
            self.mutation_bounds = (mutation_bounds,)
        else:
            self.mutation_bounds = mutation_bounds

        if mutation_methods is not None:
            if type(mutation_methods) == str:
                self.mutation_methods = (mutation_methods,)
            else:
                self.mutation_methods = mutation_methods
            mutation_method_list = ['random-reset', 'random-perturb']
            if any(mutation_method not in mutation_method_list for mutation_method in self.mutation_methods):
                raise ValueError(f'mutation_methods must be one of {mutation_method_list}')
        else:
            self.mutation_methods = mutation_methods

        if max_genes_to_mutate >= 0:
            self.max_genes_to_mutate = max_genes_to_mutate
        else:
            raise ValueError('max_genes_to_mutate must be a non-negative integer')

        if 0 <= mutation_probability <= 1:
            self.mutation_probability = mutation_probability
        else:
            raise ValueError('mutation_probability must be a valid probability (between 0 and 1, inclusive)')

        if max_mutation_attempts_per_chromosome >= 0:
            self.max_mutation_attempts_per_chromosome = max_mutation_attempts_per_chromosome
        else:
            raise ValueError('max_mutation_attempts_per_chromosome must be greater than or equal to 0')


class Chromosome:
    def __init__(self, param_dict: dict, ga_settings: CustomGASettings or None, category: str or None, generation: int,
                 population_idx: int, mea, genes: list = None, verbose: bool = True):
        """
        Chromosome class constructor. Each Chromosome is the member of a particular Population.
        """
        self.genes = genes
        self.category = category
        self.generation = generation
        self.population_idx = population_idx
        self.mea = deepcopy(mea)
        self.param_set = deepcopy(param_dict)
        self.ga_settings = ga_settings
        self.verbose = verbose
        self.airfoil_sys_generated = False
        self.intersection_checked = False
        self.valid_geometry = False
        self.mutated = False
        if self.ga_settings is not None:
            self.mutation_bounds = random.sample(self.ga_settings.mutation_bounds, 1)[0]
        else:
            self.mutation_bounds = None
        self.fitness = None
        self.forces = None

    def generate(self):
        """
        Chromosome generation flow
        :return:
        """
        if self.verbose:
            print(f'Generating chromosome idx = {self.population_idx}, gen = {self.generation}, cat = {self.category}, '
                  f'mutation_bounds = {self.mutation_bounds}')
        self.generate_airfoil_sys_from_genes()
        self.chk_self_intersection()
        if self.valid_geometry:
            if self.param_set['min_thickness_active']:
                self.chk_max_thickness()
        if self.valid_geometry:
            if self.param_set['min_area_active']:
                self.check_min_area()
        if self.valid_geometry:
            if self.param_set['internal_point_matrix'] is not None:
                if self.param_set['int_geometry_timing'] == 'Before Aerodynamic Evaluation':
                    self.check_contains_points()
        if self.valid_geometry:
            if self.param_set['external_point_matrix'] is not None:
                if self.param_set['ext_geometry_timing'] == 'Before Aerodynamic Evaluation':
                    self.check_if_inside_points()

    def generate_airfoil_sys_from_genes(self) -> dict:
        """
        Converts Chromosome's gene list into a set of discrete airfoil system coordinates
        :return:
        """
        if self.genes is not None:
            # print(f"genes = {self.genes}")
            self.mea.update_parameters(self.genes)
            # print(f"alf = {self.mea.airfoils['A0'].alf.value}")
            self.airfoil_sys_generated = True
        else:
            raise Exception('Genes of Chromosome object have not been set. Aborting airfoil system generation.')
        return self.param_set

    def chk_self_intersection(self) -> bool:
        """
        Checks if airfoil geometry is self-intersecting
        :return: Boolean flag
        """
        try:
            if self.airfoil_sys_generated:
                for airfoil in self.mea.airfoils.values():  # For each airfoil,
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

    def chk_max_thickness(self) -> bool:
        if self.airfoil_sys_generated:
            _, _, max_thickness = self.mea.airfoils['A0'].compute_thickness()
            if max_thickness < self.param_set['min_val_of_max_thickness']:
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

    def check_min_area(self):
        if self.airfoil_sys_generated:
            area = self.mea.airfoils['A0'].compute_area()
            if area < self.param_set['min_area']:
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

    def check_contains_points(self) -> bool:
        if self.airfoil_sys_generated:
            if not self.mea.airfoils['A0'].contains_line_string(self.param_set['internal_point_matrix']):
                self.valid_geometry = False
                return self.valid_geometry
        self.valid_geometry = True
        return self.valid_geometry

    def check_if_inside_points(self) -> bool:
        if self.airfoil_sys_generated:
            if not self.mea.airfoils['A0'].within_line_string_until_point(self.param_set['external_point_matrix'],
                                                                          self.param_set['cutoff_point'],
                                                                          self.param_set['ext_transform_kwargs']):
                self.valid_geometry = False
                return self.valid_geometry
        self.valid_geometry = True
        return self.valid_geometry

    def mutate_robust(self):
        """
        Tries to mutate a particular Chromosome until geometry is valid or max_mutation_attempts reached
        :return:
        """
        self.mutated = True
        mutation_attempts = 0
        max_mutation_attempts = self.ga_settings.max_mutation_attempts_per_chromosome
        while True:
            if mutation_attempts < max_mutation_attempts:
                temp_saved_genes = deepcopy(self.genes)
                mutation_attempts += 1
                num_genes_to_mutate = np.random.randint(low=1, high=1 + self.ga_settings.max_genes_to_mutate)
                genes_to_mutate = random.sample([*range(0, len(self.genes))], num_genes_to_mutate)
                for gene_idx in genes_to_mutate:
                    mutation_method = random.sample(self.ga_settings.mutation_methods, 1)[0]
                    if mutation_method == 'random-reset':
                        if self.verbose:
                            print('Mutating random-reset...')
                        self.genes[gene_idx] = np.random.uniform(low=0, high=1)
                    elif mutation_method == 'random-perturb':
                        if self.verbose:
                            print(f'Mutating random-perturb with bounds {self.mutation_bounds}...')
                        self.genes[gene_idx] = self.genes[gene_idx] + np.random.uniform(
                            low=self.mutation_bounds[0],
                            high=self.mutation_bounds[1])
                        if self.genes[gene_idx] < 0:
                            self.genes[gene_idx] = 0
                        elif self.genes[gene_idx] > 1:
                            self.genes[gene_idx] = 1
                    else:
                        raise Exception('Invalid mutation_method designation in ga_settings. Aborting GA.')
                self.generate()
                if self.valid_geometry:
                    break
                else:
                    self.genes = deepcopy(temp_saved_genes)
            else:
                raise Exception(f'Exceeded maximum number of mutation attempts ({max_mutation_attempts}). Aborting GA.')


class Population:
    def __init__(self, param_dict: dict, ga_settings: CustomGASettings or None, generation: int,
                 parents: typing.List[Chromosome] or None, mea: MEA, verbose: bool = True,
                 skip_parent_assignment: bool = False):
        self.param_set = deepcopy(param_dict)
        self.mea = deepcopy(mea)
        self.ga_settings = ga_settings
        self.generation = generation
        self.verbose = verbose
        self.population = []
        self.parent_indices = []
        self.parents = deepcopy(parents)
        if not skip_parent_assignment:
            for population_idx, chromosome in enumerate(self.parents):
                if chromosome.category != 'parent':
                    chromosome.category = 'parent'
                chromosome.population_idx = population_idx
                self.population.append(chromosome)
                # self.parents.append(chromosome)
                self.parent_indices.append(population_idx)
            self.num_parents = len(self.parent_indices)

    def generate(self):
        self.random_perturb_best()
        self.mutation()

    def regenerate(self, attempt: int):
        """Regenerates the population after a Chromosome fitness evaluation fails"""
        if self.verbose:
            print(f"Performing random perturbation...")
        self.random_perturb_best()
        if self.verbose:
            print(f"Performing mutation...")
        self.mutation()

    def random_perturb_best(self):
        best_parent = self.parents[0]
        for population_idx in range(self.ga_settings.population_size):
            chromosome = Chromosome(self.param_set, self.ga_settings, category='random',
                                    population_idx=population_idx, generation=self.generation,
                                    mea=self.mea)
            chromosome = self.random_perturb(chromosome=chromosome, best_parent=best_parent)
            self.population.append(chromosome)
        self.population = self.population[1:]

    def random_perturb(self, chromosome: Chromosome, best_parent: Chromosome):
        random_perturb_attempts = 0
        while True:
            if random_perturb_attempts >= 10:
                chromosome.mutation_bounds = random.sample(self.ga_settings.mutation_bounds, 1)[0]
                random_perturb_attempts = 0
            random_perturb_attempts += 1
            chromosome.genes = deepcopy(best_parent.genes)
            for gene_idx, gene in enumerate(chromosome.genes):
                chromosome.genes[gene_idx] = chromosome.genes[gene_idx] + np.random.uniform(
                    low=chromosome.mutation_bounds[0],
                    high=chromosome.mutation_bounds[1])
                if chromosome.genes[gene_idx] < 0:
                    chromosome.genes[gene_idx] = 0
                elif chromosome.genes[gene_idx] > 1:
                    chromosome.genes[gene_idx] = 1
            chromosome.generate()
            if chromosome.valid_geometry:
                break
        return chromosome

    def mutation(self):
        """
        Mutates all non-parent chromosomes in population at a low probability with a biased coin flip
        """
        mutation_possible_list = [chromosome for chromosome in self.population if chromosome.category != 'parent'
                                  and chromosome.fitness is None]
        for chromosome in mutation_possible_list:  # For each chromosome that is allowed to mutate,
            # Toss a coin that is heavily weighted toward the outcome of no mutation:
            biased_coin_flip = np.random.binomial(n=1, p=self.ga_settings.mutation_probability)
            if biased_coin_flip:  # If the coin flip results in a mutation,
                chromosome.mutate_robust()  # Mutate the chromosome

    def eval_chromosome_fitness(self, chromosome: Chromosome):
        """
        Evaluates the fitness of a particular chromosome
        """
        if chromosome.fitness is None:
            if self.verbose:
                print(f'Chromosome {chromosome.population_idx + 1} of {self.ga_settings.population_size} '
                      f'(generation: {chromosome.generation}, category: {chromosome.category}, '
                      f'mutated: {chromosome.mutated}): Evaluating fitness...')
            xfoil_settings, mset_settings, mses_settings, mplot_settings = None, None, None, None
            # print(f"tool = {chromosome.param_set['tool']}")
            if chromosome.param_set['tool'] == 'xfoil':
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
                                                       mea=chromosome.mea,
                                                       mea_airfoil_name='A0', tool=tool,
                                                       xfoil_settings=xfoil_settings,
                                                       mset_settings=mset_settings,
                                                       mses_settings=mses_settings,
                                                       mplot_settings=mplot_settings,
                                                       export_Cp=True, body_fixed_csys=False)
            # print(f"forces now = {chromosome.forces}")
            if chromosome.forces['converged'] and not chromosome.forces['errored_out'] and not chromosome.forces['timed_out']:
                chromosome.fitness = 1  # Set to any value that is not False and not None
            if self.verbose:
                print(f"Fitness evaluated successfully for chromosome {chromosome.population_idx + 1} of "
                      f"{self.ga_settings.population_size} with "
                      f"generation: {chromosome.generation} | category: {chromosome.category} "
                      f"| mutated: {chromosome.mutated} | "
                      f"fitness = {round(chromosome.fitness, 8)} | C_d = {round(chromosome.forces['Cd'], 8)} | C_l = "
                      f"{round(chromosome.forces['Cl'], 8)} | C_m = {round(chromosome.forces['Cm'], 8)}")
        return chromosome

    @staticmethod
    def eval_chromosome_fitness_stencil(chromosome: Chromosome):
        """Multipoint optimization not yet implemented"""
        return chromosome

    def eval_pop_fitness(self):
        """
        Evaluates the fitness of the population using parallel processing
        """
        with Pool(processes=self.param_set['num_processors']) as pool:
            if self.param_set['multi_point']:
                result = pool.map(self.eval_chromosome_fitness_stencil, self.population)
            else:
                result = pool.map(self.eval_chromosome_fitness, self.population)
            if self.verbose:
                print(f'result = {result}')
        for chromosome in result:
            if chromosome.fitness is not None:
                self.population = [chromosome if c.population_idx == chromosome.population_idx
                                   else c for c in self.population]

    def all_chromosomes_fitness_converged(self):
        for chromosome in self.population:
            if chromosome.fitness is None:
                return False
        return True

    def xfoil_timeout_callback(self, chromosome: Chromosome):
        """
        Simple print statement for MSES timeout callback
        """
        if self.verbose:
            print(f"Fitness evaluation failed for chromosome {chromosome.population_idx + 1} of "
                  f"{self.ga_settings.population_size} with "
                  f"generation: {chromosome.generation} | category: {chromosome.category} | "
                  f"mutated: {chromosome.mutated}")
