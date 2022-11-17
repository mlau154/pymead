import numpy as np
import pymoo.core.population
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.config import Config
from pymoo.core.evaluator import Evaluator
from pymoo.factory import get_reference_directions
from pymoo.core.evaluator import set_cv
from pymead.optimization.opt_setup import CustomDisplay, TPAIOPT, SelfIntersectionRepair
from pymead.utils.read_write_files import load_data, save_data
from pymead.utils.misc import make_ga_opt_dir
from pymead.optimization.pop_chrom import Chromosome, Population, CustomGASettings
from pymead.optimization.custom_ga_sampling import CustomGASampling
from pymead.optimization.opt_setup import termination_condition, calculate_warm_start_index, \
    convert_opt_settings_to_param_dict
from pymead.gui.message_box import disp_message_box
import os
from copy import deepcopy


def run(opt_settings: dict):
    Config.show_compile_hint = False

    param_dict = convert_opt_settings_to_param_dict(opt_settings)

    # population_size = 20
    # n_offsprings = 150
    par_name = 'sec_6'

    if opt_settings['Warm Start/Batch Mode']['use_current_mea']['state']:
        mea = self.copy_mea()
    mea = load_data(os.path.join(os.getcwd(), 'parametrizations', f"{par_name}.mead"))

    if opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
        opt_dir = opt_settings['Warm Start/Batch Mode']['warm_start_dir']['text']
    else:
        opt_dir = make_ga_opt_dir(opt_settings['Genetic Algorithm']['root_dir']['text'])
    name_base = 'ga_airfoil'
    name = [f"{name_base}_{i}" for i in range(opt_settings['Genetic Algorithm']['n_offspring']['value'])]

    for airfoil in mea.airfoils.values():
        airfoil.airfoil_graphs_active = False
    mea.airfoil_graphs_active = False

    parameter_list = mea.extract_parameters()
    base_folder = os.path.join(os.getcwd(), 'Analyses', 'Optimization', 'ga_test1')

    if opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
        param_dict['warm_start_generation'] = calculate_warm_start_index(
            opt_settings['Warm Start/Batch Mode']['warm_start_generation']['value'], opt_dir)
    param_dict_save = deepcopy(param_dict)
    if not opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
        save_data(param_dict_save, os.path.join(opt_dir, 'param_dict.json'))
    else:
        save_data(param_dict_save, os.path.join(
            opt_dir, f'param_dict_{param_dict["warm_start_generation"]}.json'))

    ref_dirs = get_reference_directions("energy", param_dict['n_obj'], param_dict['n_ref_dirs'],
                                        seed=param_dict['seed'])
    ga_settings = CustomGASettings(population_size=param_dict['n_offsprings'],
                                   mutation_bounds=([-0.002, 0.002]),
                                   mutation_methods=('random-reset', 'random-perturb'),
                                   max_genes_to_mutate=2,
                                   mutation_probability=0.06,
                                   max_mutation_attempts_per_chromosome=500)

    problem = TPAIOPT(n_var=param_dict['n_var'], n_obj=param_dict['n_obj'], n_constr=param_dict['n_constr'],
                      xl=param_dict['xl'], xu=param_dict['xu'], param_dict=param_dict,
                      population_size=param_dict['population_size'], target_CL=param_dict['target_CL'],
                      ga_settings=ga_settings)

    if not opt_settings['Warm Start/Batch Mode']['warm_start_active']['state']:
        tpaiga2_alg_instance = CustomGASampling(param_dict=problem.param_dict, ga_settings=ga_settings, mea=mea)
        population = Population(problem.param_dict, ga_settings, generation=0,
                                parents=[tpaiga2_alg_instance.generate_first_parent()],
                                verbose=param_dict['verbose'], mea=mea)
        population.generate()

        n_subpopulations = 0
        fully_converged_chromosomes = []
        while True:  # "Do while" loop (terminate when enough of chromosomes have fully converged solutions)
            subpopulation = deepcopy(population)
            subpopulation.population = subpopulation.population[param_dict['num_processors'] * n_subpopulations:
                                                                param_dict['num_processors'] * (
                                                                        n_subpopulations + 1)]

            subpopulation.eval_pop_fitness()

            for chromosome in subpopulation.population:
                if chromosome.fitness is not None:
                    fully_converged_chromosomes.append(chromosome)

            if len(fully_converged_chromosomes) >= param_dict['population_size']:
                # Truncate the list of fully converged chromosomes to just the first <population_size> number of
                # chromosomes:
                fully_converged_chromosomes = fully_converged_chromosomes[:param_dict['population_size']]
                break

            n_subpopulations += 1

            if n_subpopulations * (param_dict['num_processors'] + 1) > param_dict['n_offsprings']:
                raise Exception('Ran out of chromosomes to evaluate in initial population generation')

        new_X = np.array([[]])
        f1 = np.array([[]])
        f2 = np.array([[]])

        for chromosome in fully_converged_chromosomes:
            if chromosome.fitness is not None:  # This statement should always pass, but shown here for clarity
                if len(new_X) < 2:
                    new_X = np.append(new_X, np.array([chromosome.genes]))
                else:
                    new_X = np.row_stack([new_X, np.array(chromosome.genes)])
                f1_chromosome = np.array([chromosome.forces['Cd']])
                f2_chromosome = np.array([np.abs(chromosome.forces['Cl'] - problem.target_CL)])
                f1 = np.append(f1, f1_chromosome)
                f2 = np.append(f2, f2_chromosome)

                # write_F_X_data(1, chromosome, f1_chromosome[0], f2_chromosome[0],
                #                force_and_obj_fun_file, design_variable_file, f_fmt, d_fmt)

        pop_initial = pymoo.core.population.Population.new("X", new_X)
        # objectives
        pop_initial.set("F", np.column_stack([f1, f2]))
        # set_cv(pop_initial)
        Evaluator(skip_already_evaluated=True).eval(problem, pop_initial)

        algorithm = UNSGA3(ref_dirs=ref_dirs, sampling=pop_initial, repair=SelfIntersectionRepair(mea=mea),
                           n_offsprings=param_dict['n_offsprings'],
                           crossover=SimulatedBinaryCrossover(eta=param_dict['eta_crossover']),
                           mutation=PolynomialMutation(eta=param_dict['eta_mutation']))

        termination = termination_condition(param_dict)

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(problem, termination, display=CustomDisplay(), seed=param_dict['seed'], verbose=True,
                        save_history=True)

        save_data(algorithm, os.path.join(opt_dir, 'algorithm_gen_0.pkl'))

        # np.save('checkpoint', algorithm)
        # until the algorithm has no terminated
        n_generation = 0
    else:
        warm_start_index = param_dict['warm_start_generation']
        n_generation = warm_start_index
        algorithm = load_data(os.path.join(opt_settings['Warm Start/Batch Mode']['warm_start_dir']['text'],
                                           f'algorithm_gen_{warm_start_index}.pkl'))
        term = deepcopy(algorithm.termination.terminations)
        term = list(term)
        term[0].n_max_gen = param_dict['n_max_gen']
        term = tuple(term)
        algorithm.termination.terminations = term
        algorithm.has_terminated = False

    while algorithm.has_next():

        pop = algorithm.ask()

        n_generation += 1

        if n_generation > 1:

            # evaluate (objective function value arrays must be numpy column vectors)
            X = pop.get("X")
            new_X = np.array([[]])
            f1 = np.array([[]])
            f2 = np.array([[]])
            n_infeasible_solutions = 0
            search_for_feasible_idx = 0
            while True:
                gene_matrix = []
                feasible_indices = []
                while True:
                    if X[search_for_feasible_idx, 0] != 9999:
                        gene_matrix.append(X[search_for_feasible_idx, :].tolist())
                        feasible_indices.append(search_for_feasible_idx)
                    else:
                        n_infeasible_solutions += 1
                    search_for_feasible_idx += 1
                    if len(gene_matrix) == problem.num_processors:
                        break
                population = [Chromosome(problem.param_dict, ga_settings=ga_settings, category=None,
                                         generation=n_generation,
                                         population_idx=feasible_indices[idx + len(feasible_indices)
                                                                         - param_dict['num_processors']],
                                         genes=gene_list, verbose=param_dict['verbose'],
                                         mea=mea, point_matrix=point_matrix)
                              for idx, gene_list in enumerate(gene_matrix)]
                pop_obj = Population(problem.param_dict, ga_settings=ga_settings, generation=n_generation,
                                     parents=population, verbose=param_dict['verbose'], mea=mea)
                pop_obj.population = population
                for chromosome in pop_obj.population:
                    chromosome.generate()
                pop_obj.eval_pop_fitness()
                for idx, chromosome in enumerate(pop_obj.population):
                    if chromosome.fitness is not None:
                        if len(new_X) < 2:
                            new_X = np.append(new_X, np.array([chromosome.genes]))
                        else:
                            new_X = np.row_stack([new_X, np.array(chromosome.genes)])
                        f1_chromosome = np.array([1.0 * chromosome.forces['Cd']])
                        f2_chromosome = np.array([np.abs(chromosome.forces['Cl'] - problem.target_CL)])
                        f1 = np.append(f1, f1_chromosome)
                        f2 = np.append(f2, f2_chromosome)

                    else:
                        if len(new_X) < 2:
                            new_X = np.append(new_X, np.array([chromosome.genes]))
                        else:
                            new_X = np.row_stack([new_X, np.array(chromosome.genes)])
                        f1 = np.append(f1, np.array([1000.0]))
                        f2 = np.append(f2, np.array([1000.0]))
                algorithm.evaluator.n_eval += problem.num_processors
                population_full = (f1 < 1000.0).sum() >= param_dict['population_size']
                if population_full:
                    break
            # Set the objective function values of the remaining individuals to 1000.0
            for idx in range(search_for_feasible_idx, len(X)):
                new_X = np.row_stack([new_X, X[idx, :]])
                f1 = np.append(f1, np.array([1000.0]))
                f2 = np.append(f2, np.array([1000.0]))
            new_X = np.append(new_X, 9999 * np.ones(shape=(n_infeasible_solutions, param_dict['n_var'])), axis=0)
            for idx in range(n_infeasible_solutions):
                f1 = np.append(f1, np.array([1000.0]))
                f2 = np.append(f2, np.array([1000.0]))

            pop.set("X", new_X)

            # objectives
            pop.set("F", np.column_stack([f1, f2]))

            # for constraints
            # pop.set("G", the_constraint_values))

            # this line is necessary to set the CV and feasbility status - even for unconstrained
            set_cv(pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)

        # do same more things, printing, logging, storing or even modifying the algorithm object
        if n_generation % param_dict['algorithm_save_frequency'] == 0:
            save_data(algorithm, os.path.join(opt_dir, f'algorithm_gen_{n_generation}.pkl'))

    # obtain the result objective from the algorithm
    res = algorithm.result()
    save_data(res, os.path.join(opt_dir, 'res.pkl'))


if __name__ == "__main__":
    run()
