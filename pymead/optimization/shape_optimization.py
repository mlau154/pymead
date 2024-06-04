import logging
import multiprocessing.connection
import typing

import numpy as np

from copy import deepcopy

import os
import random

from pymead.core.geometry_collection import GeometryCollection
from pymead.optimization.opt_setup import CustomDisplay, TPAIOPT
from pymead.utils.read_write_files import load_data, save_data
from pymead.optimization.pop_chrom import Chromosome, Population
from pymead.optimization.objectives_and_constraints import Objective, Constraint
from pymead.optimization.sampling import ConstrictedRandomSampling
from pymead.optimization.opt_setup import termination_condition

import pymoo.core.population
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.config import Config
from pymoo.core.evaluator import Evaluator
from pymoo.factory import get_reference_directions, get_decomposition
from pymoo.core.evaluator import set_cv


def shape_optimization(conn: multiprocessing.connection.Connection or None, param_dict: dict, opt_settings: dict,
                       geo_col_dict: dict,
                       objectives: typing.List[str], constraints: typing.List[str]):

    objectives = [Objective(func_str=func_str) for func_str in objectives]
    constraints = [Constraint(func_str=func_str) for func_str in constraints]

    # print(f"Shape optimization has PID {os.getpid()}")

    def start_message(warm_start: bool):
        first_word = "Resuming" if warm_start else "Beginning"
        return f"\n{first_word} aerodynamic shape optimization with {param_dict['num_processors']} processors..."

    def write_force_dict_to_file(forces_dict, file_name: str):
        forces_temp = deepcopy(forces_dict)
        if "Cp" in forces_temp.keys():
            for el in forces_temp["Cp"]:
                if isinstance(el, list):
                    for e in el:
                        for k_, v_ in e.items():
                            if isinstance(v_, np.ndarray):
                                e[k_] = v_.tolist()
                else:
                    for k_, v_ in el.items():
                        if isinstance(v_, np.ndarray):
                            el[k_] = v_.tolist()
        save_data(forces_temp, file_name)

    def read_force_dict_from_file(file_name: str):
        return load_data(file_name)

    def send_over_pipe(data: object):
        try:
            if conn is not None:
                conn.send(data)
        except BrokenPipeError:
            pass

    forces_dict = {} if not opt_settings["General Settings"]["warm_start_active"] \
        else read_force_dict_from_file(os.path.join(param_dict["opt_dir"], "force_history.json"))

    send_over_pipe(("text", start_message(opt_settings["General Settings"]["warm_start_active"])))

    Config.show_compile_hint = False
    forces = []
    ref_dirs = get_reference_directions("energy", param_dict['n_obj'], param_dict['n_ref_dirs'],
                                        seed=param_dict['seed'])
    geo_col = GeometryCollection.set_from_dict_rep(deepcopy(geo_col_dict))
    airfoil_name, mea_name = None, None
    if param_dict["tool"] == "XFOIL":
        airfoil_name = opt_settings["XFOIL"]["airfoil"]
    elif param_dict["tool"] == "MSES":
        mea_name = opt_settings["MSET"]["mea"]
    parameter_list = geo_col.extract_design_variable_values(bounds_normalized=True)
    num_parameters = len(parameter_list)

    send_over_pipe(("text", f"Number of design variables: {num_parameters}"))

    problem = TPAIOPT(n_var=param_dict['n_var'], n_obj=param_dict['n_obj'], n_constr=param_dict['n_constr'],
                      xl=param_dict['xl'], xu=param_dict['xu'], param_dict=param_dict)

    if not opt_settings['General Settings']['warm_start_active']:
        if param_dict['seed'] is not None:
            np.random.seed(param_dict['seed'])
            random.seed(param_dict['seed'])

        sampling = ConstrictedRandomSampling(n_samples=param_dict['n_offsprings'], norm_param_list=parameter_list,
                                             max_sampling_width=param_dict['max_sampling_width'])
        X_list = sampling.sample()
        parents = [Chromosome(geo_col_dict=geo_col_dict, param_dict=param_dict, generation=0,
                              airfoil_name=airfoil_name, mea_name=mea_name, population_idx=idx, genes=individual)
                   for idx, individual in enumerate(X_list)]
        population = Population(param_dict=param_dict, generation=0, parents=parents,
                                verbose=param_dict['verbose'])

        n_eval = population.eval_pop_fitness(sig=conn)
        print(f"Finished evaluating population fitness. Continuing...")

        new_X = None
        J = None
        G = None

        for chromosome in population.converged_chromosomes:
            forces.append(chromosome.forces)
            if new_X is None:
                if param_dict['n_offsprings'] > 1:
                    new_X = np.array([chromosome.genes])
                else:
                    new_X = np.array(chromosome.genes)
            else:
                new_X = np.row_stack((new_X, np.array(chromosome.genes)))
            # print(f"{self.objectives = }")
            # print(f"Before objective and constraint update, {chromosome.forces = }")
            for objective in objectives:
                objective.update(chromosome.forces)
            for constraint in constraints:
                constraint.update(chromosome.forces)
                print(f"{constraint.value = }")
            if J is None:
                J = np.array([obj.value for obj in objectives])
            else:
                J = np.row_stack((J, np.array([obj.value for obj in objectives])))
            if len(constraints) > 0:
                if G is None:
                    G = np.array([constraint.value for constraint in constraints])
                else:
                    G = np.row_stack((G, np.array([
                        constraint.value for constraint in constraints])))

            # print(f"{J = }, {self.objectives = }")

        if new_X is None:
            send_over_pipe(("text", f"Could not converge any individuals in the population. Optimization terminated."))
            return

        if new_X.ndim == 1:
            new_X = np.array([new_X])

        if J.ndim == 1:
            J = np.array([J])

        if len(constraints) > 0 and G.ndim == 1:
            G = np.array([G])

        pop_initial = pymoo.core.population.Population.new("X", new_X)
        # objectives
        pop_initial.set("F", J)
        # print(f"Initially, {J = }")
        if len(constraints) > 0:
            if G is not None:
                pop_initial.set("G", G)
        set_cv(pop_initial)
        for individual in pop_initial:
            individual.evaluated = {"F", "G", "CV", "feasible"}
        Evaluator(skip_already_evaluated=True).eval(problem, pop_initial)

        algorithm = UNSGA3(ref_dirs=ref_dirs, sampling=pop_initial,
                           n_offsprings=param_dict['n_offsprings'],
                           crossover=SimulatedBinaryCrossover(eta=param_dict['eta_crossover']),
                           mutation=PolynomialMutation(eta=param_dict['eta_mutation']))

        termination = termination_condition(param_dict)

        display = CustomDisplay()

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(problem, termination, display=display, seed=param_dict['seed'], verbose=True,
                        save_history=False)

        algorithm.evaluator.n_eval += n_eval

        # save_data(algorithm, os.path.join(param_dict['opt_dir'], 'algorithm_gen_0.pkl'))

        # np.save('checkpoint', algorithm)
        # until the algorithm has no terminated
        n_generation = 0
    else:
        logging.debug('Starting from where we left off...')
        warm_start_index = param_dict['warm_start_generation']
        n_generation = warm_start_index
        warm_start_alg_file = os.path.join(opt_settings['General Settings']['warm_start_dir'],
                                           f'algorithm_gen_{warm_start_index}.pkl')
        algorithm = load_data(warm_start_alg_file)
        read_force_dict_from_file(os.path.join(param_dict["opt_dir"], "force_history.json"))
        logging.debug(f'Loaded {warm_start_alg_file}.')
        if not opt_settings['General Settings']['use_initial_settings']:
            # Currently only set up to change n_offsprings
            previous_offsprings = deepcopy(algorithm.n_offsprings)
            algorithm.n_offsprings = opt_settings['Genetic Algorithm']['n_offspring']
            algorithm.problem.param_dict['n_offsprings'] = algorithm.n_offsprings
            if previous_offsprings != algorithm.n_offsprings:
                logging.debug(f'Number of offspring changed from {previous_offsprings} '
                              f'to {algorithm.n_offsprings}.')
        term = deepcopy(algorithm.termination.terminations)
        term = list(term)
        term[0].n_max_gen = param_dict['n_max_gen']
        term = tuple(term)
        algorithm.termination.terminations = term
        algorithm.has_terminated = False

    while algorithm.has_next():

        logging.debug(f'Asking the algorithm to get the next population...')

        pop = algorithm.ask()

        logging.debug(f'Acquired new population. pop = {pop}')

        n_generation += 1

        if n_generation > 1:

            logging.debug(f'Starting generation {n_generation}...')

            forces = []

            # evaluate (objective function value arrays must be numpy column vectors)
            X = pop.get("X")
            logging.debug(f'Input matrix has shape {X.shape} ({X.shape[0]} chromosomes with {X.shape[1]} genes).')
            new_X = None
            J = None
            G = None

            parents = [Chromosome(param_dict=param_dict, generation=n_generation, population_idx=idx,
                                  airfoil_name=airfoil_name, mea_name=mea_name,
                                  geo_col_dict=geo_col_dict, genes=individual) for idx, individual in enumerate(X)]
            population = Population(problem.param_dict, generation=n_generation,
                                    parents=parents, verbose=param_dict['verbose'])
            n_eval = population.eval_pop_fitness(sig=conn)

            for chromosome in population.converged_chromosomes:
                forces.append(chromosome.forces)
                if new_X is None:
                    if param_dict['n_offsprings'] > 1:
                        new_X = np.array([chromosome.genes])
                    else:
                        new_X = np.array(chromosome.genes)
                else:
                    new_X = np.row_stack((new_X, np.array(chromosome.genes)))
                for objective in objectives:
                    objective.update(chromosome.forces)
                for constraint in constraints:
                    constraint.update(chromosome.forces)
                if J is None:
                    J = np.array([obj.value for obj in objectives])
                else:
                    J = np.row_stack((J, np.array([obj.value for obj in objectives])))
                if len(constraints) > 0:
                    if G is None:
                        G = np.array([constraint.value for constraint in constraints])
                    else:
                        G = np.row_stack((G, np.array([
                            constraint.value for constraint in constraints])))

                # print(f"{J = }, {self.objectives = }")

            algorithm.evaluator.n_eval += n_eval

            if new_X is None:
                send_over_pipe(
                    ("text", f"Could not converge any individuals in the population. Optimization terminated."))
                return

            for idx in range(param_dict['n_offsprings'] - len(new_X)):
                # f1 = np.append(f1, np.array([1000.0]))
                # f2 = np.append(f2, np.array([1000.0]))
                new_X = np.row_stack((new_X, 9999 * np.ones(param_dict['n_var'])))
                J = np.row_stack((J, 1000.0 * np.ones(param_dict['n_obj'])))
                if len(constraints) > 0:
                    G = np.row_stack((G, 1000.0 * np.ones(param_dict['n_constr'])))

            if new_X.ndim == 1:
                new_X = np.array([new_X])

            if J.ndim == 1:
                J = np.array([J])

            if len(constraints) > 0 and G.ndim == 1:
                G = np.array([G])

            pop.set("X", new_X)

            # objectives
            pop.set("F", J)

            # print(f"{pop.get('F') = }")

            # for constraints
            if len(constraints) > 0:
                pop.set("G", G)

            # this line is necessary to set the CV and feasbility status - even for unconstrained
            set_cv(pop)

        # returned the evaluated individuals which have been evaluated or even modified
        # print(f"{pop.get('X') = }, {pop.get('F') = }")
        algorithm.tell(infills=pop)

        # print(f"{algorithm.opt.get('F') = }")

        warm_start_gen = None
        if opt_settings["General Settings"]["warm_start_active"]:
            warm_start_gen = param_dict["warm_start_generation"]

        send_over_pipe(("opt_progress", {"text": algorithm.display.progress_dict, "completed": not algorithm.has_next(),
                                         "warm_start_gen": warm_start_gen}))

        # print(f"{algorithm.display.progress_dict = }")

        if len(objectives) == 1:
            if n_generation > 1:
                X = algorithm.opt.get("X")[0]
            else:
                X = algorithm.pop.get("X")[0, :]
        else:
            if n_generation > 1:
                X = algorithm.opt.get("X")
                F = algorithm.opt.get("F")
            else:
                X = algorithm.pop.get("X")
                F = algorithm.pop.get("F")

            # Use a "50-50" decomposition to get a representative optimal solution for plotting
            weights = np.array([1.0 / len(objectives) for _ in objectives])
            decomp = get_decomposition("asf")
            I = decomp.do(F, weights).argmin()
            X = X[I, :]

        best_in_previous_generation = False
        forces_index = 0
        try:
            forces_index = np.where((new_X == X).all(axis=1))[0][0]
        except IndexError:
            best_in_previous_generation = True

        if best_in_previous_generation:
            for k, v in forces_dict.items():
                if k not in forces_dict.keys():
                    forces_dict[k] = []
                forces_dict[k].append(v[-1])
        else:
            best_forces = forces[forces_index]
            for k, v in best_forces.items():
                if param_dict['tool'] in ['xfoil', 'XFOIL', 'mses', 'MSES', 'Mses']:
                    if k not in ['converged', 'timed_out', 'errored_out']:
                        if k not in forces_dict.keys():
                            forces_dict[k] = []
                        forces_dict[k].append(v)

        write_force_dict_to_file(forces_dict, os.path.join(param_dict["opt_dir"], "force_history.json"))

        geo_col.assign_design_variable_values(X.tolist(), bounds_normalized=True)
        mea = None if mea_name is None else geo_col.container()["mea"][mea_name]
        airfoil = None if airfoil_name is None else geo_col.container()["airfoils"][airfoil_name]
        if airfoil is not None:
            coords = [airfoil.get_coords_selig_format()]
        else:
            coords = mea.get_coords_list()

        send_over_pipe(("airfoil_coords", coords))

        norm_val_list = geo_col.extract_design_variable_values(bounds_normalized=True)

        send_over_pipe(("parallel_coords", (norm_val_list, [desvar for desvar in geo_col.container()["desvar"]])))

        if param_dict["tool"] == "XFOIL":
            Cp = forces_dict["Cp"][-1]
            if isinstance(Cp, list):
                Cp = Cp[param_dict["design_idx"]]
            send_over_pipe(("cp_xfoil", Cp))

            Cd = forces_dict['Cd']
            Cdp = forces_dict['Cdp']
            Cdf = forces_dict['Cdf']
            Cd = Cd if not isinstance(Cd[0], list) else [d[param_dict["design_idx"]] for d in Cd]
            Cdp = Cdp if not isinstance(Cdp[0], list) else [d[param_dict["design_idx"]] for d in Cdp]
            Cdf = Cdf if not isinstance(Cdf[0], list) else [d[param_dict["design_idx"]] for d in Cdf]
            send_over_pipe(("drag_xfoil", (Cd, Cdp, Cdf)))
        elif param_dict["tool"] == "MSES":
            Cp = forces_dict['BL'][-1] if isinstance(
                forces_dict['BL'][-1][0], dict) else forces_dict['BL'][-1][param_dict["design_idx"]]
            send_over_pipe(("cp_mses", Cp))

            Cd = forces_dict['Cd']
            Cdp = forces_dict['Cdp']
            Cdf = forces_dict['Cdf']
            Cdv = forces_dict['Cdv']
            Cdw = forces_dict['Cdw']
            Cd = Cd if not isinstance(Cd[0], list) else [d[param_dict["design_idx"]] for d in Cd]
            Cdp = Cdp if not isinstance(Cdp[0], list) else [d[param_dict["design_idx"]] for d in Cdp]
            Cdf = Cdf if not isinstance(Cdf[0], list) else [d[param_dict["design_idx"]] for d in Cdf]
            Cdv = Cdv if not isinstance(Cdv[0], list) else [d[param_dict["design_idx"]] for d in Cdv]
            Cdw = Cdw if not isinstance(Cdw[0], list) else [d[param_dict["design_idx"]] for d in Cdw]
            send_over_pipe(("drag_mses", (Cd, Cdp, Cdf, Cdv, Cdw)))

        if n_generation % param_dict['algorithm_save_frequency'] == 0:
            save_data(algorithm, os.path.join(param_dict['opt_dir'], f'algorithm_gen_{n_generation}.pkl'))

    # obtain the result objective from the algorithm
    res = algorithm.result()
    save_data(res, os.path.join(param_dict['opt_dir'], 'res.pkl'))
    write_force_dict_to_file(forces_dict, os.path.join(param_dict["opt_dir"], "force_history.json"))
    np.savetxt(os.path.join(param_dict['opt_dir'], 'opt_X.dat'), res.X)

    send_over_pipe(("finished", None))
    # self.save_opt_plots(param_dict['opt_dir'])  # not working at the moment
