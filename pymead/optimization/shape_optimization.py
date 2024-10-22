import multiprocessing.connection
import typing

import numpy as np

from copy import deepcopy

import os
import random

from pymead.core.geometry_collection import GeometryCollection
from pymead.optimization.opt_setup import CustomDisplay, PymeadGAProblem
from pymead.utils.read_write_files import load_data, save_data
from pymead.optimization.pop_chrom import Chromosome, Population
from pymead.optimization.objectives_and_constraints import Objective, Constraint
from pymead.optimization.sampling import ConstrictedRandomSampling
from pymead.optimization.opt_setup import termination_condition

import pymoo.core.population
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.algorithm import Algorithm
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.config import Config
from pymoo.core.evaluator import Evaluator
from pymoo.decomposition.asf import ASF

try:
    from pymoo.factory import get_reference_directions
except ModuleNotFoundError:
    from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.evaluator import set_cv


def write_alg_data(algorithm: Algorithm, param_dict: dict):
    opt, pop, off = algorithm.opt, algorithm.pop, algorithm.off
    opt_dir = param_dict["opt_dir"]
    header = (f"n_gen={algorithm.n_gen},n_eval={algorithm.evaluator.n_eval},n_obj={algorithm.problem.n_obj},"
              f"n_constr={algorithm.problem.n_constr},n_var={algorithm.problem.n_var},n_opt={len(opt)},"
              f"n_pop={len(pop)},n_off={len(off)}")

    # Get the data stacks for objectives, constraints, and design variables
    J = np.vstack((opt.get("F"), pop.get("F"), off.get("F")))
    G = None
    if algorithm.problem.n_constr > 0:
        G = np.vstack((opt.get("G"), pop.get("G"), off.get("G")))
    X = np.vstack((opt.get("X"), pop.get("X"), off.get("X")))

    # Combine the data
    if G is not None:
        data_coalesced = np.column_stack((J, G, X))
    else:
        data_coalesced = np.column_stack((J, X))

    # Save the data
    data_file_name = os.path.join(opt_dir, f"alg_data_{algorithm.n_gen}.csv")
    np.savetxt(data_file_name, data_coalesced, header=header, delimiter=",", comments="")


def read_alg_data(file_name: str) -> dict:
    data_dict = {}

    # Read the header data into the dictionary
    with open(file_name, "r") as f:
        header_line = f.readline().strip()
    for header_item in header_line.split(","):
        equal_split = header_item.split("=")
        data_dict[equal_split[0]] = int(equal_split[1])  # Key-value pair from header

    # Read the optimum, population, and offspring data into the dictionary
    n_opt, n_pop, n_off = data_dict["n_opt"], data_dict["n_pop"], data_dict["n_off"]
    n_obj, n_constr, n_var = data_dict["n_obj"], data_dict["n_constr"], data_dict["n_var"]
    data_dict["opt_G"], data_dict["pop_G"], data_dict["off_G"] = None, None, None
    data_arr = np.loadtxt(fname=file_name, skiprows=1, delimiter=",")
    data_dict["opt_J"] = data_arr[:n_opt, :n_obj]
    data_dict["pop_J"] = data_arr[n_opt:n_opt + n_pop, :n_obj]
    data_dict["off_J"] = data_arr[n_opt + n_pop:, :n_obj]
    if n_constr > 0:
        data_dict["opt_G"] = data_arr[:n_opt, n_obj:n_obj + n_constr]
        data_dict["pop_G"] = data_arr[n_opt:n_opt + n_pop, n_obj:n_obj + n_constr]
        data_dict["off_G"] = data_arr[n_opt + n_pop:, n_obj:n_obj + n_constr]
    data_dict["opt_X"] = data_arr[:n_opt, n_obj + n_constr:]
    data_dict["pop_X"] = data_arr[n_opt:n_opt + n_pop, n_obj + n_constr:]
    data_dict["off_X"] = data_arr[n_opt + n_pop:, n_obj + n_constr:]

    return data_dict


def setup_ga_optimization(param_dict: dict, J: np.ndarray, G: np.ndarray or None, X: np.ndarray,
                          n_eval: int, n_gen: int) -> Algorithm:
    # Get the reference directions for the problem
    ref_dirs = get_reference_directions("energy", param_dict['n_obj'], param_dict['n_ref_dirs'],
                                        seed=param_dict['seed'])

    # Set up the problem statement
    problem = PymeadGAProblem(n_var=param_dict['n_var'], n_obj=param_dict['n_obj'], n_constr=param_dict['n_constr'],
                              xl=param_dict['xl'], xu=param_dict['xu'], param_dict=param_dict)

    # Create the initial or warm-start population and set the evaluated values
    pop_initial = pymoo.core.population.Population.new("X", X)
    pop_initial.set("F", J)
    if G is not None:
        pop_initial.set("G", G)
    set_cv(pop_initial)
    for individual in pop_initial:
        individual.evaluated = {"F", "G", "CV", "feasible"}
    Evaluator(skip_already_evaluated=True).eval(problem, pop_initial)

    # Create the algorithm objects
    algorithm = UNSGA3(
        ref_dirs=ref_dirs,
        sampling=pop_initial,
        n_offsprings=param_dict['n_offsprings'],
        crossover=SimulatedBinaryCrossover(eta=param_dict['eta_crossover']),
        mutation=PolynomialMutation(eta=param_dict['eta_mutation'])
    )

    # Set up the termination conditions and display
    termination = termination_condition(param_dict)
    display = CustomDisplay()

    # Set up the algorithm
    algorithm.setup(problem, termination, display=display, seed=param_dict['seed'], verbose=True,
                    save_history=False)

    # Set the number of evaluations and the current generation number
    algorithm.evaluator.n_eval = n_eval
    algorithm.n_gen = n_gen

    return algorithm


def patch_termination(algorithm: Algorithm, param_dict: dict):
    """Needed for legacy algorithm load method (.pkl) only"""
    term = deepcopy(algorithm.termination.terminations)
    term = list(term)
    term[0].n_max_gen = param_dict['n_max_gen']
    term = tuple(term)
    algorithm.termination.terminations = term
    algorithm.has_terminated = False


def compute_objectives_and_forces_from_evaluated_population(
        evaluated_population: Population,
        objectives: typing.List[Objective],
        constraints: typing.List[Constraint] or None) -> (np.ndarray, np.ndarray, typing.List[dict]):
    n_offspring = len(evaluated_population.population)
    forces = []
    constraints = [] if constraints is None else constraints
    J = 1000.0 * np.ones((n_offspring, len(objectives)))
    G = 1000.0 * np.ones((n_offspring, len(constraints))) if constraints else None

    for pop_idx, chromosome in enumerate(evaluated_population.population):
        forces.append(chromosome.forces)
        if chromosome.forces is None:
            continue

        # Update objective and constraint objects; assign values to J and G, respectively
        for obj_idx, objective in enumerate(objectives):
            objective.update(chromosome.forces)
            J[pop_idx, obj_idx] = objective.value
        for cnstr_idx, constraint in enumerate(constraints):
            constraint.update(chromosome.forces)
            G[pop_idx, cnstr_idx] = constraint.value

    return J, G, forces


def get_airfoil_and_mea_names(opt_settings: dict, param_dict: dict) -> (str or None, str or None):
    airfoil_name, mea_name = None, None
    if param_dict["tool"] == "XFOIL":
        airfoil_name = opt_settings["XFOIL"]["airfoil"]
    elif param_dict["tool"] == "MSES":
        mea_name = opt_settings["MSET"]["mea"]
    return airfoil_name, mea_name


def do_sampling(param_dict: dict, norm_parm_list: list) -> list or np.ndarray:
    """
    Performs the sampling stage for the genetic algorithm.

    Parameters
    ----------
    param_dict: dict
        Parameter dictionary assigned to the population and chromosomes (copied and modified form of the optimization settings)
    norm_parm_list: list
        List of normalized geometry collection design variable values (from the baseline geometry)
    """
    # Set the psuedo-random seeds
    if param_dict["seed"] is not None:
        np.random.seed(param_dict["seed"])
        random.seed(param_dict["seed"])

    # Create the sampling object
    sampling = ConstrictedRandomSampling(
        n_samples=param_dict["n_offsprings"],
        norm_param_list=norm_parm_list,
        max_sampling_width=param_dict["max_sampling_width"]
    )

    # Perform the sampling
    X_list = sampling.sample()

    return X_list


def shape_optimization(conn: multiprocessing.connection.Connection or None, param_dict: dict, opt_settings: dict,
                       geo_col_dict: dict,
                       objectives: typing.List[str], constraints: typing.List[str]):
    objectives = [Objective(func_str=func_str) for func_str in objectives]
    constraints = [Constraint(func_str=func_str) for func_str in constraints]

    def start_message(warm_start: bool):
        first_word = "Resuming" if warm_start else "Beginning"
        return f"\n{first_word} aerodynamic shape optimization with {param_dict['num_processors']} processors..."

    def write_force_dict_to_file(force_dictionary, file_name: str):
        forces_temp = deepcopy(force_dictionary)
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

    def save_baseline_airfoil_system(current_geo_col: GeometryCollection):
        """
        Saves the currently loaded airfoil system as the baseline airfoil system in the optimization directory

        Parameters
        ----------
        current_geo_col: GeometryCollection
            Currently loaded geometry collection
        """
        save_data(current_geo_col.get_dict_rep(), os.path.join(param_dict["opt_dir"], "baseline.jmea"))

    def save_opt_airfoil_system(current_geo_col: GeometryCollection, generation: int,
                                dv_norm_list: typing.List[float]):
        """
        Loads in a set of intermediate normalized design variable values from the optimization, updates the currently
        loaded geometry collection, and saves the geometry collection to the optimization directory.

        Parameters
        ----------
        current_geo_col: GeometryCollection
            Currently loaded geometry collection
        generation: int
            Genetic algorithm generation corresponding to the design variable list
        dv_norm_list: typing.List[float]
            List of normalized design variable values used to update the ``GeometryCollection``
        """
        geo_col_copy = deepcopy(current_geo_col)
        geo_col_copy.assign_design_variable_values(dv_norm_list, bounds_normalized=True)
        save_data(geo_col_copy.get_dict_rep(), os.path.join(param_dict["opt_dir"], f"opt_gen_{generation}.jmea"))

    forces_dict = {} if not opt_settings["General Settings"]["warm_start_active"] \
        else read_force_dict_from_file(os.path.join(param_dict["opt_dir"], "force_history.json"))

    send_over_pipe(("text", start_message(opt_settings["General Settings"]["warm_start_active"])))

    Config.show_compile_hint = False

    geo_col = GeometryCollection.set_from_dict_rep(deepcopy(geo_col_dict))
    save_baseline_airfoil_system(geo_col)
    parameter_list = geo_col.extract_design_variable_values(bounds_normalized=True)
    num_parameters = len(parameter_list)
    send_over_pipe(("text", f"Number of design variables: {num_parameters}"))

    airfoil_name, mea_name = get_airfoil_and_mea_names(opt_settings, param_dict)

    if not opt_settings['General Settings']['warm_start_active']:

        X_list = do_sampling(param_dict, norm_parm_list=parameter_list)

        parents = [Chromosome(geo_col_dict=geo_col_dict, param_dict=param_dict, generation=0,
                              airfoil_name=airfoil_name, mea_name=mea_name, population_idx=idx, genes=individual)
                   for idx, individual in enumerate(X_list)]
        population = Population(param_dict=param_dict, generation=0, parents=parents,
                                verbose=param_dict['verbose'])

        n_eval = population.eval_pop_fitness(sig=conn)
        print(f"Finished evaluating population fitness. Continuing...")

        J, G, forces = compute_objectives_and_forces_from_evaluated_population(population, objectives, constraints)

        if all([obj_value == 1000.0 for obj_value in J[:, 0]]):
            send_over_pipe(("disp_message_box",
                            f"Could not converge any individuals in the population. Optimization terminated."))
            return

        n_generation = 0

        algorithm = setup_ga_optimization(param_dict, J, G, np.array(X_list), n_eval, n_gen=n_generation)
    else:
        warm_start_index = param_dict['warm_start_generation']
        n_generation = warm_start_index
        warm_start_alg_file_legacy = os.path.join(
            opt_settings["General Settings"]["warm_start_dir"], f"algorithm_gen_{warm_start_index}.pkl"
        )
        if os.path.exists(warm_start_alg_file_legacy):
            algorithm = load_data(warm_start_alg_file_legacy)
            patch_termination(algorithm, param_dict)
        else:
            warm_start_file = os.path.join(
                opt_settings["General Settings"]["warm_start_dir"], f"alg_data_{warm_start_index}.csv"
            )
            alg_data_dict = read_alg_data(warm_start_file)
            algorithm = setup_ga_optimization(
                param_dict, alg_data_dict["off_J"], alg_data_dict["off_G"], alg_data_dict["off_X"],
                n_eval=alg_data_dict["n_eval"], n_gen=alg_data_dict["n_gen"]
            )

        forces_dict = read_force_dict_from_file(os.path.join(param_dict["opt_dir"], "force_history.json"))
        forces = []

        if not opt_settings['General Settings']['use_initial_settings']:
            # Currently only set up to change n_offsprings
            algorithm.n_offsprings = opt_settings['Genetic Algorithm']['n_offspring']
            algorithm.problem.param_dict['n_offsprings'] = algorithm.n_offsprings

    while algorithm.has_next():

        pop = algorithm.ask()

        n_generation += 1

        if n_generation > 1:

            X = pop.get("X")

            parents = [Chromosome(param_dict=param_dict, generation=n_generation, population_idx=idx,
                                  airfoil_name=airfoil_name, mea_name=mea_name,
                                  geo_col_dict=geo_col_dict, genes=individual) for idx, individual in enumerate(X)]
            population = Population(algorithm.problem.param_dict, generation=n_generation,
                                    parents=parents, verbose=param_dict['verbose'])
            n_eval = population.eval_pop_fitness(sig=conn)

            J, G, forces = compute_objectives_and_forces_from_evaluated_population(population, objectives, constraints)

            algorithm.evaluator.n_eval += n_eval

            if all([obj_value == 1000.0 for obj_value in J[:, 0]]):
                send_over_pipe(
                    ("disp_message_box",
                     f"Could not converge any individuals in the population. Optimization terminated."))
                return

            pop.set("F", J)
            if len(constraints) > 0 and G is not None:
                pop.set("G", G)
            set_cv(pop)  # this line is necessary to set the CV and feasbility status - even for unconstrained

        algorithm.tell(infills=pop)

        warm_start_gen = None
        if opt_settings["General Settings"]["warm_start_active"]:
            warm_start_gen = param_dict["warm_start_generation"]

        send_over_pipe(("opt_progress", {"text": algorithm.display.progress_dict, "completed": not algorithm.has_next(),
                                         "warm_start_gen": warm_start_gen}))

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
            decomp = ASF()
            I = decomp.do(F, weights).argmin()
            X = X[I, :]

        # Save the optimized airfoil system
        save_opt_airfoil_system(geo_col, n_generation, X.tolist())

        best_in_previous_generation = False
        forces_index = 0
        try:
            forces_index = np.where((pop.get("X") == X).all(axis=1))[0][0]
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
            # write_alg_data(algorithm, param_dict)
            save_data(algorithm, os.path.join(param_dict['opt_dir'], f'algorithm_gen_{n_generation}.pkl'))  # Legacy

    # obtain the result objective from the algorithm
    res = algorithm.result()
    write_force_dict_to_file(forces_dict, os.path.join(param_dict["opt_dir"], "force_history.json"))
    np.savetxt(os.path.join(param_dict['opt_dir'], 'opt_X.dat'), res.X)

    send_over_pipe(("finished", None))
    # self.save_opt_plots(param_dict['opt_dir'])  # not working at the moment
