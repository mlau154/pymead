from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.display import Display
from pymoo.factory import get_decomposition
from copy import deepcopy
from pymead.optimization.pop_chrom import Chromosome
from pymead.gui.input_dialog import convert_dialog_to_mset_settings, convert_dialog_to_mses_settings, \
    convert_dialog_to_mplot_settings
import os
import numpy as np
import re


def termination_condition(param_dict):
    termination = MultiObjectiveDefaultTermination(
        x_tol=param_dict['x_tol'],
        cv_tol=param_dict['cv_tol'],
        f_tol=param_dict['f_tol'],
        nth_gen=param_dict['nth_gen'],
        n_last=param_dict['n_last'],
        n_max_gen=param_dict['n_max_gen'],
        n_max_evals=param_dict['n_max_evals']
    )
    return termination


def calculate_warm_start_index(warm_start_generation, warm_start_directory):
    generations = []
    for root, _, files in os.walk(warm_start_directory):
        for idx, f in enumerate(files):
            file_without_ext = os.path.splitext(f)[0]
            if re.split('_', file_without_ext)[0] == 'algorithm':
                idx = re.split('_', file_without_ext)[-1]
                generations.append(int(idx))
    generations = sorted(generations)
    if warm_start_generation not in generations and warm_start_generation != -1:
        raise ValueError(f'Invalid warm start generation. A value of {warm_start_generation} was input, and the valid '
                         f'generations in the directory are {generations}')
    warm_start_index = generations[warm_start_generation]
    return warm_start_index


def convert_opt_settings_to_param_dict(opt_settings: dict) -> dict:
    param_dict = {'tool': opt_settings['Genetic Algorithm']['tool']['current_text'],
                  'algorithm_save_frequency': opt_settings['Genetic Algorithm']['algorithm_save_frequency']['value'],
                  'n_obj': len(opt_settings['Genetic Algorithm']['J']['text'].split(',')),
                  'n_constr': len(opt_settings['Genetic Algorithm']['G']['text'].split(',')) if opt_settings['Genetic Algorithm']['G']['text'] != '' else 0,
                  'population_size': opt_settings['Genetic Algorithm']['pop_size']['value'],
                  'n_ref_dirs': opt_settings['Genetic Algorithm']['pop_size']['value'],
                  'n_offsprings': opt_settings['Genetic Algorithm']['n_offspring']['value'],
                  'xl': 0.0,
                  'xu': 1.0,
                  'seed': opt_settings['Genetic Algorithm']['random_seed']['value'],
                  'multi_point': False,
                  'num_processors': opt_settings['Genetic Algorithm']['num_processors']['value'],
                  'x_tol': opt_settings['Termination']['x_tol']['value'],
                  'cv_tol': opt_settings['Termination']['cv_tol']['value'],
                  'f_tol': opt_settings['Termination']['f_tol']['value'],
                  'nth_gen': opt_settings['Termination']['nth_gen']['value'],
                  'n_last': opt_settings['Termination']['n_last']['value'],
                  'n_max_gen': opt_settings['Termination']['n_max_gen']['value'],
                  'n_max_evals': opt_settings['Termination']['n_max_evals']['value'],
                  'xfoil_settings': {
                      'Re': opt_settings['XFOIL']['Re']['value'],
                      'Ma': opt_settings['XFOIL']['Ma']['value'],
                      'xtr': [opt_settings['XFOIL']['xtr_upper']['value'], opt_settings['XFOIL']['xtr_lower']['value']],
                      'N': opt_settings['XFOIL']['N']['value'],
                      'iter': opt_settings['XFOIL']['iter']['value'],
                      'timeout': opt_settings['XFOIL']['timeout']['value'],
                  },
                  'mset_settings': convert_dialog_to_mset_settings(opt_settings['MSET']),
                  'mses_settings': convert_dialog_to_mses_settings(opt_settings['MSES']),
                  'mplot_settings': convert_dialog_to_mplot_settings(opt_settings['MPLOT']),
                  'min_val_of_max_thickness': opt_settings['Constraints/Validation']['min_val_of_max_thickness'][
                      'value'],
                  'min_thickness_active': opt_settings['Constraints/Validation']['min_val_of_max_thickness'][
                      'active_checkbox'],
                  'min_area': opt_settings['Constraints/Validation']['min_area']['value'],
                  'min_area_active': opt_settings['Constraints/Validation']['min_area']['active_checkbox'],
                  'verbose': False,
                  'eta_crossover': opt_settings['Genetic Algorithm']['eta_crossover']['value'],
                  'eta_mutation': opt_settings['Genetic Algorithm']['eta_mutation']['value'],
                  }

    if opt_settings['XFOIL']['prescribe']['current_text'] == 'Angle of Attack (deg)':
        param_dict['xfoil_settings']['alfa'] = opt_settings['XFOIL']['alfa']['value']
    elif opt_settings['XFOIL']['prescribe']['current_text'] == 'Viscous Cl':
        param_dict['xfoil_settings']['Cl'] = opt_settings['XFOIL']['Cl']['value']
    elif opt_settings['XFOIL']['prescribe']['current_text'] == 'Viscous Cl':
        param_dict['xfoil_settings']['CLI'] = opt_settings['XFOIL']['CLI']['value']
    param_dict['mses_settings']['n_airfoils'] = param_dict['mset_settings']['n_airfoils']
    return param_dict


class TPAIOPT(Problem):
    def __init__(self, n_var, n_obj, n_constr, xl, xu, param_dict, ga_settings):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.param_dict = deepcopy(param_dict)
        self.ga_settings = ga_settings

    def _evaluate(self, X, out, *args, **kwargs):
        pass


class SelfIntersectionRepair(Repair):
    def __init__(self, mea):
        super().__init__()
        self.mea = mea

    def _do(self, problem, pop, **kwargs):
        # print('Repairing...')

        # the packing plan for the whole population (each row one individual)
        Z = pop.get("X")

        # now repair each indvidiual i
        for i in range(len(Z)):

            # the packing plan for i
            z = deepcopy(Z[i])

            genes = z.tolist()

            # Create a chromosome with "genes" and "ga_setup_params" being the only relevant parameters (this is just to
            # check for geometry validity)
            chromosome = Chromosome(genes=genes, param_dict=problem.param_dict, ga_settings=problem.ga_settings,
                                    category=None,
                                    population_idx=0, generation=0, verbose=False, mea=self.mea)
            chromosome.generate()

            if not chromosome.valid_geometry:
                Z[int(i), :] = 9999

        # set the design variables for the population
        pop.set("X", Z)
        return pop


class CustomDisplay(Display):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.term = MultiObjectiveSpaceToleranceTermination()
        self.progress_dict = None

    def set_progress_dict(self, progress_dict: dict):
        self.progress_dict = progress_dict

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        F = algorithm.pop.get("F")
        # G = algorithm.pop.get("G")
        # CV = algorithm.pop.get("CV")
        weights = np.array([0.5, 0.5])
        decomp = get_decomposition("asf")
        I = decomp.do(F, weights).argmin()
        n_nds = len(algorithm.opt)
        f1_best = F[I][0]
        f1_min = F[:, 0].min()
        f1_mean = np.mean(F[:, 0])
        # g1_min = G[:, 0].min()
        # cv_min = CV[:, 0].min()
        # g1_mean = np.mean(G[:, 0])
        # cv_mean = np.mean(CV[:, 0])
        self.set_progress_dict({'f1_best': f1_best})
        self.output.append("n_nds", n_nds, width=7)
        self.term.do_continue(algorithm)

        max_from, eps = "-", "-"

        if len(self.term.metrics) > 0:
            metric = self.term.metrics[-1]
            tol = self.term.tol
            delta_ideal, delta_nadir, delta_f = metric["delta_ideal"], metric["delta_nadir"], metric["delta_f"]

            if delta_ideal > tol:
                max_from = "ideal"
                eps = delta_ideal
            elif delta_nadir > tol:
                max_from = "nadir"
                eps = delta_nadir
            else:
                max_from = "f"
                eps = delta_f

        self.output.append("eps", eps)
        self.output.append("indicator", max_from)
        self.output.append("f1_best", f1_best)
        # self.output.append("f2_best", f2_best)
        self.output.append("f1_min", f1_min)
        # self.output.append("f2_min", f2_min)
        self.output.append("f1_mean", f1_mean)
        # self.output.append("f2_mean", f2_mean)
        # self.output.append("g1_min", g1_min)
        # self.output.append("cv_min", cv_min)
        # self.output.append("g1_mean", g1_mean)
        # self.output.append("cv_mean", cv_mean)
        pass
