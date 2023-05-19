from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.problem import Problem
from pymead.utils.read_write_files import load_data, save_data
from pymead.core.mea import MEA
from matplotlib import pyplot as plt
from random import randint, random
import numpy as np
from copy import deepcopy
import os
import typing
from pymead.optimization.pop_chrom import Chromosome, Population
from abc import abstractmethod


class Sampling:
    def __init__(self, n_samples, norm_param_list: list):
        self.n_samples = n_samples
        self.norm_param_list = norm_param_list

    @abstractmethod
    def sample(self) -> list or np.ndarray:
        pass


class ConstrictedRandomSampling(Sampling):
    def __init__(self, n_samples: int, norm_param_list: list, max_sampling_width: float):
        self.max_sampling_width = max_sampling_width
        super().__init__(n_samples, norm_param_list)

    def sample(self):
        X_list = [self.norm_param_list]
        for i in range(self.n_samples - 1):
            individual = []
            norm_list_copy = deepcopy(self.norm_param_list)
            for p in norm_list_copy:
                sign_bool = randint(0, 1)
                perturbation = random()
                if sign_bool:
                    distance_to_upper_bound = 1 - p
                    if self.max_sampling_width < distance_to_upper_bound:
                        new_p = p + self.max_sampling_width * perturbation
                    else:
                        new_p = p + distance_to_upper_bound * perturbation
                else:
                    distance_to_lower_bound = p
                    if self.max_sampling_width < distance_to_lower_bound:
                        new_p = p - self.max_sampling_width * perturbation
                    else:
                        new_p = p - distance_to_lower_bound * perturbation
                individual.append(new_p)
            X_list.append(individual)
        return X_list


class PymooLHS(Sampling):
    def __init__(self, n_samples, norm_param_list):
        super().__init__(n_samples, norm_param_list)

    def sample(self):
        problem = Problem(n_var=len(self.norm_param_list), l=0.0, xu=1.0)
        sampling = LatinHypercubeSampling()
        return sampling._do(problem, self.n_samples)


def run_analysis(analysis_dir: str, index: typing.Iterable, X_list, mea: dict, param_dict: dict, evaluate: bool = True,
                 save_coords: bool = False,
                 save_control_points: bool = False,
                 save_airfoil_state: bool = False):

    chromosomes = []

    if not os.path.exists(os.path.join(analysis_dir, 'analysis')):
        os.mkdir(os.path.join(analysis_dir, 'analysis'))

    for i in index:
        param_set = deepcopy(param_dict)
        param_set['mset_settings']['airfoil_analysis_dir'] = os.path.join(
            analysis_dir, 'analysis', f'analysis_{i}')
        param_set['mset_settings']['airfoil_coord_file_name'] = f'analysis_{i}'
        param_set['base_folder'] = os.path.join(analysis_dir, 'analysis')
        param_set['name'] = [f"analysis_{j}" for j in index]

        # parent_chromosomes.append(Chromosome(param_set=param_set, population_idx=s, mea=mea, X=X))
        chromosomes.append(Chromosome(param_dict=param_set, population_idx=i, mea=mea, genes=X_list[i],
                                      ga_settings=None, category=None, generation=0))

    population = Population(param_dict=param_dict, ga_settings=None, generation=0, parents=chromosomes,
                            mea=mea, verbose=True, skip_parent_assignment=False)
    population.generate_chromosomes_parallel()
    if save_coords:
        if not os.path.exists(os.path.join(analysis_dir, 'coords')):
            os.mkdir(os.path.join(analysis_dir, 'coords'))
        for idx, c in enumerate(population.population):
            save_data(c.coords, os.path.join(analysis_dir, 'coords', f'coords_{idx}.json'))
    if save_control_points:
        if not os.path.exists(os.path.join(analysis_dir, 'control_points')):
            os.mkdir(os.path.join(analysis_dir, 'control_points'))
        for idx, c in enumerate(population.population):
            save_data(c.control_points, os.path.join(analysis_dir,
                                                     'control_points', f'control_points_{idx}.json'))
    if save_airfoil_state:
        if not os.path.exists(os.path.join(analysis_dir, 'airfoil_state')):
            os.mkdir(os.path.join(analysis_dir, 'airfoil_state'))
        for idx, c in enumerate(population.population):
            save_data(c.airfoil_state, os.path.join(analysis_dir,
                                                    'airfoil_state', f'airfoil_state_{idx}.json'))

    if evaluate:
        population.eval_pop_fitness()
        # population_forces = {k: [c.forces[k] for c in population.population]
        #                      for k in population.population[0].forces.keys()}
        # print(f"{population_forces = }")
        # save_data(population_forces, post_process_force_file)


def main():
    parametrization = r'C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\2023_03_16_A\pai_underwing.jmea'
    jmea_dict = load_data(parametrization)
    jmea_dict['airfoil_graphs_active'] = False
    analysis_dir = r'C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\test'
    param_dict = load_data(r'C:\Users\mlauer2\Documents\pymead\pymead\pymead\tests\pai\root_underwing_opt\opt_runs\test\param_dict.json')
    param_dict['mplot_settings']['flow_field'] = 2
    if param_dict['mses_settings']['multi_point_stencil'] is not None:
        for idx, stencil_var in enumerate(param_dict['mses_settings']['multi_point_stencil']):
            stencil_var['points'] = stencil_var['points'][:2]  # Stop at the design point (idx = 1)
            param_dict['mses_settings']['multi_point_stencil'][idx] = stencil_var
    param_dict['mses_settings']['timeout'] = 40.0
    mea = MEA.generate_from_param_dict(jmea_dict)
    norm_param_list, _ = mea.extract_parameters()
    sampling = ConstrictedRandomSampling(n_samples=50, norm_param_list=norm_param_list, max_sampling_width=0.08)
    fig, axs = plt.subplots()
    X_list = sampling.sample()
    for individual in X_list:
        mea.update_parameters(individual)
        color = np.random.choice(range(256), size=3) / 255
        for a in mea.airfoils.values():
            a.plot_airfoil(axs, color=color, lw=1.0)
    axs.set_aspect('equal')
    plt.show()
    # run_analysis(analysis_dir=analysis_dir, index=[i for i in range(n_samples)], X_list=X_list,
    #              mea=jmea_dict, param_dict=param_dict)


if __name__ == '__main__':
    main()
