import typing
from abc import abstractmethod
from copy import deepcopy
from random import randint, random

import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LatinHypercubeSampling


class Sampling:
    def __init__(self, n_samples: int, norm_param_list: list):
        """
        High-level generic class for any sampling method.

        Parameters
        ----------
        n_samples: int
            Number of samples

        norm_param_list: list
            List of normalized parameter values (usually from ``extract_parameters`` in ``MEA``).
        """
        self.n_samples = n_samples
        self.norm_param_list = deepcopy(norm_param_list)

    @abstractmethod
    def sample(self) -> list or np.ndarray:
        pass


class ConstrictedRandomSampling(Sampling):
    def __init__(self, n_samples: int, norm_param_list: list, max_sampling_width: float):
        """
        A custom sampling method where the probability distribution for the new parameter value is uniform with a
        specified width on either side of the original parameter value. The probability of falling above or below the
        original parameter value is 0.5 and 0.5, respectively. The sampling width is truncated by the parameter bounds
        if necessary, retaining a probability of 0.5 on both sides of the original parameter value.

        Parameters
        ==========
        n_samples: int
            Number of samples
        norm_param_list: list
            List of normalized parameter values (usually from ``extract_parameters`` in ``MEA``).

        max_sampling_width: float
            Maximum distance of the new parameter value from the original parameter value. The actual new parameter
            value falls randomly in between the original parameter value and this maximum distance in either direction.
        """
        self.max_sampling_width = max_sampling_width
        super().__init__(n_samples, norm_param_list)

    def sample(self) -> typing.List[typing.List[float]]:
        """
        Randomly samples the design space.

        Returns
        -------
        typing.List[typing.List[float]]
            A new nested list of normalized parameter values to be used to update the airfoil system. Each row
            of the 2-D list represents a set of genes to apply to each Chromosome, and each value in the row
            represents an individual gene. These individual genes are bounds-normalized values representing a new value
            for a particular design variable in the geometry collection.
        """
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
        """
        Latin-Hypercube sampling method from the ``pymoo`` package.

        Parameters
        ----------
        n_samples: int
            Number of samples
        norm_param_list: list
            List of normalized parameter values (usually from ``extract_parameters`` in ``MEA``).
        """
        super().__init__(n_samples, norm_param_list)

    def sample(self):
        """
        Randomly samples the design space.

        Returns
        -------
        norm_param_list: np.ndarray
            1-D array of normalized parameter values (usually from ``extract_parameters`` in ``MEA``).
        """
        problem = Problem(n_var=len(self.norm_param_list), l=0.0, xu=1.0)
        sampling = LatinHypercubeSampling()
        return sampling._do(problem, self.n_samples)
