
import os
import re
from copy import deepcopy

import numpy as np
from pymoo.core.problem import Problem
from pymoo.decomposition.asf import ASF
from pymoo.util.display import Display
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination

multi_point_keys_mses = {
    0: 'MACHIN',
    1: 'REYNIN',
    2: 'ALFAIN',
    3: 'CLIFIN',
    4: 'P',
    5: 'T',
    6: 'L',
    7: 'R',
    8: 'rho',
    9: 'gam',
    10: 'ACRIT',
    11: 'MCRIT',
    12: 'MUCON',
    13: 'XTRSupper',
    14: 'XTRSlower',
    15: 'ISDELH',
    16: 'XCDELH',
    17: 'PTRHIN',
    18: 'ETAH',
    19: 'ISMOM',
    20: 'IFFBC',
}

multi_point_keys_xfoil = {
    0: "Ma",
    1: "Re",
    2: "alfa",
    3: "Cl",
    4: "CLI",
    5: "P",
    6: "T",
    7: "L",
    8: "R",
    9: "rho",
    10: "gam",
    11: "N",
    12: "xtr_upper",
    13: "xtr_lower",
}


def read_stencil_from_array(data: np.ndarray, tool: str):
    """Read the Multi-Point stencil from a text file and converts it to a list of dictionaries"""
    if tool == "MSES":
        multi_point_keys = multi_point_keys_mses
    elif tool == "XFOIL":
        multi_point_keys = multi_point_keys_xfoil
    else:
        raise ValueError("Either 'MSES' or 'XFOIL' must be selected for the tool in the multipoint stencil")

    if data.ndim == 1:
        return [{"variable": multi_point_keys[int(data[0])], "index": int(data[1]), "points": data[2:].tolist()}]
    elif data.ndim == 2:
        return [{'variable': multi_point_keys[int(col[0])], 'index': int(col[1]),
                 'points': col[2:].tolist()}
                for col in data.T]
    else:
        raise ValueError(f"Discovered multipoint data that was not 1-D or 2-D. {data.ndim = }")


def termination_condition(param_dict: dict):
    """
    Termination criteria for the optimization.

    Parameters
    ==========
    param_dict: dict
        Parameter dictionary for the optimization

    pymoo.util.termination.default.MultiObjectiveDefaultTermination
        Termination object used to define convergence for the genetic algorithm
    """
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


def calculate_warm_start_index(warm_start_generation: int, warm_start_directory: str):
    """
    Calculates the generation from which to restart the optimization. If the specified warm start generation is not
    found in the ``algorithm_gen_xxx.pkl`` files, an error is raised.

    Parameters
    ==========
    warm_start_generation: int
        Generation from which to restart the optimization. If this value is ``-1``, the optimization will start
        from the most recent point.

    warm_start_directory: str
        Path to the optimization directory containing the ``algorithm_gen_xxx.pkl`` files

    Returns
    =======
    int
        The index of the generation to start from.
    """
    generations = []
    for root, _, files in os.walk(warm_start_directory):
        for idx, f in enumerate(files):
            file_without_ext = os.path.splitext(f)[0]
            if re.split('_', file_without_ext)[0] in ["alg", "algorithm"]:
                idx = re.split('_', file_without_ext)[-1]
                generations.append(int(idx))
    generations = sorted(generations)
    if warm_start_generation not in generations and warm_start_generation != -1:
        raise ValueError(f'Invalid warm start generation. A value of {warm_start_generation} was input, and the valid '
                         f'generations in the directory are {generations}')
    if len(generations) > 0:
        warm_start_index = generations[warm_start_generation]
    else:
        warm_start_index = 0

    return warm_start_index


class PymeadGAProblem(Problem):
    def __init__(self, n_var: int, n_obj: int, n_constr: int, xl: int or list or np.ndarray,
                 xu: int or list or np.ndarray, param_dict: dict):
        """
        Simple problem statement for the ``pymoo`` package.

        Parameters
        ==========
        n_var: int
            Number of design variables (equal to the length of the normalized paramter list)

        n_obj: int
            Number of objectives

        n_constr: int
            Number of constraints

        xl: int or list or np.ndarray
            Lower bounds for the parameters. If ``int``, all lower bounds are equal.

        xu: int or list or np.ndarray
            Upper bounds for the parameters. If ``int``, all lower bounds are equal.

        param_dict: dict
            Parameter dictionary used for the shape optimization.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

        self.param_dict = deepcopy(param_dict)

    def _evaluate(self, X, out, *args, **kwargs):
        pass


class TPAIOPT(PymeadGAProblem):  # Included for legacy serialized algorithm data import
    pass


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
        G = algorithm.pop.get("G")
        # CV = algorithm.pop.get("CV")
        weights = np.array([0.5, 0.5])
        decomp = ASF()
        I = decomp.do(F, weights).argmin()
        n_nds = len(algorithm.opt)
        f_best = [F[I][i] for i in range(F.shape[1])]
        f_min = [F[:, i].min() for i in range(F.shape[1])]
        f_mean = [np.mean(F[:, i]) for i in range(F.shape[1])]
        g_best, g_min, g_mean = None, None, None
        if G[0] is not None:
            g_best = [G[I][i] for i in range(G.shape[1])]
            g_min = [G[:, i].min() for i in range(G.shape[1])]
            # cv_min = CV[:, 0].min()
            g_mean = [np.mean(G[:, i]) for i in range(G.shape[1])]
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
        for i, f in enumerate(f_best):
            self.output.append(f"f{i + 1}_best", f)
        for i, f in enumerate(f_min):
            self.output.append(f"f{i + 1}_min", f)
        for i, f in enumerate(f_mean):
            self.output.append(f"f{i + 1}_mean", f)

        if G[0] is not None:
            for i, g in enumerate(g_best):
                self.output.append(f"g{i + 1}_best", g)
            for i, g in enumerate(g_min):
                self.output.append(f"g{i + 1}_min", g)
            for i, g in enumerate(g_mean):
                self.output.append(f"g{i + 1}_mean", g)

        self.set_progress_dict(self.output.attrs)
