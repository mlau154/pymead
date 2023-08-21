import typing

import shapely.errors

import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import BSpline, splrep
import scipy
from shapely.geometry import Polygon

from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.core.mea import MEA

from copy import deepcopy

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize


def airfoil_symmetric_area_difference(parameters: list, mea: MEA, target_airfoil: str, airfoil_to_match_xy: np.ndarray):
    r"""
    ### Description:

    This method uses the shapely package to convert the parametrized airfoil and the "discrete" airfoil to
    [Polygon](https://shapely.readthedocs.io/en/stable/manual.html#Polygon)
    objects and calculate the boolean
    [symmetric difference](https://shapely.readthedocs.io/en/stable/manual.html#object.symmetric_difference)
    (a similarity metric) between the two airfoils

    ### Args:

    `parameters`: A list of parameters used to override the `pymead.core.param.Param` values in the
    `pymead.core.airfoil.Airfoil` class

    `airfoil`: instance of the Airfoil (or any sub-class) used to describe the airfoil parametrization

    `airfoil_to_match_xy`: [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) of
    `shape=(N, 2)` describing the "discrete" airfoil to match, where `N` is the number of coordinates,
    and the columns represent \(x\) and \(y\)

    ### Returns:

    The boolean symmetric area difference between the parametrized airfoil and the discrete airfoil
    """

    # Override airfoil parameters with supplied sequence of parameters
    mea.update_parameters(parameters)
    airfoil = mea.airfoils[target_airfoil]
    coords = airfoil.get_coords(body_fixed_csys=False)

    # Calculate the boolean symmetric area difference. If there is a topological error, such as a self-intersection,
    # which prevents Polygon.area() from running, then make the symmetric area difference a large value to discourage
    # the optimizer from continuing in that direction.
    try:
        airfoil_shapely_points = list(map(tuple, coords))
        airfoil_polygon = Polygon(airfoil_shapely_points)
        airfoil_to_match_shapely_points = list(map(tuple, airfoil_to_match_xy))
        airfoil_to_match_polygon = Polygon(airfoil_to_match_shapely_points)
        symmetric_area_difference_polygon = airfoil_polygon.symmetric_difference(airfoil_to_match_polygon)
        symmetric_area_difference = symmetric_area_difference_polygon.area
        print(symmetric_area_difference)
    except shapely.errors.TopologicalError:
        symmetric_area_difference = 1  # Set the boolean symmetric area difference to a large value

    return symmetric_area_difference


def match_airfoil(mea: MEA, target_airfoil: str, airfoil_to_match: str or list or np.ndarray,
                  repair: typing.Callable or None = None):
    r"""
    ### Description:

    This method uses the [`scipy.optimize.minimize(method='SLSQP')`](
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) optimization scheme to minimize the
    boolean symmetric area difference between the input `pymead.core.airfoil.Airfoil` and set of "discrete" airfoil
    coordinates

    ### Args:

    `airfoil`: instance of the `pymead.core.airfoil.Airfoil` class (or any sub-class) which describes the
    parametrized airfoil. The initial guess for the optimizer is the set of input Param values set in the
    `pymead.core.airfoil.Airfoil` instance. The bounds for the optimizer must be set by also supplying inputs to
    the `pymead.core.param.Param`'s `bounds` attribute in the `pymead.core.airfoil.Airfoil` instance.

    `airfoil_to_match`: set of \(x\) - \(y\) airfoil coordinates to be matched, or a string representing the name
    of the airfoil to be fetched from [Airfoil Tools](http://airfoiltools.com/).

    ### Returns:

    The [`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) result.
    """
    if isinstance(airfoil_to_match, str):
        airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match, repair=repair)
    elif isinstance(airfoil_to_match, list):
        airfoil_to_match_xy = np.array(airfoil_to_match)
    elif isinstance(airfoil_to_match, np.ndarray):
        airfoil_to_match_xy = airfoil_to_match
    else:
        raise TypeError(f'airfoil_to_match be of type str, list, or np.ndarray, '
                        f'and type {type(airfoil_to_match)} was used')

    # t, c, k = splrep(airfoil_to_match_xy[0:66, 0][::-1], airfoil_to_match_xy[0:66, 1][::-1], s=0, k=5)
    # spline = BSpline(t, c, k, extrapolate=False)
    # # hh = np.linspace(0, 1, 350)
    # hh = np.concatenate((np.linspace(0, 0.01, 75), np.linspace(0.01, 1.0)[1:]))
    # yy = spline(hh)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(hh, yy)
    # ax.plot(airfoil_to_match_xy[:, 0], airfoil_to_match_xy[:, 1], marker="*", ls="none", color="indianred")
    # ax.set_aspect("equal")
    # plt.show()

    # mea.deactivate_airfoil_matching_params(target_airfoil)
    new_mea = mea.deepcopy()
    new_mea.remove_airfoil_graphs()
    # new_mea = MEA.generate_from_param_dict(mea_dict)
    initial_guess = np.array(mea.extract_parameters()[0])
    # initial_guess = np.array([param.value for param in parameter_list])
    # bounds = [param.bounds for param in parameter_list]
    bounds = np.repeat(np.array([[0.0, 1.0]]), len(initial_guess), axis=0)
    # try:
    res = scipy.optimize.minimize(airfoil_symmetric_area_difference, initial_guess, method='SLSQP',
                   bounds=bounds, args=(new_mea, target_airfoil, airfoil_to_match_xy), options={'disp': True})
    # finally:
    #     mea.activate_airfoil_matching_params(target_airfoil)
    return res


class MatchAirfoilProblem(ElementwiseProblem):

    def __init__(self, n_var: int, mea: MEA, target_airfoil: str, airfoil_to_match_xy: np.ndarray):
        self.mea = mea
        self.airfoil_to_match_xy = airfoil_to_match_xy
        self.target_airfoil = target_airfoil
        super().__init__(n_var=n_var, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = airfoil_symmetric_area_difference(parameters=x.tolist(),
                                                     mea=self.mea,
                                                     target_airfoil='A0',
                                                     airfoil_to_match_xy=self.airfoil_to_match_xy)


def match_airfoil_ga(mea: MEA, target_airfoil: str, airfoil_to_match: str or list or np.ndarray,
                     repair: typing.Callable or None = None):

    if isinstance(airfoil_to_match, str):
        airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match, repair=repair)
    elif isinstance(airfoil_to_match, list):
        airfoil_to_match_xy = np.array(airfoil_to_match)
    elif isinstance(airfoil_to_match, np.ndarray):
        airfoil_to_match_xy = airfoil_to_match
    else:
        raise TypeError(f'airfoil_to_match be of type str, list, or np.ndarray, '
                        f'and type {type(airfoil_to_match)} was used')
    # mea.deactivate_airfoil_matching_params(target_airfoil)
    new_mea = mea.deepcopy()
    new_mea.remove_airfoil_graphs()
    # new_mea = MEA.generate_from_param_dict(mea_dict)
    initial_guess = np.array(mea.extract_parameters()[0])

    problem = MatchAirfoilProblem(n_var=len(initial_guess), mea=new_mea, target_airfoil=target_airfoil,
                                  airfoil_to_match_xy=airfoil_to_match_xy)

    algorithm = DE(
        pop_size=100,
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.3,
        dither="vector",
        jitter=False
    )

    res = minimize(problem,
                   algorithm,
                   seed=1,
                   verbose=True)

    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

    return res
