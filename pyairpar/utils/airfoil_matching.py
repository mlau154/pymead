import typing

import shapely.errors

import numpy as np

from scipy.optimize import minimize
from shapely.geometry import Polygon

from pyairpar.utils.get_airfoil import extract_data_from_airfoiltools
from pyairpar.core.airfoil import Airfoil


def airfoil_symmetric_area_difference(parameters: list, airfoil: Airfoil, airfoil_to_match_xy: np.ndarray):
    r"""
    ### Description:

    This method uses the shapely package to convert the parametrized airfoil and the "discrete" airfoil to
    [Polygon](https://shapely.readthedocs.io/en/stable/manual.html#Polygon)
    objects and calculate the boolean
    [symmetric difference](https://shapely.readthedocs.io/en/stable/manual.html#object.symmetric_difference)
    (a similarity metric) between the two airfoils

    ### Args:

    `parameters`: A list of parameters used to override the `pyairpar.core.param.Param` values in the
    `pyairpar.core.airfoil.Airfoil` class

    `airfoil`: instance of the Airfoil (or any sub-class) used to describe the airfoil parametrization

    `airfoil_to_match_xy`: [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) of
    `shape=(N, 2)` describing the "discrete" airfoil to match, where `N` is the number of coordinates,
    and the columns represent \(x\) and \(y\)

    ### Returns:

    The boolean symmetric area difference between the parametrized airfoil and the discrete airfoil
    """

    # Override airfoil parameters with supplied sequence of parameters
    airfoil.override(parameters)

    # Calculate the boolean symmetric area difference. If there is a topological error, such as a self-intersection,
    # which prevents Polygon.area() from running, then make the symmetric area difference a large value to discourage
    # the optimizer from continuing in that direction.
    try:
        airfoil_shapely_points = list(map(tuple, airfoil.coords))
        airfoil_polygon = Polygon(airfoil_shapely_points)
        airfoil_to_match_shapely_points = list(map(tuple, airfoil_to_match_xy))
        airfoil_to_match_polygon = Polygon(airfoil_to_match_shapely_points)
        symmetric_area_difference_polygon = airfoil_polygon.symmetric_difference(airfoil_to_match_polygon)
        symmetric_area_difference = symmetric_area_difference_polygon.area
        print(symmetric_area_difference)
    except shapely.errors.TopologicalError:
        symmetric_area_difference = 1  # Set the boolean symmetric area difference to a large value

    return symmetric_area_difference


def match_airfoil(airfoil: Airfoil, airfoil_to_match: str or list or np.ndarray,
                  repair: typing.Callable or None = None):
    r"""
    ### Description:

    This method uses the [`scipy.optimize.minimize(method='SLSQP')`](
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html) optimization scheme to minimize the
    boolean symmetric area difference between the input `pyairpar.core.airfoil.Airfoil` and set of "discrete" airfoil
    coordinates

    ### Args:

    `airfoil`: instance of the `pyairpar.core.airfoil.Airfoil` class (or any sub-class) which describes the
    parametrized airfoil. The initial guess for the optimizer is the set of input Param values set in the
    `pyairpar.core.airfoil.Airfoil` instance. The bounds for the optimizer must be set by also supplying inputs to
    the `pyairpar.core.param.Param`'s `bounds` attribute in the `pyairpar.core.airfoil.Airfoil` instance.

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
    initial_guess = np.array([param.value for param in airfoil.params])
    bounds = [param.bounds for param in airfoil.params]
    res = minimize(airfoil_symmetric_area_difference, initial_guess, method='SLSQP',
                   bounds=bounds, args=(airfoil, airfoil_to_match_xy), options={'disp': True})
    return res
