import typing
from copy import deepcopy

import numpy as np
import shapely.errors
from scipy.optimize import minimize
from shapely.geometry import Polygon

from pymead.core.geometry_collection import GeometryCollection
from pymead.utils.get_airfoil import extract_data_from_airfoiltools


def airfoil_symmetric_area_difference(parameters: list, geo_col: GeometryCollection, target_airfoil: str,
                                      airfoil_to_match_xy: np.ndarray):
    r"""
    This method uses the ``shapely`` package to convert the parametrized airfoil and the "discrete" airfoil to
    `Polygon <https://shapely.readthedocs.io/en/stable/manual.html#Polygon>`_
    objects and calculate the boolean
    `symmetric difference <https://shapely.readthedocs.io/en/stable/manual.html#object.symmetric_difference>`_
    (a similarity metric) between the two airfoils.

    Parameters
    ----------
    parameters: list
        A list of parameter values used to override the design variable values found in the geometry collection.

    geo_col: GeometryCollection
        Geometry collection from which the ``Airfoil`` is selected by the ``target_airfoil`` name and where the
        design variables are stored. During the optimization, any values in the design variable sub-container will
        be updated to produce an airfoil geometry that closely matches the airfoil coordinates specified by
        ``airfoil_to_match``.

    target_airfoil: str
        Airfoil from the geometry collection to match. For example, use ``"Airfoil-1"`` to match the airfoil
        with the same name found in the airfoil sub-container of the geometry collection. Only one airfoil may be
        matched at a time.

    airfoil_to_match_xy: np.ndarray
        Set of airfoil :math:`xy`-coordinates to be matched

    Returns
    -------
    float
        The boolean symmetric area difference between the parametrized airfoil and the discrete airfoil
    """

    # Override airfoil parameters with supplied sequence of parameters
    geo_col.assign_design_variable_values(parameters, bounds_normalized=True)
    airfoil = geo_col.container()["airfoils"][target_airfoil]
    coords = airfoil.get_coords_selig_format()

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


def match_airfoil(geo_col: GeometryCollection, target_airfoil: str, airfoil_to_match: str or np.ndarray,
                  repair: typing.Callable or None = None):
    r"""
    This method uses the `sequential least-squares programming
    <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html>`_ optimization scheme to minimize the
    boolean symmetric area difference between the input ``Airfoil`` and a set of "discrete" airfoil
    coordinates

    Parameters
    ----------
    geo_col: GeometryCollection
        Geometry collection from which the ``Airfoil`` is selected by the ``target_airfoil`` name and where the
        design variables are stored. During the optimization, any values in the design variable sub-container will
        be updated to produce an airfoil geometry that closely matches the airfoil coordinates specified by
        ``airfoil_to_match``.

    target_airfoil: str
        Airfoil from the geometry collection to match. For example, use ``"Airfoil-1"`` to match the airfoil
        with the same name found in the airfoil sub-container of the geometry collection. Only one airfoil may be
        matched at a time.

    airfoil_to_match: str or np.ndarray
        Set of airfoil :math:`xy`-coordinates to be matched, or a string representing the name
        of the airfoil to be fetched from `Airfoil Tools <http://airfoiltools.com/>`_.

    repair: typing.Callable or None
        An optional function that takes that makes modifications to the set of :math:`xy`-coordinates loaded from
        Airfoil Tools. This function should take exactly one input (the :math:`N \times 2` ``numpy.ndarray``
        representing the :math:`xy`-coordinates downloaded from Airfoil Tools) and return this array as the output.
        Default: ``None``

    Returns
    -------
    scipy.optimize.OptimizeResult
        `Results object <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult>`_
        returned by the optimizer
    """
    if isinstance(airfoil_to_match, str):
        airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match, repair=repair)
    elif isinstance(airfoil_to_match, np.ndarray):
        airfoil_to_match_xy = airfoil_to_match
    else:
        raise TypeError(f'airfoil_to_match be of type str or np.ndarray, '
                        f'and type {type(airfoil_to_match)} was used')

    if target_airfoil not in geo_col.container()["airfoils"]:
        raise ValueError(f"Target airfoil {target_airfoil} not found in the specified geometry collection. Available"
                         f" airfoils in this geometry collection: "
                         f"{[k for k in geo_col.container()['airfoils'].keys()]}")

    if len(geo_col.container()["desvar"]) == 0:
        raise ValueError(f"No design variables were found in the geometry collection. Promote at least "
                         f"one parameter to a design variable to run an airfoil matching optimization.")

    geo_col_deepcopy = deepcopy(geo_col)
    initial_guess = np.array(geo_col.extract_design_variable_values())
    bounds = np.repeat(np.array([[0.0, 1.0]]), len(initial_guess), axis=0)
    res = minimize(
        airfoil_symmetric_area_difference,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        args=(geo_col_deepcopy, target_airfoil, airfoil_to_match_xy),
        options=dict(disp=True)
    )
    return res
