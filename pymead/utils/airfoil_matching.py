import typing

import shapely.errors

import numpy as np

from scipy.optimize import minimize
from shapely.geometry import Polygon

from pymead.utils.get_airfoil import extract_data_from_airfoiltools
from pymead.core.mea import MEA


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
    coords = airfoil.get_coords(body_fixed_csys=True)

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
    mea.deactivate_airfoil_matching_params(target_airfoil)
    _, parameter_list = mea.extract_parameters()
    initial_guess = np.array([param.value for param in parameter_list])
    bounds = [param.bounds for param in parameter_list]
    try:
        res = minimize(airfoil_symmetric_area_difference, initial_guess, method='SLSQP',
                       bounds=bounds, args=(mea, target_airfoil, airfoil_to_match_xy), options={'disp': True})
    finally:
        mea.activate_airfoil_matching_params(target_airfoil)
    return res


if __name__ == '__main__':
    from pymead.core.airfoil import Airfoil
    from pymead.core.base_airfoil_params import BaseAirfoilParams
    from pymead.core.param import Param
    base = BaseAirfoilParams(t_te=Param(0.005),
                             R_le=Param(0.1, bounds=np.array([0.02, 0.2])),
                             L_le=Param(0.1, bounds=np.array([0.01, 0.25])),
                             psi1_le=Param(0.0, bounds=np.array([-0.3, 0.4])),
                             psi2_le=Param(0.0, bounds=np.array([-0.3, 0.4])),
                             L1_te=Param(0.15, bounds=np.array([0.01, 0.5])),
                             L2_te=Param(0.15, bounds=np.array([0.01, 0.5])),
                             theta1_te=Param(0.05, bounds=np.array([0.0, 0.3])),
                             theta2_te=Param(0.05, bounds=np.array([0.0, 0.3]))
                             )
    base.phi_le.active = False
    base.r_le.active = False
    airfoil = Airfoil(base_airfoil_params=base)
    mea = MEA(None, airfoils=[airfoil], airfoil_graphs_active=False)
    match_airfoil(mea, 'A0', 'sc20010-il')
