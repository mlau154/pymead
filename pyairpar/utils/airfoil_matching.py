import shapely.errors

from pyairpar.utils.get_airfoil import extract_data_from_airfoiltools
import numpy as np
from pyairpar.core.param import Param
from pyairpar.core.free_point import FreePoint
from pyairpar.core.airfoil import Airfoil
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from shapely.geometry import Polygon


def airfoil_symmetric_area_difference(parameters: list, airfoil: Airfoil, airfoil_to_match_xy: np.ndarray):
    """
    ### Description:

    This method uses the shapely package to convert the parametrized airfoil and the "discrete" airfoil to
    [Polygon](https://shapely.readthedocs.io/en/stable/manual.html#Polygon)
    objects and calculate the boolean symmetric area difference (a similarity metric) between the two airfoils
    Shapely documentation for Polygon class:
        https://shapely.readthedocs.io/en/stable/manual.html#Polygon
    Shapely documentation for object.symmetric_difference() method:
        https://shapely.readthedocs.io/en/stable/manual.html#object.symmetric_difference

    ### Args:

    `parameters`: A list of parameters used to override the `Param` values in the `pyairpar.core.airfoil.Airfoil` class
    `airfoil`: instance of the Airfoil (or any sub-class) used to describe the airfoil parametrization
    `airfoil_to_match_xy`: `numpy.ndarray` of `shape=(N, 2)` describing the "discrete" airfoil to match, where N
    is the number of coordinates, and the columns represent `x` and `y`

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


def match_airfoil(airfoil: Airfoil, airfoil_to_match: str or list or np.ndarray):
    """
    This method uses the scipy.optimize.minimize(method='SLSQP') optimization scheme to minimize the boolean symmetric
    area difference between the input Airfoil and set of "discrete" airfoil coordinates
    :param airfoil: instance of the Airfoil class (or any sub-class) which describes the parametrized airfoil. The
    initial guess for the optimizer is the set of input Param values set in the Airfoil instance. The bounds for the
    optimizer must be set by also applying inputs to the Param.bounds in the Airfoil instance.
    :param airfoil_to_match: set of xy airfoil coordinates to be matched, or a string representing the name of the
    airfoil to be fetched from http://airfoiltools.com/
    :return:
    """
    if isinstance(airfoil_to_match, str):
        airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match, repair=repair_example_sd7062)
    elif isinstance(airfoil_to_match, list):
        airfoil_to_match_xy = np.array(airfoil_to_match)
    elif isinstance(airfoil_to_match, np.ndarray):
        airfoil_to_match_xy = airfoil_to_match
    else:
        raise TypeError(f'airfoil_to_match be of type str, list, or np.ndarray, '
                        f'and type {type(airfoil_to_match)} was used')
    initial_guess = airfoil.params
    bounds = airfoil.bounds
    res = minimize(airfoil_symmetric_area_difference, initial_guess, method='SLSQP',
                   bounds=bounds, args=(airfoil, airfoil_to_match_xy))
    return res


def repair_example_sd7062(xy: np.ndarray):
    """
    An example of how to repair a poorly-defined set of discrete airfoil coordinates.
    :param xy: np.ndarray of shape=(N, 2) describing the set of "discrete" airfoil coordinates, where N is the number
    of coordinates, and the columns represent x and y
    :return: the modified xy array
    """
    xy[-1, 0] = 1
    xy[-1, 1] = 0
    return xy


def main():
    """
    An example of an implementation of match_airfoil() using the sd7062-il airfoil. Note that the match is imperfect
    because the set of discrete airfoil coordinates does not have its leading edge at exactly (0,0), and the trailing
    edge had to be repaired because of a self-intersection. Future implementations of match_airfoil() may attempt to
    address these issues.
    :return:
    """
    airfoil_to_match = 'sd7062-il'
    airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match, repair=repair_example_sd7062)
    base_airfoil_params = BaseAirfoilParams(c=Param(1.0, active=False),
                                            alf=Param(np.deg2rad(0.0), active=False),
                                            R_le=Param(0.06, 'length', bounds=np.array([1e-5, 0.3])),
                                            L_le=Param(0.08, 'length', bounds=np.array([1e-5, 0.2])),
                                            r_le=Param(0.6, bounds=np.array([0.01, 0.99])),
                                            phi_le=Param(np.deg2rad(5.0), bounds=np.array([-np.pi/4, np.pi/4])),
                                            psi1_le=Param(np.deg2rad(10.0), bounds=np.array([-np.pi/2.1, np.pi/2.1])),
                                            psi2_le=Param(np.deg2rad(15.0), bounds=np.array([-np.pi/2.1, np.pi/2.1])),
                                            L1_te=Param(0.25, 'length', bounds=np.array([1e-5, 1.0])),
                                            L2_te=Param(0.3, 'length', bounds=np.array([1e-5, 1.0])),
                                            theta1_te=Param(np.deg2rad(14.0), bounds=np.array([-np.pi/2.1, np.pi/2.1])),
                                            theta2_te=Param(np.deg2rad(1.0), bounds=np.array([-np.pi/2.1, np.pi/2.1])),
                                            t_te=Param(0.0, 'length', active=False),
                                            r_te=Param(0.5, active=False),
                                            phi_te=Param(np.deg2rad(0.0), active=False),
                                            non_dim_by_chord=True
                                            )

    free_point1 = FreePoint(x=Param(0.35, 'length', bounds=np.array([0.05, 0.95])),
                            y=Param(0.2, 'length', bounds=np.array([0.05, 0.95])), previous_anchor_point='te_1',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point2 = FreePoint(x=Param(0.5, 'length', bounds=np.array([0.05, 0.95])),
                            y=Param(-0.05, 'length', bounds=np.array([0.05, 0.95])), previous_anchor_point='le',
                            length_scale_dimension=base_airfoil_params.c.value)

    free_point_tuple = (free_point1, free_point2)

    airfoil = Airfoil(number_coordinates=100,
                      base_airfoil_params=base_airfoil_params,
                      anchor_point_tuple=(),
                      free_point_tuple=free_point_tuple)

    fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False,
                            plot_kwargs=[{'marker': '*'}, {'marker': '*'}])
    axs.plot(airfoil_to_match_xy[:, 0], airfoil_to_match_xy[:, 1], color='green', marker='x')
    plt.show()

    res = match_airfoil(airfoil, airfoil_to_match=airfoil_to_match)
    print(res)

    fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False,
                            plot_kwargs=[{'marker': '*'}, {'marker': '*'}])
    axs.plot(airfoil_to_match_xy[:, 0], airfoil_to_match_xy[:, 1], color='green', marker='x')
    plt.show()


if __name__ == '__main__':
    main()
