import shapely.errors

from utils.get_airfoil import extract_data_from_airfoiltools
import numpy as np
from core.param import Param
from core.anchor_point import AnchorPoint
from core.free_point import FreePoint
from core.airfoil import Airfoil
from core.base_airfoil_params import BaseAirfoilParams
from symmetric.symmetric_airfoil import SymmetricAirfoil
from symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from shapely.geometry import Polygon


def airfoil_symmetric_area_difference(parameters, airfoil: Airfoil, airfoil_to_match_xy: np.ndarray):
    #
    # if plot:
    #     fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False)
    #     axs.plot(xy[:, 0], xy[:, 1], color='green')
    #     plt.show()
    airfoil.override(parameters)
    try:
        airfoil_shapely_points = list(map(tuple, airfoil.coords))
        airfoil_polygon = Polygon(airfoil_shapely_points)
        airfoil_to_match_shapely_points = list(map(tuple, airfoil_to_match_xy))
        airfoil_to_match_polygon = Polygon(airfoil_to_match_shapely_points)
        symmetric_area_difference_polygon = airfoil_polygon.symmetric_difference(airfoil_to_match_polygon)
        symmetric_area_difference = symmetric_area_difference_polygon.area
        print(symmetric_area_difference)
    except shapely.errors.TopologicalError:
        symmetric_area_difference = 1

    return symmetric_area_difference


def match_airfoil(airfoil: Airfoil, airfoil_to_match: str):
    airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match)
    airfoil_to_match_xy[-1, 0] = 1
    airfoil_to_match_xy[-1, 1] = 0
    # initial_guess = np.array([0.03, 0.06, 5.0, 0.3, 10.0, 0.2, 0.11, 0.3, 0.10, 0.4, 0.10])
    # bounds = np.array([[1e-4, 0.05], [1e-4, 0.2], [-45, 45], [1e-4, 0.6], [1e-4, 20], [0.1, 0.3], [1e-4, 0.2],
    #                    [0.2, 0.4], [1e-4, 2], [0.3, 0.5], [1e-4, 0.2]])
    initial_guess = airfoil.params
    bounds = airfoil.bounds
    res = minimize(airfoil_symmetric_area_difference, initial_guess, method='SLSQP',
                   bounds=bounds, args=(airfoil, airfoil_to_match_xy))
    return res


def main():
    airfoil_to_match = 'sd7062-il'
    airfoil_to_match_xy = extract_data_from_airfoiltools(airfoil_to_match)
    airfoil_to_match_xy[-1, 0] = 1
    airfoil_to_match_xy[-1, 1] = 0
    print(airfoil_to_match_xy)

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

    points_shapely = list(map(tuple, airfoil.coords))
    polygon = Polygon(points_shapely)
    points_shapely2 = list(map(tuple, airfoil_to_match_xy))
    polygon2 = Polygon(points_shapely2)
    print(polygon.area)
    print(polygon2.area)
    # new_polygon = polygon.symmetric_difference(polygon2)
    # area = new_polygon.area
    # print(f"area = {area}")

    res = match_airfoil(airfoil, airfoil_to_match=airfoil_to_match)
    print(res)

    fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False,
                            plot_kwargs=[{'marker': '*'}, {'marker': '*'}])
    axs.plot(airfoil_to_match_xy[:, 0], airfoil_to_match_xy[:, 1], color='green', marker='x')
    plt.show()


if __name__ == '__main__':
    main()
