import shapely.errors

from utils.get_airfoil import extract_data_from_airfoiltools
import numpy as np
from core.param import Param
from core.anchor_point import AnchorPoint
from core.free_point import FreePoint
from symmetric.symmetric_airfoil import SymmetricAirfoil
from symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from shapely.geometry import Polygon


def try_to_match_airfoil(param_list: np.ndarray, plot: bool = False):
    print(param_list)
    xy = extract_data_from_airfoiltools('n0009sm-il')
    c = 1.0

    base_airfoil_params = SymmetricBaseAirfoilParams(c=Param(c),
                                                     alf=Param(np.deg2rad(0.0)),
                                                     R_le=Param(param_list[0], 'length', c=c),
                                                     L_le=Param(param_list[1], 'length', c=c),
                                                     psi1_le=Param(np.deg2rad(param_list[2])),
                                                     L1_te=Param(param_list[3], 'length', c=c),
                                                     theta1_te=Param(np.deg2rad(param_list[4])),
                                                     t_te=Param(0.00252, 'length', c=c),
                                                     )

    free_point1 = FreePoint(x=param_list[5], y=param_list[6], previous_anchor_point='te_1', scale_dimension=c)
    free_point2 = FreePoint(x=param_list[7], y=param_list[8], previous_anchor_point='te_1', scale_dimension=c)
    free_point3 = FreePoint(x=param_list[9], y=param_list[10], previous_anchor_point='te_1', scale_dimension=c)
    free_point_tuple = (free_point1, free_point2, free_point3)

    airfoil = SymmetricAirfoil(number_coordinates=100,
                               base_airfoil_params=base_airfoil_params,
                               anchor_point_tuple=(),
                               free_point_tuple=free_point_tuple)
    if plot:
        fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False)
        axs.plot(xy[:, 0], xy[:, 1], color='green')
        plt.show()

    try:
        points_shapely = list(map(tuple, airfoil.coords))
        polygon = Polygon(points_shapely)
        points_shapely2 = list(map(tuple, xy))
        polygon2 = Polygon(points_shapely2)
        new_polygon = polygon.symmetric_difference(polygon2)
        area = new_polygon.area
        print(area)
    except shapely.errors.TopologicalError:
        area = 1

    return area


def match_airfoil():
    initial_guess = np.array([0.03, 0.06, 5.0, 0.3, 10.0, 0.2, 0.11, 0.3, 0.10, 0.4, 0.10])
    bounds = np.array([[1e-4, 0.05], [1e-4, 0.2], [-45, 45], [1e-4, 0.6], [1e-4, 20], [0.1, 0.3], [1e-4, 0.2],
                       [0.2, 0.4], [1e-4, 2], [0.3, 0.5], [1e-4, 0.2]])
    res = minimize(try_to_match_airfoil, initial_guess, method='SLSQP', bounds=bounds)
    print(res)
    try_to_match_airfoil(res.x, plot=True)


def main():
    xy = extract_data_from_airfoiltools('n0012-il')
    c = 1.0

    base_airfoil_params = SymmetricBaseAirfoilParams(c=Param(c),
                                                     alf=Param(np.deg2rad(0.0)),
                                                     R_le=Param(0.03, 'length', c=c),
                                                     L_le=Param(0.06, 'length', c=c),
                                                     psi1_le=Param(np.deg2rad(5.0)),
                                                     L1_te=Param(0.3, 'length', c=c),
                                                     theta1_te=Param(np.deg2rad(10.0)),
                                                     t_te=Param(0.002, 'length', c=c),
                                                     )

    free_point1 = FreePoint(x=0.2, y=0.11, previous_anchor_point='te_1', scale_dimension=c)

    free_point_tuple = (free_point1,)

    airfoil = SymmetricAirfoil(number_coordinates=100,
                               base_airfoil_params=base_airfoil_params,
                               anchor_point_tuple=(),
                               free_point_tuple=free_point_tuple)

    fig, axs = airfoil.plot(('airfoil',), show_plot=False, show_legend=False,
                            plot_kwargs=[{'marker': '*'}, {'marker': '*'}])
    axs.plot(xy[:, 0], xy[:, 1], color='green', marker='x')
    plt.show()

    points_shapely = list(map(tuple, airfoil.coords))
    polygon = Polygon(points_shapely)
    points_shapely2 = list(map(tuple, xy))
    polygon2 = Polygon(points_shapely2)
    new_polygon = polygon.symmetric_difference(polygon2)
    area = new_polygon.area
    print(f"area = {area}")


def main2():
    match_airfoil()


def main3():
    param_list = np.array([1.00000000e-04, 4.84560773e-02, 5.00061985e+00, 2.58156903e-01,
                           9.99681835e+00, 1.85686695e-01, 3.06493223e-02, 3.13276176e-01,
                           6.57763130e-02, 4.25340698e-01, 8.53589806e-02])
    try_to_match_airfoil(param_list=param_list, plot=True)


if __name__ == '__main__':
    main2()
