import numpy as np
from shapely.geometry import LineString
import typing


def check_airfoil_self_intersection(coords: typing.Tuple[tuple]):
    """Determines whether the airfoil intersects itself using the `is_simple()` function of the
    `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

    Returns
    =======
    bool
      Describes whether the airfoil intersects itself
    """
    line_string = LineString(coords)
    is_simple = line_string.is_simple
    return not is_simple


def convert_numpy_array_to_shapely_points(arr: np.ndarray):
    return list(map(tuple, arr))


def convert_numpy_array_to_shapely_LineString(arr: np.ndarray):
    return LineString(list(map(tuple, arr)))


def map_angle_m180_p180(v: float):
    v = v % (2 * np.pi)
    if v > np.pi:
        v -= 2 * np.pi
    return v
