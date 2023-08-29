import typing

import numpy as np
from shapely.geometry import LineString


def check_airfoil_self_intersection(coords: typing.Tuple[tuple]):
    """Determines whether the airfoil intersects itself using the `is_simple()` function of the
    `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

    Parameters
    ==========
    coords: typing.Tuple[tuple]
        Set of :math:`x`-:math:`y` coordinates representing the airfoil shape, where each inner tuple is an
        :math:`x`-:math:`y` pair.

    Returns
    =======
    bool
      Describes whether the airfoil intersects itself
    """
    line_string = LineString(coords)
    is_simple = line_string.is_simple
    return not is_simple


def convert_numpy_array_to_shapely_points(arr: np.ndarray):
    """
    Converts a 2-D numpy array of :math:`x`-:math:`y` coordinates to the format used by the
    `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

    Parameters
    ==========
    arr: np.ndarray
        2-D array of :math:`x`-:math:`y` coordinates

    Returns
    =======
    typing.List[tuple]
        List of tuples where each tuple represents an :math:`x`-:math:`y` coordinate pair
    """
    return list(map(tuple, arr))


def convert_numpy_array_to_shapely_LineString(arr: np.ndarray):
    """
    Converts a 2-D numpy array of :math:`x`-:math:`y` coordinates to a
    `LineString <https://shapely.readthedocs.io/en/stable/reference/shapely.LineString.html>`_.

    Parameters
    ==========
    arr: np.ndarray
        2-D array of :math:`x`-:math:`y` coordinates

    Returns
    =======
    shapely.geometry.LineString
        Geometric object defined by a set of points connected, in order, by lines.
    """
    return LineString(list(map(tuple, arr)))


def map_angle_m180_p180(v: float):
    """
    Maps an angle from any value in radians to the range :math:`[-pi, pi]`

    Parameters
    ==========
    v: float
        Angle in radians

    Returns
    =======
    float
        Angle in radians between :math:`-pi` and :math:`pi`, inclusive
    """
    v = v % (2 * np.pi)
    if v > np.pi:
        v -= 2 * np.pi
    return v


def calculate_area_triangle_heron(a: float, b: float, c: float):
    """
    Heron's formula for area of a triangle

    Parameters
    ==========
    a: float
        Side length 1

    b: float
        Side length 2

    c: float
        Side length 3

    Returns
    =======
    float
        Area of the specified triangle
    """
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))
