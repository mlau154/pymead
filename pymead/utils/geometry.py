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
