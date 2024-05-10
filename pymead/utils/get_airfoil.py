import typing
from copy import deepcopy

import numpy as np
import requests


def extract_data_from_airfoiltools(name: str, repair: typing.Callable or None = None) -> np.ndarray:
    r"""
    Extracts the xy-coordinates of a specified airfoil from [Airfoil Tools](http://airfoiltools.com/) and
    returns the xy-coordinates as a numpy array of `shape=(N,2)` where `N` is the number of airfoil coordinates, and the
    columns represent `x` and `y`

    Parameters
    ----------
    name: str
        Name of the airfoil to be requested from `Airfoil Tools <http://airfoiltools.com/>`_. The name must
        exactly match the airfoil name inside the parentheses on Airfoil Tools.
        For example, ``"naca0012"`` does not work, but ``"n0012-il"`` does.
    repair: typing.Callable or None
        An optional function that takes that makes modifications to the set of :math:`xy`-coordinates loaded from
        Airfoil Tools. This function should take exactly one input (the :math:`N \times 2` ``numpy.ndarray``
        representing the :math:`xy`-coordinates downloaded from Airfoil Tools) and return this array as the output.
        Default: ``None``

    Returns
    -------
    np.ndarray
        The set of :math:`xy`-coordinates of type ``numpy.ndarray`` with ``shape=(N,2)``, where ``N`` is the number of
        airfoil coordinates and the columns represent :math:`x` and :math:`y`
    """
    url = f'http://airfoiltools.com/airfoil/seligdatfile?airfoil={name}'
    data = requests.get(url)
    if data.status_code == 404:
        raise AirfoilNotFoundError(f"Airfoil {name} not found at http://airfoiltools.com/")
    text = deepcopy(data.text)
    coords_str = text.split('\n')
    xy_str_list = [coord_str.split() for coord_str in coords_str]
    xy = np.array([[float(xy_str[0]), float(xy_str[1])] for xy_str in xy_str_list[1:-1]])
    if repair is not None:
        xy = repair(xy)
    return xy


class AirfoilNotFoundError(Exception):
    pass
