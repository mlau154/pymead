import requests
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import typing


def extract_data_from_airfoiltools(name: str, repair: typing.Callable or None = None):
    """
    ### Description:

    Extracts the xy-coordinates of a specified airfoil from [Airfoil Tools](http://airfoiltools.com/) and
    returns the xy-coordinates as a numpy array of `shape=(N,2)` where `N` is the number of airfoil coordinates, and the
    columns represent `x` and `y`

    ### Args:

    `name`: Name of the airfoil to be requested from [Airfoil Tools](http://airfoiltools.com/). The name must
    exactly match the airfoil name inside the parentheses on [Airfoil Tools](http://airfoiltools.com/). For example,
    `"naca0012"` does not work, but `"n0012-il"` does.

    `repair`: `callable` that makes modifications to the output set of xy-coordinates

    ### Returns:

    The set of xy-coordinates of type `numpy.ndarray` with `shape=(N,2)`, where `N` is the number of airfoil coordinates
    and the columns represent `x` and `y`
    """
    url = f'http://airfoiltools.com/airfoil/seligdatfile?airfoil={name}'
    data = requests.get(url)
    text = deepcopy(data.text)
    coords_str = text.split('\n')
    xy_str_list = [coord_str.split() for coord_str in coords_str]
    xy_list = []
    for xy_str in xy_str_list[1:-1]:
        xy = [float(xy_str[0]), float(xy_str[1])]
        xy_list.append(xy)
    xy = np.array(xy_list)
    xy = repair(xy)
    return xy


def main():
    """
    ### Description:

    Example usage of `extract_data_from_airfoiltools`: download the `NACA 0012` airfoil and
    plot
    """
    xy = extract_data_from_airfoiltools('n0012-il')
    plt.plot(xy[:, 0], xy[:, 1], marker='*')
    plt.show()


if __name__ == '__main__':
    main()
