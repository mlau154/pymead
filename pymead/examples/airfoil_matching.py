import numpy as np


def repair_example_sd7062(xy: np.ndarray):
    r"""
    ### Description:

    An example of how to repair a poorly-defined set of discrete airfoil coordinates.

    ### Args:

    `xy`: `np.ndarray` of `shape=(N, 2)` describing the set of "discrete" airfoil coordinates, where `N` is the number
    of coordinates, and the columns represent \(x\) and \(y\)

    ### Returns:

    The modified xy array
    """
    xy[-1, 0] = 1
    xy[-1, 1] = 0
    return xy
