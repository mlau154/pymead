import numpy as np
from decimal import Decimal


def nchoosek(n: int, k: int):
    r"""
    Computes the mathematical combination

    .. math::

      n \choose k

    Parameters
    ==========
    n: int
      Number of elements in the set

    k: int
      Number of items to select from the set

    Returns
    =======
    :math:`n \choose k`
    """
    f = np.math.factorial
    return float(Decimal(f(n)) / Decimal(f(k)) / Decimal(f(n - k)))
