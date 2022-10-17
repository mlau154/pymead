import numpy as np
from decimal import Decimal
from math import comb


def nchoosek(n, k):
    """
    ### Description:

    Simple function that computes the mathematical combination $$n \\choose k$$

    ### Args:

    `n`

    `k`

    ### Returns

    $$n \\choose k$$
    """
    f = np.math.factorial
    return float(Decimal(f(n)) / Decimal(f(k)) / Decimal(f(n - k)))


def nchoosek_matrix(n: np.ndarray or list, k: np.ndarray or list):
    out = np.zeros(shape=(len(n),))
    if len(n) != len(k):
        raise ValueError('Lengths of n and k arrays must be equal!')
    for idx in range(len(n)):
        out[idx] = comb(n[idx], k[idx])
    return out
