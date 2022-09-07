import numpy as np
from decimal import Decimal


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
