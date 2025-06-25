import lsv_panel
import numpy as np


def single_element_inviscid(coord: np.ndarray, alpha: float):
    r"""
    A linear strength vortex panel method for the inviscid solution of a single airfoil, sped up using the
    just-in-time compiler in *numba*. Directly adapted from "Program 7" of Appendix D in [1].

    [1] J. Katz and A. Plotkin, Low-Speed Aerodynamics, Second Edition, 2nd ed. New York, NY,
    USA: Cambridge University Press, 2004. Accessed: Mar. 07, 2023. [Online].
    Available: `<https://asmedigitalcollection.asme.org/fluidsengineering/article/126/2/293/458666/LowSpeed-Aerodynamics-Second-Edition>`_

    Parameters
    ----------
    coord: np.ndarray
        An :math:`N \times 2` array of airfoil coordinates, where :math:`N` is the number of coordinates, and the columns
        represent :math:`x` and :math:`y`

    alpha: float
        Angle of attack of the airfoil in degrees

    Returns
    -------
    np.ndarray, np.ndarray, float
        The first returned array is of size :math:`(N-1) \times 2` and represents the :math:`x`- and :math:`y`-locations
        of the collocation points, where :math:`N` is the number of airfoil coordinates. The second returned array is a
        one-dimensional array with length :math:`(N-1)` representing the surface pressure coefficient at each collocation
        point. The final returned value is the lift coefficient.
    """
    co, cp, cl = lsv_panel.solve(coord, alpha)
    return np.array(co), np.array(cp), cl
