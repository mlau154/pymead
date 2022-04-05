import numpy as np


class Param:

    def __init__(self, value: float, units: str or None = None,
                 bounds: list or np.ndarray = np.array([-np.inf, np.inf]),
                 length_scale_dimension: float or None = None, active: bool = True, linked: bool = False):
        """
        ### Description:

        This is the class used to define parameters used for the airfoil and airfoil parametrization definitions
        in `pyairpar`.

        ### Args:

        `value`: a `float` representing the value of the parameter

        `units`: a `str` where, if not set to `None`, scales the parameters by the value contained in
        `length_scale_dimension`. Must be one of `"length"`, `"inverse-length"`, or `None`. Default: `None`.

        `bounds`: a `list` or 1D `np.ndarray` with two elements of the form `[<lower bound>, <upper bound>]`. Used in
        `pyairpar.utils.airfoil_matching` and for normalization during parameter extraction. Default:
        `np.array([-np.inf, np.inf])` (no normalization).

        `length_scale_dimension`: length scale used to non-dimensionalize the parameter if `units` is not `None`.
        Default value: `None`.

        `active`: a `bool` stating whether the parameter is active (used in parameter extraction: if inactive,
        the parameter will not be extracted). Default: `True`.

        `linked`: a `bool` stating whether the parameter is linked to/set by another parameter. If `True`, the
        parameter will not be extracted). Default: `False`.

        ### Returns:

        An instance of the `pyairpar.core.param.Param` class.
        """

        self.units = units
        self.length_scale_dimension = length_scale_dimension

        if self.units == 'length' and self.length_scale_dimension is not None:
            self.value = value * self.length_scale_dimension
        elif self.units == 'inverse-length' and self.length_scale_dimension is not None:
            self.value = value / self.length_scale_dimension
        else:
            self.value = value

        self.bounds = bounds
        self.active = active
        self.linked = linked
