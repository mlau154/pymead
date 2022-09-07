import numpy as np


class Param:

    def __init__(self, value: float, units: str or None = None,
                 bounds: list or np.ndarray = np.array([-np.inf, np.inf]), scale_value: float or None = None,
                 active: bool = True, linked: bool = False, name: str = None):
        """
        ### Description:

        This is the class used to define parameters used for the airfoil and airfoil parametrization definitions
        in `pymead`.

        ### Args:

        `value`: a `float` representing the value of the parameter

        `units`: a `str` where, if not set to `None`, scales the parameters by the value contained in
        `length_scale_dimension`. Must be one of `"length"`, `"inverse-length"`, or `None`. Default: `None`.

        `bounds`: a `list` or 1D `np.ndarray` with two elements of the form `[<lower bound>, <upper bound>]`. Used in
        `pymead.utils.airfoil_matching` and for normalization during parameter extraction. Default:
        `np.array([-np.inf, np.inf])` (no normalization).

        `scale_value`: length scale used to non-dimensionalize the parameter if `units` is not `None`.
        Default value: `None`.

        `active`: a `bool` stating whether the parameter is active (used in parameter extraction: if inactive,
        the parameter will not be extracted). Default: `True`.

        `linked`: a `bool` stating whether the parameter is linked to/set by another parameter. If `True`, the
        parameter will not be extracted). Default: `False`.

        `name`: an optional `str` that gives the name of the parameter. Can be useful in identifying extracted
        parameters.

        ### Returns:

        An instance of the `pymead.core.param.Param` class.
        """

        self.units = units
        self.scale_value = scale_value

        if self.units == 'length' and self.scale_value is not None:
            self.value = value * self.scale_value
        elif self.units == 'inverse-length' and self.scale_value is not None:
            self.value = value / self.scale_value
        else:
            self.value = value

        self.bounds = bounds
        self.active = active
        self.linked = linked
        self.name = name
