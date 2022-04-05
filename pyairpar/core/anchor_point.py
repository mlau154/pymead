from pyairpar.core.param import Param
import numpy as np


class AnchorPoint:

    def __init__(self,
                 x: Param,
                 y: Param,
                 name: str,
                 previous_anchor_point: str,
                 L: Param,
                 R: Param,
                 r: Param,
                 phi: Param,
                 psi1: Param,
                 psi2: Param,
                 length_scale_dimension: float = None):
        """
        ### Description:

        The `AnchorPoint` in `pyairpar` is the way to split a Bézier curve within an `pyairpar.core.airfoil.Airfoil`
        into two Bézier curves and satisfy \\(G^0\\), \\(G^1\\), and \\(G^2\\) continuity at the joint between the
        curves

        ### Args:

        `x`: `pyairpar.core.param.Param` describing the x-location of the `AnchorPoint`

        `y`: `pyairpar.core.param.Param` describing the y-location of the `AnchorPoint`

        `previous_anchor_point`: a `str` representing the previous `AnchorPoint` (counter-clockwise ordering)

        `L`: `pyairpar.core.param.Param` describing the distance between the control points on either side of the
        `pyairpar.core.anchor_point.AnchorPoint`

        .. WARNING:: Need to fill out rest of parameters

        `length_scale_dimension`: a `float` giving the length scale by which to non-dimensionalize the `x` and `y`
        values (optional)

        ### Returns:

        An instance of the `AnchorPoint` class
        """

        self.x = x
        self.y = y
        self.name = name
        self.previous_anchor_point = previous_anchor_point
        self.L = L
        self.R = R
        self.r = r
        self.phi = phi
        self.psi1 = psi1
        self.psi2 = psi2
        self.length_scale_dimension = length_scale_dimension
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()
        self.xy = np.array([x.value, y.value])

    def scale_vars(self):
        """
        ### Description:

        Scales all of the `pyairpar.core.param.Param`s in the `AnchorPoint` with `units == 'length'` by the
        `length_scale_dimension`. Scaling only occurs for each parameter if the `pyairpar.core.param.Param` has not yet
        been scaled.
        """
        if self.length_scale_dimension is not None:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.length_scale_dimension is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.length_scale_dimension

    def count_overrideable_variables(self):
        """
        ### Description:

        Counts all the overrideable `pyairpar.core.param.Param`s in the `AnchorPoint` (criteria:
        `pyairpar.core.param.Param().active == True`, `pyairpar.core.param.Param().linked == False`)

        ### Returns:

        Number of overrideable variables (`int`)
        """
        n_overrideable_variables = len([var for var in vars(self).values()
                                        if isinstance(var, Param) and var.active and not var.linked])
        return n_overrideable_variables

    def override(self, parameters: list):
        """
        ### Description:

        Overrides all the `pyairpar.core.param.Param`s in `AnchorPoint` which are active and not linked using a list of
        parameters. This list of parameters is likely a subset of parameters passed to either
        `pyairpar.core.airfoil.Airfoil` or `pyairpar.core.parametrization.AirfoilParametrization`. This function is
        useful whenever iteration over only the relevant parameters is required.

        ### Args:

        `parameters`: a `list` of parameters
        """
        override_param_obj_list = [var for var in vars(self).values()
                                   if isinstance(var, Param) and var.active and not var.linked]
        if len(parameters) != len(override_param_obj_list):
            raise Exception('Number of base airfoil parameters does not match length of input override parameter list')
        param_idx = 0
        for param in override_param_obj_list:
            setattr(param, 'value', parameters[param_idx])
            param_idx += 1
        self.scale_vars()
        self.xy = np.array([self.x.value, self.y.value])

    def set_all_as_linked(self):
        """
        ### Description:

        Sets `linked=True` on all `pyairpar.core.param.Param`s in the `AnchorPoint`
        """
        for param in [var for var in vars(self).values() if isinstance(var, Param)]:
            param.linked = True
