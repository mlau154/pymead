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
        curves. Examples of implemented `AnchorPoint`s in an unnecessarily strange airfoil shape are shown in the
        image below. It may be helpful to enlarge the image by opening it in a new tab.

        .. image:: complex_airfoil_anchor_points.png

        ### Args:

        `x`: ( \\(x\\) ) `pyairpar.core.param.Param` describing the x-location of the `AnchorPoint`

        `y`: ( \\(y\\) ) `pyairpar.core.param.Param` describing the y-location of the `AnchorPoint`

        `previous_anchor_point`: a `str` representing the previous `AnchorPoint` (counter-clockwise ordering)

        `L`: ( \\(L\\) ) `pyairpar.core.param.Param` describing the distance between the control points on either
        side of the `pyairpar.core.anchor_point.AnchorPoint`

        `R`: ( \\(R\\) ) `pyairpar.core.param.Param` representing the radius of curvature at the \\(x\\) - \\(y\\)
        location of the `AnchorPoint`. A positive value makes the airfoil convex at the `AnchorPoint` location,
        and a negative value makes the airfoil concave at the `AnchorPoint` location. The valid range is
        \\( \\{R \\in \\mathbb{R} \\, | \\, R \\neq 0 \\} \\).

        `r`: ( \\(r\\) ) `pyairpar.core.param.Param` representing the ratio of the distance from the `AnchorPoint`
        location to the neighboring control point closest to the trailing edge to the distance between the
        `AnchorPoint`'s neighboring control points ( \\(L_{fore} / L\\) ). The valid range is \\(r \\in (0,1)\\).

        `phi`: ( \\(\\phi\\) ) `pyairpar.core.param.Param` representing the angle of the line passing through the
        `AnchorPoint`'s neighboring control points, referenced counter-clockwise from the chordline if the
        `AnchorPoint` is on the upper airfoil surface and clockwise from the chordline if the `AnchorPoint` is on the
        lower airfoil surface. The valid range is \\(\\psi_1 \\in (-90^{\\circ},90^{\\circ})\\).

        `psi1`: ( \\(\\psi_1\\) ) `pyairpar.core.param.Param` representing the angle of the aft curvature control "arm."
        Regardless of the sign of \\(R\\) or which surface the `AnchorPoint` lies on, an angle of \\(90^{\\circ}\\)
        always means that the curvature control arm points perpendicular to the line passing through the
        neighboring control points of the `AnchorPoint`. Angles below \\(90^{\\circ}\\) "tuck" the arms in, and angles
        above \\(90^{\\circ}\\) "spread" the arms out. The valid range is \\(\\psi_1 \\in (0^{\\circ},180^{\\circ})\\).

        `psi2`: ( \\(\\psi_2\\) ) `pyairpar.core.param.Param` representing the angle of the fore curvature control
        "arm." Regardless of the sign of \\(R\\) or which surface the `AnchorPoint` lies on, an angle of \\(90^{
        \\circ}\\) always means that the curvature control arm points perpendicular to the line passing through the
        neighboring control points of the `AnchorPoint`. Angles below \\(90^{\\circ}\\) "tuck" the arms in,
        and angles above \\(90^{\\circ}\\) "spread" the arms out. The valid range is \\(\\psi_2 \\in (0^{\\circ},
        180^{\\circ})\\).

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
                if param.scale_value is None:  # only scale if the parameter has not yet been scaled
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
