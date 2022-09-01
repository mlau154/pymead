from pyairpar.core.param import Param
from pyairpar.core.control_point import ControlPoint
import numpy as np


class AnchorPoint(ControlPoint):

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
        and a negative value makes the airfoil concave at the `AnchorPoint` location. A value of 0 creates a
        flat-plate-type leading edge. The valid range is \\( R \\in [-\\infty, \\infty] \\). Inclusive brackets are
        used here because setting \\(R=\\pm \\infty\\) creates an anchor point with no curvature (infinite radius of
        curvature).

        `r`: ( \\(r\\) ) `pyairpar.core.param.Param` representing the ratio of the distance from the `AnchorPoint`
        location to the neighboring control point closest to the trailing edge to the distance between the
        `AnchorPoint`'s neighboring control points ( \\(L_{fore} / L\\) ). The valid range is \\(r \\in (0,1)\\).

        `phi`: ( \\(\\phi\\) ) `pyairpar.core.param.Param` representing the angle of the line passing through the
        `AnchorPoint`'s neighboring control points, referenced counter-clockwise from the chordline if the
        `AnchorPoint` is on the upper airfoil surface and clockwise from the chordline if the `AnchorPoint` is on the
        lower airfoil surface. The valid range is \\(\\psi_1 \\in [-180^{\\circ},180^{\\circ}]\\). A value of \\(0^{
        \\circ}\\) creates an anchor point with local slope equal to the slope of the chordline.

        `psi1`: ( \\(\\psi_1\\) ) `pyairpar.core.param.Param` representing the angle of the aft curvature control "arm."
        Regardless of the sign of \\(R\\) or which surface the `AnchorPoint` lies on, an angle of \\(90^{\\circ}\\)
        always means that the curvature control arm points perpendicular to the line passing through the
        neighboring control points of the `AnchorPoint`. Angles below \\(90^{\\circ}\\) "tuck" the arms in, and angles
        above \\(90^{\\circ}\\) "spread" the arms out. The valid range is \\(\\psi_1 \\in [0^{\\circ},180^{\\circ}]\\).

        `psi2`: ( \\(\\psi_2\\) ) `pyairpar.core.param.Param` representing the angle of the fore curvature control
        "arm." Regardless of the sign of \\(R\\) or which surface the `AnchorPoint` lies on, an angle of \\(90^{
        \\circ}\\) always means that the curvature control arm points perpendicular to the line passing through the
        neighboring control points of the `AnchorPoint`. Angles below \\(90^{\\circ}\\) "tuck" the arms in,
        and angles above \\(90^{\\circ}\\) "spread" the arms out. The valid range is \\(\\psi_2 \\in [0^{\\circ},
        180^{\\circ}]\\).

        `length_scale_dimension`: a `float` giving the length scale by which to non-dimensionalize the `x` and `y`
        values (optional)

        ### Returns:

        An instance of the `AnchorPoint` class
        """

        super().__init__(x.value, y.value, name, previous_anchor_point)

        self.ctrlpt = ControlPoint(x.value, y.value, name, previous_anchor_point, cp_type='anchor_point')
        self.ctrlpt_branch_list = None

        self.n1 = None
        self.n2 = None

        self.anchor_type = None

        self.x = x
        self.y = y
        self.L = L
        self.R = R

        self.Lt_minus = None
        self.Lt_plus = None
        self.Lc_minus = None
        self.Lc_plus = None
        self.abs_psi1 = None
        self.abs_psi2 = None
        self.abs_phi1 = None
        self.abs_phi2 = None

        # self._Lt_minus = None
        # self._Lt_plus = None
        # self._Lc_minus = None
        # self._Lc_plus = None
        # self._abs_psi1 = None
        # self._abs_psi2 = None
        # self._abs_phi = None

        self.g1_minus_ctrlpt = None
        self.g1_plus_ctrlpt = None
        self.g2_minus_ctrlpt = None
        self.g2_plus_ctrlpt = None

        self.ctrlpt_branch_array = None

        self.ctrlpt_branch_generated = False

        if 0 < r.value < 1:
            self.r = r
        else:
            raise ValueError(f'The distance fraction, r, must be between 0 and 1. A value of {r.value} was entered.')

        if -np.pi <= phi.value <= np.pi:
            self.phi = phi
        else:
            raise ValueError(f'The anchor point neighboring control point angle, phi, must be between -180 degrees and'
                             f' 180 degrees, inclusive. A value of {phi.value} was entered.')

        if 0 <= psi1.value <= np.pi:
            self.psi1 = psi1
        else:
            raise ValueError(f'The aft curvature control arm angle, psi1, must be between 0 degrees and 180 degrees, '
                             f'inclusive. '
                             f'A value of {psi1.value} was entered.')

        if 0 <= psi2.value <= np.pi:
            self.psi2 = psi2
        else:
            raise ValueError(f'The fore curvature control arm angle, psi2, must be between 0 degrees and 180 degrees,'
                             f'inclusive. '
                             f'A value of {psi2.value} was entered.')

        self.length_scale_dimension = length_scale_dimension
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()
        self.xy = np.array([x.value, y.value])

    # @property
    # def Lt_minus(self):
    #     return self._Lt_minus
    #
    # @property
    # def Lt_plus(self):
    #     return self._Lt_plus
    #
    # @property
    # def Lc_minus(self):
    #     return self._Lc_minus
    #
    # @property
    # def Lc_plus(self):
    #     return self._Lc_plus
    #
    # @property
    # def abs_psi1(self):
    #     return self._abs_psi1
    #
    # @property
    # def abs_psi2(self):
    #     return self._abs_psi2
    #
    # @property
    # def abs_phi(self):
    #     return self._abs_phi

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

    def __repr__(self):
        return f"anchor_point_{self.name}"

    def get_anchor_type(self, anchor_point_order):
        if self.name == 'le':
            self.anchor_type = self.name
        elif anchor_point_order.index(self.name) < anchor_point_order.index('le'):
            self.anchor_type = 'upper_surf'
        elif anchor_point_order.index(self.name) > anchor_point_order.index('le'):
            self.anchor_type = 'lower_surf'

    def set_minus_plus_bezier_curve_orders(self, n1, n2):
        self.n1 = n1
        self.n2 = n2

    def generate_anchor_point_branch(self, anchor_point_order):
        r = self.r.value
        L = self.L.value
        phi = self.phi.value
        R = self.R.value
        psi1 = self.psi1.value
        psi2 = self.psi2.value
        name = self.name
        x0 = self.x_val
        y0 = self.y_val

        if self.n1 is None:
            raise ValueError('Order of Bezier curve before anchor point was not set before generating the anchor'
                             'point branch')
        else:
            n1 = self.n1

        if self.n2 is None:
            raise ValueError('Order of Bezier curve after anchor point was not set before generating the anchor'
                             'point branch')
        else:
            n2 = self.n2

        self.get_anchor_type(anchor_point_order)

        def generate_tangent_seg_ctrlpts(minus_plus: str):
            if R == 0:  # degenerate case 1: infinite curvature (sharp corner)
                return ControlPoint(x0, y0, f'{repr(self)}_g1_{minus_plus}', name)

            def evaluate_tangent_segment_length():
                if self.anchor_type == 'upper_surf':
                    if minus_plus == 'minus':
                        self.Lt_minus = (1 - r) * L
                    else:
                        self.Lt_plus = r * L
                elif self.anchor_type in ['lower_surf', 'le']:
                    if minus_plus == 'minus':
                        self.Lt_minus = r * L
                    else:
                        self.Lt_plus = (1 - r) * L
                else:
                    raise ValueError('Invalid anchor type')

            def map_tilt_angle():
                if self.anchor_type == 'upper_surf':
                    if minus_plus == 'minus':
                        self.abs_phi1 = phi
                    else:
                        self.abs_phi2 = np.pi + phi
                elif self.anchor_type == 'lower_surf':
                    if minus_plus == 'minus':
                        self.abs_phi1 = np.pi - phi
                    else:
                        self.abs_phi2 = -phi
                elif self.anchor_type == 'le':
                    if minus_plus == 'minus':
                        self.abs_phi1 = np.pi / 2 + phi
                    else:
                        self.abs_phi2 = 3 * np.pi / 2 + phi
                else:
                    raise ValueError('Invalid anchor type')

            evaluate_tangent_segment_length()
            map_tilt_angle()

            if minus_plus == 'minus':
                xy = np.array([x0, y0]) + self.Lt_minus * np.array([np.cos(self.abs_phi1), np.sin(self.abs_phi1)])
            else:
                xy = np.array([x0, y0]) + self.Lt_plus * np.array([np.cos(self.abs_phi2), np.sin(self.abs_phi2)])
            return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_{minus_plus}', name, cp_type=f'g1_{minus_plus}')

        def generate_curvature_seg_ctrlpts(psi, tangent_ctrlpt: ControlPoint, n, minus_plus):
            if R == 0:  # degenerate case 1: infinite curvature (sharp corner)
                return ControlPoint(x0, y0, f'{repr(self)}_g2_{minus_plus}', name)
            with np.errstate(divide='ignore'):  # accept divide by 0 as valid
                if name == 'le':
                    if np.sin(psi + np.pi / 2) == 0 or np.true_divide(1, R) == 0:
                        # degenerate case 2: zero curvature (straight line)
                        return ControlPoint(tangent_ctrlpt.x_val, tangent_ctrlpt.y_val, f'{repr(self)}_g2_{minus_plus}',
                                            name, cp_type=f'g2_{minus_plus}')
                else:
                    if np.sin(psi) == 0 or np.true_divide(1, R) == 0:
                        # degenerate case 2: zero curvature (straight line)
                        return ControlPoint(tangent_ctrlpt.x_val, tangent_ctrlpt.y_val, f'{repr(self)}_g2_{minus_plus}',
                                            name, cp_type=f'g2_{minus_plus}')

            if name == 'le':
                if minus_plus == 'minus':
                    self.Lc_minus = self.Lt_minus ** 2 / (R * (1 - 1 / n) * np.sin(psi + np.pi / 2))
                else:
                    self.Lc_plus = self.Lt_plus ** 2 / (R * (1 - 1 / n) * np.sin(psi + np.pi / 2))
            else:
                if minus_plus == 'minus':
                    self.Lc_minus = self.Lt_minus ** 2 / (R * (1 - 1 / n) * np.sin(psi))
                else:
                    self.Lc_plus = self.Lt_plus ** 2 / (R * (1 - 1 / n) * np.sin(psi))

            def map_psi_to_airfoil_csys():
                if minus_plus == 'minus':
                    if self.anchor_type == 'upper_surf':
                        if R > 0:
                            self.abs_psi1 = np.pi + psi + phi
                        else:
                            self.abs_psi1 = np.pi - psi + phi
                    elif self.anchor_type == 'lower_surf':
                        if R > 0:
                            self.abs_psi1 = psi - phi
                        else:
                            self.abs_psi1 = -psi - phi
                    elif self.anchor_type == 'le':
                        if R > 0:
                            self.abs_psi1 = psi + phi
                        else:
                            self.abs_psi1 = np.pi - psi + phi
                    else:
                        raise ValueError("Anchor is of invalid type")
                else:
                    if self.anchor_type == 'upper_surf':
                        if R > 0:
                            self.abs_psi2 = -psi + phi
                        else:
                            self.abs_psi2 = psi + phi
                    elif self.anchor_type == 'lower_surf':
                        if R > 0:
                            self.abs_psi2 = np.pi - psi - phi
                        else:
                            self.abs_psi2 = np.pi + psi - phi
                    elif self.anchor_type == 'le':
                        if R > 0:
                            self.abs_psi2 = -psi + phi
                        else:
                            self.abs_psi2 = np.pi + psi + phi
                    else:
                        raise ValueError("Anchor is of invalid type")

            map_psi_to_airfoil_csys()

            if minus_plus == 'minus':
                xy = np.array([tangent_ctrlpt.x_val, tangent_ctrlpt.y_val]) + abs(self.Lc_minus) * \
                     np.array([np.cos(self.abs_psi1), np.sin(self.abs_psi1)])
            else:
                xy = np.array([tangent_ctrlpt.x_val, tangent_ctrlpt.y_val]) + abs(self.Lc_plus) * \
                     np.array([np.cos(self.abs_psi2), np.sin(self.abs_psi2)])

            return ControlPoint(xy[0], xy[1], f'{repr(self)}_g2_{minus_plus}', name, cp_type=f'g2_{minus_plus}')

        self.g1_minus_ctrlpt = generate_tangent_seg_ctrlpts('minus')
        self.g1_plus_ctrlpt = generate_tangent_seg_ctrlpts('plus')
        self.g2_minus_ctrlpt = generate_curvature_seg_ctrlpts(psi1, self.g1_minus_ctrlpt, n1, 'minus')
        self.g2_plus_ctrlpt = generate_curvature_seg_ctrlpts(psi2, self.g1_plus_ctrlpt, n2, 'plus')

        self.ctrlpt_branch_list = [self.g2_minus_ctrlpt, self.g1_minus_ctrlpt, self.ctrlpt, self.g1_plus_ctrlpt,
                                   self.g2_plus_ctrlpt]

        self.ctrlpt_branch_array = np.array([[self.g2_minus_ctrlpt.xp, self.g2_minus_ctrlpt.yp],
                                             [self.g1_minus_ctrlpt.xp, self.g1_minus_ctrlpt.yp],
                                             [self.xp, self.yp],
                                             [self.g1_plus_ctrlpt.xp, self.g1_plus_ctrlpt.yp],
                                             [self.g2_plus_ctrlpt.xp, self.g2_plus_ctrlpt.yp]])

        self.ctrlpt_branch_generated = True

    def recalculate_ap_branch_props_from_g2_pt(self, minus_plus: str, measured_psi, measured_Lc):
        if minus_plus == 'minus':
            # The following logic block is to ensure that the curvature control arm angle (psi) uses the correct
            # coordinate system:
            self.abs_psi1 = measured_psi
            self.Lc_minus = measured_Lc
            if 0 < np.arctan2(np.sin(self.abs_psi1 - self.abs_phi1), np.cos(self.abs_psi1 - self.abs_phi1)) < np.pi:
                sign_R = -1
            else:
                sign_R = 1
        else:
            self.abs_psi2 = measured_psi
            self.Lc_plus = measured_Lc
            # print(f"angle_diff = {np.arctan2(np.sin(self.abs_psi2 - self.abs_phi2), np.cos(self.abs_psi2 - self.abs_phi2)) * 180/np.pi}")
            if 0 < np.arctan2(np.sin(self.abs_psi2 - self.abs_phi2), np.cos(self.abs_psi2 - self.abs_phi2)) < np.pi:
                sign_R = 1
            else:
                sign_R = -1

        # print(f"self.abs_phi1 = {self.abs_phi1 * 180/np.pi}")
        # print(f"self.abs_phi2 = {self.abs_phi2 * 180/np.pi}")
        # print(f"self.abs_psi1 = {self.abs_psi1 * 180/np.pi}")
        # print(f"self.abs_psi2 = {self.abs_psi2 * 180/np.pi}")

        # print(f"sign(Rvalue) = {int(np.sign(self.R.value))}")
        if int(np.sign(self.R.value)) * sign_R == -1:
            # print('Flipping sign!')
            self.R.value *= -1  # Flip the sign of the radius of curvature if different than current value
        else:
            pass

        # Since we will be overriding the radius of curvature (R.value) with a value that is always positive, we need to
        # determine whether the sign of the radius of curvature should be positive or negative:
        negate_R = False
        if self.R.value < 0:
            negate_R = True

        def map_psi_to_airfoil_csys_inverse():
            if minus_plus == 'minus':
                if self.anchor_type == 'upper_surf':
                    if self.R.value > 0:
                        # angle = np.pi + psi + phi
                        self.psi1.value = self.abs_psi1 - np.pi - self.phi.value
                    else:
                        # angle = np.pi - psi + phi
                        self.psi1.value = -self.abs_psi1 + np.pi + self.phi.value
                elif self.anchor_type == 'lower_surf':
                    if self.R.value > 0:
                        # angle = psi - phi
                        self.psi1.value = self.abs_psi1 + self.phi.value
                    else:
                        # angle = -psi - phi
                        self.psi1.value = -self.abs_psi1 - self.phi.value
                elif self.anchor_type == 'le':
                    if self.R.value > 0:
                        # angle = psi + phi
                        self.psi1.value = self.abs_psi1 - self.phi.value
                    else:
                        # angle = np.pi - psi + phi
                        self.psi1.value = np.pi - self.abs_psi1 + self.phi.value
                else:
                    raise ValueError("Anchor is of invalid type")
            else:
                if self.anchor_type == 'upper_surf':
                    if self.R.value > 0:
                        # angle = -psi + phi
                        self.psi2.value = -self.abs_psi2 + self.phi.value
                    else:
                        # angle = psi + phi
                        self.psi2.value = self.abs_psi2 - self.phi.value
                elif self.anchor_type == 'lower_surf':
                    if self.R.value > 0:
                        # angle = np.pi - psi - phi
                        self.psi2.value = np.pi - self.abs_psi2 - self.phi.value
                    else:
                        # angle = np.pi + psi - phi
                        self.psi2.value = self.abs_psi2 - np.pi + self.phi.value
                elif self.anchor_type == 'le':
                    if self.R.value > 0:
                        # angle = -psi + phi
                        self.psi2.value = -self.abs_psi2 + self.phi.value
                    else:
                        # angle = np.pi + psi + phi
                        self.psi2.value = self.abs_psi2 - np.pi - self.phi.value
                else:
                    raise ValueError("Anchor is of invalid type")

        map_psi_to_airfoil_csys_inverse()
        # print(f"new_psi = {airfoil.anchor_points[1].psi1.value * 180 / np.pi}")
        # Lt = evaluate_tangent_segment_length('minus', anchor_point)
        # print(anchor_point.name)
        if self.name == 'le':
            # Lc = Lt ** 2 / (R * (1 - 1 / n) * np.sin(psi + np.pi / 2))
            if minus_plus == 'minus':
                self.R.value = self.Lt_minus ** 2 / (
                            self.Lc_minus * (1 - 1 / self.n1) * np.sin(self.psi1.value + np.pi / 2))
            else:
                self.R.value = self.Lt_plus ** 2 / (
                        self.Lc_plus * (1 - 1 / self.n2) * np.sin(self.psi2.value + np.pi / 2))
                # print(f"self.psi2.value = {self.psi2.value * 180/ np.pi}")
        else:
            # Lc = Lt ** 2 / (R * (1 - 1 / n) * np.sin(psi))
            if minus_plus == 'minus':
                self.R.value = self.Lt_minus ** 2 / (self.Lc_minus * (1 - 1 / self.n1) * np.sin(self.psi1.value))
            else:
                self.R.value = self.Lt_plus ** 2 / (self.Lc_plus * (1 - 1 / self.n2) * np.sin(self.psi2.value))
        if negate_R:
            self.R.value *= -1

    def recalculate_ap_branch_props_from_g1_pt(self, minus_plus: str, measured_phi, measured_Lt):

        if minus_plus == 'minus':
            self.Lt_minus = measured_Lt
            self.abs_phi1 = measured_phi
        else:
            self.Lt_plus = measured_Lt
            self.abs_phi2 = measured_phi

        def evaluate_g1_length_and_ratio():
            self.L.value = self.Lt_minus + self.Lt_plus
            if self.anchor_type == 'upper_surf':
                self.r.value = self.Lt_plus / self.L.value
            elif self.anchor_type in ['lower_surf', 'le']:
                self.r.value = self.Lt_minus / self.L.value
            else:
                raise ValueError('Invalid anchor type')

        def map_tilt_angle_inverse():
            if self.anchor_type == 'upper_surf':
                if minus_plus == 'minus':
                    # self.abs_phi1 = phi
                    self.phi.value = self.abs_phi1
                else:
                    # self.abs_phi2 = np.pi + phi
                    self.phi.value = self.abs_phi2 - np.pi
            elif self.anchor_type == 'lower_surf':
                if minus_plus == 'minus':
                    # self.abs_phi1 = np.pi - phi
                    self.phi.value = np.pi - self.abs_phi1
                else:
                    # self.abs_phi2 = -phi
                    self.phi.value = -self.abs_phi2
            elif self.anchor_type == 'le':
                if minus_plus == 'minus':
                    # self.abs_phi1 = np.pi / 2 + phi
                    self.phi.value = self.abs_phi1 - np.pi / 2
                else:
                    # self.abs_phi2 = 3 * np.pi / 2 + phi
                    self.phi.value = self.abs_phi2 - 3 * np.pi / 2
            else:
                raise ValueError('Invalid anchor type')

        evaluate_g1_length_and_ratio()
        map_tilt_angle_inverse()

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

    def set_all_as_linked(self):
        """
        ### Description:

        Sets `linked=True` on all `pyairpar.core.param.Param`s in the `AnchorPoint`
        """
        for param in [var for var in vars(self).values() if isinstance(var, Param)]:
            param.linked = True
