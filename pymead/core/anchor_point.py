from pymead.core.param import Param
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import rotate, translate, scale, transform
import numpy as np


class AnchorPoint(ControlPoint):

    def __init__(self,
                 x: Param,
                 y: Param,
                 tag: str,
                 previous_anchor_point: str,
                 airfoil_tag: str,
                 L: Param,
                 R: Param,
                 r: Param,
                 phi: Param,
                 psi1: Param,
                 psi2: Param,
                 length_scale_dimension: float = None):
        """
        ### Description:

        The `AnchorPoint` in `pymead` is the way to split a Bézier curve within an `pymead.core.airfoil.Airfoil`
        into two Bézier curves and satisfy \\(G^0\\), \\(G^1\\), and \\(G^2\\) continuity at the joint between the
        curves. Examples of implemented `AnchorPoint`s in an unnecessarily strange airfoil shape are shown in the
        image below. It may be helpful to enlarge the image by opening it in a new tab.

        .. image:: complex_airfoil_anchor_points.png

        ### Args:

        `x`: ( \\(x\\) ) `pymead.core.param.Param` describing the x-location of the `AnchorPoint`

        `y`: ( \\(y\\) ) `pymead.core.param.Param` describing the y-location of the `AnchorPoint`

        `previous_anchor_point`: a `str` representing the previous `AnchorPoint` (counter-clockwise ordering)

        `L`: ( \\(L\\) ) `pymead.core.param.Param` describing the distance between the control points on either
        side of the `pymead.core.anchor_point.AnchorPoint`

        `R`: ( \\(R\\) ) `pymead.core.param.Param` representing the radius of curvature at the \\(x\\) - \\(y\\)
        location of the `AnchorPoint`. A positive value makes the airfoil convex at the `AnchorPoint` location,
        and a negative value makes the airfoil concave at the `AnchorPoint` location. A value of 0 creates a
        flat-plate-type leading edge. The valid range is \\( R \\in [-\\infty, \\infty] \\). Inclusive brackets are
        used here because setting \\(R=\\pm \\infty\\) creates an anchor point with no curvature (infinite radius of
        curvature).

        `r`: ( \\(r\\) ) `pymead.core.param.Param` representing the ratio of the distance from the `AnchorPoint`
        location to the neighboring control point closest to the trailing edge to the distance between the
        `AnchorPoint`'s neighboring control points ( \\(L_{fore} / L\\) ). The valid range is \\(r \\in (0,1)\\).

        `phi`: ( \\(\\phi\\) ) `pymead.core.param.Param` representing the angle of the line passing through the
        `AnchorPoint`'s neighboring control points, referenced counter-clockwise from the chordline if the
        `AnchorPoint` is on the upper airfoil surface and clockwise from the chordline if the `AnchorPoint` is on the
        lower airfoil surface. The valid range is \\(\\psi_1 \\in [-180^{\\circ},180^{\\circ}]\\). A value of \\(0^{
        \\circ}\\) creates an anchor point with local slope equal to the slope of the chordline.

        `psi1`: ( \\(\\psi_1\\) ) `pymead.core.param.Param` representing the angle of the aft curvature control "arm."
        Regardless of the sign of \\(R\\) or which surface the `AnchorPoint` lies on, an angle of \\(90^{\\circ}\\)
        always means that the curvature control arm points perpendicular to the line passing through the
        neighboring control points of the `AnchorPoint`. Angles below \\(90^{\\circ}\\) "tuck" the arms in, and angles
        above \\(90^{\\circ}\\) "spread" the arms out. The valid range is \\(\\psi_1 \\in [0^{\\circ},180^{\\circ}]\\).

        `psi2`: ( \\(\\psi_2\\) ) `pymead.core.param.Param` representing the angle of the fore curvature control
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

        super().__init__(x.value, y.value, tag, previous_anchor_point)

        self.ctrlpt = ControlPoint(x.value, y.value, tag, previous_anchor_point, cp_type='anchor_point')
        self.ctrlpt_branch_list = None

        self.n1 = None
        self.n2 = None

        self.anchor_type = None
        self.tag = tag
        self.previous_anchor_point = previous_anchor_point

        self.x = x
        self.y = y
        self.xp = Param(self.x.value)
        self.yp = Param(self.y.value)
        self.airfoil_transformation = None
        self.airfoil_tag = airfoil_tag
        self.L = L
        self.R = R
        self.x.anchor_point = self
        self.y.anchor_point = self
        self.xp.anchor_point = self
        self.yp.anchor_point = self

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

    def scale_vars(self):
        """
        ### Description:

        Scales all the `pymead.core.param.Param`s in the `AnchorPoint` with `units == 'length'` by the
        `length_scale_dimension`. Scaling only occurs for each parameter if the `pymead.core.param.Param` has not yet
        been scaled.
        """
        if self.length_scale_dimension is not None:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.scale_value is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.length_scale_dimension

    def __repr__(self):
        return f"anchor_point_{self.tag}"

    def set_xy(self, x=None, y=None, xp=None, yp=None):
        # other_airfoils_affected = []
        # print("Made it to the start")
        # print(f"Made it 2")
        for xy in [['x', 'y'], ['xp', 'yp'], ['x', 'yp'], ['xp', 'y']]:
            if getattr(getattr(self, xy[0]), 'linked') and getattr(getattr(self, xy[1]), 'linked') or (
                    not getattr(getattr(self, xy[0]), 'active')) and (not getattr(getattr(self, xy[1]), 'active')):
                return []  # Early return if both x and y degrees of freedom are locked
        if (x is not None or y is not None) and xp is None and yp is None:
            # if self.xp.linked and self.yp.linked or (not self.xp.active) and (not self.yp.active):
            #     return []
            if x is not None and (not self.x.linked) and self.x.active:
                self.x.value = x
            if y is not None and (not self.y.linked) and self.y.active:
                self.y.value = y
            if self.xp.linked or not self.xp.active:
                self.yp.value = self.get_yp_from_x_y(x, y)
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            elif self.yp.linked or not self.yp.active:
                self.xp.value = self.get_xp_from_x_y(x, y)
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            else:
                self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
                                                         self.airfoil_transformation['dx'].value,
                                                         self.airfoil_transformation['dy'].value,
                                                         -self.airfoil_transformation['alf'].value,
                                                         self.airfoil_transformation['c'].value,
                                                         ['scale', 'rotate', 'translate'])
            # If other airfoils are affected by this change in FreePoint location, we need to mark the airfoil for
            # change:
            # other_airfoils_affected.extend(self.update_xy())
        elif (xp is not None or yp is not None) and x is None and y is None:
            # if self.x.linked and self.y.linked or (not self.x.active) and (not self.y.active):
            #     return []
            # if xp is not None and (not self.xp.linked) and self.xp.active:
            #     self.xp.value = xp
            #     print(f"Setting xp value! xp = {self.xp.value}")
            # if yp is not None and (not self.yp.linked) and self.yp.active:
            #     self.yp.value = yp
            if self.x.linked or not self.x.active:
                self.y.value = self.get_y_from_xp_yp(xp, yp)
                self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
                                                         self.airfoil_transformation['dx'].value,
                                                         self.airfoil_transformation['dy'].value,
                                                         -self.airfoil_transformation['alf'].value,
                                                         self.airfoil_transformation['c'].value,
                                                         ['scale', 'rotate', 'translate'])
            elif self.y.linked or not self.y.active:
                self.x.value = self.get_x_from_xp_yp(xp, yp)
                self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
                                                         self.airfoil_transformation['dx'].value,
                                                         self.airfoil_transformation['dy'].value,
                                                         -self.airfoil_transformation['alf'].value,
                                                         self.airfoil_transformation['c'].value,
                                                         ['scale', 'rotate', 'translate'])
            elif self.xp.linked or not self.xp.active:
                self.yp.value = yp
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            elif self.yp.linked or not self.yp.active:
                self.xp.value = xp
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            else:
                self.xp.value = xp
                self.yp.value = yp
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
        else:
            raise ValueError("Either (\'x\' or \'y\') or (\'xp\' or \'yp\') must be specified or \'only_update_xp_yp\' "
                             "must be set to True")
        other_airfoils_affected = self.update_xy()
        # print(f"other_airfoils_affected now is {other_airfoils_affected}")
        # if only_update_xy:
        #     self.set_ctrlpt_value2()
        # else:
        self.set_ctrlpt_value()
        # print(f"set ctrlpt value to {self.ctrlpt.xp}, {self.ctrlpt.yp}")
        return other_airfoils_affected

    def update_xy(self):
        other_airfoils_affected = []
        for xy in ['x', 'y', 'xp', 'yp']:
            getattr(self, xy).update()
            # print(f"updating affects")
            # print(f"affects = {getattr(self, xy).affects}")
            for affects in getattr(self, xy).affects:
                # print(f"affects.anchor_point = {affects.anchor_point}")
                # print(f"affects = {affects}")
                affects.update()
                if affects.free_point is not None:
                    fp = affects.free_point
                    if fp.airfoil_tag != self.airfoil_tag:
                        other_airfoils_affected.append(fp.airfoil_tag)
                    if affects.x or affects.y:
                        fp.xp.value, fp.yp.value = transform(fp.x.value, fp.y.value,
                                                             fp.airfoil_transformation['dx'].value,
                                                             fp.airfoil_transformation['dy'].value,
                                                             -fp.airfoil_transformation['alf'].value,
                                                             fp.airfoil_transformation['c'].value,
                                                             ['scale', 'rotate', 'translate'])
                    if affects.xp or affects.yp:
                        fp.x.value, fp.y.value = transform(fp.xp.value, fp.yp.value,
                                                           -fp.airfoil_transformation['dx'].value,
                                                           -fp.airfoil_transformation['dy'].value,
                                                           fp.airfoil_transformation['alf'].value,
                                                           1 / fp.airfoil_transformation['c'].value,
                                                           ['translate', 'rotate', 'scale'])
                    if affects.x or affects.y or affects.xp or affects.yp:
                        fp.set_ctrlpt_value()
                if affects.anchor_point is not None:

                    ap = affects.anchor_point
                    # print(f"ap airfoil tag = {ap.airfoil_tag}")
                    # print(f"self.airfoil_tag = {self.airfoil_tag}")
                    # print(f"affects.x = {affects.x}")
                    # print(f"affects.y = {affects.y}")
                    # print(f"affects.xp = {affects.xp}")
                    # print(f"affects.yp = {affects.yp}")
                    if ap.airfoil_tag != self.airfoil_tag:
                        other_airfoils_affected.append(ap.airfoil_tag)
                    if affects.x or affects.y:
                        ap.xp.value, ap.yp.value = transform(ap.x.value, ap.y.value,
                                                             ap.airfoil_transformation['dx'].value,
                                                             ap.airfoil_transformation['dy'].value,
                                                             -ap.airfoil_transformation['alf'].value,
                                                             ap.airfoil_transformation['c'].value,
                                                             ['scale', 'rotate', 'translate'])
                    if affects.xp or affects.yp:
                        ap.x.value, ap.y.value = transform(ap.xp.value, ap.yp.value,
                                                           -ap.airfoil_transformation['dx'].value,
                                                           -ap.airfoil_transformation['dy'].value,
                                                           ap.airfoil_transformation['alf'].value,
                                                           1 / ap.airfoil_transformation['c'].value,
                                                           ['translate', 'rotate', 'scale'])
                    if affects.x or affects.y or affects.xp or affects.yp:
                        ap.set_ctrlpt_value()
        return other_airfoils_affected

    def get_x_from_xp_yp(self, xp, yp):
        x, _ = transform(xp, yp, -self.airfoil_transformation['dx'].value,
                         -self.airfoil_transformation['dy'].value,
                         self.airfoil_transformation['alf'].value,
                         1 / self.airfoil_transformation['c'].value,
                         ['translate', 'rotate', 'scale'])
        return x

    def get_y_from_xp_yp(self, xp, yp):
        _, y = transform(xp, yp, -self.airfoil_transformation['dx'].value, -self.airfoil_transformation['dy'].value,
                         self.airfoil_transformation['alf'].value, 1 / self.airfoil_transformation['c'].value,
                         ['translate', 'rotate', 'scale'])
        return y

    def get_xp_from_x_y(self, x, y):
        xp, _ = transform(x, y, self.airfoil_transformation['dx'].value,
                          self.airfoil_transformation['dy'].value,
                          -self.airfoil_transformation['alf'].value,
                          self.airfoil_transformation['c'].value,
                          ['scale', 'rotate', 'translate'])
        return xp

    def get_yp_from_x_y(self, x, y):
        _, yp = transform(x, y, self.airfoil_transformation['dx'].value, self.airfoil_transformation['dy'].value,
                          -self.airfoil_transformation['alf'].value, self.airfoil_transformation['c'].value,
                          ['scale', 'rotate', 'translate'])
        return yp

    def more_than_one_xy_linked_or_inactive(self):
        linked_or_inactive_counter = 0
        for xy in ['x', 'y', 'xp', 'yp']:
            if getattr(self, xy).linked or not getattr(self, xy).active:
                linked_or_inactive_counter += 1
        if linked_or_inactive_counter > 1:
            return True
        else:
            return False

    def x_or_y_linked_or_inactive(self):
        for xy in ['x', 'y']:
            if getattr(self, xy).linked or not getattr(self, xy).active:
                return True
        return False

    def xp_or_yp_linked_or_inactive(self):
        for xy in ['xp', 'yp']:
            if getattr(self, xy).linked or not getattr(self, xy).active:
                return True
        return False

    def set_x_value(self, value):
        if value is not None:
            self.x.value = value
        self.xp.value, self.yp.value = scale(self.x.value, self.y.value,
                                             self.airfoil_transformation['c'].value)
        self.xp.value, self.yp.value = rotate(self.xp.value, self.yp.value, -self.airfoil_transformation['alf'].value)
        self.xp.value, self.yp.value = translate(self.xp.value, self.yp.value, self.airfoil_transformation['dx'].value,
                                                 self.airfoil_transformation['dy'].value)
        if value is not None:
            self.set_ctrlpt_value()

    def set_y_value(self, value):
        if value is not None:
            self.y.value = value
        self.xp.value, self.yp.value = scale(self.x.value, self.y.value,
                                             self.airfoil_transformation['c'].value)
        self.xp.value, self.yp.value = rotate(self.xp.value, self.yp.value, -self.airfoil_transformation['alf'].value)
        self.xp.value, self.yp.value = translate(self.xp.value, self.yp.value, self.airfoil_transformation['dx'].value,
                                                 self.airfoil_transformation['dy'].value)
        if value is not None:
            self.set_ctrlpt_value()

    def set_xp_value(self, value):
        if value is not None:
            self.xp.value = value
        # self.x.value, self.y.value = translate(self.xp.value, self.yp.value, -self.airfoil_transformation['dx'].value,
        #                                        -self.airfoil_transformation['dy'].value)
        # self.x.value, self.y.value = rotate(self.x.value, self.y.value, self.airfoil_transformation['alf'].value)
        # self.x.value, self.y.value = scale(self.x.value, self.y.value, 1 / self.airfoil_transformation['c'].value)
        # print(f"Setting xp value! x = {self.x.value}, y = {self.y.value}")
        skip_x = False
        skip_y = False
        if not skip_x or not skip_y:
            # print(f"not skip_x or not skip_y!")
            self.x.value, self.y.value = transform(self.xp.value, self.yp.value, -self.airfoil_transformation['dx'].value,
                                                   -self.airfoil_transformation['dy'].value,
                                                   self.airfoil_transformation['alf'].value,
                                                   1 / self.airfoil_transformation['c'].value,
                                                   ['translate', 'rotate', 'scale'], skip_x=skip_x, skip_y=skip_y)
        self.set_ctrlpt_value()

    def set_yp_value(self, value):
        if value is not None:
            self.yp.value = value
        skip_x = False
        skip_y = False
        if not skip_x or not skip_y:
            # print(f"not skip_x or not skip_y!")
            self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                   -self.airfoil_transformation['dx'].value,
                                                   -self.airfoil_transformation['dy'].value,
                                                   self.airfoil_transformation['alf'].value,
                                                   1 / self.airfoil_transformation['c'].value,
                                                   ['translate', 'rotate', 'scale'], skip_x=skip_x, skip_y=skip_y)
        # print(f"Setting yp value! x = {self.x.value}, y = {self.y.value}")
        self.set_ctrlpt_value()

    def set_xp_yp_value(self, xp_value, yp_value):
        if xp_value is not None:
            self.xp.value = xp_value
        if yp_value is not None:
            self.yp.value = yp_value
        self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                               -self.airfoil_transformation['dx'].value,
                                               -self.airfoil_transformation['dy'].value,
                                               self.airfoil_transformation['alf'].value,
                                               1 / self.airfoil_transformation['c'].value,
                                               ['translate', 'rotate', 'scale'], skip_x=False, skip_y=False)
        self.set_ctrlpt_value()

    def set_ctrlpt_value(self):
        self.ctrlpt.xp = self.xp.value
        self.ctrlpt.yp = self.yp.value
        self.ctrlpt.x_val = self.x.value
        self.ctrlpt.y_val = self.y.value

    def get_anchor_type(self, anchor_point_order):
        if self.tag == 'le':
            self.anchor_type = self.tag
        elif anchor_point_order.index(self.tag) < anchor_point_order.index('le'):
            self.anchor_type = 'upper_surf'
        elif anchor_point_order.index(self.tag) > anchor_point_order.index('le'):
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
        tag = self.tag
        x0 = self.x.value
        y0 = self.y.value
        # if self.tag == 'ap0':
        #     print(f"Generating anchor point branch! x0 = {x0}, y0 = {y0}")

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
                return ControlPoint(x0, y0, f'anchor_point_{tag}_g1_{minus_plus}', tag)

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
            return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_{minus_plus}', tag, cp_type=f'g1_{minus_plus}')

        def generate_curvature_seg_ctrlpts(psi, tangent_ctrlpt: ControlPoint, n, minus_plus):
            if R == 0:  # degenerate case 1: infinite curvature (sharp corner)
                return ControlPoint(x0, y0, f'{repr(self)}_g2_{minus_plus}', tag)
            with np.errstate(divide='ignore'):  # accept divide by 0 as valid
                if tag == 'le':
                    if np.sin(psi + np.pi / 2) == 0 or np.true_divide(1, R) == 0:
                        # degenerate case 2: zero curvature (straight line)
                        return ControlPoint(tangent_ctrlpt.x_val, tangent_ctrlpt.y_val, f'{repr(self)}_g2_{minus_plus}',
                                            tag, cp_type=f'g2_{minus_plus}')
                else:
                    if np.sin(psi) == 0 or np.true_divide(1, R) == 0:
                        # degenerate case 2: zero curvature (straight line)
                        return ControlPoint(tangent_ctrlpt.x_val, tangent_ctrlpt.y_val, f'{repr(self)}_g2_{minus_plus}',
                                            tag, cp_type=f'g2_{minus_plus}')

            if tag == 'le':
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

            return ControlPoint(xy[0], xy[1], f'{repr(self)}_g2_{minus_plus}', tag, cp_type=f'g2_{minus_plus}')

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
            if measured_psi is not None:
                self.abs_psi1 = measured_psi
            if measured_Lc is not None:
                self.Lc_minus = measured_Lc
            if 0 < np.arctan2(np.sin(self.abs_psi1 - self.abs_phi1), np.cos(self.abs_psi1 - self.abs_phi1)) < np.pi:
                sign_R = -1
            else:
                sign_R = 1
        else:
            if measured_psi is not None:
                self.abs_psi2 = measured_psi
            if measured_Lc is not None:
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
        if self.R.active and not self.R.linked:
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
                if self.psi1.active and not self.psi1.linked:
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
                if self.psi2.active and not self.psi2.linked:
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
        # print(anchor_point.tag)
        if self.R.active and not self.R.linked:
            if self.tag == 'le':
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
            if measured_Lt is not None:
                self.Lt_minus = measured_Lt
            if measured_phi is not None:
                self.abs_phi1 = measured_phi
        else:
            if measured_Lt is not None:
                self.Lt_plus = measured_Lt
            if measured_phi is not None:
                self.abs_phi2 = measured_phi

        def evaluate_g1_length_and_ratio():
            if self.L.active and not self.L.linked:
                self.L.value = self.Lt_minus + self.Lt_plus
            if self.r.active and not self.r.linked:
                if self.anchor_type == 'upper_surf':
                    self.r.value = self.Lt_plus / self.L.value
                elif self.anchor_type in ['lower_surf', 'le']:
                    self.r.value = self.Lt_minus / self.L.value
                else:
                    raise ValueError('Invalid anchor type')

        def map_tilt_angle_inverse():
            if self.phi.active and not self.phi.linked:
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

        Counts all the overrideable `pymead.core.param.Param`s in the `AnchorPoint` (criteria:
        `pymead.core.param.Param().active == True`, `pymead.core.param.Param().linked == False`)

        ### Returns:

        Number of overrideable variables (`int`)
        """
        n_overrideable_variables = len([var for var in vars(self).values()
                                        if isinstance(var, Param) and var.active and not var.linked])
        return n_overrideable_variables

    def override(self, parameters: list):
        """
        ### Description:

        Overrides all the `pymead.core.param.Param`s in `AnchorPoint` which are active and not linked using a list of
        parameters. This list of parameters is likely a subset of parameters passed to either
        `pymead.core.airfoil.Airfoil` or `pymead.core.parametrization.AirfoilParametrization`. This function is
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

        Sets `linked=True` on all `pymead.core.param.Param`s in the `AnchorPoint`
        """
        for param in [var for var in vars(self).values() if isinstance(var, Param)]:
            param.linked = True
