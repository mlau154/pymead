import numpy as np
import typing

from pymead.core.param import Param
from pymead.core.pos_param import PosParam
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import transform_matrix
from pymead.core.transformation import AirfoilTransformation


class AnchorPoint(ControlPoint):
    """

    """

    def __init__(self,
                 xy: PosParam,
                 tag: str,
                 previous_anchor_point: str,
                 airfoil_tag: str,
                 L: Param,
                 R: Param,
                 r: Param,
                 phi: Param,
                 psi1: Param,
                 psi2: Param):
        r"""
        The :class:`AnchorPoint` in :class:`pymead` is the way to split a Bézier curve within an
        :class:`pymead.core.airfoil.Airfoil` into two Bézier curves and satisfy :math:`G^0`, :math:`G^1`, and
        :math:`G^2` continuity at the joint between the curves. Examples of implemented :class:`AnchorPoint`\ s in an
        unnecessarily strange airfoil shape are shown in the image below (click on the image to zoom).

        .. image:: ../images/complex_1.png
            :width: 400
            :align: center

        Parameters
        ==========

        xy: PosParam
          :math:`x`- and :math:`y`-location of the :class:`AnchorPoint`

        previous_anchor_point: str
          Name of the previous :class:`AnchorPoint` (counter-clockwise ordering)

        L: Param
          Distance between the control points before and after the :class:`AnchorPoint`

        R: Param
          The radius of curvature at the location of the :class:`AnchorPoint`. A positive value makes the airfoil
          convex at the :class:`AnchorPoint` location, and a negative value makes the airfoil concave at the
          :class:`AnchorPoint` location. A value of ``0`` creates a flat-plate-type leading edge. The valid range is
          :math:`R \in [-\infty, \infty]`. Inclusive brackets are used here because setting :math:`R=\pm \infty`
          is valid and creates an anchor point with no curvature (infinite radius of curvature).

        r: Param
          The ratio of the distance from the :class:`AnchorPoint` location to the neighboring control point closest to
          the trailing edge to the distance between the :class:`AnchorPoint`'s neighboring control points
          ( :math:`L_{fore} / L` ). The valid range is :math:`r \in (0,1)`.

        phi: Param
          The angle of the line passing through the :class:`AnchorPoint`'s neighboring control points, referenced
          counter-clockwise from the chordline if the :class:`AnchorPoint` is on the airfoil's upper surface and
          clockwise from the chordline if the :class:`AnchorPoint` is on the airfoil's lower surface. The valid
          range is :math:`\psi_1 \in [-180^{\circ},180^{\circ}]`. A value of :math:`0^{\circ}` creates an anchor point
          with local slope equal to the slope of the chordline.

        psi1: Param
          The angle of the aft curvature control "arm." Regardless of the sign of :math:`R` or which surface the
          :class:`AnchorPoint` lies on, an angle of :math:`90^{\circ}` always means that the curvature control arm
          points perpendicular to the line passing through the neighboring control points of the :class:`AnchorPoint`.
          Angles below :math:`90^{\circ}` "tuck" the arms in, and angles above :math:`90^{\circ}` "spread" the arms
          out. The valid range is :math:`\psi_1 \in [0^{\circ},180^{\circ}]`.

        psi2: Param
          The angle of the fore curvature control "arm." Regardless of the sign of :math:`R` or which surface
          the :class:`AnchorPoint` lies on, an angle of :math:`90^{\circ}` always means that the curvature control
          arm points perpendicular to the line passing through the neighboring control points of
          the :class:`AnchorPoint`. Angles below :math:`90^{\circ}` "tuck" the arms in, and angles above
          :math:`90^{\circ}` "spread" the arms out. The valid range is :math:`\psi_2 \in [0^{\circ},180^{\circ}]`.

        Returns
        =======

        An :class:`AnchorPoint` instance
        """

        super().__init__(xy.value[0], xy.value[1], tag, previous_anchor_point)

        self.ctrlpt = ControlPoint(xy.value[0], xy.value[1], tag, previous_anchor_point, cp_type='anchor_point')
        self.ctrlpt_branch_list = None

        self.n1 = None
        self.n2 = None

        self.anchor_type = None
        self.tag = tag
        self.previous_anchor_point = previous_anchor_point

        self.xy = xy
        self.airfoil_transformation = None
        self.airfoil_tag = airfoil_tag
        self.L = L
        self.R = R
        self.xy.anchor_point = self

        self.Lt_minus = None
        self.Lt_plus = None
        self.Lc_minus = None
        self.Lc_plus = None
        self.abs_psi1 = None
        self.abs_psi2 = None
        self.abs_phi1 = None
        self.abs_phi2 = None

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

        # if -np.pi <= phi.value <= np.pi:
        self.phi = phi
        # else:
        #     raise ValueError(f'The anchor point neighboring control point angle, phi, must be between -180 degrees and'
        #                      f' 180 degrees, inclusive. A value of {phi.value} was entered.')
        #
        # if 0 <= psi1.value <= np.pi:
        self.psi1 = psi1
        # else:
        #     raise ValueError(f'The aft curvature control arm angle, psi1, must be between 0 degrees and 180 degrees, '
        #                      f'inclusive. '
        #                      f'A value of {psi1.value} was entered.')
        #
        # if 0 <= psi2.value <= np.pi:
        self.psi2 = psi2
        # else:
        #     raise ValueError(f'The fore curvature control arm angle, psi2, must be between 0 degrees and 180 degrees,'
        #                      f'inclusive. '
        #                      f'A value of {psi2.value} was entered.')

    def __repr__(self):
        return f"anchor_point_{self.tag}"

    def set_xp_yp_value(self, xp, yp):
        """
        Setter for the AnchorPoint's ``xy`` attribute where the changes are only applied individually for :math:`x` and
        :math:`y` if ``linked==False`` and ``active==True``.

        xp
          Value to assign to ``self.xy.value[0]``

        yp
          Value to assign to ``self.xy.value[1]``
        """
        x_changed, y_changed = False, False
        if self.xy.active[0] and not self.xy.linked[0]:
            new_x = xp
            x_changed = True
        else:
            new_x = self.xy.value[0]
        if self.xy.active[1] and not self.xy.linked[1]:
            new_y = yp
            y_changed = True
        else:
            new_y = self.xy.value[1]
        self.xy.value = [new_x, new_y]

        # If x or y was changed, set the location of the control point to reflect this:
        if x_changed or y_changed:
            self.set_ctrlpt_value()

    def transform_xy(self, dx, dy, angle, sf, transformation_order: typing.List[str]):
        """
        Transforms the ``xy``-location of the AnchorPoint.

        Parameters
        ==========
        dx
          Units to translate the ``AnchorPoint`` in the :math:`x`-direction.

        dy
          Units to translate the ``AnchorPoint`` in the :math:`y`-direction.

        angle
          Angle, in radians, by which to rotate the AnchorPoint's location about the origin.

        sf
          Scale factor to apply to the AnchorPoint's ``xy``-location

        transformation_order: typing.List[str]
          Order in which to apply the transformations. Use ``"s"`` for scale, ``"t"`` for translate, and ``"r"`` for
          rotate
        """
        mat = np.array([self.xy.value])
        new_mat = transform_matrix(mat, dx, dy, angle, sf, transformation_order)
        self.xy.value = new_mat[0].tolist()

    def set_ctrlpt_value(self):
        """
        Sets the :math:`x`- and :math:`y`-values of the AnchorPoints's ``pymead.core.control_point.ControlPoint``.
        """
        self.ctrlpt.x_val = self.xy.value[0]
        self.ctrlpt.y_val = self.xy.value[1]
        self.ctrlpt.xp = self.xy.value[0]
        self.ctrlpt.yp = self.xy.value[1]

    def get_anchor_type(self, anchor_point_order: list):
        """
        Sets the type of :class:`AnchorPoint` based on the position within the airfoil's anchor point order.

        Parameters
        ==========
        anchor_point_order: list
          Counter-clockwise order of the anchor points in the airfoil
        """
        if self.tag == 'le':
            self.anchor_type = self.tag
        elif anchor_point_order.index(self.tag) < anchor_point_order.index('le'):
            self.anchor_type = 'upper_surf'
        elif anchor_point_order.index(self.tag) > anchor_point_order.index('le'):
            self.anchor_type = 'lower_surf'

    def set_degree_adjacent_bezier_curves(self, n1: int, n2: int):
        """
        Sets the degree of the Bézier curves in contact with the :class:`AnchorPoint`.

        Parameters
        ==========

        n1: int
          Degree of the Bézier curve preceding the :class:`AnchorPoint` in counter-clockwise ordering

        n2: int
          Degree of the Bézier curve proceeding the :class:`AnchorPoint` in counter-clockwise ordering
        """
        self.n1 = n1
        self.n2 = n2

    def generate_anchor_point_branch(self, anchor_point_order: typing.List[str]):
        """
        Generates the set of 5 :class:`ControlPoint`\ s associated with the :class:`AnchorPoint`.

        Parameters
        ==========
        anchor_point_order: typing.List[str]
          The counter-clockwise order of the :class:`AnchorPoint`\ s by name
        """
        r = self.r.value
        L = self.L.value
        phi = self.phi.value
        R = self.R.value
        psi1 = self.psi1.value
        psi2 = self.psi2.value
        tag = self.tag
        if all(t not in tag for t in ['te_1', 'le', 'te_2']):  # for all inserted AnchorPoints:
            xy0_abs = self.xy.value  # the absolute x-y position of the Anchor
            abs_coords = np.array([xy0_abs])
            transform = AirfoilTransformation(**{k: v.value for k, v in self.airfoil_transformation.items()})
            xy = transform.transform_rel(abs_coords)
            xy0 = xy[0].tolist()  # get the position of the Anchor relative to the airfoil coordinate system
        else:
            xy0 = self.xy.value

        if self.n1 is None:
            raise ValueError('Degree of Bezier curve before anchor point was not set before generating the anchor'
                             'point branch')
        else:
            n1 = self.n1

        if self.n2 is None:
            raise ValueError('Degree of Bezier curve after anchor point was not set before generating the anchor'
                             'point branch')
        else:
            n2 = self.n2

        self.get_anchor_type(anchor_point_order)

        def generate_tangent_seg_ctrlpts(minus_plus: str):
            # Generates the control points for the two control point segments adjacent to the AnchorPoint

            if R == 0:  # degenerate case 1: infinite curvature (sharp corner)
                return ControlPoint(xy0[0], xy0[1], f'anchor_point_{tag}_g1_{minus_plus}', tag)

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
                xy = np.array(xy0) + self.Lt_minus * np.array([np.cos(self.abs_phi1), np.sin(self.abs_phi1)])
            else:
                xy = np.array(xy0) + self.Lt_plus * np.array([np.cos(self.abs_phi2), np.sin(self.abs_phi2)])
            return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_{minus_plus}', tag, cp_type=f'g1_{minus_plus}')

        def generate_curvature_seg_ctrlpts(psi, tangent_ctrlpt: ControlPoint, n, minus_plus):
            if R == 0:  # degenerate case 1: infinite curvature (sharp corner)
                return ControlPoint(xy0[0], xy0[1], f'{repr(self)}_g2_{minus_plus}', tag)
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
                            self.abs_psi2 = psi - phi
                        else:
                            self.abs_psi2 = -psi - phi
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
                            self.abs_psi1 = np.pi - psi - phi
                        else:
                            self.abs_psi1 = np.pi + psi - phi
                    elif self.anchor_type == 'le':
                        if R > 0:
                            self.abs_psi2 = -psi + phi
                        else:
                            self.abs_psi2 = np.pi + psi + phi
                    else:
                        raise ValueError("Anchor is of invalid type")

            map_psi_to_airfoil_csys()

            if (minus_plus == 'minus' and self.anchor_type in ['upper_surf', 'le']) or (
                    minus_plus == 'plus' and self.anchor_type == 'lower_surf'):
                apsi = self.abs_psi1
            else:
                apsi = self.abs_psi2

            if minus_plus == 'minus':
                xy = np.array([tangent_ctrlpt.x_val, tangent_ctrlpt.y_val]) + abs(self.Lc_minus) * \
                     np.array([np.cos(apsi), np.sin(apsi)])
            else:
                xy = np.array([tangent_ctrlpt.x_val, tangent_ctrlpt.y_val]) + abs(self.Lc_plus) * \
                     np.array([np.cos(apsi), np.sin(apsi)])

            return ControlPoint(xy[0], xy[1], f'{repr(self)}_g2_{minus_plus}', tag, cp_type=f'g2_{minus_plus}')

        self.g1_minus_ctrlpt = generate_tangent_seg_ctrlpts('minus')
        self.g1_plus_ctrlpt = generate_tangent_seg_ctrlpts('plus')
        if self.anchor_type in ['upper_surf', 'le']:
            self.g2_minus_ctrlpt = generate_curvature_seg_ctrlpts(psi1, self.g1_minus_ctrlpt, n1, 'minus')
            self.g2_plus_ctrlpt = generate_curvature_seg_ctrlpts(psi2, self.g1_plus_ctrlpt, n2, 'plus')
        else:
            self.g2_minus_ctrlpt = generate_curvature_seg_ctrlpts(psi2, self.g1_minus_ctrlpt, n1, 'minus')
            self.g2_plus_ctrlpt = generate_curvature_seg_ctrlpts(psi1, self.g1_plus_ctrlpt, n2, 'plus')

        self.ctrlpt_branch_list = [self.g2_minus_ctrlpt, self.g1_minus_ctrlpt, self.ctrlpt, self.g1_plus_ctrlpt,
                                   self.g2_plus_ctrlpt]

        self.ctrlpt_branch_array = np.array([[self.g2_minus_ctrlpt.xp, self.g2_minus_ctrlpt.yp],
                                             [self.g1_minus_ctrlpt.xp, self.g1_minus_ctrlpt.yp],
                                             [self.xp, self.yp],
                                             [self.g1_plus_ctrlpt.xp, self.g1_plus_ctrlpt.yp],
                                             [self.g2_plus_ctrlpt.xp, self.g2_plus_ctrlpt.yp]])

        self.ctrlpt_branch_generated = True

    def recalculate_ap_branch_props_from_g2_pt(self, minus_plus: str, measured_psi, measured_Lc):
        """
        Recalculates ``R`` and ``psi1`` or ``psi2`` based on the measured length and absolute angle of the curvature
        control arm segment. Used only in the GUI.

        Parameters
        ==========
        minus_plus: str
          Whether the :math:`G^2` point comes before or after the :class:`AnchorPoint` in counter-clockwise ordering.
          If before, input ``"minus"``. If after, input ``"plus"``.

        measured_psi
          The measured absolute angle of the curvature control arm

        measured_Lc
          The measured length of the curvature control arm
        """
        apsi = None
        if measured_psi is not None:
            if (minus_plus == 'minus' and self.anchor_type in ['upper_surf', 'le']) or (
                    minus_plus == 'plus' and self.anchor_type == 'lower_surf'):
                self.abs_psi1 = measured_psi
                apsi = self.abs_psi1
            else:
                self.abs_psi2 = measured_psi
                apsi = self.abs_psi2

        if minus_plus == 'minus':
            # The following logic block is to ensure that the curvature control arm angle (psi) uses the correct
            # coordinate system:
            if measured_Lc is not None:
                self.Lc_minus = measured_Lc
            if 0 < np.arctan2(np.sin(apsi - self.abs_phi1), np.cos(apsi - self.abs_phi1)) < np.pi:
                sign_R = -1
            else:
                sign_R = 1
        else:
            if measured_Lc is not None:
                self.Lc_plus = measured_Lc
            if 0 < np.arctan2(np.sin(apsi - self.abs_phi2), np.cos(apsi - self.abs_phi2)) < np.pi:
                sign_R = 1
            else:
                sign_R = -1

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
                            self.psi2.value = self.abs_psi2 + self.phi.value
                        else:
                            # angle = -psi - phi
                            self.psi2.value = -self.abs_psi2 - self.phi.value
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
                            self.psi1.value = np.pi - self.abs_psi1 - self.phi.value
                        else:
                            # angle = np.pi + psi - phi
                            self.psi1.value = self.abs_psi1 - np.pi + self.phi.value
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

        if (minus_plus == 'minus' and self.anchor_type in ['upper_surf', 'le']) or (
                minus_plus == 'plus' and self.anchor_type == 'lower_surf'):
            apsi = self.psi1.value
        else:
            apsi = self.psi2.value

        if self.R.active and not self.R.linked:
            if self.tag == 'le':
                if minus_plus == 'minus':
                    self.R.value = self.Lt_minus ** 2 / (
                                self.Lc_minus * (1 - 1 / self.n1) * np.sin(apsi + np.pi / 2))
                else:
                    self.R.value = self.Lt_plus ** 2 / (
                            self.Lc_plus * (1 - 1 / self.n2) * np.sin(apsi + np.pi / 2))
            else:
                if minus_plus == 'minus':
                    self.R.value = self.Lt_minus ** 2 / (self.Lc_minus * (1 - 1 / self.n1) * np.sin(apsi))
                else:
                    self.R.value = self.Lt_plus ** 2 / (self.Lc_plus * (1 - 1 / self.n2) * np.sin(apsi))
            if negate_R:
                self.R.value *= -1

    def recalculate_ap_branch_props_from_g1_pt(self, minus_plus: str, measured_phi, measured_Lt):
        """
        Recalculates ``phi``, ``r``, and ``L`` based on the measured length and absolute angle of the tangent
        control arm segment. Used only in the GUI.

        Parameters
        ==========
        minus_plus: str
          Whether the :math:`G^1` point comes before or after the :class:`AnchorPoint` in counter-clockwise ordering.
          If before, input ``"minus"``. If after, input ``"plus"``.

        measured_phi
          The measured absolute angle of the tangent control arm

        measured_Lt
          The measured length of the tangent control arm
        """

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
