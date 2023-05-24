import numpy as np
from pymead.core.pos_param import PosParam
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import transform_matrix


class FreePoint(ControlPoint):

    def __init__(self,
                 xy: PosParam,
                 previous_anchor_point: str,
                 airfoil_tag: str,
                 previous_free_point: str or None = None,
                 tag: str or None = None,
                 ):
        r"""
        The FreePoint in pymead is the way to add a control point to a Bézier curve within an Airfoil
        without requiring the Bézier curve to pass through that particular point. In other words, a FreePoint allows
        an :math:`x`-:math:`y` coordinate pair to be added to the ``P`` matrix (see ``pymead.core.airfoil.bezier`` for
        usage). An example showing some possible locations of FreePoints is shown below.

        Parameters
        ==========
        xy: PosParam
          The location of the FreePoint in :math:`x`-:math:`y` space

        previous_anchor_point: str
          The previous ``AnchorPoint`` (counter-clockwise ordering)

        airfoil_tag: str
          The Airfoil to which this FreePoint belongs

        previous_free_point: str or None
          The previous FreePoint associated with the current FreePoint's AnchorPoint (counter-clockwise ordering). If
          ``None``, the current FreePoint immediately follows the last ControlPoint associated with its AnchorPoint.
          Default: ``None``.

        tag: str or None
          A description of this FreePoint. Default: ``None``.

        Returns
        =======
        FreePoint
          An instance of the ``FreePoint`` class
        """

        super().__init__(xy.value[0], xy.value[1], tag, previous_anchor_point, cp_type='free_point')

        self.ctrlpt = ControlPoint(xy.value[0], xy.value[1], tag, previous_anchor_point, cp_type='free_point')

        self.xy = xy
        self.xy.free_point = self
        self.airfoil_transformation = None
        self.airfoil_tag = airfoil_tag
        self.tag = tag
        self.previous_free_point = previous_free_point

    def set_tag(self, tag: str):
        """
        Sets the tag for this FreePoint and its associated ControlPoint.

        Parameters
        ==========
        tag: str
          A description of the FreePoint
        """
        self.tag = tag
        self.ctrlpt.tag = tag

    def set_xp_yp_value(self, xp, yp):
        """
        Setter for the FreePoint's ``xy`` attribute where the changes are only applied individually for :math:`x` and
        :math:`y` if ``linked==False`` and ``active==True``.

        Parameters
        ==========
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

        # If x or y was changed, set the location of the control point to reflect this
        if x_changed or y_changed:
            self.set_ctrlpt_value()

    def transform_xy(self, dx, dy, angle, sf, transformation_order):
        """
        Transforms the ``xy``-location of the FreePoint.

        Parameters
        ==========
        dx
          Units to translate the ``FreePoint`` in the :math:`x`-direction.

        dy
          Units to translate the ``FreePoint`` in the :math:`y`-direction.

        angle
          Angle, in radians, by which to rotate the FreePoint's location about the origin.

        sf
          Scale factor to apply to the FreePoint's ``xy``-location

        transformation_order: typing.List[str]
          Order in which to apply the transformations. Use ``"s"`` for scale, ``"t"`` for translate, and ``"r"`` for
          rotate
        """
        mat = np.array([self.xy.value])
        new_mat = transform_matrix(mat, dx, dy, angle, sf, transformation_order)
        self.xy.value = new_mat[0].tolist()

    def set_ctrlpt_value(self):
        """
        Sets the :math:`x`- and :math:`y`-values of the FreePoints's ``pymead.core.control_point.ControlPoint``.
        """
        self.ctrlpt.x_val = self.xy.value[0]
        self.ctrlpt.y_val = self.xy.value[1]
        self.ctrlpt.xp = self.xy.value[0]
        self.ctrlpt.yp = self.xy.value[1]

    def __repr__(self):
        return f"free_point_{self.tag}"
