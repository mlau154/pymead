import numpy as np
from pymead.core.param import Param
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import translate, rotate, scale, transform, transform_matrix


class FreePoint(ControlPoint):

    def __init__(self,
                 x: Param,
                 y: Param,
                 previous_anchor_point: str,
                 airfoil_tag: str,
                 previous_free_point: str or None = None,
                 tag: str or None = None,
                 ):
        """
        ### Description:

        The `FreePoint` in `pymead` is the way to add a control point to a Bézier curve within an `Airfoil`
        without requiring the Bézier curve to pass through that particular point. In other words, a FreePoint allows
        an \\(x\\) - \\(y\\) coordinate pair to be added to the `P` matrix (see `pymead.core.airfoil.bezier` for
        usage). An example showing some possible locations of `FreePoint`s is shown below.

        ### Args:

        `x`: `Param` describing the x-location of the `FreePoint`

        `y`: `Param` describing the y-location of the `FreePoint`

        `previous_anchor_point`: a `str` representing the previous `AnchorPoint` (counter-clockwise ordering)

        ### Returns:

        An instance of the `FreePoint` class
        """

        super().__init__(x.value, y.value, tag, previous_anchor_point, cp_type='free_point')

        self.ctrlpt = ControlPoint(x.value, y.value, tag, previous_anchor_point, cp_type='free_point')

        self.x = x
        self.y = y
        self.x.free_point = self
        self.y.free_point = self
        self.airfoil_transformation = None
        self.airfoil_tag = airfoil_tag
        self.tag = tag
        self.previous_free_point = previous_free_point

    def set_tag(self, tag: str):
        self.tag = tag
        self.ctrlpt.tag = tag

    def set_xp_yp_value(self, xp, yp):
        mat = np.array([[xp, yp]])
        new_mat = transform_matrix(mat, -self.airfoil_transformation['dx'].value,
                                   -self.airfoil_transformation['dy'].value,
                                   self.airfoil_transformation['alf'].value,
                                   1 / self.airfoil_transformation['c'].value,
                                   ['translate', 'rotate', 'scale'])
        x_changed, y_changed = False, False
        if self.x.active and not self.x.linked:
            self.x.value = new_mat[0, 0]
            x_changed = True
        if self.y.active and not self.y.linked:
            self.y.value = new_mat[0, 1]
            y_changed = True

        # If x or y was changed, set the location of the control point to reflect this
        if x_changed or y_changed:
            self.set_ctrlpt_value()

    def set_ctrlpt_value(self):
        self.ctrlpt.x_val = self.x.value
        self.ctrlpt.y_val = self.y.value
        self.ctrlpt.xp = self.x.value
        self.ctrlpt.yp = self.y.value
        self.ctrlpt.transform(self.airfoil_transformation['dx'].value,
                              self.airfoil_transformation['dy'].value,
                              -self.airfoil_transformation['alf'].value,
                              self.airfoil_transformation['c'].value,
                              transformation_order=['scale', 'rotate', 'translate'])

    def __repr__(self):
        return f"free_point_{self.tag}"
