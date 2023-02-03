import numpy as np
from pymead.core.pos_param import PosParam
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import translate, rotate, scale, transform, transform_matrix


class FreePoint(ControlPoint):

    def __init__(self,
                 xy: PosParam,
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

        super().__init__(xy.value[0], xy.value[1], tag, previous_anchor_point, cp_type='free_point')

        self.ctrlpt = ControlPoint(xy.value[0], xy.value[1], tag, previous_anchor_point, cp_type='free_point')

        self.xy = xy
        self.xy.free_point = self
        self.airfoil_transformation = None
        self.airfoil_tag = airfoil_tag
        self.tag = tag
        self.previous_free_point = previous_free_point

    def set_tag(self, tag: str):
        self.tag = tag
        self.ctrlpt.tag = tag

    def set_xp_yp_value(self, xp, yp):
        # mat = np.array([[xp, yp]])
        # new_mat = transform_matrix(mat, -self.airfoil_transformation['dx'].value,
        #                            -self.airfoil_transformation['dy'].value,
        #                            self.airfoil_transformation['alf'].value,
        #                            1 / self.airfoil_transformation['c'].value,
        #                            ['translate', 'rotate', 'scale'])
        x_changed, y_changed = False, False
        if self.xy.active[0] and not self.xy.linked[0]:
            new_x = xp
            if self.anchor_point_tag == 'le':
                print(f"{new_x = }")
            x_changed = True
        else:
            new_x = self.xy.value[0]
        if self.xy.active[1] and not self.xy.linked[1]:
            new_y = yp
            if self.anchor_point_tag == 'le':
                print(f"{new_y = }")
            y_changed = True
        else:
            new_y = self.xy.value[1]
        self.xy.value = [new_x, new_y]
        # print(f"New FreePoint xy value is {self.xy.value}")

        # If x or y was changed, set the location of the control point to reflect this
        if x_changed or y_changed:
            self.set_ctrlpt_value()

    def transform_xy(self, dx, dy, angle, sf, transformation_order):
        mat = np.array([self.xy.value])
        new_mat = transform_matrix(mat, dx, dy, angle, sf, transformation_order)
        self.xy.value = new_mat[0].tolist()

    def set_ctrlpt_value(self):
        self.ctrlpt.x_val = self.xy.value[0]
        self.ctrlpt.y_val = self.xy.value[1]
        self.ctrlpt.xp = self.xy.value[0]
        self.ctrlpt.yp = self.xy.value[1]
        # self.ctrlpt.transform(self.airfoil_transformation['dx'].value,
        #                       self.airfoil_transformation['dy'].value,
        #                       -self.airfoil_transformation['alf'].value,
        #                       self.airfoil_transformation['c'].value,
        #                       transformation_order=['scale', 'rotate', 'translate'])

    def __repr__(self):
        return f"free_point_{self.tag}"
