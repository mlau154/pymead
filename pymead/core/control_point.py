from pymead.core.param import Param
from math import pi, sin, cos
from pymead.utils.transformations import transform_matrix
import numpy as np

from copy import deepcopy


class ControlPoint:
    def __init__(self, x, y, tag: str, anchor_point_tag: str, cp_type: str = None):
        r"""
        ### Description:

        Base class for `pymead.core.anchor_point.AnchorPoint`s and `pymead.core.free_point.FreePoint`s.

        ### Args:

        `x`: a `pymead.core.param.Param` indicating the $x$-location
        """
        self.x_val = x
        self.y_val = y
        self.anchor_point_tag = anchor_point_tag
        self.tag = tag
        self.xp = deepcopy(self.x_val)
        self.yp = deepcopy(self.y_val)
        self.cp_type = cp_type

    def __repr__(self):
        return f"control_point_{self.tag}"

    # def translate(self, dx, dy):
    #     self.xp += dx
    #     self.yp += dy
    #
    # def rotate(self, angle):
    #     self.xp = self.xp * cos(angle) - self.yp * sin(angle)
    #     self.yp = self.yp * cos(angle) + self.xp * sin(angle)
    #
    # def scale(self, sf):
    #     self.xp *= sf
    #     self.yp *= sf

    # def transform(self, dx, dy, angle, sf, transformation_order):
    #     mat = np.array([[self.xp, self.yp]])
    #     new_mat = transform_matrix(mat, dx, dy, angle, sf, transformation_order)
    #     self.xp = new_mat[0][0]
    #     self.yp = new_mat[0][1]

    def transform(self, dx, dy, angle, sf, transformation_order):
        mat = np.array([[self.x_val, self.y_val]])
        new_mat = transform_matrix(mat, -dx, -dy, -angle, 1 / sf, transformation_order[::-1])
        self.x_val = new_mat[0][0]
        self.y_val = new_mat[0][1]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ctrlpt = ControlPoint(Param(0.5), Param(0.0), 'cpt', 'le')
    ctrlpt.transform(0.0, -0.1, 90)
    plt.plot(ctrlpt.x_val, ctrlpt.y_val, 'bo')
    plt.plot(ctrlpt.xp, ctrlpt.yp, 'go')
    plt.show()
