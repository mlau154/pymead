from pyairpar.core.param import Param
from math import pi, sin, cos

from copy import deepcopy


class ControlPoint:
    def __init__(self, x, y, name: str, anchor_point_tag: str, cp_type: str = None):
        r"""
        ### Description:

        Base class for `pyairpar.core.anchor_point.AnchorPoint`s and `pyairpar.core.free_point.FreePoint`s.

        ### Args:

        `x`: a `pyairpar.core.param.Param` indicating the $x$-location
        """
        self.x_val = x
        self.y_val = y
        self.anchor_point_tag = anchor_point_tag
        self.name = name
        self.xp = deepcopy(self.x_val)
        self.yp = deepcopy(self.y_val)
        self.cp_type = cp_type

    def __repr__(self):
        return f"control_point_{self.name}"

    def translate(self, dx, dy):
        self.xp += dx
        self.yp += dy

    def rotate(self, angle, units: str = 'deg'):
        if units not in ['rad', 'deg']:
            raise ValueError('Units must be either radians (\'rad\') or degrees (\'deg\')')
        if units == 'deg':
            angle = angle * pi / 180

        self.xp = self.xp * cos(angle) - self.yp * sin(angle)
        self.yp = self.yp * cos(angle) + self.xp * sin(angle)

    def transform(self, dx: int or float, dy: int or float, angle: int or float, rotation_units: str = 'deg'):
        self.rotate(angle, rotation_units)
        self.translate(dx, dy)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ctrlpt = ControlPoint(Param(0.5), Param(0.0), 'cpt', 'le')
    ctrlpt.transform(0.0, -0.1, 90)
    plt.plot(ctrlpt.x_val, ctrlpt.y_val, 'bo')
    plt.plot(ctrlpt.xp, ctrlpt.yp, 'go')
    plt.show()
