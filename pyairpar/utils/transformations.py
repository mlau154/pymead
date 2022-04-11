import numpy as np


def translate(x, y, dx, dy):
    r"""
    ### Description:

    Translates a point \((x, y)\) by \((\Delta x, \Delta y)\)

    ### Args:

    `x`: \(x\)-location of the point

    `y`: \(y\)-location of the point

    `dx`: \(\Delta x\) by which to translate the \(x\)-coordinate

    `dy`: \(\Delta y\) by which to translate the \(y\)-coordinate

    ### Returns:

    The translated \(x\)- and \(y\)-coordinates
    """
    x += dx
    y += dy
    return x, y


def rotate(x, y, theta):
    r"""
    ### Description:

    Rotates a point \((x, y)\) by an angle \(\theta\)

    ### Args:

    `x`: \(x\)-location of the point

    `y`: \(y\)-location of the point

    `theta`: \(\theta\), the angle by which to rotate the point

    ### Returns:

    The rotated \(x\)- and \(y\)-coordinates
    """
    xy = np.array([[x, y]]).T
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
    new_xy = (rotation_mat @ xy).T.flatten()
    new_x = new_xy[0]
    new_y = new_xy[1]
    return new_x, new_y
