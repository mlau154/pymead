import numpy as np


def translate(x, y, dx, dy, skip_x: bool = False, skip_y: bool = False):
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
    if not skip_x:
        x += dx
    if not skip_y:
        y += dy
    return x, y


def translate_matrix(mat: np.ndarray, dx, dy):
    if not mat.shape[1] == 2:
        raise ValueError("Input matrix must be N x 2")
    mat[:, 0] += dx
    mat[:, 1] += dy
    return mat


def scale(x, y, scale_factor, skip_x: bool = False, skip_y: bool = False):
    if not skip_x:
        x *= scale_factor
    if not skip_y:
        y *= scale_factor
    return x, y


def scale_matrix(mat: np.ndarray, scale_factor):
    if not mat.shape[1] == 2:
        raise ValueError("Input matrix must be N x 2")
    mat *= scale_factor
    return mat


def rotate(x, y, theta, skip_x: bool = False, skip_y: bool = False):
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
    if skip_x:
        new_x = x
    else:
        new_x = new_xy[0]
    if skip_y:
        new_y = y
    else:
        new_y = new_xy[1]
    return new_x, new_y


def rotate_matrix(mat: np.ndarray, theta):
    if not mat.shape[1] == 2:
        raise ValueError("Input matrix must be N x 2")
    xy = mat.T
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
    mat = (rotation_mat @ xy).T
    return mat


def transform(x, y, dx, dy, theta, scale_factor, transformation_order: list,
              skip_x: bool = False, skip_y: bool = False):
    command_dict = {'translate': translate, 'rotate': rotate, 'scale': scale}
    argument_dict = {'translate': {'dx': dx, 'dy': dy}, 'rotate': {'theta': theta},
                     'scale': {'scale_factor': scale_factor}}
    # print(f"x, y before was {x, y}")
    for command in transformation_order:
        x, y = command_dict[command](x, y, **argument_dict[command], skip_x=skip_x, skip_y=skip_y)
        # print(f"x, y after {command} is {x, y}")
    return x, y


def transform_matrix(mat: np.ndarray, dx, dy, theta, scale_factor, transformation_order: list):
    command_dict = {'translate': translate_matrix, 'rotate': rotate_matrix, 'scale': scale_matrix}
    argument_dict = {'translate': {'dx': dx, 'dy': dy}, 'rotate': {'theta': theta},
                     'scale': {'scale_factor': scale_factor}}
    for command in transformation_order:
        mat = command_dict[command](mat, **argument_dict[command])
    return mat
