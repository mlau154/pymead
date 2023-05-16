import numpy as np


def translate_matrix(mat: np.ndarray, dx, dy):
    if not mat.shape[1] == 2:
        raise ValueError("Input matrix must be N x 2")
    mat[:, 0] += dx
    mat[:, 1] += dy
    return mat


def scale_matrix(mat: np.ndarray, scale_factor):
    if not mat.shape[1] == 2:
        raise ValueError("Input matrix must be N x 2")
    mat *= scale_factor
    return mat


def rotate_matrix(mat: np.ndarray, theta):
    if not mat.shape[1] == 2:
        raise ValueError("Input matrix must be N x 2")
    xy = mat.T
    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
    mat = (rotation_mat @ xy).T
    return mat


def transform_matrix(mat: np.ndarray, dx, dy, theta, scale_factor, transformation_order: list):
    command_dict = {'translate': translate_matrix, 'rotate': rotate_matrix, 'scale': scale_matrix}
    argument_dict = {'translate': {'dx': dx, 'dy': dy}, 'rotate': {'theta': theta},
                     'scale': {'scale_factor': scale_factor}}
    for command in transformation_order:
        mat = command_dict[command](mat, **argument_dict[command])
    return mat
