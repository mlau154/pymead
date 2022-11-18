import numpy as np


class Transformation2D:
    def __init__(self, tx: list = None, ty: list = None, r: list = None, sx: list = None,
                 sy: list = None, rotation_units: str = 'rad', order='r,s,t'):
        """Allows for arbitrary 2D transformations on a set of coordinates of size 2 x N"""

        self.tx = tx if tx is not None else [0.0]
        self.ty = ty if ty is not None else [0.0]
        self.r = r if r is not None else [0.0]
        self.sx = sx if sx is not None else [1.0]
        self.sy = sy if sy is not None else [1.0]

        if rotation_units == 'rad':
            self.r_rad = self.r
            self.r_deg = np.rad2deg(np.array(self.r)).tolist()
        elif rotation_units == 'deg':
            self.r_deg = self.r
            self.r_rad = np.deg2rad(np.array(self.r)).tolist()

        self.order = order

        self.r_mat = None
        self.s_mat = None
        self.t_mat = None

        self.M = np.eye(3)  # 3 x 3 identity matrix

        self.generate_rotation_matrix()
        self.generate_scale_matrix()
        self.generate_translation_matrix()
        self.generate_transformation_matrix()

    def generate_rotation_matrix(self):
        self.r_mat = np.array([np.array([[np.cos(r), -np.sin(r), 0],
                                         [np.sin(r), np.cos(r), 0],
                                         [0, 0, 1]]) for r in self.r_rad])

    def generate_scale_matrix(self):
        self.s_mat = np.array([np.array([[self.sx[idx], 0, 0],
                                         [0, self.sy[idx], 0],
                                         [0, 0, 1]]) for idx in range(len(self.sx))])

    def generate_translation_matrix(self):
        self.t_mat = np.array([np.array([[1, 0, self.tx[idx]],
                                         [0, 1, self.ty[idx]],
                                         [0, 0, 1]]) for idx in range(len(self.tx))])

    def generate_transformation_matrix(self):
        r_count = 0
        s_count = 0
        t_count = 0
        for idx, operation in enumerate(self.order.split(',')):
            if operation == 'r':
                self.M = self.M @ self.r_mat[r_count].T
                r_count += 1
            elif operation == 's':
                self.M = self.M @ self.s_mat[s_count].T
                s_count += 1
            elif operation == 't':
                self.M = self.M @ self.t_mat[t_count].T
                t_count += 1
            else:
                raise TransformationError(f'Invalid value \'{operation}\' found in 2-D transformation '
                                          f'(must be \'r\', \'s\', or \'t\'')

    def transform(self, coordinates: np.ndarray):
        return (np.column_stack((coordinates, np.ones(len(coordinates)))) @ self.M)[:, :2]  # x' = x * M


class TransformationError(Exception):
    pass


if __name__ == '__main__':
    coords = np.array([[1, 0], [2, 1], [3, 3], [2, 0]])
    transform2d = Transformation2D(tx=[1], ty=[1], r=[90], sx=[2], sy=[2], rotation_units='deg', order='r,s,t')
    new_coords = transform2d.transform(coords)
    print(f"new_coords = {new_coords}")
