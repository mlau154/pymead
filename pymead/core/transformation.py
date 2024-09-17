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
        """Computes the transformation of the coordinates.

        Parameters
        ==========
        coordinates: np.ndarray
          Size N x 2, where N is the number of coordinates. The columns represent :math:`x` and :math:`y`.
        """
        return (np.column_stack((coordinates, np.ones(len(coordinates)))) @ self.M)[:, :2]  # x' = x * M


class Transformation3D:
    def __init__(self, tx: list = None, ty: list = None, tz: list = None, rx: list = None, ry: list = None,
                 rz: list = None, sx: list = None, sy: list = None, sz: list = None,
                 rotation_units: str = 'rad', order='rx,ry,rz,s,t'):
        """Allows for arbitrary 3D transformations on a set of coordinates of size 3 x N"""

        self.tx = tx if tx is not None else [0.0]
        self.ty = ty if ty is not None else [0.0]
        self.tz = tz if tz is not None else [0.0]
        self.rx = rx if rx is not None else [0.0]
        self.ry = ry if ry is not None else [0.0]
        self.rz = rz if rz is not None else [0.0]
        self.sx = sx if sx is not None else [1.0]
        self.sy = sy if sy is not None else [1.0]
        self.sz = sz if sz is not None else [1.0]

        if rotation_units == 'rad':
            self.rx_rad = self.rx
            self.rx_deg = np.rad2deg(np.array(self.rx)).tolist()
            self.ry_rad = self.ry
            self.ry_deg = np.rad2deg(np.array(self.ry)).tolist()
            self.rz_rad = self.rz
            self.rz_deg = np.rad2deg(np.array(self.rz)).tolist()
        elif rotation_units == 'deg':
            self.rx_deg = self.rx
            self.rx_rad = np.deg2rad(np.array(self.rx)).tolist()
            self.ry_deg = self.ry
            self.ry_rad = np.deg2rad(np.array(self.ry)).tolist()
            self.rz_deg = self.rz
            self.rz_rad = np.deg2rad(np.array(self.rz)).tolist()

        self.order = order.split(',')

        self.rx_mat = None
        self.ry_mat = None
        self.rz_mat = None
        self.s_mat = None
        self.t_mat = None

        self.M = np.eye(4)  # 4 x 4 identity matrix

        self.generate_rotation_matrix_x()
        self.generate_rotation_matrix_y()
        self.generate_rotation_matrix_z()
        self.generate_scale_matrix()
        self.generate_translation_matrix()
        self.generate_transformation_matrix()

    def generate_rotation_matrix_x(self):
        self.rx_mat = np.array([np.array([[1, 0, 0, 0],
                                          [0, np.cos(r), np.sin(r), 0],
                                          [0, -np.sin(r), np.cos(r), 0],
                                          [0, 0, 0, 1]]
                                         ) for r in self.rx_rad])

    def generate_rotation_matrix_y(self):
        self.ry_mat = np.array([np.array([[np.cos(r), 0, -np.sin(r), 0],
                                          [0, 1, 0, 0],
                                          [np.sin(r), 0, np.cos(r), 0],
                                          [0, 0, 0, 1]]
                                         ) for r in self.ry_rad])

    def generate_rotation_matrix_z(self):
        self.rz_mat = np.array([np.array([[np.cos(r), -np.sin(r), 0, 0],
                                          [np.sin(r), np.cos(r), 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]]
                                         ) for r in self.rz_rad])

    def generate_scale_matrix(self):
        self.s_mat = np.array([np.array([[self.sx[idx], 0, 0, 0],
                                         [0, self.sy[idx], 0, 0],
                                         [0, 0, self.sz[idx], 0],
                                         [0, 0, 0, 1]]) for idx in range(len(self.sx))])

    def generate_translation_matrix(self):
        self.t_mat = np.array([np.array([[1, 0, 0, self.tx[idx]],
                                         [0, 1, 0, self.ty[idx]],
                                         [0, 0, 1, self.tz[idx]],
                                         [0, 0, 0, 1]]) for idx in range(len(self.tx))])

    def generate_transformation_matrix(self):
        rx_count = 0
        ry_count = 0
        rz_count = 0
        s_count = 0
        t_count = 0
        for idx, operation in enumerate(self.order):
            if operation == 'rx':
                self.M = self.M @ self.rx_mat[rx_count].T
                rx_count += 1
            elif operation == 'ry':
                self.M = self.M @ self.ry_mat[ry_count].T
                ry_count += 1
            elif operation == 'rz':
                self.M = self.M @ self.rz_mat[rz_count].T
                rz_count += 1
            elif operation == 's':
                self.M = self.M @ self.s_mat[s_count].T
                s_count += 1
            elif operation == 't':
                self.M = self.M @ self.t_mat[t_count].T
                t_count += 1
            else:
                raise TransformationError(f'Invalid value \'{operation}\' found in 3-D transformation order '
                                          f'(must be \'rx\', \'ry\', \'rz\', \'s\', or \'t\'')

    def transform(self, coordinates: np.ndarray):
        """Computes the transformation of the coordinates.

        Parameters
        ==========
        coordinates: np.ndarray
          Size N x 3, where N is the number of coordinates. The columns represent :math:`x` and :math:`y`.
        """
        return (np.column_stack((coordinates, np.ones(len(coordinates)))) @ self.M)[:, :3]  # x' = x * M


class TransformationError(Exception):
    pass


class AirfoilTransformation:
    """Convenience class for computing transformations of coordinate pairs :math:`(x,y)` to and from the
    airfoil-relative coordinate system."""
    def __init__(self, dx, dy, alf, c):
        self.transform_abs_obj = Transformation2D(tx=[dx], ty=[dy], r=[-alf], sx=[c], sy=[c],
                                                  rotation_units='rad', order='r,s,t')
        self.transform_rel_obj = Transformation2D(tx=[-dx], ty=[-dy], r=[alf], sx=[1.0 / c], sy=[1.0 / c],
                                                  rotation_units='rad', order='t,s,r')

    def transform_abs(self, coordinates: np.ndarray):
        """Computes the transformation of the coordinates from the airfoil-relative coordinate system to the absolute
        coordinate system.

        Parameters
        ==========
        coordinates: np.ndarray
          Size N x 2, where N is the number of coordinates. The columns represent :math:`x` and :math:`y`.
        """
        return self.transform_abs_obj.transform(coordinates)

    def transform_rel(self, coordinates: np.ndarray):
        """Computes the transformation of the coordinates from the absolute coordinate system to the airfoil-relative
        coordinate system.

        Parameters
        ==========
        coordinates: np.ndarray
          Size N x 2, where N is the number of coordinates. The columns represent :math:`x` and :math:`y`.
        """
        return self.transform_rel_obj.transform(coordinates)
