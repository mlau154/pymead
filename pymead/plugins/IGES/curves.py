import numpy as np

from pymead.plugins.IGES.entity import Entity
from pymead.plugins.IGES.iges_param import IGESParam


class RationalBSplineIGES(Entity):
    """
    IGES Entity #126
    """
    def __init__(self, knots: np.ndarray, weights: np.ndarray,
                 control_points_XYZ: np.ndarray, degree: int, start_parameter_value=0.0, end_parameter_value=1.0,
                 unit_normal_x=0.0, unit_normal_y=0.0, unit_normal_z=0.0, planar_flag: bool = False,
                 closed_flag: bool = False, polynomial_flag: bool = False, periodic_flag: bool = False):
        self.upper_index_of_sum = control_points_XYZ.shape[0] - 1
        self.degree = degree
        self.flag1 = int(planar_flag)
        self.flag2 = int(closed_flag)
        self.flag3 = int(polynomial_flag)
        self.flag4 = int(periodic_flag)
        self.knots = knots
        self.weights = weights
        self.control_points = control_points_XYZ
        self.v0 = start_parameter_value
        self.v1 = end_parameter_value
        self.XN = unit_normal_x
        self.YN = unit_normal_y
        self.ZN = unit_normal_z
        parameter_data = [
            IGESParam(self.upper_index_of_sum, "int"),
            IGESParam(self.degree, "int"),
            IGESParam(self.flag1, "int"),
            IGESParam(self.flag2, "int"),
            IGESParam(self.flag3, "int"),
            IGESParam(self.flag4, "int"),
            *[IGESParam(k, "real") for k in self.knots],
            *[IGESParam(w, "real") for w in self.weights],
            *[IGESParam(xyz, "real") for xyz in self.control_points.flatten()],
            IGESParam(self.v0, "real"),
            IGESParam(self.v1, "real"),
            IGESParam(self.XN, "real"),
            IGESParam(self.YN, "real"),
            IGESParam(self.ZN, "real"),
        ]
        super().__init__(126, parameter_data)


class BezierIGES(RationalBSplineIGES):
    def __init__(self, control_points_XYZ: np.ndarray, start_parameter_value=0.0, end_parameter_value=1.0,
                 unit_normal_x=0.0, unit_normal_y=0.0, unit_normal_z=0.0, planar_flag: bool = False,
                 closed_flag: bool = False, periodic_flag: bool = False):
        order = len(control_points_XYZ)
        degree = order - 1
        knots = np.concatenate((np.zeros(order), np.ones(order)))
        weights = np.ones(order)
        polynomial_flag = True
        super().__init__(knots=knots, weights=weights, control_points_XYZ=control_points_XYZ, degree=degree,
                         start_parameter_value=start_parameter_value, end_parameter_value=end_parameter_value,
                         unit_normal_x=unit_normal_x, unit_normal_y=unit_normal_y, unit_normal_z=unit_normal_z,
                         planar_flag=planar_flag, closed_flag=closed_flag, polynomial_flag=polynomial_flag,
                         periodic_flag=periodic_flag)
