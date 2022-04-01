import numpy as np
from core.param import Param


class FreePoint:

    def __init__(self,
                 x: Param,
                 y: Param,
                 previous_anchor_point: str,
                 length_scale_dimension: float = None
                 ):

        self.x = x
        self.y = y
        self.previous_anchor_point = previous_anchor_point
        self.length_scale_dimension = length_scale_dimension
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()
        self.xy = np.array([self.x.value, self.y.value])

    def scale_vars(self):
        if self.length_scale_dimension is not None:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.length_scale_dimension is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.length_scale_dimension

    def count_overrideable_variables(self):
        n_overrideable_variables = len([var for var in vars(self).values()
                                        if isinstance(var, Param) and var.active and not var.linked])
        return n_overrideable_variables

    def override(self, parameters: list):
        override_param_obj_list = [var for var in vars(self).values()
                                   if isinstance(var, Param) and var.active and not var.linked]
        if len(parameters) != len(override_param_obj_list):
            raise Exception('Number of base airfoil parameters does not match length of input override parameter list')
        param_idx = 0
        for param in override_param_obj_list:
            setattr(param, 'value', parameters[param_idx])
            param_idx += 1

        self.scale_vars()
        self.xy = np.array([self.x.value, self.y.value])
