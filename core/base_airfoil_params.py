from core.param import Param
import numpy as np


class BaseAirfoilParams:

    def __init__(self,
                 c: Param = Param(1.0),                         # chord length
                 alf: Param = Param(0.0),                       # angle of attack (rad)
                 R_le: Param = Param(0.1, 'length'),            # leading edge radius
                 L_le: Param = Param(0.1, 'length'),            # leading edge length
                 r_le: Param = Param(0.5),                      # leading edge length ratio
                 phi_le: Param = Param(0.0),                    # leading edge 'tilt' angle
                 psi1_le: Param = Param(0.0),                   # leading edge upper curvature control angle
                 psi2_le: Param = Param(0.0),                   # leading edge lower curvature control angle
                 L1_te: Param = Param(0.1, 'length'),           # trailing edge upper length
                 L2_te: Param = Param(0.1, 'length'),           # trailing edge lower length
                 theta1_te: Param = Param(np.deg2rad(10.0)),    # trailing edge upper angle
                 theta2_te: Param = Param(np.deg2rad(10.0)),    # trailing edge lower angle
                 t_te: Param = Param(0.0, 'length'),            # blunt trailing edge thickness
                 r_te: Param = Param(0.5),                      # blunt trailing edge thickness length ratio
                 phi_te: Param = Param(0.0),                    # blunt trailing edge 'tilt' angle
                 non_dim_by_chord: bool = True,
                 ):

        self.c = c
        self.alf = alf
        self.R_le = R_le
        self.L_le = L_le
        self.r_le = r_le
        self.phi_le = phi_le
        self.psi1_le = psi1_le
        self.psi2_le = psi2_le
        self.L1_te = L1_te
        self.L2_te = L2_te
        self.theta1_te = theta1_te
        self.theta2_te = theta2_te
        self.t_te = t_te
        self.r_te = r_te
        self.phi_te = phi_te
        self.non_dim_by_chord = non_dim_by_chord
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()

    def scale_vars(self):
        if self.non_dim_by_chord:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.length_scale_dimension is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.c.value

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
