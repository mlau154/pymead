from pyairpar.core.param import Param
import numpy as np


class SymmetricBaseAirfoilParams:

    def __init__(self,
                 c: Param = Param(1.0),  # chord length
                 alf: Param = Param(0.0),  # angle of attack (rad)
                 R_le: Param = Param(0.1, 'length'),  # leading edge radius
                 L_le: Param = Param(0.1, 'length'),  # leading edge length
                 psi1_le: Param = Param(0.0),  # leading edge upper curvature control angle
                 L1_te: Param = Param(0.1, 'length'),  # trailing edge upper length
                 theta1_te: Param = Param(np.deg2rad(10.0)),  # trailing edge upper angle
                 t_te: Param = Param(0.0, 'length'),  # blunt trailing edge thickness
                 dx: Param = Param(0.0, active=False),  # dx to translate
                 dy: Param = Param(0.0, active=False),  # dy to translate
                 non_dim_by_chord: bool = True):                # Non-dimensionalize by chord length?

        self.c = c
        self.alf = alf
        self.R_le = R_le
        self.L_le = L_le
        self.r_le = Param(0.5, active=False)
        self.phi_le = Param(0.0, active=False)
        self.psi1_le = psi1_le
        self.psi2_le = Param(psi1_le.value, active=False, linked=True)
        self.L1_te = L1_te
        self.L2_te = Param(L1_te.value, 'length', active=False, linked=True)
        self.theta1_te = theta1_te
        self.theta2_te = Param(theta1_te.value, active=False, linked=True)
        self.t_te = t_te
        self.r_te = Param(0.5, active=False)
        self.phi_te = Param(0.0, active=False)
        self.dx = dx
        self.dy = dy
        self.non_dim_by_chord = non_dim_by_chord
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()

    def scale_vars(self):
        """
        ### Description:

        Scales all of the `pyairpar.core.param.Param`s in the `SymmetricBaseAirfoilParams` with `units == 'length'`
        by the `length_scale_dimension`. Scaling only occurs for each parameter if the `pyairpar.core.param.Param` has
        not yet been scaled.
        """
        if self.non_dim_by_chord:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.scale_value is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.c.value

    def count_overrideable_variables(self):
        """
        ### Description:

        Counts all the overrideable `pyairpar.core.param.Param`s in the `SymmetricBaseAirfoilParams` (criteria:
        `pyairpar.core.param.Param().active == True`, `pyairpar.core.param.Param().linked == False`)

        ### Returns:

        Number of overrideable variables (`int`)
        """
        n_overrideable_variables = len([var for var in vars(self).values()
                                        if isinstance(var, Param) and var.active and not var.linked])
        return n_overrideable_variables

    def override(self, parameters: list):
        """
        ### Description:

        Overrides all the `pyairpar.core.param.Param`s in `SymmetricBaseAirfoilParams` which are active and not
        linked using a list of parameters. This list of parameters is likely a subset of parameters passed to either
        `pyairpar.core.airfoil.Airfoil` or `pyairpar.core.parametrization.AirfoilParametrization`. This function is
        useful whenever iteration over only the relevant parameters is required.

        ### Args:

        `parameters`: a `list` of parameters
        """
        override_param_obj_list = [var for var in vars(self).values()
                                   if isinstance(var, Param) and var.active and not var.linked]
        if len(parameters) != len(override_param_obj_list):
            raise Exception('Number of symmetric base airfoil parameters does not match length of input override '
                            'parameter list')
        param_idx = 0
        for param in override_param_obj_list:
            setattr(param, 'value', parameters[param_idx])
            param_idx += 1
        self.scale_vars()
