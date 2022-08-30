from pyairpar.core.param import Param
from pyairpar.core.control_point import ControlPoint
import numpy as np


class TrailingEdgePoint(ControlPoint):

    def __init__(self,
                 c: Param,
                 r: Param,
                 t: Param,
                 phi: Param,
                 L: Param,
                 theta: Param,
                 upper: bool,
                 length_scale_dimension: float = None
                 ):

        self.c = c
        self.r = r
        self.t = t
        self.phi = phi
        self.L = L
        self.theta = theta
        self.upper = upper
        self.length_scale_dimension = length_scale_dimension

        self.ctrlpt = None
        self.tangent_ctrlpt = None
        self.ctrlpt_branch_array = None
        self.ctrlpt_branch_list = None
        self.ctrlpt_branch_generated = False

        if self.upper:
            xy = np.array([c.value, 0]) + self.r.value * self.t.value * np.array([np.cos(np.pi / 2 + self.phi.value),
                                                                                  np.sin(np.pi / 2 + self.phi.value)])
            name = 'te_1'
        else:
            xy = np.array([c.value, 0]) + (1 - self.r.value) * self.t.value * \
                 np.array([np.cos(3 * np.pi / 2 + self.phi.value), np.sin(3 * np.pi / 2 + self.phi.value)])
            name = 'te_2'

        super().__init__(xy[0], xy[1], name, name)

        self.ctrlpt = ControlPoint(xy[0], xy[1], name, name, cp_type='anchor_point')

    def scale_vars(self):
        """
        ### Description:

        Scales all of the `pyairpar.core.param.Param`s in the `AnchorPoint` with `units == 'length'` by the
        `length_scale_dimension`. Scaling only occurs for each parameter if the `pyairpar.core.param.Param` has not yet
        been scaled.
        """
        if self.length_scale_dimension is not None:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.scale_value is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.length_scale_dimension

    def __repr__(self):
        return f"anchor_point_{self.name}"

    def generate_anchor_point_branch(self):

        def generate_tangent_seg_ctrlpts():

            self.ctrlpt_branch_generated = True

            if self.upper:
                xy = np.array([self.x_val, self.y_val]) + self.L.value * np.array([np.cos(np.pi - self.theta.value),
                                                                                   np.sin(np.pi - self.theta.value)])
                return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_plus', self.name, cp_type='g1_plus')
            else:
                xy = np.array([self.x_val, self.y_val]) + self.L.value * np.array([np.cos(np.pi + self.theta.value),
                                                                                   np.sin(np.pi + self.theta.value)])
                return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_minus', self.name, cp_type='g1_minus')

        self.tangent_ctrlpt = generate_tangent_seg_ctrlpts()

        if self.upper:
            self.ctrlpt_branch_array = np.array([[self.xp, self.yp],
                                                 [self.tangent_ctrlpt.xp, self.tangent_ctrlpt.yp]])
            self.ctrlpt_branch_list = [self.ctrlpt, self.tangent_ctrlpt]
        else:
            self.ctrlpt_branch_array = np.array([[self.tangent_ctrlpt.xp, self.tangent_ctrlpt.yp],
                                                 [self.xp, self.yp]])
            self.ctrlpt_branch_list = [self.tangent_ctrlpt, self.ctrlpt]

    def recalculate_ap_branch_props_from_g1_pt(self, minus_plus: str, measured_phi, measured_Lt):
        self.L.value = measured_Lt
        if minus_plus == 'minus':
            self.theta.value = measured_phi - np.pi
        else:
            self.theta.value = -measured_phi + np.pi

    def count_overrideable_variables(self):
        """
        ### Description:

        Counts all the overrideable `pyairpar.core.param.Param`s in the `AnchorPoint` (criteria:
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

        Overrides all the `pyairpar.core.param.Param`s in `AnchorPoint` which are active and not linked using a list of
        parameters. This list of parameters is likely a subset of parameters passed to either
        `pyairpar.core.airfoil.Airfoil` or `pyairpar.core.parametrization.AirfoilParametrization`. This function is
        useful whenever iteration over only the relevant parameters is required.

        ### Args:

        `parameters`: a `list` of parameters
        """
        override_param_obj_list = [var for var in vars(self).values()
                                   if isinstance(var, Param) and var.active and not var.linked]
        if len(parameters) != len(override_param_obj_list):
            raise Exception('Number of base airfoil parameters does not match length of input override parameter list')
        param_idx = 0
        for param in override_param_obj_list:
            setattr(param, 'value', parameters[param_idx])
            param_idx += 1
        self.scale_vars()

    def set_all_as_linked(self):
        """
        ### Description:

        Sets `linked=True` on all `pyairpar.core.param.Param`s in the `AnchorPoint`
        """
        for param in [var for var in vars(self).values() if isinstance(var, Param)]:
            param.linked = True
