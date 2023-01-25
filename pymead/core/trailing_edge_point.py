from pymead.core.param import Param
from pymead.core.control_point import ControlPoint
import numpy as np


class TrailingEdgePoint(ControlPoint):

    def __init__(self,
                 c: Param,
                 r: Param,
                 t: Param,
                 phi: Param,
                 L: Param,
                 theta: Param,
                 upper: bool
                 ):

        self.c = c
        self.r = r
        self.t = t
        self.phi = phi
        self.L = L
        self.theta = theta
        self.upper = upper

        self.ctrlpt = None
        self.tangent_ctrlpt = None
        self.ctrlpt_branch_array = None
        self.ctrlpt_branch_list = None
        self.ctrlpt_branch_generated = False

        if self.upper:
            xy = np.array([1, 0]) + self.r.value * self.t.value * np.array([np.cos(np.pi / 2 + self.phi.value),
                                                                                  np.sin(np.pi / 2 + self.phi.value)])
            tag = 'te_1'
        else:
            xy = np.array([1, 0]) + (1 - self.r.value) * self.t.value * \
                 np.array([np.cos(3 * np.pi / 2 + self.phi.value), np.sin(3 * np.pi / 2 + self.phi.value)])
            tag = 'te_2'

        super().__init__(xy[0], xy[1], tag, tag)

        self.ctrlpt = ControlPoint(xy[0], xy[1], tag, tag, cp_type='anchor_point')

    def __repr__(self):
        return f"anchor_point_{self.tag}"

    def set_te_points(self):
        if self.upper:
            xy = np.array([1, 0]) + self.r.value * self.t.value * np.array([np.cos(np.pi / 2 + self.phi.value),
                                                                                  np.sin(np.pi / 2 + self.phi.value)])
            tag = 'te_1'
        else:
            xy = np.array([1, 0]) + (1 - self.r.value) * self.t.value * \
                 np.array([np.cos(3 * np.pi / 2 + self.phi.value), np.sin(3 * np.pi / 2 + self.phi.value)])
            tag = 'te_2'

        super().__init__(xy[0], xy[1], tag, tag)

        self.ctrlpt = ControlPoint(xy[0], xy[1], tag, tag, cp_type='anchor_point')

    def generate_anchor_point_branch(self):

        self.set_te_points()

        def generate_tangent_seg_ctrlpts():

            self.ctrlpt_branch_generated = True

            if self.upper:
                xy = np.array([self.x_val, self.y_val]) + self.L.value * np.array([np.cos(np.pi - self.theta.value),
                                                                                   np.sin(np.pi - self.theta.value)])
                return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_plus', self.tag, cp_type='g1_plus')
            else:
                xy = np.array([self.x_val, self.y_val]) + self.L.value * np.array([np.cos(np.pi + self.theta.value),
                                                                                   np.sin(np.pi + self.theta.value)])
                return ControlPoint(xy[0], xy[1], f'{repr(self)}_g1_minus', self.tag, cp_type='g1_minus')

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
        if self.L.active and not self.L.linked:
            self.L.value = measured_Lt
        if self.theta.active and not self.theta.linked:
            if minus_plus == 'minus':
                self.theta.value = measured_phi - np.pi
            else:
                self.theta.value = -measured_phi + np.pi
