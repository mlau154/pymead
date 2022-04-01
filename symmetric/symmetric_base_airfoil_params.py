from core.param import Param
import numpy as np


class SymmetricBaseAirfoilParams:

    def __init__(self,
                 c: Param = Param(1.0),                         # chord length
                 alf: Param = Param(0.0),                       # angle of attack (rad)
                 R_le: Param = Param(0.1, 'length', c=1.0),     # leading edge radius
                 L_le: Param = Param(0.1, 'length', c=1.0),     # leading edge length
                 psi1_le: Param = Param(0.0),                   # leading edge upper curvature control angle
                 L1_te: Param = Param(0.1, 'length', c=1.0),    # trailing edge upper length
                 theta1_te: Param = Param(np.deg2rad(10.0)),    # trailing edge upper angle
                 t_te: Param = Param(0.0, 'length', c=1.0)):     # blunt trailing edge thickness

        self.c = c
        self.alf = alf
        self.R_le = R_le
        self.L_le = L_le
        self.r_le = Param(0.5)
        self.phi_le = Param(0.0)
        self.psi1_le = psi1_le
        self.psi2_le = self.psi1_le
        self.L1_te = L1_te
        self.L2_te = self.L1_te
        self.theta1_te = theta1_te
        self.theta2_te = self.theta1_te
        self.t_te = t_te
        self.r_te = Param(0.5)
        self.phi_te = Param(0.0)
