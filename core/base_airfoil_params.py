from core.param import Param
import numpy as np


class BaseAirfoilParams:

    def __init__(self,
                 c: Param = Param(1.0),                         # chord length
                 alf: Param = Param(0.0),                       # angle of attack (rad)
                 R_le: Param = Param(0.1, 'length', c=1.0),     # leading edge radius
                 L_le: Param = Param(0.1, 'length', c=1.0),     # leading edge length
                 r_le: Param = Param(0.5),                      # leading edge length ratio
                 phi_le: Param = Param(0.0),                    # leading edge 'tilt' angle
                 psi1_le: Param = Param(0.0),                   # leading edge upper curvature control angle
                 psi2_le: Param = Param(0.0),                   # leading edge lower curvature control angle
                 L1_te: Param = Param(0.1, 'length', c=1.0),    # trailing edge upper length
                 L2_te: Param = Param(0.1, 'length', c=1.0),    # trailing edge lower length
                 theta1_te: Param = Param(np.deg2rad(10.0)),    # trailing edge upper angle
                 theta2_te: Param = Param(np.deg2rad(10.0)),    # trailing edge lower angle
                 t_te: Param = Param(0.0, 'length', c=1.0),     # blunt trailing edge thickness
                 r_te: Param = Param(0.5),                      # blunt trailing edge thickness length ratio
                 phi_te: Param = Param(0.0)):                   # blunt trailing edge 'tilt' angle

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
