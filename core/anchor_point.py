from core.param import Param
import numpy as np


class AnchorPoint:

    def __init__(self,
                 x: Param,
                 y: Param,
                 name: str,
                 previous_anchor_point: str,
                 L: Param,
                 R: Param,
                 r: Param,
                 phi: Param,
                 psi1: Param,
                 psi2: Param):
        self.x = x
        self.y = y
        self.xy = np.array([x.value, y.value])
        self.name = name
        self.previous_anchor_point = previous_anchor_point
        self.L = L
        self.R = R
        self.r = r
        self.phi = phi
        self.psi1 = psi1
        self.psi2 = psi2
