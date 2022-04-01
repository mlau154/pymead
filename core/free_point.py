import numpy as np
from core.param import Param


class FreePoint:

    def __init__(self,
                 x: Param,
                 y: Param,
                 previous_anchor_point: str,
                 ):

        self.x = x
        self.y = y
        self.xy = np.array([self.x.value, self.y.value])
        self.previous_anchor_point = previous_anchor_point
