import numpy as np


class Param:

    def __init__(self, value: float, units: str or None = None, category: str = 'fixed',
                 bounds: list or np.ndarray = np.array([-np.inf, np.inf]),
                 c: float or None = None, active: bool = True, linked: bool = False):
        self.units = units
        self.c = c
        if self.units == 'length' and self.c is not None:
            self.value = value * self.c
        elif self.units == 'inverse-length' and self.c is not None:
            self.value = value / self.c
        else:
            self.value = value
        self.category = category
        self.bounds = bounds
        self.active = active
        self.linked = linked
