import numpy as np


class Param:

    def __init__(self, value: float, units: str or None = None,
                 bounds: list or np.ndarray = np.array([-np.inf, np.inf]),
                 length_scale_dimension: float or None = None, active: bool = True, linked: bool = False):
        self.units = units
        self.length_scale_dimension = length_scale_dimension

        if self.units == 'length' and self.length_scale_dimension is not None:
            self.value = value * self.length_scale_dimension
        elif self.units == 'inverse-length' and self.length_scale_dimension is not None:
            self.value = value / self.length_scale_dimension
        else:
            self.value = value

        self.bounds = bounds
        self.active = active
        self.linked = linked
