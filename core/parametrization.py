import numpy as np
from airfoil import Airfoil


class AirfoilParametrization:

    def __init__(self, n_airfoils: int):
        if not isinstance(n_airfoils, int):
            raise TypeError(f'n_airfoils must be of type int, not type {type(n_airfoils)}')
        elif n_airfoils < 1:
            raise ValueError(f'n_airfoils must be a positive integer (input: n_airfoils={n_airfoils})')
        else:
            self.n_airfoils = n_airfoils
        self.airfoil_system = [Airfoil() for _ in range(self.n_airfoils)]

    def init_default_parameters(self):
        for idx in range(self.n_airfoils):
            pass
