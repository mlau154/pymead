import typing

import numpy as np


class Param:
    """
    Base-level parameter in ``pymead``. Sub-classed by ``DesVar`` (design variable) and ``IntVar`` (intermediate
    variable). Provides operator overloading and getter/setter methods.
    """
    def __init__(self, value: float):
        """
        Parameters
        ==========
        value: float
            Starting value for the parameter.
        """
        self._value = None
        self.set_value(value)

    def value(self):
        """
        Retrieves the parameter value

        Returns
        =======
        float
            The parameter value
        """
        return self._value

    def set_value(self, value: float):
        """
        Sets the parameter value

        Parameters
        ==========
        value: float
            The parameter value
        """
        self._value = value

    def __add__(self, other):
        return Param(value=self.value() + other.value())

    def __radd__(self, other):
        return Param(value=other.value() + self.value())

    def __sub__(self, other):
        return Param(value=self.value() - other.value())

    def __rsub__(self, other):
        return Param(value=other.value() - self.value())

    def __mul__(self, other):
        return Param(value=self.value() * other.value())

    def __rmul__(self, other):
        return Param(value=other.value() * self.value())

    def __floordiv__(self, other):
        return Param(value=self.value() // other.value())

    def __truediv__(self, other):
        return Param(value=self.value() / other.value())

    def __abs__(self):
        return Param(value=abs(self.value()))

    def __pow__(self, power, modulo=None):
        return Param(value=self.value() ** power)

    def __eq__(self, other):
        return self.value() == other.value()

    def __ne__(self, other):
        return self.value() != other.value()


class ParamCollection:
    """
    Collection (list) of Params. Allows for import from/export to array.
    """
    def __init__(self, params: typing.List[Param]):
        self._params = None
        self.set_params(params)
        self.shape = len(self.params())

    def params(self):
        return self._params

    def set_params(self, params: typing.List[Param]):
        self._params = params

    def as_array(self):
        return np.array([[p.value()] for p in self.params()])

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.ndim == 1:
            return cls(params=[Param(value=v) for v in arr])
        elif arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(f"Shape of the input array must be Nx1 (found {arr.shape = })")
            return cls(params=[Param(value=v) for v in arr[:, 0]])

    def __len__(self):
        return len(self.params())


class IntVar(Param):
    """
    Designation for an intermediate variable. Used in equation definitions with design variables for parameters.
    """
    pass


class DesVar(Param):
    """
    Design variable class; subclasses the base-level Param. Adds lower and upper bound functionality.
    """
    def __init__(self, value: float, lower: float, upper: float):
        """
        Parameters
        ==========
        value: float
            Starting value of the design variable

        lower: float
            Lower bound for the design variable

        upper: float
            Upper bound for the design variable
        """
        self._value = None
        self._lower = None
        self._upper = None
        self.set_lower(lower)
        self.set_upper(upper)
        super().__init__(value)

    def set_value(self, value: float):
        """
        Sets the design variable value, adjusting the value to fit inside the bounds if necessary.

        Parameters
        ==========
        value: float
            Design variable value
        """
        if value < self._lower:  # If below the lower bound, set the value equal to the lower bound
            self._value = self._lower
        elif value > self._upper:  # If above the upper bound, set the value equal to the upper bound
            self._value = self._upper
        else:  # Otherwise, use the default behavior for Param.
            super().set_value(value)

    def lower(self):
        """
        Returns the lower bound for the design variable

        Returns
        =======
        float
            DV lower bound
        """
        return self._lower

    def upper(self):
        """
        Returns the upper bound for the design variable

        Returns
        =======
        float
            DV upper bound
        """
        return self._upper

    def set_lower(self, lower: float):
        """
        Sets the lower bound for the design variable. If called from outside ``DesVar.__init__()``, adjust the design
        variable value to fit inside the bounds if necessary.

        Parameters
        ==========
        lower: float
            Lower bound for the design variable
        """
        self._lower = lower
        if self._value is not None and self._value < self._lower:
            self._value = self._lower

    def set_upper(self, upper: float):
        """
        Sets the upper bound for the design variable. If called from outside ``DesVar.__init__()``, adjust the design
        variable value to fit inside the bounds if necessary.

        Parameters
        ==========
        upper: float
            Upper bound for the design variable
        """
        self._upper = upper
        if self._value is not None and self._value > self._upper:
            self._value = self._upper
