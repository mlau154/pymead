import typing

import numpy as np


class Param:
    """
    Base-level parameter in ``pymead``. Sub-classed by ``DesVar`` (design variable) and ``IntVar`` (intermediate
    variable). Provides operator overloading and getter/setter methods.
    """
    def __init__(self, value: float, name: str, setting_from_geo_col: bool = False):
        """
        Parameters
        ==========
        value: float
            Starting value for the parameter.
        """
        self._value = None
        self._name = None
        self.geo_col = None
        self.geo_objs = []
        self.setting_from_geo_col = setting_from_geo_col
        self.set_value(value)
        self.set_name(name)

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

    def name(self):
        """
        Retrieves the parameter name

        Returns
        =======
        str
            The parameter name
        """
        return self._name

    def set_name(self, name: str):
        """
        Sets the parameter name. Must not contain hyphens unless setting from the geometry collection.

        Parameters
        ==========
        name: str
            The parameter name
        """
        if "-" in name and not self.setting_from_geo_col:
            raise ValueError("Hyphens are reserved characters and cannot be used unless setting from geometry "
                             "collection")
        if "." in name and not self.setting_from_geo_col:
            raise ValueError("Dots are reserved characters and cannot be used unless setting from geometry "
                             "collection")
        self._name = name

    def get_dict_rep(self):
        return {"value": self.value(), "name": self.name()}

    @classmethod
    def set_from_dict_rep(cls, d: dict):
        return cls(value=d["value"], name=d["name"])

    def __add__(self, other):
        return Param(value=self.value() + other.value(), name="forward_add_result")

    def __radd__(self, other):
        return Param(value=other.value() + self.value(), name="reverse_add_result")

    def __sub__(self, other):
        return Param(value=self.value() - other.value(), name="forward_subtract_result")

    def __rsub__(self, other):
        return Param(value=other.value() - self.value(), name="reverse_subtract_result")

    def __mul__(self, other):
        return Param(value=self.value() * other.value(), name="forward_multiplication_result")

    def __rmul__(self, other):
        return Param(value=other.value() * self.value(), name="reverse_multiplication_result")

    def __floordiv__(self, other):
        return Param(value=self.value() // other.value(), name="floor_division_result")

    def __truediv__(self, other):
        return Param(value=self.value() / other.value(), name="true_division_result")

    def __abs__(self):
        return Param(value=abs(self.value()), name="absolute_value_result")

    def __pow__(self, power, modulo=None):
        return Param(value=self.value() ** power, name="power_result")

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
            return cls(params=[Param(value=v, name=f"FromArray-Index{idx}") for idx, v in enumerate(arr)])
        elif arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(f"Shape of the input array must be Nx1 (found {arr.shape = })")
            return cls(params=[Param(value=v, name=f"FromArray-Index{idx}") for idx, v in enumerate(arr[:, 0])])

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
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 setting_from_geo_col: bool = False):
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
        self._lower = None
        self._upper = None

        super().__init__(value=value, name=name, setting_from_geo_col=setting_from_geo_col)

        if upper is not None and lower is not None:
            if upper < lower:
                raise ValueError(f"Specified upper bound ({upper}) smaller than the specified lower bound ({lower})")

            if np.isclose(upper - lower, 0.0):
                raise ValueError(f"Specified upper bound ({upper}) too close to the specified lower bound ({lower})")

        if lower is not None and lower > value:
            raise ValueError(f"Specified lower bound ({lower}) larger than current parameter value ({value})"
                             f"for {name}")

        if upper is not None and upper < value:
            raise ValueError(f"Specified upper bound ({upper}) smaller than current parameter value ({value})"
                             f"for {name}")

        # Default behavior for lower bound
        if lower is None:
            if -0.1 <= value <= 0.1:
                lower = value - 0.02
            else:
                if value < 0.0:
                    lower = 1.2 * value
                else:
                    lower = 0.8 * value

        # Default behavior for upper bound
        if upper is None:
            if -0.1 <= value <= 0.1:
                upper = value + 0.02
            else:
                if value < 0.0:
                    upper = 0.8 * value
                else:
                    upper = 1.2 * value

        self.set_lower(lower)
        self.set_upper(upper)

    def value(self, bounds_normalized: bool = False):
        """
        Returns the design variable value.

        Parameters
        ==========
        bounds_normalized: bool
            Whether to return the value as normalized by the set bounds. If the value is at the lower bound, 0.0 will
            be returned. If the value is at the upper bound, 1.0 will be returned. Otherwise, a number between 0.0
            and 1.0 will be returned. Default: ``False``.

        Returns
        =======
        float
            The design variable value
        """
        if bounds_normalized:
            return (self._value - self._lower) / (self._upper - self._lower)
        else:
            return self._value

    def set_value(self, value: float, bounds_normalized: bool = False):
        """
        Sets the design variable value, adjusting the value to fit inside the bounds if necessary.

        Parameters
        ==========
        value: float
            Design variable value

        bounds_normalized: bool
            Whether the specified value is normalized by the bounds (e.g., 0 if the value is equal to the lower bound,
            1 if the value is equal to the upper bound, or a float between 0.0 and 1.0 if the value is somewhere
            between the bounds). Default: ``False``
        """
        if bounds_normalized:
            value = value * (self._upper - self._lower) + self._lower

        if self._lower is not None and value < self._lower:  # If below the lower bound,
            # set the value equal to the lower bound
            self._value = self._lower
        elif self._upper is not None and value > self._upper:  # If above the upper bound,
            # set the value equal to the upper bound
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

    def get_dict_rep(self):
        return {"value": self.value(), "name": self.name(), "lower": self.lower(), "upper": self.upper()}

    @classmethod
    def set_from_dict_rep(cls, d: dict):
        return cls(value=d["value"], name=d["name"], lower=d["lower"], upper=d["upper"])
