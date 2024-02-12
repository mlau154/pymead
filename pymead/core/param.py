import typing

import numpy as np

from pymead.core import UNITS
from pymead.core.pymead_obj import PymeadObj


class Param(PymeadObj):
    """
    Base-level parameter in ``pymead``. Sub-classed by ``DesVar`` (design variable). Provides operator overloading and
    getter/setter methods.
    """

    def __init__(self, value: float or int, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "params", setting_from_geo_col: bool = False, point=None):
        """
        Parameters
        ==========
        value: float or int
            Starting value for the parameter.

        name: str
            Name of the parameter

        lower: float
            Lower bound for the parameter

        upper: float
            Upper bound for the parameter
        """
        super().__init__(sub_container=sub_container)

        self.dtype = type(value).__name__
        self._value = None
        self._lower = None
        self._upper = None
        self.at_boundary = False
        self.point = point
        self.geo_objs = []
        self.gcs = None
        self.geo_cons = []
        self.dims = []
        self.setting_from_geo_col = setting_from_geo_col

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

        self.set_value(value)

        if lower is not None:
            self.set_lower(lower)

        if upper is not None:
            self.set_upper(upper)

        self.set_name(name)

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
            if self.lower() is None or self.upper() is None:
                raise ValueError("Lower and upper bounds must be set to extract a bounds-normalized value.")
            return (self._value - self._lower) / (self._upper - self._lower)
        else:
            return self._value

    def set_value(self, value: float or int, updated_objs: typing.List[PymeadObj] = None, bounds_normalized: bool = False):
        """
        Sets the design variable value, adjusting the value to fit inside the bounds if necessary.

        Parameters
        ==========
        value: float or int
            Design variable value

        bounds_normalized: bool
            Whether the specified value is normalized by the bounds (e.g., 0 if the value is equal to the lower bound,
            1 if the value is equal to the upper bound, or a float between 0.0 and 1.0 if the value is somewhere
            between the bounds). Default: ``False``
        """
        if bounds_normalized:
            if self.lower() is None or self.upper() is None:
                raise ValueError("Lower and upper bounds must be set to assign a bounds-normalized value.")
            value = value * (self._upper - self._lower) + self._lower

        if self._lower is not None and value < self._lower:  # If below the lower bound,
            # set the value equal to the lower bound
            self._value = self._lower
            self.at_boundary = True
        elif self._upper is not None and value > self._upper:  # If above the upper bound,
            # set the value equal to the upper bound
            self._value = self._upper
            self.at_boundary = True
        else:  # Otherwise, use the default behavior for Param.
            self._value = value
            self.at_boundary = False

        if self.at_boundary:
            return

        updated_objs = [] if updated_objs is None else updated_objs

        if self in updated_objs:
            return
        else:
            updated_objs.append(self)
            for dim in self.dims:
                dim.update_points_from_param(updated_objs=updated_objs)

        if self.gcs is not None and self.geo_cons:
            points_solved = self.gcs.solve(self.geo_cons[0])
            self.gcs.update_canvas_items(points_solved)

        if self.tree_item is not None:
            self.tree_item.treeWidget().itemWidget(self.tree_item, 1).setValue(self.value())

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
        if lower > self.value():
            return

        self._lower = lower

        if self.tree_item is not None:
            self.tree_item.treeWidget().itemWidget(self.tree_item, 1).setMinimum(self.lower())

    def set_upper(self, upper: float):
        """
        Sets the upper bound for the design variable. If called from outside ``DesVar.__init__()``, adjust the design
        variable value to fit inside the bounds if necessary.

        Parameters
        ==========
        upper: float
            Upper bound for the design variable
        """
        if upper < self.value():
            return

        self._upper = upper

        if self.tree_item is not None:
            self.tree_item.treeWidget().itemWidget(self.tree_item, 1).setMaximum(self.upper())

    def get_dict_rep(self):
        return {"value": float(self.value()) if self.dtype == "float" else int(self.value()),
                "lower": self.lower(), "upper": self.upper(),
                "unit_type": None}

    @classmethod
    def set_from_dict_rep(cls, d: dict):
        if "lower" not in d.keys():
            d["lower"] = None
        if "upper" not in d.keys():
            d["upper"] = None
        return cls(value=d["value"], name=d["name"], lower=d["lower"], upper=d["upper"])

    def __repr__(self):
        return f"{self.__class__.__name__} {self.name()}<v={self.value()}>"

    def __add__(self, other):
        return self.__class__(value=self.value() + other.value(), name="forward_add_result")

    def __radd__(self, other):
        return self.__class__(value=other.value() + self.value(), name="reverse_add_result")

    def __sub__(self, other):
        return self.__class__(value=self.value() - other.value(), name="forward_subtract_result")

    def __rsub__(self, other):
        return self.__class__(value=other.value() - self.value(), name="reverse_subtract_result")

    def __mul__(self, other):
        return self.__class__(value=self.value() * other.value(), name="forward_multiplication_result")

    def __rmul__(self, other):
        return self.__class__(value=other.value() * self.value(), name="reverse_multiplication_result")

    def __floordiv__(self, other):
        return self.__class__(value=self.value() // other.value(), name="floor_division_result")

    def __truediv__(self, other):
        return self.__class__(value=self.value() / other.value(), name="true_division_result")

    def __abs__(self):
        return self.__class__(value=abs(self.value()), name="absolute_value_result")

    def __pow__(self, power, modulo=None):
        return self.__class__(value=self.value() ** power, name="power_result")

    # def __eq__(self, other):
    #     return self.value() == other.value()
    #
    # def __ne__(self, other):
    #     return self.value() != other.value()


class LengthParam(Param):
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "params",
                 setting_from_geo_col: bool = False, point=None):
        self._unit = None
        self.set_unit(UNITS.current_length_unit())
        name = "Length-1" if name is None else name
        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container=sub_container,
                         setting_from_geo_col=setting_from_geo_col,
                         point=point)

    def unit(self):
        return self._unit

    def set_unit(self, unit: str or None = None):
        if unit is not None:
            self._unit = unit
        else:
            self._unit = UNITS.current_length_unit()

    def lower(self):
        return UNITS.convert_length_from_base(self._lower, self.unit())

    def upper(self):
        return UNITS.convert_length_from_base(self._upper, self.unit())

    def value(self, bounds_normalized: bool = False):
        new_value = super().value(bounds_normalized=bounds_normalized)
        if bounds_normalized:
            return new_value
        else:
            return UNITS.convert_length_from_base(new_value, self.unit())

    def set_lower(self, lower: float):
        if self.point is None and lower < 0.0:
            lower = 0.0

        return super().set_lower(UNITS.convert_length_to_base(lower, self.unit()))

    def set_upper(self, upper: float):
        return super().set_upper(UNITS.convert_length_to_base(upper, self.unit()))

    def set_value(self, value: float, updated_objs: typing.List[PymeadObj] = None, bounds_normalized: bool = False):

        # Negative lengths are prohibited unless this represents a point
        if self.point is None and value < 0.0:
            return

        new_value = UNITS.convert_length_to_base(value, self.unit())
        return super().set_value(new_value, updated_objs=updated_objs, bounds_normalized=bounds_normalized)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "length"}


class AngleParam(Param):
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "params",
                 setting_from_geo_col: bool = False, point=None):
        self._unit = None
        self.set_unit(UNITS.current_angle_unit())
        name = "Angle-1" if name is None else name
        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container=sub_container,
                         setting_from_geo_col=setting_from_geo_col, point=point)

    def unit(self):
        return self._unit

    def set_unit(self, unit: str or None = None):
        if unit is not None:
            self._unit = unit
        else:
            self._unit = UNITS.current_angle_unit()

    def lower(self):
        return UNITS.convert_angle_from_base(self._lower, self.unit())

    def upper(self):
        return UNITS.convert_angle_from_base(self._upper, self.unit())

    def value(self, bounds_normalized: bool = False):
        new_value = super().value(bounds_normalized=bounds_normalized)
        if bounds_normalized:
            return new_value
        else:
            return UNITS.convert_angle_from_base(new_value, self.unit())

    def rad(self):
        """
        Returns the value of the angle parameter in radians, the base angle unit of pymead

        Returns
        =======
        float
            Angle in radians
        """
        return self._value

    def set_lower(self, lower: float):
        return super().set_lower(UNITS.convert_angle_to_base(lower, self.unit()))

    def set_upper(self, upper: float):
        return super().set_upper(UNITS.convert_angle_to_base(upper, self.unit()))

    def set_value(self, value: float, updated_objs: typing.List[PymeadObj] = None, bounds_normalized: bool = False):

        new_value = UNITS.convert_angle_to_base(value, self.unit())

        zero_to_2pi_value = new_value % (2 * np.pi)

        return super().set_value(zero_to_2pi_value, updated_objs=updated_objs, bounds_normalized=bounds_normalized)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "angle"}


# class ParamCollection:
#     """
#     Collection (list) of Params. Allows for import from/export to array.
#     """
#     def __init__(self, params: typing.List[Param]):
#         self._params = None
#         self.set_params(params)
#         self.shape = len(self.params())
#
#     def params(self):
#         return self._params
#
#     def set_params(self, params: typing.List[Param]):
#         self._params = params
#
#     def as_array(self):
#         return np.array([[p.value()] for p in self.params()])
#
#     @classmethod
#     def generate_from_array(cls, arr: np.ndarray):
#         if arr.ndim == 1:
#             return cls(params=[Param(value=v, name=f"FromArrayIndex{idx}") for idx, v in enumerate(arr)])
#         elif arr.ndim == 2:
#             if arr.shape[1] != 1:
#                 raise ValueError(f"Shape of the input array must be Nx1 (found {arr.shape = })")
#             return cls(params=[Param(value=v, name=f"FromArrayIndex{idx}") for idx, v in enumerate(arr[:, 0])])
#
#     def __len__(self):
#         return len(self.params())


def default_lower(value: float):
    if -0.1 <= value <= 0.1:
        return value - 0.02
    else:
        if value < 0.0:
            return 1.2 * value
        else:
            return 0.8 * value


def default_upper(value: float):
    if -0.1 <= value <= 0.1:
        return value + 0.02
    else:
        if value < 0.0:
            return 0.8 * value
        else:
            return 1.2 * value


class DesVar(Param):
    """
    Design variable class; subclasses the base-level Param. Adds lower and upper bound default behavior.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "desvar", setting_from_geo_col: bool = False, point=None):
        """
        Parameters
        ==========
        value: float
            Starting value of the design variable

        name: str
            Name of the design variable

        lower: float or None
            Lower bound for the design variable. If ``None``, a reasonable value is chosen. Default: ``None``.

        upper: float or None
            Upper bound for the design variable. If ``None``, a reasonable value is chosen. Default: ``None``.

        setting_from_geo_col: bool
            Whether this method is being called directly from the geometric collection. Default: ``False``.
        """

        # Default behavior for lower bound
        if lower is None:
            lower = default_lower(value)

        # Default behavior for upper bound
        if upper is None:
            upper = default_upper(value)

        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container=sub_container,
                         setting_from_geo_col=setting_from_geo_col, point=point)


class LengthDesVar(LengthParam):
    """
    Design variable class for length values; subclasses the base-level Param. Adds lower and upper bound
    default behavior.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 setting_from_geo_col: bool = False, point=None):
        """
        Parameters
        ==========
        value: float
            Starting value of the design variable

        name: str
            Name of the design variable

        lower: float or None
            Lower bound for the design variable. If ``None``, a reasonable value is chosen. Default: ``None``.

        upper: float or None
            Upper bound for the design variable. If ``None``, a reasonable value is chosen. Default: ``None``.

        setting_from_geo_col: bool
            Whether this method is being called directly from the geometric collection. Default: ``False``.
        """

        # Default behavior for lower bound
        if lower is None:
            lower = default_lower(value)

        # Default behavior for upper bound
        if upper is None:
            upper = default_upper(value)

        super().__init__(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=setting_from_geo_col,
                         sub_container="desvar", point=point)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "length"}


class AngleDesVar(AngleParam):
    """
    Design variable class for angle values; subclasses the base-level Param. Adds lower and upper bound
    default behavior.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 setting_from_geo_col: bool = False, point=None):
        """
        Parameters
        ==========
        value: float
            Starting value of the design variable

        name: str
            Name of the design variable

        lower: float or None
            Lower bound for the design variable. If ``None``, a reasonable value is chosen. Default: ``None``.

        upper: float or None
            Upper bound for the design variable. If ``None``, a reasonable value is chosen. Default: ``None``.

        setting_from_geo_col: bool
            Whether this method is being called directly from the geometric collection. Default: ``False``.
        """

        # Default behavior for lower bound
        if lower is None:
            lower = default_lower(value)

        # Default behavior for upper bound
        if upper is None:
            upper = default_upper(value)

        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container="desvar",
                         setting_from_geo_col=setting_from_geo_col, point=point)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "angle"}
