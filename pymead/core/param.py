import networkx
import numpy as np
from PyQt6.QtCore import QSignalBlocker

from pymead.core.pymead_obj import PymeadObj


class Param(PymeadObj):
    """
    Base-level parameter in ``pymead``. Sub-classed by ``DesVar`` (design variable). Provides operator overloading and
    getter/setter methods.

    .. note::

       Instances of this class should never be created directly. Instead, parameters can be created by
       creating a ``GeometryCollection`` and calling its ``add_param`` method.
    """

    def __init__(self, value: float or int, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "params", setting_from_geo_col: bool = False, point=None, root=None,
                 rotation_handle=None, enabled: bool = True, equation_str: str = None, geo_col=None):
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
        super().__init__(sub_container=sub_container, geo_col=geo_col)

        self.dtype = type(value).__name__
        self._value = None
        self._lower = None
        self._upper = None
        self._enabled = None
        self.at_boundary = False
        self.point = point
        self.root = root
        self.set_enabled(enabled)
        self.rotation_handle = rotation_handle
        if rotation_handle is not None:
            self.rotation_handle.rotation_param = self
        self.geo_objs = []
        self.param_graph = None
        self.equation_str = equation_str
        self.equation = None
        self.equation_dict = None
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

        kwargs = dict(direct_user_request=False) if isinstance(self, LengthParam) else {}
        self.set_value(value, **kwargs)

        if lower is not None:
            self.set_lower(lower)

        if upper is not None:
            self.set_upper(upper)

        self.set_name(name)

    def _get_value_spin(self):
        return self.tree_item.treeWidget().itemWidget(self.tree_item, 1)

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

    def set_value(self, value: float or int, bounds_normalized: bool = False, force: bool = False,
                  param_graph_update: bool = False, from_request_move: bool = False):
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

        force: bool
            Whether to force the change in value. This keyword argument should never be set to ``True`` when using
            the API. Default: ``False``

        param_graph_update: bool
            Whether this value is being set as part of a parameter graph update. Used to prevent the parameter graph
            from triggering multiple updates for the same parameter. This keyword argument should never
            be set to ``True`` when using the API. Default: ``False``

        from_request_move: bool
            Whether this method was called from ``Point.request_move``. Default: ``False``
        """

        old_point_vals = {k: v.as_array() for k, v in self.geo_col.container()["points"].items()} \
            if self.geo_col is not None else {}

        def get_curves_to_update(_points_to_update):
            _curves_to_update = []
            for point in _points_to_update:
                for curve in point.curves:
                    if curve not in _curves_to_update:
                        _curves_to_update.append(curve)
            return _curves_to_update

        def get_airfoils_to_update(_curves_to_update):
            _airfoils_to_update = []
            for curve in _curves_to_update:
                if curve.airfoil is not None and curve.airfoil not in _airfoils_to_update:
                    _airfoils_to_update.append(curve.airfoil)
            return _airfoils_to_update

        def rotate_cluster(new_v):
            if self.gcs is None:
                return
            points_to_update, root = self.gcs.rotate_cluster(self.rotation_handle, new_rotation_angle=new_v)
            constraints_to_update = []
            for point in networkx.dfs_preorder_nodes(self.gcs, source=root):
                for geo_con in point.geo_cons:
                    if geo_con not in constraints_to_update:
                        constraints_to_update.append(geo_con)

            for geo_con in constraints_to_update:
                if geo_con.canvas_item is not None:
                    geo_con.canvas_item.update()

            _curves_to_update = get_curves_to_update(points_to_update)

            _airfoils_to_update = get_airfoils_to_update(_curves_to_update)

            # Update airfoil-relative points
            for _airfoil in _airfoils_to_update:
                _airfoil.update_relative_points(old_point_vals)

            # Visual updates to geometric objects
            for point in points_to_update:
                if point.canvas_item is not None:
                    point.canvas_item.updateCanvasItem(point.x().value(), point.y().value())

            for _curve in _curves_to_update:
                _curve.update()

            for _airfoil in _airfoils_to_update:
                _airfoil.update_coords()
                if _airfoil.canvas_item is not None:
                    _airfoil.canvas_item.generatePicture()

        points_solved = []
        if not force:

            if bounds_normalized:
                if self.lower() is None or self.upper() is None:
                    raise ValueError("Lower and upper bounds must be set to assign a bounds-normalized value.")
                value = value * (self._upper - self._lower) + self._lower

            # Bounds clipping
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

            if self.rotation_handle is not None:
                rotate_cluster(self._value)

            if self.at_boundary:
                return

            if self.gcs is not None and self.geo_cons:
                points_solved = []
                for gc in self.geo_cons:
                    points_solved.extend(self.gcs.solve(gc))

                curves_to_update = get_curves_to_update(points_solved)
                airfoils_to_update = get_airfoils_to_update(curves_to_update)
                # Update airfoil-relative points
                for airfoil in airfoils_to_update:
                    airfoil.update_relative_points(old_point_vals)

                if not from_request_move:
                    self.gcs.update_canvas_items(list(set(points_solved)))

            if self.param_graph is not None and not param_graph_update and self in self.param_graph.nodes:
                for node in networkx.dfs_preorder_nodes(self.param_graph, source=self):
                    if not node.equation_str:
                        continue
                    node.evaluate_equation()

        else:
            self._value = value

        if self.tree_item is not None:
            value_spin = self._get_value_spin()
            with QSignalBlocker(value_spin):
                self.tree_item.treeWidget().itemWidget(self.tree_item, 1).setValue(self.value())

        return list(set(points_solved))

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

    def set_lower(self, lower: float, force: bool = False):
        """
        Sets the lower bound for the design variable. If called from outside ``DesVar.__init__()``, adjust the design
        variable value to fit inside the bounds if necessary.

        Parameters
        ==========
        lower: float
            Lower bound for the design variable

        force: bool
            Setting this argument to ``True`` ignores the check for lower bound greater than value. Default: ``False``
        """
        if lower > self.value() and not force:
            return

        self._lower = lower

        if self.tree_item is not None:
            value_spin = self._get_value_spin()
            with QSignalBlocker(value_spin):
                value_spin.setMinimum(self.lower())

    def set_upper(self, upper: float, force: bool = False):
        """
        Sets the upper bound for the design variable. If called from outside ``DesVar.__init__()``, adjust the design
        variable value to fit inside the bounds if necessary.

        Parameters
        ==========
        upper: float
            Upper bound for the design variable

        force: bool
            Setting this argument to ``True`` ignores the check for upper bound less than value. Default: ``False``
        """
        if upper < self.value() and not force:
            return

        self._upper = upper

        if self.tree_item is not None:
            value_spin = self._get_value_spin()
            with QSignalBlocker(value_spin):
                value_spin.setMaximum(self.upper())

    def enabled(self):
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if self.tree_item is not None:
            value_spin = self._get_value_spin()
            with QSignalBlocker(value_spin):
                value_spin.setEnabled(enabled)

    def update_equation(self, equation_str: str = None):
        if not equation_str:  # Handles both the None and empty-string cases
            self.equation_str = equation_str
            self.equation = None
            self.equation_dict = None
            return
        if self.param_graph is None:
            return
        self.equation = "def f(): return "
        self.equation_dict = {"p": {}}
        equation_split = equation_str.split()
        param_names = [param.name() for param in self.param_graph.param_list]
        for idx, sub_str in enumerate(equation_split):
            if sub_str[0] != "$":
                self.equation += sub_str
                continue
            param_name = sub_str.strip("$")
            if param_name in param_names:
                self.equation_dict["p"][param_name] = self.param_graph.param_list[param_names.index(param_name)]
            else:
                raise EquationCompileError("Failed to compile")
            self.equation += f"p['{param_name}'].value()"

        for param in self.equation_dict["p"].values():
            self.param_graph.add_edge(param, self)
        if len(self.param_graph.nodes) != 0 and not networkx.is_forest(self.param_graph):
            # Revert to the original equation string
            self.update_equation(self.equation_str)
            raise EquationCompileError("The dependencies for this equation create a closed loop")

        self.evaluate_equation()

        self.equation_str = equation_str

    def evaluate_equation(self):
        try:
            exec(self.equation, self.equation_dict)
            self.set_value(self.equation_dict["f"](), param_graph_update=True)
        except (NameError, SyntaxError) as e:
            raise EquationCompileError(str(e))

    def get_dict_rep(self):
        return {"value": float(self.value()) if self.dtype == "float" else int(self.value()),
                "lower": self.lower(), "upper": self.upper(),
                "unit_type": None, "enabled": self.enabled(), "equation_str": self.equation_str}

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


class LengthParam(Param):
    """
    Length-type parameter in ``pymead``. Adds unit functionality and prevents negative values except in the case
    of points.

    .. note::

       Instances of this class should never be created directly. Instead, parameters can be created by
       creating a ``GeometryCollection`` and calling its ``add_param`` method with ``unit_type="length"``.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "params",
                 setting_from_geo_col: bool = False, point=None, root=None, rotation_handle=None, enabled: bool = True,
                 equation_str: str = None, geo_col=None):
        self._unit = None
        name = "Length-1" if name is None else name
        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container=sub_container,
                         setting_from_geo_col=setting_from_geo_col,
                         point=point, root=root, rotation_handle=rotation_handle, enabled=enabled,
                         equation_str=equation_str, geo_col=geo_col)

    def unit(self):
        return self._unit

    def set_unit(self, unit: str or None = None, old_unit: str = None, modify_value: bool = True) -> float or None:
        """
        This method sets the length unit to be used, called the first time when adding a parameter
        via ``GeometryCollection.add_pymead_obj_by_ref``.

        Parameters
        ----------
        unit: str or None
            The new unit to switch to. If ``None``, the current length unit will be used. Default: ``None``
        old_unit: str or None
            The old unit to switch from. If ``None``, no changes will be made to the bounds or value of the param.
            Default: ``None``
        modify_value: bool
            Whether to scale the parameter value based on the conversion to the new unit. Default: ``True``

        Returns
        -------
        float or None
            If ``old_unit==None``, ``None`` will be returned. Otherwise, the value of the parameter is returned

        """
        if unit is not None:
            self._unit = unit
        else:
            self._unit = self.geo_col.units.current_length_unit()

        if old_unit is None:
            return

        lower = self.lower()
        upper = self.upper()
        value = self.value()
        if lower is not None:
            base_lower = self.geo_col.units.convert_length_to_base(lower, old_unit)
            self.set_lower(self.geo_col.units.convert_length_from_base(base_lower, unit), force=True)
        if upper is not None:
            base_upper = self.geo_col.units.convert_length_to_base(upper, old_unit)
            self.set_upper(self.geo_col.units.convert_length_from_base(base_upper, unit), force=True)

        base_value = self.geo_col.units.convert_length_to_base(value, old_unit)
        new_value = self.geo_col.units.convert_length_from_base(base_value, unit)
        if modify_value:
            self.set_value(new_value, direct_user_request=False)

        if self.tree_item is not None:
            self.tree_item.treeWidget().itemWidget(self.tree_item, 1).setSuffix(f" {unit}")

        return new_value

    def set_lower(self, lower: float, force: bool = False):
        if self.point is None and lower < 0.0:
            lower = 0.0

        return super().set_lower(lower, force=force)

    def set_value(self, value: float, bounds_normalized: bool = False, force: bool = False,
                  param_graph_update: bool = False, from_request_move: bool = False,
                  direct_user_request: bool = True):

        # Negative lengths are prohibited unless this represents a point
        if self.point is None and value < 0.0:
            value = self.value()  # Can only happen if not bounds-normalized, so no need to pass this argument

        if direct_user_request and self.point and self.point.x() and self.point.y():
            value = value * (self._upper - self._lower) + self._lower if bounds_normalized else value
            if self is self.point.x():
                self.point.request_move(xp=value, yp=self.point.y().value())
            if self is self.point.y():
                self.point.request_move(xp=self.point.x().value(), yp=value)
            return

        return super().set_value(value, bounds_normalized=bounds_normalized, force=force,
                                 param_graph_update=param_graph_update, from_request_move=from_request_move)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "length", "enabled": self.enabled(), "equation_str": self.equation_str}


class AngleParam(Param):
    r"""
    Angle-type parameter in ``pymead``. Adds unit functionality and transforms values into the range :math:`[0, 2\pi)`.

    .. note::

       Instances of this class should never be created directly. Instead, parameters can be created by
       creating a ``GeometryCollection`` and calling its ``add_param`` method with ``unit_type="angle"``.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 sub_container: str = "params",
                 setting_from_geo_col: bool = False, point=None, root=None, rotation_handle=None,
                 enabled: bool = True, equation_str: str = None, geo_col=None):
        self._unit = None
        name = "Angle-1" if name is None else name
        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container=sub_container,
                         setting_from_geo_col=setting_from_geo_col, point=point, root=root,
                         rotation_handle=rotation_handle, enabled=enabled, equation_str=equation_str,
                         geo_col=geo_col)

    def unit(self):
        return self._unit

    def set_unit(self, unit: str or None = None, old_unit: str or None = None):
        if unit is not None:
            self._unit = unit
        else:
            self._unit = self.geo_col.units.current_angle_unit()

        if old_unit is None:
            return

        lower = self.lower()
        upper = self.upper()
        value = self.value()
        if lower is not None:
            base_lower = self.geo_col.units.convert_angle_to_base(lower, old_unit)
            self.set_lower(self.geo_col.units.convert_angle_from_base(base_lower, unit), force=True)
        if upper is not None:
            base_upper = self.geo_col.units.convert_angle_to_base(upper, old_unit)
            self.set_upper(self.geo_col.units.convert_angle_from_base(base_upper, unit), force=True)
        base_value = self.geo_col.units.convert_angle_to_base(value, old_unit)
        self.set_value(self.geo_col.units.convert_angle_from_base(base_value, unit))

        if self.tree_item is not None:
            self.tree_item.treeWidget().itemWidget(self.tree_item, 1).setSuffix(f" {unit}")

    def rad(self):
        """
        Returns the value of the angle parameter in radians, the base angle unit of pymead

        Returns
        =======
        float
            Angle in radians
        """
        return self.geo_col.units.convert_angle_to_base(self._value, self.unit())

    def set_value(self, value: float, bounds_normalized: bool = False, force: bool = False,
                  param_graph_update: bool = False, from_request_move: bool = False):

        if self.unit() is None:
            self.set_unit()
        new_value = self.geo_col.units.convert_angle_to_base(value, self.unit())
        zero_to_2pi_value = new_value % (2 * np.pi)
        new_value = self.geo_col.units.convert_angle_from_base(zero_to_2pi_value, self.unit())

        return super().set_value(new_value, bounds_normalized=bounds_normalized, force=force,
                                 param_graph_update=param_graph_update, from_request_move=from_request_move)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "angle",
                "root": self.root.name() if self.root is not None else None,
                "rotation_handle": self.rotation_handle.name() if self.rotation_handle is not None else None,
                "enabled": self.enabled(), "equation_str": self.equation_str}


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
                 sub_container: str = "desvar", setting_from_geo_col: bool = False, point=None, root=None,
                 rotation_handle=None, enabled: bool = True, equation_str: str = None, assignable: bool = True):
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

        self._assignable = None

        # Default behavior for lower bound
        if lower is None:
            lower = default_lower(value)

        # Default behavior for upper bound
        if upper is None:
            upper = default_upper(value)

        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container=sub_container,
                         setting_from_geo_col=setting_from_geo_col, point=point, root=root,
                         rotation_handle=rotation_handle, enabled=enabled, equation_str=equation_str)

        self.assignable = assignable

    @property
    def assignable(self):
        return self._assignable

    @assignable.setter
    def assignable(self, assignable: bool):
        """Currently used just for Fan Pressure Ratio design variables."""
        self._assignable = assignable


class LengthDesVar(LengthParam):
    """
    Design variable class for length values. Adds lower and upper bound default behavior.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 setting_from_geo_col: bool = False, point=None, root=None, rotation_handle=None,
                 enabled: bool = True, equation_str: str = None):
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

        self.assignable = True

        # Default behavior for lower bound
        if lower is None:
            lower = default_lower(value)

        # Default behavior for upper bound
        if upper is None:
            upper = default_upper(value)

        super().__init__(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=setting_from_geo_col,
                         sub_container="desvar", point=point, root=root, rotation_handle=rotation_handle,
                         enabled=enabled, equation_str=equation_str)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "length", "enabled": self.enabled(), "equation_str": self.equation_str}


class AngleDesVar(AngleParam):
    """
    Design variable class for angle values; subclasses the base-level Param. Adds lower and upper bound
    default behavior.
    """
    def __init__(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                 setting_from_geo_col: bool = False, point=None, root=None, rotation_handle=None, enabled: bool = True,
                 equation_str: str = None):
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

        self.assignable = True

        # Default behavior for lower bound
        if lower is None:
            lower = default_lower(value)

        # Default behavior for upper bound
        if upper is None:
            upper = default_upper(value)

        super().__init__(value=value, name=name, lower=lower, upper=upper, sub_container="desvar",
                         setting_from_geo_col=setting_from_geo_col, point=point, root=root,
                         rotation_handle=rotation_handle, enabled=enabled, equation_str=equation_str)

    def set_value(self, value: float, bounds_normalized: bool = False, force: bool = False,
                  param_graph_update: bool = False, from_request_move: bool = False):
        r"""
        In this special case of ``set_value`` for an ``AngleDesVar``, we skip over the call to the ``set_value``
        method in ``AngleParam`` and directly call the ``set_value`` method in ``Param`` (the grandparent class).
        The reason for this is that ``AngleParam`` always keeps the angle between 0 and :math:`2 \pi`, which is not
        logical behavior for a bounded variable. This method eliminates that restriction.
        """
        return Param.set_value(self, value, bounds_normalized=bounds_normalized, force=force,
                               param_graph_update=param_graph_update, from_request_move=from_request_move)

    def get_dict_rep(self):
        return {"value": float(self.value()), "lower": self.lower(), "upper": self.upper(),
                "unit_type": "angle",
                "root": self.root.name() if self.root is not None else None,
                "rotation_handle": self.rotation_handle.name() if self.rotation_handle is not None else None,
                "enabled": self.enabled(), "equation_str": self.equation_str}


class EquationCompileError(Exception):
    pass
