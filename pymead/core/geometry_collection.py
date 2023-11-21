import re
import typing

from pymead.core.param2 import Param, DesVar
from pymead.core.point import Point


class GeometryCollection:
    def __init__(self, geo_ui=None):
        self._container = {
            "desvar": {},
            "params": {},
            "points": {},
            "lines": {},
            "bezier": {},
            "geocon": {}
        }
        self.geo_canvas = geo_ui

    def container(self):
        """
        Retrieves the geometry container. Note that there is no "setter" method because geometry items should
        be added only be their respective methods. For example, a ``Param`` should be added using ``.add_param()``.
        This allows for name validation, which ensures that every geometry item in the container has a unique name.

        Returns
        =======
        dict
            Dictionary of geometry items
        """
        return self._container

    def get_name_list(self, sub_container: str):
        """
        Gets a list of all the parameter or geometry names in a specified sub-container.

        Parameters
        ==========
        sub_container: str
            Sub-container in the geometry collection. For example, ``"params"``, ``"lines"``, ``"points"``, etc.

        Returns
        =======
        typing.List[str]
            List of names found in the sub-container
        """
        return [k for k in self.container()[sub_container].keys()]

    @staticmethod
    def unique_namer(specified_name: str, name_list: typing.List[str]):
        """
        This static method creates unique names for parameters or geometry by incrementing the appended index, which
        is attached with a hyphen. A dot separator is used to distinguish between levels within the geometry
        collection hierarchy. For example, the :math:`x`-location of the second point added is given by
        ``"Point-2.x"``. Note that the first parameter added with a given name will not have an index by default.
        For example, if a ``Param`` with ``name=="my_param"`` is added three times, the resulting names, in order,
        will be ``"my_param"``, ``"my_param-2"``, and ``"my_param-3"``.

        If a parameter is removed, and another
        parameter with the same name added, the index assigned will be the maximum index plus one. In the previous
        example, if ``"my_param-2"`` is removed from the geometry collection and the addition of ``"my_param"``
        requested, the actual name assigned will be ``"my_param-4"``. This is to prevent confusion between removed and
        added parameters, as well as to give an indication of the order in which the parameter was added.

        Parameters
        ==========
        specified_name: str
            Input name to be tested or modified for uniqueness. Does not need to have an index.

        name_list: typing.List[str]
            List of names extracted from a particular sub-container in the geometry collection, used to check the
            ``specified_name`` for uniqueness.

        Returns
        =======
        str
            The (possibly modified) ``specified_name``
        """
        max_index = 0
        for name in name_list:
            dot_split = name.split(".")
            dash_split = dot_split[-1].split("-")
            if len(dot_split) == 1:  # If there were no dots,
                prefix = dash_split[0]
            else:
                prefix = ".".join(dot_split)[:-1] + dash_split[0]
            if specified_name != prefix:  # If the specified name does not match the name being analyzed, continue to
                # next name
                continue
            if len(dash_split) < 2 and max_index == 0:  # If the string after the last dot had no hyphen
                # and max_index is still 0, set the max_index to 1.
                max_index = 1
            elif len(dash_split) >= 2:  # If the string after the last dot had a hyphen and the index is greater than
                # the max_index, set the max_index to this index.
                idx = int(dash_split[-1])
                if idx > max_index:
                    max_index = idx

        if max_index == 0:
            return specified_name
        else:
            return f"{specified_name}-{max_index + 1}"

    def add_to_subcontainer(self, obj: Param or DesVar or Point, sub_container: str,
                            assign_unique_name: bool = True):
        """
        Adds an object to a sub-container within the geometry collection's ``container()``. Also performs the task
        of assigning a unique name to the object before insertion into the sub-container, if necessary.
        
        Parameters
        ==========
        obj: Param or DesVar or Point
            Object to add to the sub-container

        sub_container: str
            Specified sub-container within ``container()``
            
        assign_unique_name: bool
            Whether to assign the object a unique name before insertion into the sub-container. Default: ``True``.
        """
        # Set the object's name to a unique name if necessary
        if assign_unique_name:
            name_list = self.get_name_list(sub_container=sub_container)
            unique_name = self.unique_namer(obj.name(), name_list)
            obj.set_name(unique_name)

        # Add the object to the geometry collection sub-container
        self.container()[sub_container][obj.name()] = obj

    def remove_from_subcontainer(self, obj: Param or DesVar or Point or str, sub_container: str):
        """
        Removes an object from the specified sub-container.

        Parameters
        ==========
        obj: Param or DesVar or Point or str
            Object to remove. If a ``str`` is not specified, the objects ``name()`` method will be called to deduce
            the storage key.

        sub_container: str
            Sub-container within the ``container()``
        """
        if not isinstance(obj, str):
            obj = obj.name()

        self.container()[sub_container].pop(obj)

    def add_param(self, value: float, name: str or None = None):
        """
        Adds a parameter to the geometry collection sub-container ``"params"``, and modifies the name to make it
        unique if necessary.

        Parameters
        ==========
        value: float
            Parameter value

        name: str or None
            Parameter name
        """
        param = Param(value=value, name=name, setting_from_geo_col=True)
        param.geo_col = self

        self.add_to_subcontainer(param, "params")
        return param

    def remove_param(self, param: Param or str):
        """
        Removes a parameter from the geometry collection by name or object reference.

        Parameters
        ==========
        param: Param or str
            Parameter (or parameter name) to remove
        """
        self.remove_from_subcontainer(param, "params")

    def add_point(self, x: float, y: float):
        """
        Adds a point by value to the geometry collection

        Parameter
        =========
        x: float
            :math:`x`-location of the point

        y: float
            :math:`y`-location of the point

        Returns
        =======
        Point
            Object reference
        """
        point = Point(x=x, y=y, name="Point", setting_from_geo_col=True)

        self.add_to_subcontainer(point, "points")
        self.add_to_subcontainer(point.x(), "params", assign_unique_name=False)
        self.add_to_subcontainer(point.y(), "params", assign_unique_name=False)

        point.x().geo_col = self
        point.y().geo_col = self
        point.geo_col = self

        return point

    def remove_point(self, point: Point or str):
        """
        Removes a point by object reference or by name

        Parameters
        ==========
        point: Point or str
            Reference to or name of the point
        """
        point = point if isinstance(point, Point) else self.container()["points"][point]
        self.remove_from_subcontainer(point.x(), "params")
        self.remove_from_subcontainer(point.y(), "params")
        self.remove_from_subcontainer(point.name(), "points")

    def add_desvar(self, value: float, name: str, lower: float or None = None, upper: float or None = None):
        """
        Directly adds a design variable value to the geometry collection.

        Parameters
        ==========
        value: float
            Value of the design variable

        name: str
            Name of the design variable (might be overridden when adding to the 'desvar' sub-container).

        lower: float or None
            Lower bound for the design variable. If ``None``, a reasonable value will be chosen automatically.
            Default: ``None``.

        upper: float or none.
            Upper bound for the design variable. If ``None``, a reasonable value will be chosen automatically.
            Default: ``None``.

        Returns
        =======
        DesVar
            The generated design variable
        """
        desvar = DesVar(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=True)
        desvar.geo_col = self

        self.add_to_subcontainer(desvar, "desvar")
        return desvar

    def remove_desvar(self, desvar: DesVar or str):
        """
        Removes a design variable from the geometry collection

        Parameters
        ==========
        desvar: DesVar or str
            Design variable to remove. If a ``str`` is not specified, the objects ``name()`` method will be called
            to deduce the storage key.
        """
        desvar = desvar if isinstance(desvar, DesVar) else self.container()["desvar"][desvar]

        self.remove_from_subcontainer(desvar, "desvar")

    @staticmethod
    def replace_geo_objs(tool: Param or DesVar, target: Param or DesVar):
        """
        This static method is used to make sure that in param/desvar promotion/demotion, any references to geometric
        objects get replaced.

        Parameters
        ==========
        tool: Param or DesVar
            Object to be removed, and all geometric object references replaced with the target

        target: Param or DesVar
            Object to add
        """
        for geo_obj in tool.geo_objs:
            if isinstance(geo_obj, Point):
                if geo_obj.x() is tool:
                    geo_obj.set_x(target)
                elif geo_obj.y() is tool:
                    geo_obj.set_y(target)

    def promote_param_to_desvar(self, param: Param or str, lower: float or None = None, upper: float or None = None):
        """
        Promotes a parameter to a design variable by adding bounds. The ``Param`` will be removed from the 'params'
        sub-container, and the corresponding ``DesVar`` will be added to the 'desvar' sub-container.

        Parameters
        ==========
        param: Param or str
            Parameter to promote. If ``str``, the ``Param`` will be identified by looking in the 'params' sub-container.

        lower: float or None
            Lower bound for the design variable. If ``None``, a reasonable value will be chosen automatically.
            Default: ``None``.

        upper: float or none.
            Upper bound for the design variable. If ``None``, a reasonable value will be chosen automatically.
            Default: ``None``.

        Returns
        =======
        DesVar
            The generated design variable.
        """
        param = param if isinstance(param, Param) else self.container()["params"][param]

        desvar = self.add_desvar(value=param.value(), name=param.name(), lower=lower, upper=upper)

        # Replace the corresponding x() or y() in parameter with the new design variable
        self.replace_geo_objs(tool=param, target=desvar)

        # Make a copy of the geometry object reference list in the new design variable
        desvar.geo_objs = param.geo_objs.copy()

        # Remove the parameter
        self.remove_param(param)

        return desvar

    def demote_desvar_to_param(self, desvar: DesVar):
        """
        Demotes a design variable to a parameter by removing the bounds. The ``DesVar`` will be removed from the
        'desvar' sub-container, and the corresponding ``Param`` will be added to the 'params' sub-container.

        Parameters
        ==========
        desvar: DesVar
            Parameter to promote. If ``str``, the ``Param`` will be identified by looking in the 'params' sub-container.

        Returns
        =======
        Param
            The generated parameter
        """
        param = self.add_param(value=desvar.value(), name=desvar.name())

        # Replace the corresponding x() or y() in parameter with the new parameter
        self.replace_geo_objs(tool=desvar, target=param)

        # Make a copy of the geometry object reference list in the new parameter
        param.geo_objs = desvar.geo_objs.copy()

        # Remove the design variable
        self.remove_desvar(desvar)

        return param

    def extract_design_variable_values(self, bounds_normalized: bool = False):
        """
        Extracts the values of the design variables in the geometry collection 'desvar' container.

        Parameters
        ==========
        bounds_normalized: bool
            Whether to normalize the design variable values by the bounds before extraction. Default: ``False``.

        Returns
        =======
        typing.List[float]
            List of design variable values, normalized the bounds if specified.
        """
        return [dv.value(bounds_normalized=bounds_normalized) for dv in self.container()["desvar"].values()]

    def assign_design_variable_values(self, dv_values: typing.Union[typing.Iterable, typing.Sized],
                                      bounds_normalized: bool = False):
        """
        Assigns a list or array of design variable values, possibly normalized by the bounds, to the design variables
        in the geometry collection 'desvar' container.

        Parameters
        ==========
        dv_values: typing.Union[typing.Iterable, typing.Sized]
            List or array of design variable values. Must be equal in length to the number of design variables in the
            'desvar' container. Note that the value assignment will not necessarily take place in the same order as
            the order of the design variables in the GUI, but rather in the order they are present in the underlying
            dictionary.

        bounds_normalized: bool
            Whether the specified ``dv_values`` are normalized by the design variable bounds. Default: ``False``.
        """
        # First check that the length of the input vector and the number of design variables are equal
        dv_val_len = len(dv_values)
        num_dv = len(self.container()["desvar"])
        if dv_val_len != num_dv:
            raise ValueError(f"Length of design variable values to assign ({dv_val_len}) is different than the "
                             f"number of design variables ({num_dv})")

        # Set the values
        for dv, dv_value in zip(self.container()["desvar"].values(), dv_values):
            dv.set_value(dv_value, bounds_normalized=bounds_normalized)

    def alphabetical_sub_container_key_list(self, sub_container: str):
        """
        This method sorts a sub-container with the following rules:

        - Text case does not matter
        - Any consecutive numerical strings should appear in descending order
        - For parameters with an associated index, the parameter with an implied index of 1 should appear first,
          provided it exists. For example, ``"Point-2.x"`` should come after ``"Point.x"``, and ``"myParam-5"``
          should come after ``"myParam"``.

        Parameters
        ==========
        sub_container: str
            Sub-container from the geometry collection to sort alphabetically

        Returns
        =======
        typing.List[str]
            The alphabetically sorted list
        """
        original_list = [k for k in self.container()[sub_container].keys()]
        multiples = []
        for idx, k in enumerate(original_list):
            if "-" not in k:
                continue
            for sub_dot in k.split("."):
                if "-" not in sub_dot:
                    continue
                multiple = sub_dot.split("-")[0]
                if multiple not in multiples:
                    multiples.append(multiple)

        modified_list = []
        for idx, k in enumerate(original_list):
            if all([m not in k for m in multiples]):
                modified_list.append(k)
                continue
            k_split = k.split(".")
            for sub_idx, sub_dot in enumerate(k_split):
                if "-" in sub_dot or all([m != sub_dot for m in multiples]):
                    if sub_idx == 0:
                        modified_list.append(sub_dot)
                    else:
                        modified_list[-1] += sub_dot
                    if sub_idx < len(k_split) - 1:
                        modified_list[-1] += "."
                    continue
                modified_list.append(sub_dot + "-1")
                if sub_idx < len(k_split) - 1:
                    modified_list[-1] += "."

        sorted_list = sorted(modified_list, key=lambda k: [int(c) if c.isdigit() else c.lower()
                                                           for c in re.split("([0-9]+)", k)])
        for idx, ks in enumerate(sorted_list):
            if "-1." in ks:
                ks = ks.replace("-1.", ".")
            if ks[-2:] == "-1":
                ks = ks[:-2]
            sorted_list[idx] = ks

        return sorted_list
