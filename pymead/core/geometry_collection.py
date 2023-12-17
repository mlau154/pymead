import re
import typing

from pymead.core.airfoil2 import Airfoil
from pymead.core.bezier2 import Bezier
from pymead.core.constraints import CollinearConstraint, CurvatureConstraint, GeoCon
from pymead.core.dimensions import LengthDimension, AngleDimension, Dimension
from pymead.core.mea2 import MEA
from pymead.core.pymead_obj import DualRep, PymeadObj
from pymead.core.line2 import LineSegment
from pymead.core.param2 import Param, LengthParam, AngleParam, DesVar, LengthDesVar, AngleDesVar
from pymead.core.point import Point, PointSequence


class GeometryCollection(DualRep):
    def __init__(self):
        self._container = {
            "desvar": {},
            "params": {},
            "points": {},
            "lines": {},
            "bezier": {},
            "airfoils": {},
            "mea": {},
            "geocon": {},
            "dims": {},
        }
        self.canvas = None
        self.tree = None
        self.selected_objects = {k: [] for k in self._container.keys()}
        self.selected_airfoils = []
        self.single_step = 0.01

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
        ``"Point-2.x"``. Note that, for parameters and design variables, the first parameter added with a given name
        will not have an index by default. For example, if a ``Param`` with ``name=="my_param"`` is added three times,
        the resulting names, in order, will be ``"my_param"``, ``"my_param-2"``, and ``"my_param-3"``.

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
        specified_name = specified_name.split("-")[0]
        max_index = 0
        for name in name_list:
            dot_split = name.split(".")
            dash_split = dot_split[-1].split("-")
            if len(dot_split) == 1:  # If there were no dots,
                prefix = dash_split[0]
            else:
                prefix = ".".join(dot_split)[:-1] + dash_split[0]
            if specified_name != prefix:  # If the specified name does not match the name
                # being analyzed, continue to next name
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
            return f"{specified_name}-1"
        else:
            return f"{specified_name}-{max_index + 1}"

    def add_to_subcontainer(self, pymead_obj: PymeadObj, assign_unique_name: bool = True):
        """
        Adds an object to a sub-container within the geometry collection's ``container()``. Also performs the task
        of assigning a unique name to the object before insertion into the sub-container, if necessary.
        
        Parameters
        ==========
        pymead_obj: PymeadObj
            Object to add to the sub-container
            
        assign_unique_name: bool
            Whether to assign the object a unique name before insertion into the sub-container. Default: ``True``.
        """
        # Set the object's name to a unique name if necessary
        if assign_unique_name:
            name_list = self.get_name_list(sub_container=pymead_obj.sub_container)
            unique_name = self.unique_namer(pymead_obj.name(), name_list)
            if isinstance(pymead_obj, Param) and unique_name.split("-")[1] == "1":
                pass
            else:
                pymead_obj.set_name(unique_name)

        # Add the object to the geometry collection sub-container
        self.container()[pymead_obj.sub_container][pymead_obj.name()] = pymead_obj

    def remove_from_subcontainer(self, pymead_obj: PymeadObj):
        """
        Removes an object from the specified sub-container.

        Parameters
        ==========
        pymead_obj: PymeadObj
            Object to remove.
        """
        self.container()[pymead_obj.sub_container].pop(pymead_obj.name())

    def add_param(self, value: float, name: str or None = None, lower: float or None = None,
                  upper: float or None = None, unit_type: str or None = None, assign_unique_name: bool = True):
        """
        Adds a parameter to the geometry collection sub-container ``"params"``, and modifies the name to make it
        unique if necessary.

        Parameters
        ==========
        value: float
            Parameter value

        name: str or None
            Parameter name

        lower: float or None
            Lower bound. If ``None``, no bound will be set. Default: ``None``.

        upper: float or None
            Upper bound. If ``None``, no bound will be set. Default: ``None``.

        unit_type: str or None:
            The unit type of design variable to create. Default: ``None``.

        Returns
        =======
        Param
            The generated parameter
        """
        kwargs = dict(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=True)
        if unit_type is None:
            param = Param(**kwargs)
        elif unit_type == "length":
            param = LengthParam(**kwargs)
        elif unit_type == "angle":
            param = AngleParam(**kwargs)
        else:
            raise ValueError(f"unit_type must be None, 'length', or 'angle'. Found type: {type(unit_type)}")

        return self.add_pymead_obj_by_ref(param, assign_unique_name=assign_unique_name)

    def select_object(self, pymead_obj: PymeadObj):
        if self.tree is not None:
            self.tree.setItemStyle(pymead_obj.tree_item, "default")
            pymead_obj.tree_item.hoverable = False
            pymead_obj.tree_item.setSelected(True)

        if self.canvas is not None:
            if isinstance(pymead_obj, Point):
                self.canvas.setItemStyle(pymead_obj.canvas_item, "selected")
                pymead_obj.canvas_item.hoverable = False
            elif isinstance(pymead_obj, Airfoil):
                for curve in pymead_obj.curves:
                    self.canvas.setItemStyle(curve.canvas_item, "hovered")
                    curve.canvas_item.hoverable = False

        # if isinstance(pymead_obj, Point):
        #     if pymead_obj not in self.selected_objects:
        #         self.selected_objects.append(pymead_obj)
        # elif isinstance(pymead_obj, Airfoil):
        #     if pymead_obj not in self.selected_airfoils:
        #         self.selected_airfoils.append(pymead_obj)

        if pymead_obj not in self.selected_objects[pymead_obj.sub_container]:
            self.selected_objects[pymead_obj.sub_container].append(pymead_obj)

        # print(f"Selecting object {pymead_obj}. {self.selected_objects = }")

    def deselect_object(self, pymead_obj: PymeadObj):
        if self.tree is not None:
            if pymead_obj.tree_item is not None:
                pymead_obj.tree_item.hoverable = True
                pymead_obj.tree_item.setSelected(False)

        if self.canvas is not None:
            if isinstance(pymead_obj, Point):
                pymead_obj.canvas_item.hoverable = True
                self.canvas.setItemStyle(pymead_obj.canvas_item, "default")
            elif isinstance(pymead_obj, LineSegment):
                pymead_obj.canvas_item.hoverable = True
                self.canvas.setItemStyle(pymead_obj.canvas_item, "default")
            elif isinstance(pymead_obj, Bezier):
                pymead_obj.canvas_item.hoverable = True
                self.canvas.setItemStyle(pymead_obj.canvas_item, "default")
            elif isinstance(pymead_obj, Airfoil):
                for curve in pymead_obj.curves:
                    curve.canvas_item.hoverable = True
                    self.canvas.setItemStyle(curve.canvas_item, "default")

        # if isinstance(pymead_obj, Point):
        #     if pymead_obj in self.selected_objects:
        #         self.selected_objects.remove(pymead_obj)
        # elif isinstance(pymead_obj, Airfoil):
        #     if pymead_obj in self.selected_airfoils:
        #         self.selected_airfoils.remove(pymead_obj)

        if pymead_obj in self.selected_objects[pymead_obj.sub_container]:
            self.selected_objects[pymead_obj.sub_container].remove(pymead_obj)

    def clear_selected_objects(self):
        # for point in self.selected_objects[::-1]:
        #     self.deselect_object(point)
        # for airfoil in self.selected_airfoils[::-1]:
        #     self.deselect_object(airfoil)

        for d in self.selected_objects.values():
            for obj in d[::-1]:
                self.deselect_object(obj)

    def remove_selected_objects(self):
        # Remove only the points first for speed (points are the core object in pymead, so deleting a point deletes
        # all associated pymead objects)
        for pt in self.selected_objects["points"]:
            self.remove_pymead_obj(pt)

        # Remove all the other selected objects
        remaining_subcontainers = [k for k in self.container().keys() if k != "points"]
        for sub_container in remaining_subcontainers:
            for obj in self.selected_objects[sub_container]:
                self.remove_pymead_obj(obj)

        # Clear the selected objects
        self.clear_selected_objects()

    def hover_enter_obj(self, pymead_obj: PymeadObj):
        if self.tree is not None:
            self.tree.setItemStyle(pymead_obj.tree_item, "hovered")

        if self.canvas is not None:
            self.canvas.setItemStyle(pymead_obj.canvas_item, "hovered")

    def hover_leave_obj(self, pymead_obj: PymeadObj):
        if self.tree is not None:
            self.tree.setItemStyle(pymead_obj.tree_item, "default")

        if self.canvas is not None:
            self.canvas.setItemStyle(pymead_obj.canvas_item, "default")

    def add_pymead_obj_by_ref(self, pymead_obj: PymeadObj, assign_unique_name: bool = True):
        """
        This method adds a pymead object by passing it directly to the geometry collection. If the
        object is already associated with a geometry collection, a ``ValueError`` is raised.

        Parameters
        ==========
        pymead_obj: PymeadObj
            The pymead object to add to the collection

        Returns
        =======
        PymeadObj
            The modified pymead object
        """
        if pymead_obj.geo_col is not None:
            raise ValueError("Can only add a pymead object by reference if it has not yet been added to a "
                             "geometry collection")

        pymead_obj.geo_col = self

        self.add_to_subcontainer(pymead_obj, assign_unique_name=assign_unique_name)

        if self.tree is not None:
            self.tree.addPymeadTreeItem(pymead_obj=pymead_obj)

        if self.canvas is not None:
            self.canvas.addPymeadCanvasItem(pymead_obj=pymead_obj)

        return pymead_obj

    def remove_pymead_obj(self, pymead_obj: PymeadObj):
        """
        Removes a pymead object from the geometry collection.

        Parameters
        ==========
        pymead_obj: PymeadObj
            Pymead object to remove
        """
        # Type-specific actions
        if isinstance(pymead_obj, Bezier) or isinstance(pymead_obj, LineSegment):
            # Remove all the references to this curve in each of the curve's points
            for pt in pymead_obj.point_sequence().points():
                pt.curves.remove(pymead_obj)

            if pymead_obj.airfoil is not None:
                for curve in pymead_obj.airfoil.curves:
                    if pymead_obj is not curve:
                        curve.airfoil = None
                self.remove_pymead_obj(pymead_obj.airfoil)

        if isinstance(pymead_obj, Point):
            # Loop through the curves associated with this point to see which ones need to be deleted if one point
            # is removed from their point sequence
            curves_to_delete = []
            for curve in pymead_obj.curves:
                if curve.point_removal_deletes_curve():
                    curves_to_delete.append(curve)

            # Remove the curves that need to be removed due to insufficient points in the point sequence
            for curve in curves_to_delete:
                self.remove_pymead_obj(curve)

            # Update any remaining curves
            for curve in pymead_obj.curves:
                curve.remove_point(point=pymead_obj)
                curve.update()

            for dim in pymead_obj.dims:
                self.remove_pymead_obj(dim)

            for geo_con in pymead_obj.geo_cons:
                self.remove_pymead_obj(geo_con)

        if isinstance(pymead_obj, Dimension):
            # Remove the dimension references from the points
            if isinstance(pymead_obj.target(), Point):
                pymead_obj.target().dims.remove(pymead_obj)
            elif isinstance(pymead_obj.target(), PointSequence):
                for pt in pymead_obj.target().points():
                    pt.dims.remove(pymead_obj)

            if isinstance(pymead_obj.tool(), Point):
                pymead_obj.tool().dims.remove(pymead_obj)
            elif isinstance(pymead_obj.tool(), PointSequence):
                for pt in pymead_obj.tool().points():
                    pt.dims.remove(pymead_obj)

            # Remove the parameter associated with the dimension
            self.remove_pymead_obj(pymead_obj.param())

        if isinstance(pymead_obj, GeoCon):
            # Remove the constraint references from the points
            if isinstance(pymead_obj.target(), Point):
                pymead_obj.target().geo_cons.remove(pymead_obj)
            elif isinstance(pymead_obj.target(), PointSequence):
                for pt in pymead_obj.target().points():
                    pt.geo_cons.remove(pymead_obj)

            if isinstance(pymead_obj.tool(), Point):
                pymead_obj.tool().geo_cons.remove(pymead_obj)
            elif isinstance(pymead_obj.tool(), PointSequence):
                for pt in pymead_obj.tool().points():
                    pt.geo_cons.remove(pymead_obj)

        # Remove the item from the geometry collection subcontainer
        self.remove_from_subcontainer(pymead_obj)

        # Remove the tree item if it exists
        if self.tree is not None:
            self.tree.removePymeadTreeItem(pymead_obj)

        # Remove the canvas item if it exists
        if self.canvas is not None:
            self.canvas.removeItem(pymead_obj.canvas_item)

    def add_point(self, x: float, y: float, name: str or None = None, assign_unique_name: bool = True):
        """
        Adds a point by value to the geometry collection

        Parameters
        ==========
        x: float
            :math:`x`-location of the point

        y: float
            :math:`y`-location of the point

        name: str
            Optional name for the point

        Returns
        =======
        Point
            Object reference
        """
        point = Point(x=x, y=y, name=name)
        point.x().geo_col = self
        point.y().geo_col = self

        return self.add_pymead_obj_by_ref(point, assign_unique_name=assign_unique_name)

    def add_bezier(self, point_sequence: PointSequence, name: str or None = None, assign_unique_name: bool = True):
        bezier = Bezier(point_sequence=point_sequence, name=name)

        return self.add_pymead_obj_by_ref(bezier, assign_unique_name=assign_unique_name)

    def add_line(self, point_sequence: PointSequence, name: str or None = None, assign_unique_name: bool = True):
        line = LineSegment(point_sequence=point_sequence, name=name)

        return self.add_pymead_obj_by_ref(line, assign_unique_name=assign_unique_name)

    def add_desvar(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                   unit_type: str or None = None, assign_unique_name: bool = True):
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

        upper: float or None.
            Upper bound for the design variable. If ``None``, a reasonable value will be chosen automatically.
            Default: ``None``.

        unit_type: str or None:
            The unit type of design variable to create. Default: ``None``.

        Returns
        =======
        DesVar
            The generated design variable
        """
        kwargs = dict(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=True)
        if unit_type is None:
            desvar = DesVar(**kwargs)
        elif unit_type == "length":
            desvar = LengthDesVar(**kwargs)
        elif unit_type == "angle":
            desvar = AngleDesVar(**kwargs)
        else:
            raise ValueError(f"unit_type must be None, 'length', or 'angle'. Found type: {type(unit_type)}")

        # desvar.geo_col = self
        #
        # self.add_to_subcontainer(desvar)
        #
        # if self.tree is not None:
        #     self.tree.addPymeadTreeItem(desvar)
        #
        # return desvar
        return self.add_pymead_obj_by_ref(desvar, assign_unique_name=assign_unique_name)

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

        if isinstance(param, LengthParam):
            unit_type = "length"
        elif isinstance(param, AngleParam):
            unit_type = "angle"
        else:
            unit_type = None

        desvar = self.add_desvar(value=param.value(), name=param.name(), lower=lower, upper=upper, unit_type=unit_type)

        # Replace the corresponding x() or y() in parameter with the new design variable
        self.replace_geo_objs(tool=param, target=desvar)

        # Make a copy of the geometry object reference lists in the new design variable
        desvar.geo_objs = param.geo_objs.copy()
        desvar.geo_cons = param.geo_cons.copy()
        desvar.dims = param.dims.copy()
        for dim in desvar.dims:
            dim.set_param(desvar)

        # Remove the parameter
        if param.point is None:
            self.remove_pymead_obj(param)

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
        if isinstance(desvar, LengthDesVar):
            unit_type = "length"
        elif isinstance(desvar, AngleDesVar):
            unit_type = "angle"
        else:
            unit_type = None

        param = self.add_param(value=desvar.value(), name=desvar.name(), unit_type=unit_type)

        # Replace the corresponding x() or y() in parameter with the new parameter
        self.replace_geo_objs(tool=desvar, target=param)

        # Make a copy of the geometry object reference list in the new parameter
        param.geo_objs = desvar.geo_objs.copy()
        param.geo_cons = desvar.geo_cons.copy()
        param.dims = desvar.dims.copy()
        for dim in param.dims:
            dim.set_param(param)

        # Remove the design variable
        self.remove_pymead_obj(desvar)

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

    def assign_design_variable_values(self, dv_values: list, bounds_normalized: bool = False):
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

    def add_airfoil(self, leading_edge: Point, trailing_edge: Point, upper_surf_end: Point, lower_surf_end: Point,
                    name: str or None = None, assign_unique_name: bool = True):
        airfoil = Airfoil(leading_edge=leading_edge, trailing_edge=trailing_edge, upper_surf_end=upper_surf_end,
                          lower_surf_end=lower_surf_end, name=name)

        return self.add_pymead_obj_by_ref(airfoil, assign_unique_name=assign_unique_name)

    def add_mea(self, airfoils: typing.List[Airfoil], name: str or None = None, assign_unique_name: bool = True):
        mea = MEA(airfoils=airfoils, name=name)

        return self.add_pymead_obj_by_ref(mea, assign_unique_name=assign_unique_name)

    def add_constraint(self):
        pass

    def add_length_dimension(self, tool_point: Point, target_point: Point, length_param: LengthParam or None = None,
                             name: str or None = None, assign_unique_name: bool = True):
        length_dim = LengthDimension(tool_point=tool_point, target_point=target_point, length_param=length_param,
                                     name=name)

        if length_dim.param().geo_col is None:
            self.add_pymead_obj_by_ref(pymead_obj=length_dim.param(), assign_unique_name=assign_unique_name)

        return self.add_pymead_obj_by_ref(length_dim, assign_unique_name=assign_unique_name)

    def add_angle_dimension(self, tool_point: Point, target_point: Point, angle_param: AngleDimension or None = None,
                            name: str or None = None, assign_unique_name: bool = True):
        angle_dim = AngleDimension(tool_point=tool_point, target_point=target_point, angle_param=angle_param,
                                   name=name)

        if angle_dim.param().geo_col is None:
            self.add_pymead_obj_by_ref(pymead_obj=angle_dim.param(), assign_unique_name=assign_unique_name)

        return self.add_pymead_obj_by_ref(angle_dim, assign_unique_name=assign_unique_name)

    def add_collinear_constraint(self, start_point: Point, middle_point: Point, end_point: Point,
                                 name: str or None = None, assign_unique_name: bool = True):
        collinear_constraint = CollinearConstraint(start_point=start_point, middle_point=middle_point,
                                                   end_point=end_point, name=name)

        return self.add_pymead_obj_by_ref(collinear_constraint, assign_unique_name=assign_unique_name)

    def add_curvature_constraint(self, curve_joint: Point, name: str or None = None, assign_unique_name: bool = True):
        curvature_constraint = CurvatureConstraint(curve_joint=curve_joint, name=name)

        # Remove any collinear constraints because the CurvatureConstraint behavior provides collinear constraint
        # behavior automatically
        for con in curvature_constraint.curve_joint.geo_cons:
            if isinstance(con, CollinearConstraint):
                self.remove_pymead_obj(con)

        return self.add_pymead_obj_by_ref(curvature_constraint, assign_unique_name=assign_unique_name)

    def get_dict_rep(self):
        dict_rep = {k_outer: {k: v.get_dict_rep() for k, v in self.container()[k_outer].items()}
                    for k_outer in self.container().keys()}
        return dict_rep

    @classmethod
    def set_from_dict_rep(cls, d: dict, canvas=None, tree=None):
        geo_col = cls()
        geo_col.canvas = canvas
        geo_col.tree = tree
        for desvar_dict in d["desvar"].values():
            geo_col.add_desvar(**desvar_dict, assign_unique_name=False)
        for param_dict in d["params"].values():
            geo_col.add_param(**param_dict, assign_unique_name=False)
        for point_dict in d["points"].values():
            geo_col.add_point(**point_dict, assign_unique_name=False)
        for line_dict in d["lines"].values():
            geo_col.add_line(point_sequence=PointSequence(
                points=[geo_col.container()["points"][k] for k in line_dict["points"]]),
                name=line_dict["name"], assign_unique_name=False
            )
        for bezier_dict in d["bezier"].values():
            geo_col.add_bezier(point_sequence=PointSequence(
                points=[geo_col.container()["points"][k] for k in bezier_dict["points"]]),
                name=bezier_dict["name"], assign_unique_name=False
            )
        for geocon_dict in d["geocon"].values():
            constraint_type = geocon_dict["constraint_type"]
            if constraint_type == "curvature":
                geo_col.add_curvature_constraint(
                    curve_joint=geo_col.container()["points"][geocon_dict["curve_joint"]],
                    name=geocon_dict["name"], assign_unique_name=False
                )
            elif constraint_type == "collinear":
                geo_col.add_collinear_constraint(
                    start_point=geo_col.container()["points"][geocon_dict["start_point"]],
                    middle_point=geo_col.container()["points"][geocon_dict["middle_point"]],
                    end_point=geo_col.container()["points"][geocon_dict["end_point"]],
                    name=geocon_dict["name"], assign_unique_name=False
                )
            else:
                raise ValueError(f"Invalid constraint type: {constraint_type}")
        for dim_dict in d["dims"].values():
            if "length_param" in dim_dict.keys():
                if dim_dict["length_param"] in geo_col.container()["desvar"].keys():
                    param = geo_col.container()["desvar"][dim_dict["length_param"]]
                elif dim_dict["length_param"] in geo_col.container()["params"].keys():
                    param = geo_col.container()["params"][dim_dict["length_param"]]
                else:
                    raise ValueError(f"Could not find length_param named {dim_dict['length_param']} in either the "
                                     f"desvar or params sub-containers")
                geo_col.add_length_dimension(tool_point=geo_col.container()["points"][dim_dict["tool_point"]],
                                             target_point=geo_col.container()["points"][dim_dict["target_point"]],
                                             length_param=param,
                                             name=dim_dict["name"])
            elif "angle_param" in dim_dict.keys():
                if dim_dict["angle_param"] in geo_col.container()["desvar"].keys():
                    param = geo_col.container()["desvar"][dim_dict["angle_param"]]
                elif dim_dict["angle_param"] in geo_col.container()["params"].keys():
                    param = geo_col.container()["params"][dim_dict["angle_param"]]
                else:
                    raise ValueError(f"Could not find angle_param named {dim_dict['angle_param']} in either the "
                                     f"desvar or params sub-containers")
                geo_col.add_angle_dimension(tool_point=geo_col.container()["points"][dim_dict["tool_point"]],
                                            target_point=geo_col.container()["points"][dim_dict["target_point"]],
                                            angle_param=param,
                                            name=dim_dict["name"])
            else:
                raise ValueError("Current valid load/save dimensions are LengthDimension and AngleDimension. Could "
                                 "not find either length_param or angle_param in saved dictionary.")
        for airfoil_dict in d["airfoils"].values():
            geo_col.add_airfoil(leading_edge=geo_col.container()["points"][airfoil_dict["leading_edge"]],
                                trailing_edge=geo_col.container()["points"][airfoil_dict["trailing_edge"]],
                                upper_surf_end=geo_col.container()["points"][airfoil_dict["upper_surf_end"]],
                                lower_surf_end=geo_col.container()["points"][airfoil_dict["lower_surf_end"]],
                                name=airfoil_dict["name"], assign_unique_name=False)
        for mea_dict in d["mea"].values():
            geo_col.add_mea(airfoils=[geo_col.container()["airfoils"][k] for k in mea_dict["airfoils"]],
                            name=mea_dict["name"], assign_unique_name=False)
        return geo_col
