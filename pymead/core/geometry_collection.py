import datetime
import os
import random
import re
import sys
import typing
from copy import copy

from pymead.core.ferguson import Ferguson
from pymead.core.param_graph import ParamGraph
from pymead.core.units import Units
from pymead.core.airfoil import Airfoil
from pymead.core.bezier import Bezier
from pymead.core.constraints import *
from pymead.core.gcs import GCS
from pymead.core.mea import MEA
from pymead.core.parametric_curve import ParametricCurve
from pymead.core.pymead_obj import DualRep, PymeadObj
from pymead.core.line import LineSegment, PolyLine, ReferencePolyline
from pymead.core.param import Param, LengthParam, AngleParam, DesVar, LengthDesVar, AngleDesVar
from pymead.core.point import Point, PointSequence
from pymead.core.transformation import Transformation3D
from pymead.plugins.IGES.curves import BezierIGES
from pymead.plugins.IGES.iges_generator import IGESGenerator
from pymead.version import __version__


class GeometryCollection(DualRep):
    def __init__(self, gui_obj=None):
        """
        The geometry collection is the primary class in pymead for housing all the available fundamental geometry
        types. Geometry, parameters, and constraints can be added using the nomenclature ``add_<object-name>()``.
        For example, points can be added using ``geo_col.add_point(x=<x-value>, y=<y-value>)``.
        """
        self._container = {
            "desvar": {},
            "params": {},
            "points": {},
            "reference": {},
            "lines": {},
            "polylines": {},
            "bezier": {},
            "ferguson": {},
            "airfoils": {},
            "mea": {},
            "geocon": {},
        }
        self.gcs = GCS()
        self.param_graph = ParamGraph()
        self.gcs.geo_col = self
        self.gui_obj = gui_obj
        self.canvas = None
        self.tree = None
        self.selected_objects = {k: [] for k in self._container.keys()}
        self.selected_airfoils = []
        self.single_step = 0.01
        self.units = Units()

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

    def clear_container(self):
        """
        Clears all the entities in the geometry container.
        """
        for sub_container in self.container().values():
            sub_container.clear()

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
            if pymead_obj.sub_container == "params":
                name_list.extend(self.get_name_list("desvar"))
            elif pymead_obj.sub_container == "desvar":
                name_list.extend(self.get_name_list("params"))
            unique_name = self.unique_namer(pymead_obj.name(), name_list)
            if isinstance(pymead_obj, Param) and unique_name.split("-")[1] == "1":
                pass
            else:
                pymead_obj.set_name(unique_name)

            if isinstance(pymead_obj, GeoCon):
                if pymead_obj.param() is not None and pymead_obj.param().name() == "unnamed":
                    pymead_obj.param().set_name(f"{pymead_obj.name()}.par")

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
        if pymead_obj.name() not in self.container()[pymead_obj.sub_container]:
            return
        self.container()[pymead_obj.sub_container].pop(pymead_obj.name())

    def add_param(self, value: float, name: str or None = None, lower: float or None = None,
                  upper: float or None = None, unit_type: str or None = None, assign_unique_name: bool = True,
                  point: Point = None, root: Point = None, rotation_handle: Point = None, enabled: bool = True,
                  equation_str: str = None):
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
        kwargs = dict(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=True, point=point,
                      root=root, rotation_handle=rotation_handle, enabled=enabled, equation_str=equation_str)
        if unit_type is None:
            param = Param(**kwargs)
        elif unit_type == "length":
            param = LengthParam(**kwargs, geo_col=self)
        elif unit_type == "angle":
            param = AngleParam(**kwargs, geo_col=self)
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
                    self.canvas.setItemStyle(curve.canvas_item, "selected")
                    curve.canvas_item.hoverable = False
            elif isinstance(pymead_obj, ParametricCurve):
                self.canvas.setItemStyle(pymead_obj.canvas_item, "selected")
                pymead_obj.canvas_item.hoverable = False
            elif isinstance(pymead_obj, GeoCon):
                self.canvas.setItemStyle(pymead_obj.canvas_item, "selected")
                pymead_obj.canvas_item.hoverable = False

        if pymead_obj not in self.selected_objects[pymead_obj.sub_container]:
            self.selected_objects[pymead_obj.sub_container].append(pymead_obj)

    def deselect_object(self, pymead_obj: PymeadObj):
        if self.tree is not None:
            if pymead_obj.tree_item is not None:
                pymead_obj.tree_item.hoverable = True
                pymead_obj.tree_item.setSelected(False)

        if self.canvas is not None:
            if isinstance(pymead_obj, Point):
                pymead_obj.canvas_item.hoverable = True
                self.canvas.setItemStyle(pymead_obj.canvas_item, "default")
            elif isinstance(pymead_obj, ParametricCurve):
                pymead_obj.canvas_item.hoverable = True
                self.canvas.setItemStyle(pymead_obj.canvas_item, "default")
            elif isinstance(pymead_obj, Airfoil):
                for curve in pymead_obj.curves:
                    curve.canvas_item.hoverable = True
                    self.canvas.setItemStyle(curve.canvas_item, "default")
            elif isinstance(pymead_obj, GeoCon):
                pymead_obj.canvas_item.hoverable = True
                self.canvas.setItemStyle(pymead_obj.canvas_item, "default")
                # pymead_obj.canvas_item.setStyle(theme=self.gui_obj.themes[self.gui_obj.current_theme])

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

    def add_pymead_obj_by_ref(self, pymead_obj: PymeadObj, assign_unique_name: bool = True) -> PymeadObj:
        """
        This method adds a pymead object by passing it directly to the geometry collection. If the
        object is already associated with a geometry collection, a ``ValueError`` is raised.

        Parameters
        ----------
        pymead_obj: PymeadObj
            The pymead object to add to the collection
        assign_unique_name: bool
            Whether to assign a unique name to the pymead object (by appending ``"-1"`` to the end of the name
            of the object if there are no objects with the same name, ``"-2"`` if there is one object with the same
            name, etc.). Default: ``True``

        Returns
        -------
        PymeadObj
            The modified pymead object
        """
        if pymead_obj.geo_col is not None:
            if isinstance(pymead_obj, Param) and pymead_obj.point is not None:
                pass
            elif isinstance(pymead_obj, LengthParam) or isinstance(pymead_obj, AngleParam):
                pass
            elif isinstance(pymead_obj, GeoCon):
                pass
            elif isinstance(pymead_obj, MEA):
                pass
            else:
                raise ValueError("Can only add a pymead object by reference if it has not yet been added to a "
                                 "geometry collection")
        else:
            pymead_obj.geo_col = self

        self.add_to_subcontainer(pymead_obj, assign_unique_name=assign_unique_name)

        if self.gui_obj is not None:
            pymead_obj.gui_obj = self.gui_obj

        if self.tree is not None:
            pymead_obj.tree = self.tree
            self.tree.addPymeadTreeItem(pymead_obj=pymead_obj)

        if self.canvas is not None:
            pymead_obj.canvas = self.canvas
            self.canvas.addPymeadCanvasItem(pymead_obj=pymead_obj)

        if isinstance(pymead_obj, Point):
            self.gcs.add_point(pymead_obj)
            for param in [pymead_obj.x(), pymead_obj.y()]:
                param.param_graph = self.param_graph
                if param not in param.param_graph.param_list:
                    param.param_graph.param_list.append(param)
                param.update_equation(param.equation_str)

        if isinstance(pymead_obj, Param):
            pymead_obj.param_graph = self.param_graph
            if pymead_obj not in pymead_obj.param_graph.param_list:
                pymead_obj.param_graph.param_list.append(pymead_obj)
            pymead_obj.update_equation(pymead_obj.equation_str)
            pymead_obj.set_enabled(pymead_obj.enabled())

        if isinstance(pymead_obj, LengthParam):
            pymead_obj.set_unit(unit=self.units.current_length_unit(), old_unit=self.units.current_length_unit())

        if isinstance(pymead_obj, AngleParam):
            pymead_obj.set_unit(unit=self.units.current_angle_unit(), old_unit=self.units.current_angle_unit())

        return pymead_obj

    def remove_pymead_obj(self, pymead_obj: PymeadObj, promotion_demotion: bool = False,
                          constraint_removal: bool = False, equating_constraints: bool = False):
        """
        Removes a pymead object from the geometry collection.

        Parameters
        ==========
        pymead_obj: PymeadObj
            Pymead object to remove

        promotion_demotion: bool
            When this flag is set to ``True``, the ``ValueError`` normally raised when directly deleting a ``Param``
            associated with a ``GeoCon`` is ignored. Default: ``False``

        constraint_removal: bool
            When this flag is set to ``True``, the ``ValueError`` normally raise when directly deleting a ``Param``
            associated with a constraint cluster rotation is ignored. Default: ``False``

        equating_constraints: bool
            When this flag is set to ``True`` and the ``pymead_obj`` is a ``Param``, the associated constraints are
            not deleted
        """
        # Type-specific actions
        if isinstance(pymead_obj, Param):

            if pymead_obj.rotation_handle and not constraint_removal and not promotion_demotion:
                error_message = f"This parameter can only be removed by deleting its associated constraint cluster"
                if self.gui_obj is None:
                    raise ValueError(error_message)
                else:
                    self.gui_obj.disp_message_box(error_message, message_mode="error")
                    return

            if not promotion_demotion and not equating_constraints:  # Do not remove the constraints if this is a
                # promotion/demotion action or an equating constraints action
                for geo_con in pymead_obj.geo_cons:
                    self.remove_pymead_obj(geo_con)

            if pymead_obj in self.param_graph.param_list:
                self.param_graph.param_list.remove(pymead_obj)
            if pymead_obj in self.param_graph.nodes:
                self.param_graph.remove_node(pymead_obj)

        elif isinstance(pymead_obj, Bezier) or isinstance(pymead_obj, LineSegment) or isinstance(
                pymead_obj, PolyLine) or isinstance(pymead_obj, Ferguson):
            # Remove all the references to this curve in each of the curve's points
            for pt in pymead_obj.point_sequence().points():
                if pymead_obj in pt.curves:
                    pt.curves.remove(pymead_obj)

            # If this is an airfoil curve, delete the airfoil
            if pymead_obj.airfoil is not None:
                for curve in pymead_obj.airfoil.curves:
                    if pymead_obj is not curve:
                        curve.airfoil = None
                self.remove_pymead_obj(pymead_obj.airfoil)

            # mark airfoils for removal if this is a trailing edge line for those airfoils
            if isinstance(pymead_obj, LineSegment):
                airfoils_to_delete = []
                for airfoil in self.container()["airfoils"].values():
                    te_references = [airfoil.trailing_edge, airfoil.upper_surf_end, airfoil.lower_surf_end]
                    if pymead_obj.points()[0] in te_references and pymead_obj.points()[1] in te_references:
                        airfoils_to_delete.append(airfoil)

                # Remove the airfoils that need to be removed due to this line being a trailing edge line
                for airfoil in airfoils_to_delete:
                    try:
                        self.remove_pymead_obj(airfoil)
                    except KeyError:
                        pass

        elif isinstance(pymead_obj, Point):

            # Demote and cover both x and y parameters if necessary
            if pymead_obj.x() in self.container()["desvar"].values():
                self.demote_desvar_to_param(pymead_obj.x())
            if pymead_obj.y() in self.container()["desvar"].values():
                self.demote_desvar_to_param(pymead_obj.y())
            if pymead_obj.x() in self.container()["params"].values() or pymead_obj.y() in self.container()["params"].values():
                self.cover_point_xy(pymead_obj)

            # Remove the x and y parameters from the parameter graph
            for param in [pymead_obj.x(), pymead_obj.y()]:
                if param in param.param_graph.param_list:
                    param.param_graph.param_list.remove(param)
                if param in param.param_graph.nodes:
                    param.param_graph.remove_node(param)

            # Remove any constraints associated with this point
            for geo_con in pymead_obj.geo_cons:
                self.remove_pymead_obj(geo_con)

            # Loop through the curves associated with this point to see which ones need to be deleted if one point
            # is removed from their point sequence
            curves_to_delete = []
            for curve in pymead_obj.curves:
                if curve.point_removal_deletes_curve():
                    curves_to_delete.append(curve)

            # If this point is a trailing edge of one or more airfoils, remove those airfoils
            airfoils_to_delete = []
            for airfoil in self.container()["airfoils"].values():
                if pymead_obj is airfoil.trailing_edge:
                    airfoils_to_delete.append(airfoil)

            # Remove the curves that need to be removed due to insufficient points in the point sequence
            for curve in curves_to_delete:
                try:
                    self.remove_pymead_obj(curve)
                except KeyError:  # This curve may have already been deleted in a prior step, so catch this exception
                    pass

            # Remove the airfoils that need to be removed due to this point being a trailing edge
            for airfoil in airfoils_to_delete:
                try:
                    self.remove_pymead_obj(airfoil)
                except KeyError:  # This airfoil may have already been deleted in a prior step, so catch this exception
                    pass

            # Update any remaining curves
            for curve in pymead_obj.curves:
                if pymead_obj in curve.point_sequence().points():
                    curve.remove_point(point=pymead_obj)
                    curve.update()

            for geo_con in pymead_obj.geo_cons[::-1]:
                self.remove_pymead_obj(geo_con)

            self.gcs.remove_point(pymead_obj)

        elif isinstance(pymead_obj, GeoCon):
            # First, remove the parameter associated with the constraint if necessary (i.e., if that parameter is not
            # tied to any other constraints
            if pymead_obj.param() is not None:
                pymead_obj.param().geo_cons.remove(pymead_obj)
                if len(pymead_obj.param().geo_cons) == 0:
                    self.remove_pymead_obj(pymead_obj.param())

            # if (isinstance(pymead_obj, DistanceConstraint) or isinstance(pymead_obj, RelAngle3Constraint) or
            #     isinstance(pymead_obj, Perp3Constraint) or isinstance(pymead_obj, AntiParallel3Constraint)):
            #     if pymead_obj.p2.rotation_handle and pymead_obj.p2.rotation_param is not None:
            #         self.remove_pymead_obj(pymead_obj.p2.rotation_param, constraint_removal=True)

            # Remove the constraint from the ConstraintGraph
            self.gcs.remove_constraint(pymead_obj)

        elif isinstance(pymead_obj, Airfoil):
            for curve in pymead_obj.curves:
                curve.airfoil = None
            mea_to_delete = []
            for mea in self.container()["mea"].values():
                if pymead_obj not in mea.airfoils:
                    continue
                mea_to_delete.append(mea)
            for mea in mea_to_delete:
                self.remove_pymead_obj(mea)

        # Remove the item from the geometry collection subcontainer
        self.remove_from_subcontainer(pymead_obj)

        # Remove the tree item if it exists
        if self.tree is not None:
            self.tree.removePymeadTreeItem(pymead_obj)

        # Remove the canvas item if it exists
        if self.canvas is not None:
            if hasattr(pymead_obj.canvas_item, "canvas_items"):  # This is the case for GeoCons
                for canvas_item in pymead_obj.canvas_item.canvas_items:
                    self.canvas.removeItem(canvas_item)
            else:
                self.canvas.removeItem(pymead_obj.canvas_item)

        if isinstance(pymead_obj, Airfoil):
            self.gui_obj.permanent_widget.updateAirfoils()

    def add_point(self, x: float, y: float, name: str or None = None, relative_airfoil_name: str = None,
                  assign_unique_name: bool = True):
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
        point = Point(x=x, y=y, name=name, relative_airfoil_name=relative_airfoil_name)
        point.x().geo_col = self
        point.y().geo_col = self
        point.x().set_unit(unit=self.units.current_length_unit(), old_unit=self.units.current_length_unit())
        point.y().set_unit(unit=self.units.current_length_unit(), old_unit=self.units.current_length_unit())
        self.add_pymead_obj_by_ref(point, assign_unique_name=assign_unique_name)

        return point

    def add_bezier(self, point_sequence: PointSequence or typing.List[Point],
                   default_nt: int or None = None,
                   name: str or None = None,
                   t_start: float = None, t_end: float = None, assign_unique_name: bool = True):
        bezier = Bezier(point_sequence=point_sequence, default_nt=default_nt, name=name, t_start=t_start, t_end=t_end)

        return self.add_pymead_obj_by_ref(bezier, assign_unique_name=assign_unique_name)

    def add_ferguson(self, point_sequence: PointSequence or typing.List[Point],
                     default_nt: int or None = None, name: str or None = None,
                     t_start: float = None, t_end: float = None, assign_unique_name: bool = True):
        ferguson = Ferguson(point_sequence=point_sequence, default_nt=default_nt, name=name,
                            t_start=t_start, t_end=t_end)

        return self.add_pymead_obj_by_ref(ferguson, assign_unique_name=assign_unique_name)

    def add_line(self, point_sequence: PointSequence or typing.List[Point], name: str or None = None,
                 assign_unique_name: bool = True):
        line = LineSegment(point_sequence=point_sequence, name=name)

        return self.add_pymead_obj_by_ref(line, assign_unique_name=assign_unique_name)

    def add_polyline(self, source: str, coords: np.ndarray = None, start: int or float = None, end: int or float = None,
                     point_sequence: PointSequence = None,
                     name: str or None = None, assign_unique_name: bool = True):
        polyline = PolyLine(source=source, coords=coords, start=start, end=end, point_sequence=point_sequence,
                            name=name)
        if point_sequence is None:
            for point in polyline.point_sequence().points():
                if point not in self.container()["points"].values():
                    self.add_pymead_obj_by_ref(point)
        return self.add_pymead_obj_by_ref(polyline, assign_unique_name=assign_unique_name)

    def add_reference_polyline(self, points: typing.List[typing.List[float]] or np.ndarray = None, source: str = None,
                               num_header_rows: int = 0, delimiter: str or None = None,
                               color: tuple = (245, 37, 106), lw: float = 1.0, name: str = None,
                               assign_unique_name: bool = True):
        ref_polyline = ReferencePolyline(points=points, source=source, num_header_rows=num_header_rows,
                                         delimiter=delimiter, color=color, lw=lw, name=name)

        return self.add_pymead_obj_by_ref(ref_polyline, assign_unique_name=assign_unique_name)

    def split_polyline(self, polyline: PolyLine, split: int or float):
        new_polylines = polyline.split(split)

        # Remove the old polyline and its dependent points
        # if polyline.start is None and polyline.end is None:
        #     for point in polyline.point_sequence().points():
        #         self.remove_pymead_obj(point)
        #     polyline.point_sequence().points().clear()
        # else:
        #     self.remove_pymead_obj(polyline)
        self.remove_pymead_obj(polyline)
        for new_polyline in new_polylines:
            for point in new_polyline.point_sequence().points():
                if point not in self.container()["points"].values():
                    self.add_pymead_obj_by_ref(point)
            self.add_pymead_obj_by_ref(new_polyline)

    def add_desvar(self, value: float, name: str, lower: float or None = None, upper: float or None = None,
                   unit_type: str or None = None, assign_unique_name: bool = True, point: Point = None,
                   root: Point = None, rotation_handle: Point = None, enabled: bool = True, equation_str: str = None):
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
        kwargs = dict(value=value, name=name, lower=lower, upper=upper, setting_from_geo_col=True, point=point,
                      root=root, rotation_handle=rotation_handle, enabled=enabled, equation_str=equation_str)
        if unit_type is None:
            desvar = DesVar(**kwargs)
        elif unit_type == "length":
            desvar = LengthDesVar(**kwargs)
        elif unit_type == "angle":
            desvar = AngleDesVar(**kwargs)
        else:
            raise ValueError(f"unit_type must be None, 'length', or 'angle'. Found type: {type(unit_type)}")

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

        if param.point is not None and len(param.point.geo_cons) > 0 and not param.point.root:
            raise ValueError(f"Promotion of {param.name()} to a design variable is not allowed because the "
                             f"parent point ({param.point.name()}) contains direct constraints to one or "
                             f"more other points and is not the root of a constraint cluster. "
                             f"Try removing the constraints to allow promotion.")

        if isinstance(param, LengthParam):
            unit_type = "length"
        elif isinstance(param, AngleParam):
            unit_type = "angle"
        else:
            unit_type = None

        desvar = self.add_desvar(value=param.value(), name=param.name(), lower=lower, upper=upper, unit_type=unit_type,
                                 point=copy(param.point), root=param.root,
                                 rotation_handle=param.rotation_handle, assign_unique_name=False)

        # Replace the corresponding x() or y() in parameter with the new design variable
        self.replace_geo_objs(tool=param, target=desvar)

        # Make a copy of the geometry object reference lists in the new design variable
        desvar.geo_objs = param.geo_objs.copy()

        # Copy constraint information
        desvar.geo_cons = param.geo_cons
        desvar.gcs = self.gcs
        for constraint in desvar.geo_cons:
            constraint.set_param(desvar)
        # desvar.gcs.constraint_params[self.gcs.constraint_params.index(param)] = desvar

        # Remove the parameter
        self.remove_pymead_obj(param, promotion_demotion=True)

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

        param = self.add_param(value=desvar.value(), name=desvar.name(), unit_type=unit_type, point=copy(desvar.point),
                               root=desvar.root, rotation_handle=desvar.rotation_handle, assign_unique_name=False)

        # Replace the corresponding x() or y() in parameter with the new parameter
        self.replace_geo_objs(tool=desvar, target=param)

        # Make a copy of the geometry object reference list in the new parameter
        param.geo_objs = desvar.geo_objs.copy()

        # Copy constraint information
        param.geo_cons = desvar.geo_cons
        param.gcs = self.gcs
        for constraint in param.geo_cons:
            constraint.set_param(param)
        # param.gcs.constraint_params[self.gcs.constraint_params.index(desvar)] = param

        # Remove the design variable
        self.remove_pymead_obj(desvar, promotion_demotion=True)

        return param

    def expose_point_xy(self, point: Point):
        self.add_pymead_obj_by_ref(point.x(), assign_unique_name=False)
        self.add_pymead_obj_by_ref(point.y(), assign_unique_name=False)

    def cover_point_xy(self, point: Point):
        self.remove_pymead_obj(point.x())
        self.remove_pymead_obj(point.y())

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
        num_assignable_dvs = len([dv for dv in self.container()["desvar"].values() if dv.assignable])
        if dv_val_len != num_assignable_dvs:
            raise ValueError(f"Length of design variable values to assign ({dv_val_len}) is different than the "
                             f"number of assignable design variables ({num_assignable_dvs})")

        # Set the values
        for dv, dv_value in zip(self.container()["desvar"].values(), dv_values):
            if not dv.assignable:
                continue
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

    def add_airfoil(self,
                    leading_edge: Point,
                    trailing_edge: Point,
                    upper_surf_end: Point or None = None,
                    lower_surf_end: Point or None = None,
                    name: str or None = None, assign_unique_name: bool = True):
        airfoil = Airfoil(leading_edge=leading_edge, trailing_edge=trailing_edge, upper_surf_end=upper_surf_end,
                          lower_surf_end=lower_surf_end, name=name)

        return self.add_pymead_obj_by_ref(airfoil, assign_unique_name=assign_unique_name)

    def add_mea(self, airfoils: typing.List[Airfoil], name: str or None = None, assign_unique_name: bool = True):
        mea = MEA(airfoils=airfoils, name=name, geo_col=self)

        return self.add_pymead_obj_by_ref(mea, assign_unique_name=assign_unique_name)

    def add_constraint(self, constraint_type: str, *constraint_args, assign_unique_name: bool = True,
                       **constraint_kwargs):
        constraint = getattr(sys.modules[__name__], constraint_type)(*constraint_args, geo_col=self, **constraint_kwargs)
        self.gcs.check_constraint_for_duplicates(constraint)
        self.add_pymead_obj_by_ref(constraint, assign_unique_name=assign_unique_name)
        if (constraint.param() is not None and constraint.param() not in self.container()["params"].values() and
                constraint.param() not in self.container()["desvar"].values()):
            self.add_pymead_obj_by_ref(constraint.param())
        try:
            self.gcs.add_constraint(constraint)
            if isinstance(constraint, AntiParallel3Constraint):
                if any([isinstance(curve, PolyLine) for curve in constraint.p1.curves]) and not constraint.p1.root:
                    self.gcs.move_root(constraint.p1)
                elif any([isinstance(curve, PolyLine) for curve in constraint.p3.curves]) and not constraint.p3.root:
                    self.gcs.move_root(constraint.p3)

            if (isinstance(constraint, AntiParallel3Constraint) or isinstance(constraint, Perp3Constraint) or
                    isinstance(constraint, RelAngle3Constraint) or isinstance(constraint, DistanceConstraint)):
                points_solved = self.gcs.solve(constraint)
                self.gcs.update_canvas_items(points_solved)
        except ValueError as e:
            self.remove_pymead_obj(constraint)
            self.clear_selected_objects()
            if self.gui_obj is not None:
                # self.gui_obj.showColoredMessage("Constraint cluster is over-constrained. Removing constraint...",
                #                                 4000, "#eb4034")
                self.gui_obj.disp_message_box(str(e), message_mode="error")
            return
        return constraint

    def equate_constraints(self, constraint1: GeoCon, constraint2: GeoCon):
        if constraint1.__class__.__name__ != constraint2.__class__.__name__:
            raise ValueError("Constraints must be of the same type to equate")

        self.remove_pymead_obj(constraint2.param(), equating_constraints=True)
        constraint2.set_param(constraint1.param())
        constraint1.param().geo_cons.append(constraint2)

        # Manually trigger an update by setting the value to the current value
        constraint1.param().set_value(constraint1.param().value())

    def get_dict_rep(self):
        dict_rep = {k_outer: {k: v.get_dict_rep() for k, v in self.container()[k_outer].items()}
                    for k_outer in self.container().keys()}
        dict_rep["metadata"] = self.get_metadata()
        return dict_rep

    def get_metadata(self):
        return {
            "pymead_version": __version__,
            "save_datetime": str(datetime.datetime.now()),
            "length_unit": self.units.current_length_unit(),
            "angle_unit":  self.units.current_angle_unit(),
            "area_unit": self.units.current_area_unit()
        }

    @classmethod
    def set_from_dict_rep(cls, d: dict, canvas=None, tree=None, gui_obj=None):
        geo_col = cls(gui_obj=gui_obj)
        geo_col.canvas = canvas
        geo_col.tree = tree
        geo_col.units.set_current_length_unit(d["metadata"]["length_unit"])
        geo_col.units.set_current_angle_unit(d["metadata"]["angle_unit"])
        geo_col.units.set_current_area_unit(d["metadata"]["area_unit"])
        if "reference" in d:
            for name, ref_dict in d["reference"].items():
                geo_col.add_reference_polyline(**ref_dict, name=name, assign_unique_name=False)
        for name, point_dict in d["points"].items():
            geo_col.add_point(**point_dict, name=name, assign_unique_name=False)
        for name, desvar_dict in d["desvar"].items():
            if ".x" in name:
                point = geo_col.container()["points"][name.split(".")[0]]
                geo_col.add_pymead_obj_by_ref(point.x(), assign_unique_name=False)
                geo_col.promote_param_to_desvar(point.x(), lower=desvar_dict["lower"], upper=desvar_dict["upper"])
            elif ".y" in name:
                point = geo_col.container()["points"][name.split(".")[0]]
                geo_col.add_pymead_obj_by_ref(point.y(), assign_unique_name=False)
                geo_col.promote_param_to_desvar(point.y(), lower=desvar_dict["lower"], upper=desvar_dict["upper"])
            else:
                root_name = desvar_dict.pop("root") if "root" in desvar_dict else None
                root = geo_col.container()["points"][root_name] if root_name is not None else None
                rotation_handle_name = desvar_dict.pop("rotation_handle") if "rotation_handle" in desvar_dict else None
                rotation_handle = geo_col.container()["points"][rotation_handle_name] \
                    if rotation_handle_name is not None else None
                geo_col.add_desvar(**desvar_dict, name=name, assign_unique_name=False, rotation_handle=rotation_handle,
                                   root=root)
        for name, param_dict in d["params"].items():
            if ".x" in name:
                point = geo_col.container()["points"][name.split(".")[0]]
                geo_col.add_pymead_obj_by_ref(point.x(), assign_unique_name=False)
            elif ".y" in name:
                point = geo_col.container()["points"][name.split(".")[0]]
                geo_col.add_pymead_obj_by_ref(point.y(), assign_unique_name=False)
            else:
                root_name = param_dict.pop("root") if "root" in param_dict else None
                root = geo_col.container()["points"][root_name] if root_name is not None else None
                rotation_handle_name = param_dict.pop("rotation_handle") if "rotation_handle" in param_dict else None
                rotation_handle = geo_col.container()["points"][rotation_handle_name] \
                    if rotation_handle_name is not None else None
                geo_col.add_param(**param_dict, name=name, assign_unique_name=False, rotation_handle=rotation_handle,
                                  root=root)
        for name, line_dict in d["lines"].items():
            geo_col.add_line(point_sequence=PointSequence(
                points=[geo_col.container()["points"][k] for k in line_dict["points"]]),
                name=name, assign_unique_name=False
            )
        if "polylines" in d:
            for name, polyline_dict in d["polylines"].items():
                geo_col.add_polyline(point_sequence=PointSequence(
                    points=[geo_col.container()["points"][k] for k in polyline_dict["points"]]),
                    coords=np.array(polyline_dict["coords"]) if "coords" in polyline_dict else None,
                    source=polyline_dict["source"], start=polyline_dict["start"], end=polyline_dict["end"], name=name,
                    assign_unique_name=False)
        for name, bezier_dict in d["bezier"].items():
            geo_col.add_bezier(point_sequence=PointSequence(
                points=[geo_col.container()["points"][k] for k in bezier_dict["points"]]),
                name=name, assign_unique_name=False,
                default_nt=bezier_dict["default_nt"] if "default_nt" in bezier_dict else None
            )
        if "ferguson" in d:
            for name, ferguson_dict in d["ferguson"].items():
                geo_col.add_ferguson(point_sequence=PointSequence(
                    points=[geo_col.container()["points"][k] for k in ferguson_dict["points"]]),
                    name=name, assign_unique_name=False,
                    default_nt=ferguson_dict["default_nt"] if "default_nt" in ferguson_dict else None
                )

        constraints_added = []
        for name, geocon_dict in d["geocon"].items():
            for k, v in geocon_dict.items():
                if v in d["points"].keys():
                    geocon_dict[k] = geo_col.container()["points"][v]
                elif v in d["params"].keys():
                    geocon_dict[k] = geo_col.container()["params"][v]
                elif v in d["desvar"].keys():
                    geocon_dict[k] = geo_col.container()["desvar"][v]
                elif v in d["lines"].keys():
                    geocon_dict[k] = geo_col.container()["lines"][v]
                elif "polylines" in d and v in d["polylines"].keys():
                    geocon_dict[k] = geo_col.container()["polylines"][v]
                elif v in d["bezier"].keys():
                    geocon_dict[k] = geo_col.container()["bezier"][v]
                elif "ferguson" in d and v in d["ferguson"].keys():
                    geocon_dict[k] = geo_col.container()["ferguson"][v]
                else:
                    pass
            constraint_type = geocon_dict.pop("constraint_type")
            geocon_dict["name"] = name
            constraint = geo_col.add_constraint(constraint_type, **geocon_dict, assign_unique_name=False)
            constraints_added.append(constraint)
        for name, airfoil_dict in d["airfoils"].items():
            geo_col.add_airfoil(leading_edge=geo_col.container()["points"][airfoil_dict["leading_edge"]],
                                trailing_edge=geo_col.container()["points"][airfoil_dict["trailing_edge"]],
                                upper_surf_end=geo_col.container()["points"][airfoil_dict["upper_surf_end"]],
                                lower_surf_end=geo_col.container()["points"][airfoil_dict["lower_surf_end"]],
                                name=name, assign_unique_name=False)
        for name, mea_dict in d["mea"].items():
            geo_col.add_mea(airfoils=[geo_col.container()["airfoils"][k] for k in mea_dict["airfoils"]],
                            name=name, assign_unique_name=False)
        for point in geo_col.container()["points"].values():
            if point.relative_airfoil_name is None:
                continue
            airfoil = geo_col.container()["airfoils"][point.relative_airfoil_name]
            point.relative_airfoil = airfoil
            if point in airfoil.relative_points:
                continue
            airfoil.relative_points.append(point)
        return geo_col

    def switch_units(self, unit_type: str, old_unit: str, new_unit: str):

        if old_unit == new_unit:
            return

        if unit_type == "length":
            self.units.set_current_length_unit(new_unit)
            self.units.set_current_area_unit(new_unit + "2")
        elif unit_type == "angle":
            self.units.set_current_angle_unit(new_unit)

        def switch_unit_for_param(p: Param):
            if isinstance(p, LengthParam) and unit_type == "length":
                p.set_unit(new_unit, old_unit)

            elif isinstance(p, AngleParam) and unit_type == "angle":
                p.set_unit(new_unit, old_unit)

        def switch_unit_for_point(p: Point, force: bool = False):
            if unit_type != "length":
                return

            new_x = p.x().set_unit(new_unit, old_unit, modify_value=False)
            new_y = p.y().set_unit(new_unit, old_unit, modify_value=False)

            p.request_move(new_x, new_y, force=force)

        def switch_unit_for_polyline(p: PolyLine):
            if unit_type != "length":
                return

            p.coords = self.units.convert_length_to_base(p.coords, old_unit)
            p.coords = self.units.convert_length_from_base(p.coords, new_unit)
            p.update()

            for pt in p.point_sequence().points():
                switch_unit_for_point(pt, force=True)

        def switch_unit_for_ref_polyline(p: ReferencePolyline):
            if unit_type != "length":
                return

            p.points = self.units.convert_length_to_base(p.points, old_unit)
            p.points = self.units.convert_length_from_base(p.points, new_unit)
            p.update()

        for param in self.container()["params"].values():
            if param.point is not None:
                continue
            switch_unit_for_param(param)

        for desvar in self.container()["desvar"].values():
            if desvar.point is not None:
                continue
            switch_unit_for_param(desvar)

        for point in self.container()["points"].values():
            switch_unit_for_point(point)

        for poly in self.container()["polylines"].values():
            switch_unit_for_polyline(poly)

        for ref_poly in self.container()["reference"].values():
            switch_unit_for_ref_polyline(ref_poly)

        if unit_type == "length" and self.canvas is not None:
            x_data_range, y_data_range = self.canvas.getPointRange()
            self.canvas.plot.getViewBox().setRange(xRange=x_data_range, yRange=y_data_range)
            self.canvas.plot.setLabel(axis="bottom", text=f"x [{self.units.current_length_unit()}]")
            self.canvas.plot.setLabel(axis="left", text=f"y [{self.units.current_length_unit()}]")

    def write_to_iges(self, base_dir: str, file_name: str, translation: typing.List[float] = None,
                      scaling: typing.List[float] = None, rotation: typing.List[float] = None,
                      transformation_order: str = None):
        """
        Writes all the Bézier curves in the geometry collection to an IGES file.

        Parameters
        ----------
        base_dir: str
            Directory where the IGES file will be stored

        file_name: str
            Name of the IGES file (should include the .igs extension)

        translation: typing.List[float]
            How to translate the curves from the X-Z plane (in the form [tx, ty, tz])

        scaling: typing.List[float]
            How to scale the curves from the X-Z plane (in the form [sx, sy, sz])

        rotation: typing.List[float]
            How to rotate the curves from the X-Z plane (in the form [rx, ry, rz])

        transformation_order: str
            Order in which the transformation takes place

        Returns
        -------
        str
            The full file path to the created IGES file
        """

        # Create the full file path
        full_file = os.path.join(base_dir, file_name)

        # Grab the control points for each Bézier curve in the geometry collection
        control_point_list = [curve.point_sequence().as_array() for curve in self.container()["bezier"].values()]

        # Create the transformation object
        translation = [0., 0., 0.] if translation is None else translation
        scaling = [1., 1., 1.] if scaling is None else scaling
        rotation = [0., 0., 0.] if rotation is None else rotation
        transformation_order = "rx,ry,rz,s,t" if transformation_order is None else transformation_order
        transform_3d = Transformation3D(tx=[translation[0]], ty=[translation[1]], tz=[translation[2]],
                                        sx=[scaling[0]], sy=[scaling[1]], sz=[scaling[2]],
                                        rx=[rotation[0]], ry=[rotation[1]], rz=[rotation[2]],
                                        rotation_units="deg",
                                        order=transformation_order)

        # Add a third dimension by inserting a vector of zeros to the matrix
        cp_list_3d = []
        for cp in control_point_list:
            cp = np.insert(cp, 1, 0, axis=1)
            cp_list_3d.append(cp)

        # Transform the points
        transformed_cp_list = [transform_3d.transform(P) for P in cp_list_3d]

        # Create the list of IGES Bezier objects
        bez_IGES_list = [BezierIGES(P) for P in transformed_cp_list]

        # Generate the IGES file
        iges_generator = IGESGenerator(bez_IGES_list)
        iges_generator.generate(full_file)

        return full_file

    def verify_constraints(self):
        for geo_con in self.container()["geocon"].values():
            assert geo_con.verify()

        return True

    def verify_point_movement(self):
        random.seed(1)
        for point in self.container()["points"].values():
            # Record the starting xy position of the point
            old_xy = [point.x().value(), point.y().value()]
            random_xy = [random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)]
            point.request_move(random_xy[0], random_xy[1])

            self.verify_constraints()

            # Return the point to its original position
            point.request_move(old_xy[0], old_xy[1])

        return True

    def verify_desvar(self):
        for desvar in self.container()["desvar"].values():

            # Record the starting value of the design variable
            old_value = desvar.value()

            # To avoid issues with setting a length of 0, set the value close to 0 if the lower bound is 0
            if isinstance(desvar, LengthParam) and np.isclose(desvar.lower(), 0.0):
                desvar.set_value(0.0001)
            else:
                desvar.set_value(desvar.lower())

            self.verify_constraints()

            desvar.set_value(desvar.upper())

            self.verify_constraints()

            # Set the design variable to its original value
            desvar.set_value(old_value)

        return True

    def verify_params(self):
        random.seed(1)
        for param in self.container()["params"].values():
            # Record the starting value of the param
            old_value = param.value()

            if isinstance(param, LengthParam):
                param.set_value(random.uniform(0.0001, 10.0))
            elif isinstance(param, AngleParam):
                param.set_value(random.uniform(0.0, 2 * np.pi))
            else:
                param.set_value(random.uniform(-10.0, 10.0))

            self.verify_constraints()

            param.set_value(old_value)

        return True

    def verify_all(self):
        self.verify_constraints()
        self.verify_point_movement()
        self.verify_desvar()
        self.verify_params()

        return True
