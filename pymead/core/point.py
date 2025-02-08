import typing

import networkx
import numpy as np

from pymead.core.param import LengthParam
from pymead.core.pymead_obj import PymeadObj


class Point(PymeadObj):
    """
    The ``Point`` is the lowest-level geometry object in `pymead`. All curves in `pymead` are tied directly to ``Point``
    objects. For example, instances of this class are used to define the endpoints of finite lines and control points of
    Bézier curves.
    """
    def __init__(self, x: float, y: float, name: str or None = None, relative_airfoil_name: str = None,
                 setting_from_geo_col: bool = False):
        super().__init__(sub_container="points")
        self._x = None
        self._y = None
        self.relative_airfoil_name = relative_airfoil_name
        self.relative_airfoil = None
        self.gcs = None
        self.root = False
        self.rotation_handle = False
        self.rotation_param = None
        self.geo_cons = []
        self.dims = []
        self.curves = []
        self.setting_from_geo_col = setting_from_geo_col
        name = "Point-1" if name is None else name
        self.set_name(name)
        self.set_x(x)
        self.set_y(y)

    def x(self):
        """
        Getter for the point's ``x`` parameter.

        Returns
        -------
        LengthParam
            The point's ``x`` parameter
        """
        return self._x

    def y(self):
        """
        Getter for the point's ``y`` parameter.

        Returns
        -------
        LengthParam
            The point's ``y`` parameter
        """
        return self._y

    def set_x(self, x: LengthParam or float):
        self._x = x if isinstance(x, LengthParam) else LengthParam(
            value=x, name=self.name() + ".x", setting_from_geo_col=self.setting_from_geo_col, point=self)
        if self not in self._x.geo_objs:
            self._x.geo_objs.append(self)
        self._x.point = self

    def set_y(self, y: LengthParam or float):
        self._y = y if isinstance(y, LengthParam) else LengthParam(
            value=y, name=self.name() + ".y", setting_from_geo_col=self.setting_from_geo_col, point=self)
        if self not in self._y.geo_objs:
            self._y.geo_objs.append(self)
        self._y.point = self

    def set_name(self, name: str):
        """
        Extends the base ``set_name()`` by also renaming the point's ``x`` and ``y`` parameters.

        Parameters
        ----------
        name: str
            Name for the point
        """
        # Rename the x and y parameters of the Point
        if self.x() is not None:
            self.x().set_name(f"{name}.x")
        if self.y() is not None:
            self.y().set_name(f"{name}.y")

        super().set_name(name)

    def as_array(self):
        """
        Gives a one-dimensional, two-element array representation of the point (:math:`x` and :math:`y` values)

        Returns
        -------
        np.ndarray
            One-dimensional array containing the point's :math:`x` and :math:`y` values
        """
        return np.array([self.x().value(), self.y().value()])

    @classmethod
    def generate_from_array(cls, arr: np.ndarray, name: str or None = None):
        if arr.ndim != 1:
            raise ValueError("Points can only be generated from 1-dimensional arrays")
        return cls(x=arr[0], y=arr[1], name=name)

    def measure_distance(self, other: "Point") -> float:
        """
        Measures the distance from this point to another point.

        Parameters
        ----------
        other: Point
            Other point (the endpoint of the line whose distance is measured)

        Returns
        -------
        float
            The distance between ``self`` and ``other``
        """
        # Note: the quotes around "Point" are necessary here because this type hint is a forward reference.
        # This means that an object of type <Class> is specified as a hint somewhere inside the definition for <Class>
        return np.hypot(other.x().value() - self.x().value(), other.y().value() - self.y().value())

    def measure_angle(self, other: "Point"):
        """
        Measures the angle (in radians) of the line starting at this point and ending at ``other``

        Parameters
        ----------
        other: Point
            Other point (the endpoint of the line whose angle is measured)

        Returns
        -------
        float
            The angle of the line connecting ``self`` and ``other``
        """
        return np.arctan2(other.y().value() - self.y().value(), other.x().value() - self.x().value())

    def is_coincident(self, other: "Point", rtol: float = 1.0e-14) -> bool:
        """
        Determines whether this point is coincident with another point (within a tight tolerance)

        Parameters
        ----------
        other: Point
            Other point to test for coincidence with this point

        rtol: float
            Relative tolerance used for the coincidence test. Default: ``1e-14``

        Returns
        -------
        bool
            Whether the points are coincident
        """
        dist = self.measure_distance(other)
        if np.isclose(dist, 0.0, rtol=rtol):
            return True
        return False

    def _is_symmetry_123_and_no_edges(self) -> list or bool:
        """
        Checks if this point is a member of a symmetry constraint (but not the target of the symmetry constraint) and
        has no attached edges in the constraint graph. Used to determine if the point is allowed to move in
        ``request_move()``.

        Returns
        -------
        list or bool
            The list of symmetry constraints this point is a member of if the above conditions are met,
            ``False`` otherwise
        """
        if self.gcs is None:
            return False
        if len([edge for edge in self.gcs.in_edges(nbunch=self)]) != 0:
            return False
        if len([edge for edge in self.gcs.out_edges(nbunch=self)]) != 0:
            return False
        symmetry_constraints = []
        for geo_con in self.geo_cons:
            if geo_con.__class__.__name__ == "SymmetryConstraint":
                symmetry_constraints.append(geo_con)
        for symmetry_constraint in symmetry_constraints:
            if self is symmetry_constraint.p4:
                return False
        return symmetry_constraints

    def _is_symmetry_target_and_has_edges(self) -> bool:
        if self.gcs is None:
            return False
        if (not [edge for edge in self.gcs.in_edges(nbunch=self)] and
            not [edge for edge in self.gcs.out_edges(nbunch=self)]):
            return False
        symmetry_constraints = []
        for geo_con in self.geo_cons:
            if geo_con.__class__.__name__ == "SymmetryConstraint":
                symmetry_constraints.append(geo_con)
        if not symmetry_constraints:
            return False
        if not any([self is symmetry_constraint.p4 for symmetry_constraint in symmetry_constraints]):
            return False
        return True

    def is_movement_allowed(self) -> bool:
        """
        This method determines if movement is allowed for the point. Movement is allowed in these cases:

        - Where the constraint solver has not been set or there are no geometric constraints attached
        - Where the point is a root or rotation handle of a constraint cluster. If a rotation handle, movement is
          allowed, but the movement gets accepted as a rotation about the root point with a fixed distance to the
          root point
        - Where the point is one of the first three out of the four points in a symmetry constraint, and no edges are
          attached to this point in the constraint graph

        Returns
        -------
        bool
            ``True`` if movement is allowed for this point, ``False`` otherwise
        """
        if any([curve.__class__.__name__ == "PolyLine" for curve in self.curves]):
            return False
        if self.gcs is None:
            return True
        if self.gcs is not None and len(self.geo_cons) == 0:
            return True
        if self.gcs is not None and self.root:
            return True
        if self.gcs is not None and self.rotation_handle:
            # In this case, movement is allowed, but movements get translated to a rotation about the root point with
            # a fixed distance to the root point
            return True
        if self._is_symmetry_123_and_no_edges():
            return True
        return False

    def request_move(self, xp: float, yp: float, force: bool = False):
        """
        Updates the location of the point and updates any curves and canvas items associated with the point movement.

        Parameters
        ----------
        xp: float
            New :math:`x`-value for the point

        yp: float
            New :math:`y`-value for the point

        force: bool
            Force the movement of this point. Overrides ``is_movement_allowed``.

        Warning
        -------
        The ``force`` keyword argument should **never** be called directly from the API, or unexpected behavior
        may result. This argument is used in the backend code for the constraint solver in the symmetry and curvature
        constraints.
        """

        if not self.is_movement_allowed() and not force:
            return

        old_point_vals = {k: v.as_array() for k, v in self.geo_col.container()["points"].items()} \
            if self.geo_col is not None else {}

        if self.root:
            # Bounds checks
            if (self.x().lower() is not None and self.x().upper() is not None and
                    not self.x().lower() <= xp <= self.x().upper()):
                return
            if (self.y().lower() is not None and self.y().upper() is not None and
                    not self.y().lower() <= yp <= self.y().upper()):
                return

            points_to_update = self.gcs.translate_cluster(self, dx=xp - self.x().value(), dy=yp - self.y().value())
            constraints_to_update = []
            for point in networkx.dfs_preorder_nodes(self.gcs, source=self):
                for geo_con in point.geo_cons:
                    if geo_con not in constraints_to_update:
                        constraints_to_update.append(geo_con)

            for geo_con in constraints_to_update:
                if geo_con.canvas_item is not None:
                    geo_con.canvas_item.update()
        elif self.rotation_handle:
            if self.rotation_param is None:
                return
            points_to_update = self.rotation_param.set_value(
                self.geo_col.units.convert_angle_from_base(
                    self.rotation_param.root.measure_angle(Point(xp, yp)),
                    self.geo_col.units.current_angle_unit()
                ),
                from_request_move=True
            )
        else:
            self.x().set_value(xp, direct_user_request=False)
            self.y().set_value(yp, direct_user_request=False)
            points_to_update = [self]
            symmetry_constraints = self._is_symmetry_123_and_no_edges()
            if symmetry_constraints:
                for symmetry_constraint in symmetry_constraints:
                    self.gcs.solve_symmetry_constraint(symmetry_constraint)
                    points_to_update.extend(symmetry_constraint.child_nodes)
                points_to_update = list(set(points_to_update))  # Get only the unique points
                for symmetry_constraint in symmetry_constraints:
                    if symmetry_constraint.canvas_item is not None:
                        symmetry_constraint.canvas_item.update()

        if self._is_symmetry_target_and_has_edges():
            points_solved = []
            for gc in self.geo_cons:
                points_solved.extend(self.gcs.solve(gc))
            self.gcs.update_canvas_items(list(set(points_solved)))

        # Update the GUI object, if there is one
        if self.canvas_item is not None:
            self.canvas_item.updateCanvasItem(self.x().value(), self.y().value())

        if points_to_update is None:
            return

        curves_to_update = []
        for point in points_to_update:
            for curve in point.curves:
                if curve not in curves_to_update:
                    curves_to_update.append(curve)

        airfoils_to_update = []
        for curve in curves_to_update:
            if curve.airfoil is not None and curve.airfoil not in airfoils_to_update:
                airfoils_to_update.append(curve.airfoil)

        # Update airfoil-relative points
        if not self.relative_airfoil:
            for airfoil in airfoils_to_update:
                airfoil.update_relative_points(old_point_vals)

        # Visual updates to geometric objects
        for point in points_to_update:
            if point.canvas_item is not None:
                point.canvas_item.updateCanvasItem(point.x().value(), point.y().value())

        for curve in curves_to_update:
            curve.update()

        for airfoil in airfoils_to_update:
            airfoil.update_coords()
            if airfoil.canvas_item is not None:
                airfoil.canvas_item.generatePicture()

    def __repr__(self):
        return f"Point {self.name()}<x={self.x().value():.6f}, y={self.y().value():.6f}>"

    def get_dict_rep(self):
        return {"x": float(self.x().value()), "y": float(self.y().value()),
                "relative_airfoil_name": self.relative_airfoil_name}

    def __add__(self, other: "Point"):
        return Point(x=self.x().value() + other.x().value(), y=self.y().value() + other.y().value())

    def __sub__(self, other: "Point"):
        return Point(x=self.x().value() - other.x().value(), y=self.y().value() - other.y().value())

    def __mul__(self, other: float):
        if isinstance(other, float):
            return Point(x=self.x().value() * other, y=self.y().value() * other)
        else:
            raise ValueError("Only multiplication between points and scalars is currently supported")

    def __rmul__(self, other: float):
        return self.__mul__(other)


class PointSequence:
    def __init__(self, points: typing.List[Point]):
        self._points = None
        self.set_points(points)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.generate_from_slice(self, idx)
        else:
            return self.points()[idx]

    def __setitem__(self, idx, val):
        self.points()[idx] = val

    def __len__(self):
        return len(self.points())

    @classmethod
    def generate_from_slice(cls, original_point_seq, s):
        return cls(points=original_point_seq.points()[s].copy())

    @classmethod
    def generate_from_array(cls, arr: np.ndarray):
        if arr.shape[1] != 2:
            raise ValueError(f"Array must have two columns, x and y. Found {arr.shape[1]} columns.")
        return cls(points=[Point(x=x, y=y, name=f"PointFromArray-Index{idx}")
                           for idx, (x, y) in enumerate(zip(arr[:, 0], arr[:, 1]))])

    def points(self):
        return self._points

    def point_idx_from_ref(self, point: Point):
        return self.points().index(point)

    def reverse(self):
        self.points().reverse()

    def set_points(self, points: typing.List[Point]):
        self._points = points

    def insert_point(self, idx: int, point: Point):
        self._points.insert(idx, point)

    def append_point(self, point: Point):
        self._points.append(point)

    def remove_point(self, idx: int):
        self._points.pop(idx)

    def as_array(self):
        return np.array([[p.x().value(), p.y().value()] for p in self.points()])

    def extract_subsequence(self, indices: list):
        return PointSequence(points=[self.points()[idx] for idx in indices])
