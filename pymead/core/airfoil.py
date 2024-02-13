from copy import deepcopy

import numpy as np
from shapely.geometry import Polygon, LineString

from pymead.core.point import Point
from pymead.core.pymead_obj import PymeadObj
from pymead.core.transformation import Transformation2D


class Airfoil(PymeadObj):
    def __init__(self, leading_edge: Point, trailing_edge: Point,
                 upper_surf_end: Point, lower_surf_end: Point, name: str or None = None):

        super().__init__(sub_container="airfoils")

        # Point inputs
        self.leading_edge = leading_edge
        self.trailing_edge = trailing_edge
        self.upper_surf_end = upper_surf_end
        self.lower_surf_end = lower_surf_end

        # Name the airfoil
        name = "Airfoil-1" if name is None else name
        self._name = None
        self.set_name(name)

        # Properties to set during closure check
        self.upper_te_curve = None
        self.lower_te_curve = None
        self.curves = []
        self.curves_to_reverse = []

        # Check if the curves in the curve list form a single closed loop
        self.check_closed()

        # Add the airfoil reference to the curves
        for curve in self.curves:
            curve.airfoil = self

        self.coords = self.get_coords_selig_format()

    def check_closed(self):
        # Get the trailing edge upper curve
        if self.trailing_edge is self.upper_surf_end:
            self.upper_te_curve = None
        else:
            for curve in self.trailing_edge.curves:
                if self.upper_surf_end in curve.point_sequence().points():
                    self.upper_te_curve = curve
                    break
            if self.upper_te_curve is None:
                raise ClosureError("Could not identify the upper trailing edge line/curve")

        # Get the trailing edge lower curve
        if self.trailing_edge is self.lower_surf_end:
            self.lower_te_curve = None
        else:
            for curve in self.trailing_edge.curves:
                if self.lower_surf_end in curve.point_sequence().points():
                    self.lower_te_curve = curve
                    break
            if self.lower_te_curve is None:
                raise ClosureError("Could not identify the lower trailing edge line/curve")

        # Loop through the rest of the curves
        current_curve = None
        if self.upper_te_curve is None:
            current_curve = self.upper_surf_end.curves[0]
        else:
            for curve in self.upper_surf_end.curves:
                if curve is not self.upper_te_curve:
                    current_curve = curve
                    break

        if current_curve is None:
            raise ClosureError("Curve loop is not closed")

        for point in current_curve.point_sequence().points():
            if len(point.curves) > 2:
                raise BranchError("Found more than two curves associated with a curve endpoint in the loop")

        previous_point = self.upper_surf_end
        closed = False
        while not closed:
            self.curves.append(current_curve)

            idx_of_prev_point = current_curve.point_sequence().points().index(previous_point)
            if idx_of_prev_point == 0:
                idx_of_next_point = -1
            else:
                self.curves_to_reverse.append(current_curve)
                idx_of_next_point = 0

            next_point = current_curve.point_sequence().points()[idx_of_next_point]

            if next_point is self.lower_surf_end:
                closed = True
                break

            for curve in next_point.curves:
                if curve is not current_curve:
                    current_curve = curve
                    break

            previous_point = next_point
            pass

        if not closed:
            raise ClosureError("Curve loop not closed")

    def get_coords_selig_format(self) -> np.ndarray:
        coords = None
        for curve in self.curves:
            p_curve_data = curve.evaluate()
            arr = p_curve_data.xy
            if curve in self.curves_to_reverse:
                arr = np.flipud(arr)
            if coords is None:
                coords = arr
            else:
                coords = np.row_stack((coords, arr[1:, :]))
        return coords

    def update_coords(self):
        self.coords = self.get_coords_selig_format()
        if self.canvas_item is not None:
            self.canvas_item.data = self.coords

    def save_coords_selig_format(self, file_name: str) -> None:
        """
        Saves this airfoil's coords in Selig file format: :math:`x`- and :math:`y`-columns in counter-clockwise order,
        starting at the upper surface trailing edge and ending at the lower surface trailing edge. File saves as a
        text file (usually a :code:`.txt` or :code:`.dat` file extension).

        Parameters
        ==========
        file_name: str
            Full path to the coordinate file save location
        """
        coords = self.get_coords_selig_format()
        np.savetxt(file_name, coords)

    def measure_chord(self):
        """
        Measures the chord length

        Returns
        -------
        float
            The airfoil's current chord length

        """
        return self.leading_edge.measure_distance(self.trailing_edge)

    def measure_alpha(self):
        """
        Measures the angle of attack in radians

        Returns
        -------
        float
            The airfoil's current angle of attack

        """
        return -self.leading_edge.measure_angle(self.trailing_edge)

    def get_chord_relative_coords(self, coords: np.ndarray = None) -> np.ndarray:
        """
        Gets the chord-relative values of the airfoil coordinates. The airfoil is transformed such that the leading
        edge is at :math:`(0,0)` and the trailing edge is at :math:`(1,0)`.

        Parameters
        ----------
        coords: np.ndarray or None
            Optional Selig format airfoil coordinates (only specified if computational speed is important).
            If the coordinates are not specified, they are calculated.

        Returns
        -------
        np.ndarray
            An :math:`N \times 2` array of transformed airfoil coordinates
        """
        # Get the chord length and angle of attack
        chord_length = self.measure_chord()
        angle_of_attack = self.measure_alpha()

        # Get the transformation object
        transformation = Transformation2D(
            tx=[-self.leading_edge.x().value()],
            ty=[-self.leading_edge.y().value()],
            r=[angle_of_attack],
            sx=[1 / chord_length],
            sy=[1 / chord_length],
            rotation_units="rad",
            order="t,s,r"
        )

        coords = self.get_coords_selig_format() if coords is None else coords
        return transformation.transform(coords)

    @staticmethod
    def convert_coords_to_shapely_format(coords: np.ndarray):
        return list(map(tuple, coords))

    @staticmethod
    def create_line_string(coords_shapely_format: list):
        return LineString(coords_shapely_format)

    @staticmethod
    def create_shapely_polygon(line_string: LineString):
        return Polygon(line_string)

    def compute_area(self, airfoil_polygon: Polygon = None):
        """Computes the area of the airfoil as the area of a many-sided polygon enclosed by the airfoil coordinates
        using the `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

        Parameters
        ----------
        airfoil_polygon: Polygon or None
            Optional specification of the airfoil's ``shapely.geometry.Polygon`` for speed. If ``None``, it
            will be computed.

        Returns
        -------
        float
            The area of the airfoil
        """
        airfoil_polygon = self.create_shapely_polygon(
            self.create_line_string(
                self.convert_coords_to_shapely_format(
                    self.get_chord_relative_coords(
                        self.get_coords_selig_format()
                    )
                )
            )
        ) if airfoil_polygon is None else airfoil_polygon
        return airfoil_polygon.area

    def check_self_intersection(self, airfoil_line_string: LineString = None):
        """Determines whether the airfoil intersects itself using the `is_simple()` function of the
        `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

        Parameters
        ----------
        airfoil_line_string: LineString or None
            Optional specification of the airfoil's ``shapely.geometry.LineString`` for speed. If ``None``, it
            will be computed.

        Returns
        -------
        bool
            Describes whether the airfoil intersects itself
        """
        airfoil_line_string = self.create_line_string(
            self.convert_coords_to_shapely_format(
                self.get_chord_relative_coords(
                    self.get_coords_selig_format()
                )
            )
        ) if airfoil_line_string is None else airfoil_line_string
        return not airfoil_line_string.is_simple

    def compute_min_radius(self) -> float:
        """
        Computes the minimum radius of curvature for the airfoil.

        Returns
        -------
        float:
            The minimum radius of curvature

        """
        return min([np.abs(curve.evaluate().R).min() for curve in self.curves])

    def compute_thickness(self, airfoil_line_string: LineString = None, n_lines: int = 201):
        r"""Calculates the thickness distribution and maximum thickness of the airfoil.

        Parameters
        ----------
        airfoil_line_string: LineString or None
            Optional specification of the airfoil's ``shapely.geometry.LineString`` for speed. If ``None``, it
            will be computed.

        n_lines: int
          Describes the number of lines evenly spaced along the chordline produced to determine the thickness
          distribution. Default: 201

        Returns
        =======
        dict
          The list of :math:`x`-values used for the thickness distribution calculation, the thickness distribution, the
          maximum value of the thickness distribution, and, if ``return_max_thickness_location=True``,
          the :math:`x/c`-location of the maximum thickness value.
        """
        airfoil_line_string = self.create_line_string(
            self.convert_coords_to_shapely_format(
                self.get_chord_relative_coords(
                    self.get_coords_selig_format()
                )
            )
        ) if airfoil_line_string is None else airfoil_line_string
        x_thickness = np.linspace(0.0, 1.0, n_lines)
        thickness = []
        for idx in range(n_lines):
            line_string = LineString([(x_thickness[idx], -1), (x_thickness[idx], 1)])
            x_inters = line_string.intersection(airfoil_line_string)
            if x_inters.is_empty:
                thickness.append(0.0)
            else:
                thickness.append(x_inters.convex_hull.length)
        x_thickness = x_thickness
        thickness = thickness
        max_thickness = max(thickness)
        x_c_loc_idx = np.argmax(thickness)
        x_c_loc = x_thickness[x_c_loc_idx]
        return {
            "x/c": x_thickness,
            "t/c": thickness,
            "t/c_max": max_thickness,
            "t/c_max_x/c_loc": x_c_loc
        }

    def compute_thickness_at_points(self, x_over_c: np.ndarray, airfoil_line_string: LineString = None,
                                    start_y_over_c: float = -1.0, end_y_over_c: float = 1.0):
        """
        Calculates the thickness (t/c) at a set of x-locations (x/c)

        Parameters
        ----------
        x_over_c: float or list or np.ndarray
            The :math:`x/c` locations at which to evaluate the thickness

        airfoil_line_string: LineString or None
            Optional specification of the airfoil's ``shapely.geometry.LineString`` for speed. If ``None``, it
            will be computed.

        start_y_over_c: float
            The :math:`y/c` location to draw the first point in a line whose intersection with the airfoil is checked.
            May need to decrease this value for unusually thick airfoils

        end_y_over_c: float
            The :math:`y/c` location to draw the last point in a line whose intersection with the airfoil is checked.
            May need to increase this value for unusually thick airfoils

        Returns
        -------
        np.ndarray
            An array of thickness (:math:`t/c`) values corresponding to the input :math:`x/c` values
        """
        airfoil_line_string = self.create_line_string(
            self.convert_coords_to_shapely_format(
                self.get_chord_relative_coords(
                    self.get_coords_selig_format()
                )
            )
        ) if airfoil_line_string is None else airfoil_line_string
        thickness = np.array([])
        for pt in x_over_c:
            line_string = LineString([(pt, start_y_over_c), (pt, end_y_over_c)])
            x_inters = line_string.intersection(airfoil_line_string)
            if pt == 1.0:
                thickness = np.append(thickness, self.t_te.value)
            elif x_inters.is_empty:  # If no intersection between line and airfoil LineString,
                thickness = np.append(thickness, 0.0)
            else:
                thickness = np.append(thickness, x_inters.convex_hull.length)
        return thickness  # Return an array of t/c values corresponding to the x/c locations

    def compute_camber_at_points(self, x_over_c: np.ndarray, airfoil_line_string: LineString = None,
                                 start_y_over_c: float = -1.0, end_y_over_c: float = 1.0):
        """Calculates the thickness (t/c) at a set of x-locations (x/c)

        Parameters
        ----------
        x_over_c: float or list or np.ndarray
            The :math:`x/c` locations at which to evaluate the camber

        airfoil_line_string: LineString or None
            Optional specification of the airfoil's ``shapely.geometry.LineString`` for speed. If ``None``, it
            will be computed.

        start_y_over_c: float
            The :math:`y/c` location to draw the first point in a line whose intersection with the airfoil is checked.
            May need to decrease this value for unusually thick airfoils

        end_y_over_c: float
            The :math:`y/c` location to draw the last point in a line whose intersection with the airfoil is checked.
            May need to increase this value for unusually thick airfoils

        Returns
        -------
        np.ndarray
            An array of thickness (:math:`t/c`) values corresponding to the input :math:`x/c` values
        """
        airfoil_line_string = self.create_line_string(
            self.convert_coords_to_shapely_format(
                self.get_chord_relative_coords(
                    self.get_coords_selig_format()
                )
            )
        ) if airfoil_line_string is None else airfoil_line_string
        camber = np.array([])
        for pt in x_over_c:
            line_string = LineString([(pt, start_y_over_c), (pt, end_y_over_c)])
            x_inters = line_string.intersection(airfoil_line_string)
            if pt == 0.0 or pt == 1.0 or x_inters.is_empty:
                camber = np.append(camber, 0.0)
            else:
                camber = np.append(camber, x_inters.convex_hull.centroid.xy[1])
        return camber  # Return an array of h/c values corresponding to the x/c locations

    def contains_point(self, point: np.ndarray, airfoil_polygon: Polygon = None):
        """Determines whether a point is contained inside the airfoil

        Parameters
        ----------
        point: np.ndarray or list
            The point to test. Should be either a 1-D ``ndarray`` of the format ``array([<x_val>,<y_val>])`` or
            a list of the format ``[<x_val>,<y_val>]``

        airfoil_polygon: Polygon or None
            Optional specification of the airfoil's ``shapely.geometry.Polygon`` for speed. If ``None``, it
            will be computed.

        Returns
        -------
        bool
            Whether the point is contained inside the airfoil
        """
        airfoil_polygon = self.create_shapely_polygon(
            self.create_line_string(
                self.convert_coords_to_shapely_format(
                    self.get_chord_relative_coords(
                        self.get_coords_selig_format()
                    )
                )
            )
        ) if airfoil_polygon is None else airfoil_polygon
        return airfoil_polygon.contains(Point(point[0], point[1]))

    def contains_line_string(self, points: np.ndarray or list, airfoil_polygon: Polygon = None) -> bool:
        """
        Whether a connected string of points is contained the airfoil

        Parameters
        ----------
        points: np.ndarray or list
            Should be a 2-D array or list of the form ``[[<x_val_1>, <y_val_1>], [<x_val_2>, <y_val_2>], ...]``

        airfoil_polygon: Polygon or None
            Optional specification of the airfoil's ``shapely.geometry.Polygon`` for speed. If ``None``, it
            will be computed.

        Returns
        -------
        bool
            Whether the line string is contained inside the airfoil
        """
        airfoil_polygon = self.create_shapely_polygon(
            self.create_line_string(
                self.convert_coords_to_shapely_format(
                    self.get_chord_relative_coords(
                        self.get_coords_selig_format()
                    )
                )
            )
        ) if airfoil_polygon is None else airfoil_polygon
        line_string = LineString(list(map(tuple, points)))
        return airfoil_polygon.contains(line_string)

    def downsample(self, max_airfoil_points: int, curvature_exp: float = 2.0):
        r"""
        Downsamples the airfoil coordinates based on a curvature exponent. This method works by evaluating each
        Bézier curve using a set number of points (150) and then calculating
        :math:`\mathbf{R_e} = \mathbf{R}^{1/e_c}`, where :math:`\mathbf{R}` is the radius of curvature vector and
        :math:`e_c` is the curvature exponent (an input to this method). Then, :math:`\mathbf{R_e}` is
        normalized by its maximum value and concatenated to a single array for all curves in a given airfoil.
        Finally, ``max_airfoil_points`` are chosen from this array to create a new set of parameter vectors
        for the airfoil.

        Parameters
        ==========
        max_airfoil_points: int
            Maximum number of points in the airfoil (the actual number in the final airfoil may be slightly less)

        curvature_exp: float
            Curvature exponent used to scale the radius of curvature. Values close to 0 place high emphasis on
            curvature, while values close to :math:`\infty` place low emphasis on curvature (creating nearly
            uniform spacing)


        Returns
        =======
        list[np.ndarray]
            List of parameter vectors (one for each Bézier curve)
        """

        if max_airfoil_points > sum([len(curve.t) for curve in self.curves]):
            for curve in self.curves:
                curve.update(P=curve.P, nt=np.ceil(max_airfoil_points / len(self.curves)).astype(int))

        new_param_vec_list = []
        new_t_concat = np.array([])

        for c_idx, curve in enumerate(self.curves):
            temp_R = deepcopy(curve.R)
            for r_idx, r in enumerate(temp_R):
                if np.isinf(r) and r > 0:
                    temp_R[r_idx] = 10000
                elif np.isinf(r) and r < 0:
                    temp_R[r_idx] = -10000

            exp_R = np.abs(temp_R) ** (1 / curvature_exp)
            new_t = np.zeros(exp_R.shape)
            for i in range(1, new_t.shape[0]):
                new_t[i] = new_t[i - 1] + (exp_R[i] + exp_R[i - 1]) / 2
            new_t = new_t / np.max(new_t)
            new_t_concat = np.concatenate((new_t_concat, new_t))

        indices_to_select = np.linspace(0, new_t_concat.shape[0] - 1,
                                        max_airfoil_points - 2 * len(self.curves)).astype(int)

        t_old = 0.0
        for selection_idx in indices_to_select:
            t = new_t_concat[selection_idx]

            if t == 0.0 and selection_idx == 0:
                new_param_vec_list.append(np.array([0.0]))
            elif t < t_old:
                if t_old != 1.0:
                    new_param_vec_list[-1] = np.append(new_param_vec_list[-1], 1.0)
                if t == 0.0:
                    new_param_vec_list.append(np.array([]))
                else:
                    new_param_vec_list.append(np.array([0.0]))
                new_param_vec_list[-1] = np.append(new_param_vec_list[-1], t)
                t_old = t
            else:
                new_param_vec_list[-1] = np.append(new_param_vec_list[-1], t)
                t_old = t

        return new_param_vec_list

    def get_dict_rep(self):
        return {"leading_edge": self.leading_edge.name(), "trailing_edge": self.trailing_edge.name(),
                "upper_surf_end": self.upper_surf_end.name(), "lower_surf_end": self.lower_surf_end.name()}


class BranchError(Exception):
    pass


class ClosureError(Exception):
    pass
