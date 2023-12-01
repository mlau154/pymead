import numpy as np
from shapely.geometry import Polygon, LineString

from pymead.core.point import Point
from pymead.core.dual_rep import DualRep


class Airfoil(DualRep):
    def __init__(self, leading_edge: Point, trailing_edge: Point,
                 upper_surf_end: Point, lower_surf_end: Point, name: str or None = None):

        # Point inputs
        self.leading_edge = leading_edge
        self.trailing_edge = trailing_edge
        self.upper_surf_end = upper_surf_end
        self.lower_surf_end = lower_surf_end

        # References
        self.geo_col = None
        self.tree_item = None

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

    def name(self):
        return self._name

    def set_name(self, name: str):
        # Rename the reference in the geometry collection
        if self.geo_col is not None and self.name() in self.geo_col.container()["airfoils"].keys():
            self.geo_col.container()["airfoils"][name] = self.geo_col.container()["airfoils"][self.name()]
            self.geo_col.container()["airfoils"].pop(self.name())

        self._name = name

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

    @staticmethod
    def convert_coords_to_shapely_format(coords: np.ndarray):
        return list(map(tuple, coords))

    def compute_area(self, airfoil_polygon: Polygon):
        """Computes the area of the airfoil as the area of a many-sided polygon enclosed by the airfoil coordinates
        using the `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

        Returns
        =======
        float
          The area of the airfoil
        """
        #polygon = Polygon(points_shapely)
        return airfoil_polygon.area

    def check_self_intersection(self, airfoil_line_string: LineString):
        """Determines whether the airfoil intersects itself using the `is_simple()` function of the
        `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

        Returns
        =======
        bool
          Describes whether the airfoil intersects itself
        """
        # self.get_coords(body_fixed_csys=True)
        # points_shapely = list(map(tuple, self.coords))
        # line_string = LineString(points_shapely)
        # is_simple = line_string.is_simple
        # return not is_simple
        return not airfoil_line_string.is_simple

    @staticmethod
    def compute_thickness(airfoil_line_string: LineString, n_lines: int = 201):
        r"""Calculates the thickness distribution and maximum thickness of the airfoil.

        Parameters
        ==========
        n_lines: int
          Describes the number of lines evenly spaced along the chordline produced to determine the thickness
          distribution. Default: :code:`201`

        return_max_thickness_loc: bool
          Whether to return the :math:`x/c`-location of the maximum thickness. Return type will be a :code:`dict`
          rather than a :code:`tuple` if this value is selected to be :code:`True`. Default: :code:`False`

        Returns
        =======
        tuple or dict
          The list of \(x\)-values used for the thickness distribution calculation, the thickness distribution, the
          maximum value of the thickness distribution, and, if :code:`return_max_thickness_location=True`,
          the :math:`x/c`-location of the maximum thickness value.
        """
        # self.get_coords(body_fixed_csys=True)
        # points_shapely = list(map(tuple, self.coords))
        # airfoil_line_string = LineString(points_shapely)
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

    def compute_thickness_at_points(self, airfoil_line_string: LineString, x_over_c: np.ndarray,
                                    start_y_over_c=-1.0, end_y_over_c=1.0):
        """Calculates the thickness (t/c) at a set of x-locations (x/c)

        Parameters
        ==========
        x_over_c: float or list or np.ndarray
          The :math:`x/c` locations at which to evaluate the thickness

        start_y_over_c: float
          The :math:`y/c` location to draw the first point in a line whose intersection with the airfoil is checked. May
          need to decrease this value for unusually thick airfoils

        end_y_over_c: float
          The :math:`y/c` location to draw the last point in a line whose intersection with the airfoil is checked. May
          need to increase this value for unusually thick airfoils

        Returns
        =======
        np.ndarray
          An array of thickness (:math:`t/c`) values corresponding to the input :math:`x/c` values
        """
        # self.get_coords(body_fixed_csys=True)  # Get the airfoil coordinates
        # points_shapely = list(map(tuple, self.coords))  # Convert the coordinates to Shapely input format
        # airfoil_line_string = LineString(points_shapely)  # Create a LineString from the points
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

    def compute_camber_at_points(self, airfoil_line_string: LineString, x_over_c: np.ndarray,
                                 start_y_over_c=-1.0, end_y_over_c=1.0):
        """Calculates the thickness (t/c) at a set of x-locations (x/c)

        Parameters
        ==========
        x_over_c: float or list or np.ndarray
          The :math:`x/c` locations at which to evaluate the camber

        start_y_over_c: float
          The :math:`y/c` location to draw the first point in a line whose intersection with the airfoil is checked. May
          need to decrease this value for unusually thick airfoils

        end_y_over_c: float
          The :math:`y/c` location to draw the last point in a line whose intersection with the airfoil is checked. May
          need to increase this value for unusually thick airfoils

        Returns
        =======
        np.ndarray
          An array of thickness (:math:`t/c`) values corresponding to the input :math:`x/c` values
        """
        # self.get_coords(body_fixed_csys=True)  # Get the airfoil coordinates
        # points_shapely = list(map(tuple, self.coords))  # Convert the coordinates to Shapely input format
        # airfoil_line_string = LineString(points_shapely)  # Create a LineString from the points
        camber = np.array([])
        for pt in x_over_c:
            line_string = LineString([(pt, start_y_over_c), (pt, end_y_over_c)])
            x_inters = line_string.intersection(airfoil_line_string)
            if pt == 0.0 or pt == 1.0 or x_inters.is_empty:
                camber = np.append(camber, 0.0)
            else:
                camber = np.append(camber, x_inters.convex_hull.centroid.xy[1])
        return camber  # Return an array of h/c values corresponding to the x/c locations

    def contains_point(self, airfoil_polygon: Polygon, point: np.ndarray):
        """Determines whether a point is contained inside the airfoil

        Parameters
        ==========
        point: np.ndarray or list
          The point to test. Should be either a 1-D :code:`ndarray` of the format :code:`array([<x_val>,<y_val>])` or a
          list of the format :code:`[<x_val>,<y_val>]`

        Returns
        =======
        bool
          Whether the point is contained inside the airfoil
        """
        # if isinstance(point, list):
        #     point = np.array(point)
        # self.get_coords(body_fixed_csys=False)
        # points_shapely = list(map(tuple, self.coords))
        # polygon = Polygon(points_shapely)
        return airfoil_polygon.contains(Point(point[0], point[1]))

    def contains_line_string(self, airfoil_polygon: Polygon, points: np.ndarray or list) -> bool:
        """Whether a connected string of points is contained the airfoil

        Parameters
        ==========
        points: np.ndarray or list
          Should be a 2-D array or list of the form :code:`[[<x_val_1>, <y_val_1>], [<x_val_2>, <y_val_2>], ...]`

        Returns
        =======
        bool
          Whether the line string is contained inside the airfoil
        """
        # if isinstance(points, list):
        #     points = np.array(points)
        # self.get_coords(body_fixed_csys=False)
        # points_shapely = list(map(tuple, self.coords))
        # polygon = Polygon(points_shapely)
        line_string = LineString(list(map(tuple, points)))
        return airfoil_polygon.contains(line_string)


class BranchError(Exception):
    pass


class ClosureError(Exception):
    pass
