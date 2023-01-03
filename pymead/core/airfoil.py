import numpy as np
from pymead.core.anchor_point import AnchorPoint
from pymead.core.free_point import FreePoint
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.bezier import Bezier
from pymead.core.trailing_edge_point import TrailingEdgePoint
from pymead.utils.increment_string_index import increment_string_index, decrement_string_index, get_prefix_and_index_from_string
from pymead.utils.transformations import translate_matrix, rotate_matrix, scale_matrix
from pymead.utils.downsampling_schemes import fractal_downsampler2
from pymead.core.transformation import Transformation2D
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from copy import deepcopy
from pymead import DATA_DIR
import os
import subprocess
import pandas as pd


class Airfoil:
    """A class for Bézier-parametrized airfoil creation."""
    def __init__(self,
                 number_coordinates: int = 300,
                 base_airfoil_params: BaseAirfoilParams = None,
                 tag: str = None
                 ):
        """
        Parameters
        __________

        number_coordinates : int
         Represents the number of discrete \\(x\\) - \\(y\\) coordinate pairs in each Bézier curve. Gets \\
         passed to a :class:`pymead.core.bezier.Bezier` object.

        base_airfoil_params: BaseAirfoilParams
            Defines the base set of parameters to be used (chord length, angle of attack, leading edge parameters, \\
            and trailing edge parameters)

        tag: str
            Specifies the name of the Airfoil
        """

        self.nt = number_coordinates
        self.tag = tag
        self.mea = None
        self.base_airfoil_params = base_airfoil_params
        if not self.base_airfoil_params:
            self.base_airfoil_params = BaseAirfoilParams(airfoil_tag=self.tag)

        self.param_dicts = {'Base': {}, 'AnchorPoints': {}, 'FreePoints': {'te_1': {}, 'le': {}}, 'Custom': {}}

        self.c = self.base_airfoil_params.c
        self.alf = self.base_airfoil_params.alf
        self.R_le = self.base_airfoil_params.R_le
        self.L_le = self.base_airfoil_params.L_le
        self.r_le = self.base_airfoil_params.r_le
        self.phi_le = self.base_airfoil_params.phi_le
        self.psi1_le = self.base_airfoil_params.psi1_le
        self.psi2_le = self.base_airfoil_params.psi2_le
        self.L1_te = self.base_airfoil_params.L1_te
        self.L2_te = self.base_airfoil_params.L2_te
        self.theta1_te = self.base_airfoil_params.theta1_te
        self.theta2_te = self.base_airfoil_params.theta2_te
        self.t_te = self.base_airfoil_params.t_te
        self.r_te = self.base_airfoil_params.r_te
        self.phi_te = self.base_airfoil_params.phi_te
        self.dx = self.base_airfoil_params.dx
        self.dy = self.base_airfoil_params.dy

        for bp in ['c', 'alf', 'R_le', 'L_le', 'r_le', 'phi_le', 'psi1_le', 'psi2_le', 'L1_te', 'L2_te', 'theta1_te',
                   'theta2_te', 't_te', 'r_te', 'phi_te', 'dx', 'dy']:
            # setattr(self, bp, getattr(self.base_airfoil_params, bp))
            self.param_dicts['Base'][bp] = getattr(self, bp)

        self.control_point_array = None
        self.n_control_points = None
        self.curve_list = None
        self.curve_list_generated = None

        # # Ensure that all the trailing edge parameters are no longer active if the trailing edge thickness is set to 0.0
        # if self.t_te.value == 0.0:
        #     self.r_te.active = False
        #     self.phi_te.active = False

        self.coords = None
        self.non_transformed_coords = None
        self.curvature = None
        self.area = None
        self.x_thickness = None
        self.thickness = None
        self.max_thickness = None

        self.curvature_combs_active = False
        self.curvature_scale_factor = None
        self.normalized_curvature_scale_factor = None
        self.plt_normals = None
        self.plt_comb_curves = None

        self.Cl = None
        self.Cp = None

        self.needs_update = True

        self.transformed_anchor_points = None
        self.anchor_point_order = ['te_1', 'le', 'te_2']
        self.free_points = {'te_1': {}, 'le': {}}
        self.free_point_order = {'te_1': [], 'le': []}
        self.anchor_point_array = np.array([])

        self.N = {
            'te_1': 4,
            'le': 4
        }

        self.anchor_points = self.base_airfoil_params.generate_main_anchor_points()
        self.control_points = []

        self.airfoil_graph = None

        self.update()

    def __getstate__(self):
        # Reimplemented to ensure MEA picklability
        # Do not need to reimplement __setstate__ for unpickling because the __setstate__ reimplementation in
        # pymead.core.mea.MEA re-adds the airfoil_graph (and thus pg_curve_handle) objects to each airfoil
        state = self.__dict__.copy()
        for curve in state['curve_list']:
            if hasattr(curve, 'pg_curve_handle'):
                # curve.pg_curve_handle.clear()
                curve.pg_curve_handle = None  # Delete unpicklable PlotDataItem object from state
        state['airfoil_graph'] = None  # Delete GraphItem object from state (contains several unpicklable object)
        return state

    def insert_free_point(self, free_point: FreePoint):
        """Method used to insert a :class:`pymead.core.free_point.FreePoint` into an already instantiated
        :class:`pymead.core.airfoil.Airfoil`.

        Parameters
        ==========
        free_point: FreePoint
          FreePoint to add to a Bézier curve
        """
        fp_dict = self.free_points[free_point.anchor_point_tag]
        free_point.x.x = True
        free_point.x.airfoil_tag = self.tag
        free_point.y.y = True
        free_point.y.airfoil_tag = self.tag
        free_point.xp.xp = True
        free_point.xp.airfoil_tag = self.tag
        free_point.yp.yp = True
        free_point.yp.airfoil_tag = self.tag
        free_point.airfoil_transformation = {'dx': self.dx, 'dy': self.dy, 'alf': self.alf, 'c': self.c}

        if free_point.previous_free_point is None:
            free_point.set_tag('FP0')
            fp_idx = 0
        else:
            # Name the free point the previous free point with index incremented by one
            free_point.set_tag(increment_string_index(free_point.previous_free_point))
            fp_idx = get_prefix_and_index_from_string(free_point.tag)[1]

        if free_point.anchor_point_tag in self.free_points.keys():
            temp_dict = {}
            keys_to_pop = []
            for key, fp in fp_dict.items():
                idx = get_prefix_and_index_from_string(key)[1]
                if idx >= fp_idx:
                    new_key = increment_string_index(fp.tag)
                    temp_dict[new_key] = fp
                    fp.set_tag(new_key)
                    keys_to_pop.append(key)
            for k in keys_to_pop[::-1]:
                fp_dict.pop(k)
            fp_dict = {**fp_dict, **temp_dict}
            fp_dict[free_point.tag] = free_point
            # print(f"id of fp_dict is {hex(id(fp_dict))}")
            self.free_points[free_point.anchor_point_tag] = fp_dict
            # print(f"id of fp_dict after assignment is {hex(id(self.free_points[free_point.previous_anchor_point]))}")

        key_pairs = []
        for k in self.param_dicts['FreePoints'][free_point.anchor_point_tag].keys():
            idx = get_prefix_and_index_from_string(k)[1]
            if idx >= fp_idx:
                new_key = increment_string_index(k)
                key_pairs.append((k, new_key))
        for kp in key_pairs:
            self.param_dicts['FreePoints'][free_point.anchor_point_tag][kp[1]] = self.param_dicts['FreePoints'][
                free_point.anchor_point_tag].pop(kp[0])
        self.free_point_order[free_point.anchor_point_tag] = [f"FP{idx}" for idx in range(len(fp_dict))]
        self.N[free_point.anchor_point_tag] += 1
        self.param_dicts['FreePoints'][free_point.anchor_point_tag][free_point.tag] = {
            'x': free_point.x, 'y': free_point.y, 'xp': free_point.xp, 'yp': free_point.yp}

    def delete_free_point(self, free_point_tag: str, anchor_point_tag: str):
        """Deletes a :class:`pymead.core.free_point.FreePoint` from the Airfoil.

        Parameters
        ==========
        free_point_tag: str
          Label identifying the FreePoint from a dictionary

        anchor_point_tag: str
          Label identifying the FreePoint's previous AnchorPoint from a dictionary
        """
        fp_dict = self.free_points[anchor_point_tag]
        fp_idx = next((idx for idx, fp_tag in enumerate(self.free_point_order[anchor_point_tag])
                       if fp_tag == free_point_tag))
        # self.free_point_order[anchor_point_tag].pop(fp_idx)
        self.free_points[anchor_point_tag].pop(free_point_tag)
        self.param_dicts['FreePoints'][anchor_point_tag].pop(free_point_tag)
        temp_dict = {}
        keys_to_pop = []
        for key, fp in fp_dict.items():
            idx = get_prefix_and_index_from_string(key)[1]
            # print(f"free_point.tag = {free_point.tag}")
            if idx >= fp_idx:
                new_key = decrement_string_index(key)
                temp_dict[new_key] = fp
                fp.set_tag(new_key)
                keys_to_pop.append(key)
        for k in keys_to_pop[::-1]:
            fp_dict.pop(k)
        fp_dict = {**fp_dict, **temp_dict}
        self.free_points[anchor_point_tag] = fp_dict
        self.free_point_order[anchor_point_tag] = [f"FP{idx}" for idx in range(len(fp_dict))]
        self.N[anchor_point_tag] += 1
        key_pairs = []
        for k in self.param_dicts['FreePoints'][anchor_point_tag].keys():
            idx = get_prefix_and_index_from_string(k)[1]
            if idx >= fp_idx:
                new_key = decrement_string_index(k)
                key_pairs.append((k, new_key))
        for kp in key_pairs:
            self.param_dicts['FreePoints'][anchor_point_tag][kp[1]] = self.param_dicts['FreePoints'][anchor_point_tag].pop(kp[0])

    def insert_anchor_point(self, ap: AnchorPoint):
        """Method used to insert a :class:`pymead.core.anchor_point.AnchorPoint` into an already instantiated
        :class:`pymead.core.airfoil.Airfoil`.

        Parameters
        ==========
        ap: AnchorPoint
          AnchorPoint to add to a Bézier curve
        """
        ap.x.x = True
        ap.x.airfoil_tag = self.tag
        ap.y.y = True
        ap.y.airfoil_tag = self.tag
        ap.xp.xp = True
        ap.xp.airfoil_tag = self.tag
        ap.yp.yp = True
        ap.yp.airfoil_tag = self.tag
        order_idx = next((idx for idx, anchor_point in enumerate(self.anchor_point_order)
                          if anchor_point == ap.previous_anchor_point))
        self.anchor_points[order_idx + 1].previous_anchor_point = ap.tag
        self.anchor_point_order.insert(order_idx + 1, ap.tag)
        self.anchor_points.insert(order_idx + 1, ap)
        if self.anchor_point_order[order_idx + 1] == 'te_2':
            self.N[ap.tag] = 4
        else:
            self.N[ap.tag] = 5
        if ap.previous_anchor_point == 'le':
            self.N['le'] = 5
        ap.airfoil_transformation = {'c': self.c, 'alf': self.alf, 'dx': self.dx, 'dy': self.dy}
        ap_param_list = ['x', 'y', 'xp', 'yp', 'L', 'R', 'r', 'phi', 'psi1', 'psi2']
        self.param_dicts['AnchorPoints'][ap.tag] = {p: getattr(ap, p) for p in ap_param_list}
        self.free_points[ap.tag] = {}
        self.param_dicts['FreePoints'][ap.tag] = {}

    def update(self, skip_fp_ap_regen: bool = False, generate_curves: bool = True, update_ap_fp: bool = True):
        """Used to update the state of the airfoil, including the Bézier curves, after a change in any parameter

        Parameters
        ==========
        generate_curves: bool
          Determines whether the curves should be re-generated during the update
        """
        # print('Called update!')
        # Translate back to origin if not already at origin
        if self.control_points is not None and self.control_points != []:
            self.translate(-self.dx.value, -self.dy.value)

        # Rotate to zero degree angle of attack
        if self.control_points is not None and self.control_points != []:
            self.rotate(self.alf.value)

        # Scale so that the chord length is equal to 1.0
        if self.control_points is not None and self.control_points != []:
            self.scale(1 / self.c.value)

        if not skip_fp_ap_regen:
            # Generate anchor point branches
            for ap in self.anchor_points:
                if isinstance(ap, AnchorPoint):
                    ap.set_minus_plus_bezier_curve_orders(self.N[ap.previous_anchor_point], self.N[ap.tag])
                    ap.generate_anchor_point_branch(self.anchor_point_order)
                elif isinstance(ap, TrailingEdgePoint):
                    ap.generate_anchor_point_branch()  # trailing edge anchor points do not need to know the ap order

            # Get the control points from all the anchor points
            self.control_points = []
            for ap_tag in self.anchor_point_order:
                self.control_points.extend(next((ap.ctrlpt_branch_list for ap in self.anchor_points if ap.tag == ap_tag)))

            for key, fp_dict in self.free_points.items():
                if len(fp_dict) > 0:
                    if key == 'te_1':
                        insertion_index = 2
                    else:
                        insertion_index = next((idx for idx, cp in enumerate(self.control_points)
                                                if cp.cp_type == 'g2_plus' and cp.anchor_point_tag == key)) + 1
                    self.control_points[insertion_index:insertion_index] = [fp_dict[k].ctrlpt for k in self.free_point_order[key]]

        # Scale airfoil by chord length
        self.scale(self.c.value)

        # Rotate by airfoil angle of attack (alf)
        self.rotate(-self.alf.value)

        # Translate by airfoil dx, dy
        self.translate(self.dx.value, self.dy.value)

        # Get the control point array
        self.update_control_point_array()

        if update_ap_fp:
            for ap_key, ap_val in self.free_points.items():
                for fp_key, fp_val in ap_val.items():
                    fp_val.xp.value = fp_val.x.value * self.c.value
                    fp_val.yp.value = fp_val.y.value * self.c.value
                    rot_mat = np.array([[np.cos(-self.alf.value), -np.sin(-self.alf.value)],
                                        [np.sin(-self.alf.value), np.cos(-self.alf.value)]])
                    rotated_point = (rot_mat @ np.array([[fp_val.xp.value], [fp_val.yp.value]])).flatten()
                    fp_val.xp.value = rotated_point[0]
                    fp_val.yp.value = rotated_point[1]
                    fp_val.xp.value += self.dx.value
                    fp_val.yp.value += self.dy.value
            for ap in self.anchor_points:
                if ap.tag not in ['te_1', 'le', 'te_2']:
                    ap.xp.value = ap.x.value * self.c.value
                    ap.yp.value = ap.y.value * self.c.value
                    rot_mat = np.array([[np.cos(-self.alf.value), -np.sin(-self.alf.value)],
                                        [np.sin(-self.alf.value), np.cos(-self.alf.value)]])
                    rotated_point = (rot_mat @ np.array([[ap.xp.value], [ap.yp.value]])).flatten()
                    ap.xp.value = rotated_point[0]
                    ap.yp.value = rotated_point[1]
                    ap.xp.value += self.dx.value
                    ap.yp.value += self.dy.value

        if generate_curves:
            self.generate_curves()

    def generate_curves(self):
        """Generates the Bézier curves from the control point array"""
        # Make Bézier curves from the control point array
        self.curve_list_generated = True
        previous_number_of_curves = 0
        if self.curve_list is None or len(self.curve_list) != len(self.anchor_point_order) - 1:
            self.curve_list = []
            self.curve_list_generated = False
        else:
            previous_number_of_curves = len(self.curve_list)

        P_list = self.get_list_of_control_point_arrays()
        for idx, P in enumerate(P_list):
            if self.curve_list_generated and previous_number_of_curves == len(self.anchor_point_order) - 1:
                self.curve_list[idx].update(P, 150)
            else:
                self.curve_list.append(Bezier(P, 150))
        self.curve_list_generated = True

        self.n_control_points = len(self.control_points)

    def get_list_of_control_point_arrays(self):
        """Converts the control point array for the Airfoil into a separate control point array for each Bézier curve

        Returns
        =======
        list
          A list of control point arrays for each Bézier curve
        """
        P_list = []
        cp_end_idx, cp_start_idx = 0, 1
        for idx, ap_tag in enumerate(self.anchor_point_order[:-1]):
            if idx == 0:
                cp_end_idx += self.N[ap_tag] + 1
            else:
                cp_end_idx += self.N[ap_tag]
            P = self.control_point_array[cp_start_idx - 1:cp_end_idx]
            P_list.append(P)
            cp_start_idx = deepcopy(cp_end_idx)
        return P_list

    def update_control_point_array(self):
        r"""Updates the control point array from the list of :class:`pymead.core.control_point.ControlPoint` objects

        Returns
        =======
        np.ndarray
          2D array of control points (each row is a different control point, and the two columns are :math:`x` and
          :math:`y`)
        """
        self.control_point_array = np.array([[cp.xp, cp.yp] for cp in self.control_points])
        return self.control_point_array

    def translate(self, dx: float, dy: float):
        r"""Translates all the control points and anchor points by :math:`\Delta x` and :math:`\Delta y`.

        Parameters
        ==========
        dx: float
          :math:`x`-direction translation magnitude

        dy: float
          :math:`y`-direction translation magnitude
        """
        for cp in self.control_points:
            if cp.tag == 'le':
                cp.xp = self.dx.value
                cp.yp = self.dy.value
            else:
                cp.xp += dx
                cp.yp += dy

    def rotate(self, angle: float):
        """Rotates all the control points and anchor points by a specified angle. Used to implement the angle of attack.

        Parameters
        ==========
        angle: float
          Angle (in radians) by which to rotate the airfoil.
        """
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        for cp in self.control_points:
            rotated_point = (rot_mat @ np.array([[cp.xp], [cp.yp]])).flatten()
            cp.xp = rotated_point[0]
            cp.yp = rotated_point[1]

    def scale(self, scale_value):
        """
        Scales the airfoil about the origin.

        Parameters
        ==========
        scale_value: float
          A value by which to scale the airfoil uniformly in both the :math:`x`- and :math:`y`-directions
        """
        for cp in self.control_points:
            cp.xp *= scale_value
            cp.yp *= scale_value

    def compute_area(self):
        """Computes the area of the airfoil as the area of a many-sided polygon enclosed by the airfoil coordinates
        using the `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

        Returns
        =======
        float
          The area of the airfoil
        """
        if self.needs_update:
            self.update()
        points_shapely = list(map(tuple, self.coords))
        polygon = Polygon(points_shapely)
        area = polygon.area
        self.area = area
        return area

    def check_self_intersection(self):
        """Determines whether the airfoil intersects itself using the `is_simple()` function of the
        `shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ library.

        Returns
        =======
        bool
          Describes whether the airfoil intersects itself
        """
        if self.needs_update:
            # print("Calling update in self intersection!")
            self.update()
        self.get_coords(body_fixed_csys=True)
        points_shapely = list(map(tuple, self.coords))
        line_string = LineString(points_shapely)
        is_simple = line_string.is_simple
        return not is_simple

    def compute_thickness(self, n_lines: int = 201, return_max_thickness_loc: bool = False):
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
        self.get_coords(body_fixed_csys=True)
        points_shapely = list(map(tuple, self.coords))
        airfoil_line_string = LineString(points_shapely)
        x_thickness = np.linspace(0.0, 1.0, n_lines)
        thickness = []
        for idx in range(n_lines):
            line_string = LineString([(x_thickness[idx], -1), (x_thickness[idx], 1)])
            x_inters = line_string.intersection(airfoil_line_string)
            if x_inters.is_empty:
                thickness.append(0.0)
            else:
                thickness.append(x_inters.convex_hull.length)
        self.x_thickness = x_thickness
        self.thickness = thickness
        self.max_thickness = max(thickness)
        if return_max_thickness_loc:
            x_c_loc_idx = np.argmax(thickness)
            x_c_loc = self.x_thickness[x_c_loc_idx]
            return {
                'x/c': self.x_thickness,
                't/c': self.thickness,
                't/c_max': self.max_thickness,
                't/c_max_x/c_loc': x_c_loc
            }
        else:
            return self.x_thickness, self.thickness, self.max_thickness

    def compute_thickness_at_points(self, x_over_c: float or list or np.ndarray, start_y_over_c=-1.0, end_y_over_c=1.0):
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
        # If x_over_c is not iterable (i.e., just a float), convert to list
        if not hasattr(x_over_c, '__iter__'):
            x_over_c = [x_over_c]

        self.get_coords(body_fixed_csys=True)  # Get the airfoil coordinates
        points_shapely = list(map(tuple, self.coords))  # Convert the coordinates to Shapely input format
        airfoil_line_string = LineString(points_shapely)  # Create a LineString from the points
        thickness = np.array([])
        for pt in x_over_c:
            line_string = LineString([(pt, start_y_over_c), (pt, end_y_over_c)])
            x_inters = line_string.intersection(airfoil_line_string)
            if x_inters.is_empty:  # If no intersection between line and airfoil LineString,
                thickness = np.append(thickness, 0.0)
            else:
                thickness = np.append(thickness, x_inters.convex_hull.length)
        return thickness  # Return an array of t/c values corresponding to the x/c locations

    def contains_point(self, point: np.ndarray or list):
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
        if isinstance(point, list):
            point = np.array(point)
        self.get_coords(body_fixed_csys=False)
        points_shapely = list(map(tuple, self.coords))
        polygon = Polygon(points_shapely)
        return polygon.contains(Point(point[0], point[1]))

    def contains_line_string(self, points: np.ndarray or list) -> bool:
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
        if isinstance(points, list):
            points = np.array(points)
        self.get_coords(body_fixed_csys=False)
        points_shapely = list(map(tuple, self.coords))
        polygon = Polygon(points_shapely)
        line_string = LineString(list(map(tuple, points)))
        return polygon.contains(line_string)

    def within_line_string_until_point(self, points: np.ndarray or list, cutoff_point,
                                       **transformation_kwargs) -> bool:
        """Whether the airfoil is contained inside a connected string of points until a cutoff point

        Parameters
        ==========
        points: np.ndarray or list
          Should be a 2-D array or list of the form :code:`[[<x_val_1>, <y_val_1>], [<x_val_2>, <y_val_2>], ...]`

        cutoff_point: float
          The :math:`x`-location to set the end of the constraint

        **transformation_kwargs
          Keyword arguments to be fed into a transformation function to transform the airfoil prior to the line string
          containment check

        Returns
        =======
        bool
          Whether the airfoil is contained inside the connected string of points before the cutoff point
        """
        if isinstance(points, list):
            points = np.array(points)
        points_shapely = list(map(tuple, points))
        exterior_polygon = Polygon(points_shapely)

        self.get_coords(body_fixed_csys=True)
        airfoil_points = self.coords[self.coords[:, 0] < cutoff_point, :]
        transform2d = Transformation2D(**transformation_kwargs)
        airfoil_points = transform2d.transform(airfoil_points)
        line_string = LineString(list(map(tuple, airfoil_points)))

        return exterior_polygon.contains(line_string)

    def plot_airfoil(self, axs: plt.axes, **plot_kwargs):
        """Plots each of the airfoil's Bézier curves on a specified matplotlib axis

        Parameters
        ==========
        axs: plt.axes
          A :code:`matplotlib.axes.Axes` object on which to plot each of the airfoil's Bézier curves

        **plot_kwargs
          Arguments to feed to the `matplotlib` "plot" function

        Returns
        =======
        list
          A list of the `matplotlib` plot handles
        """
        plt_curves = []
        for curve in self.curve_list:
            plt_curve = curve.plot_curve(axs, **plot_kwargs)
            plt_curves.append(plt_curve)
        return plt_curves

    def plot_control_points(self, axs: plt.axes, **plot_kwargs):
        """Plots the airfoil's control point skeleton on a specified `matplotlib` axis

        Parameters
        ==========
        axs: plt.axes
          A :code:`matplotlib.axes.Axes` object on which to plot each of the airfoil's control point skeleton

        **plot_kwargs
          Arguments to feed to the `matplotlib` "plot" function
        """
        axs.plot(self.control_point_array[:, 0], self.control_point_array[:, 1], **plot_kwargs)

    def init_airfoil_curve_pg(self, v, pen):
        """Initializes the `pyqtgraph.PlotDataItem` for each of the airfoil's Bézier curves

        Parameters
        ==========
        v
          The `pyqtgraph` axis on which to draw the airfoil

        pen: QPen
          The pen to use to draw the airfoil curves
        """
        for curve in self.curve_list:
            curve.init_curve_pg(v, pen)

    def set_airfoil_pen(self, pen):
        """Sets the QPen for each curve in the airfoil object
        """
        for curve in self.curve_list:
            if curve.pg_curve_handle:
                curve.pg_curve_handle.setPen(pen)

    def update_airfoil_curve(self):
        """Updates each airfoil `matplotlib` axis curve handle"""
        for curve in self.curve_list:
            curve.update_curve()

    def update_airfoil_curve_pg(self):
        """Updates each airfoil `pyqtgraph` axis curve handle"""
        for curve in self.curve_list:
            curve.update_curve_pg()

    def get_coords(self, body_fixed_csys: bool = False):
        """Gets the set of discrete airfoil coordinates for the airfoil

        Parameters
        ==========
        body_fixed_csys: bool
          Whether to internally transform the airfoil such that :math:`(0,0)` is located at the leading edge and
          :math:`(1,0)` is located at the trailing edge prior to the coordinate output

        Returns
        =======
        np.ndarray
          A 2-D array of the airfoil coordinates
        """
        x = np.array([])
        y = np.array([])
        self.coords = []
        for idx, curve in enumerate(self.curve_list):
            if idx == 0:
                x = curve.x
                y = curve.y
            else:
                x = np.append(x, curve.x[1:])
                y = np.append(y, curve.y[1:])
        self.coords = np.column_stack((x, y))
        if body_fixed_csys:
            self.coords = translate_matrix(self.coords, -self.dx.value, -self.dy.value)
            self.coords = rotate_matrix(self.coords, self.alf.value)
            self.coords = scale_matrix(self.coords, 1 / self.c.value)
        return self.coords

    def write_coords_to_file(self, f: str, read_write_mode: str, body_fixed_csys: bool = False,
                             scale_factor: float = None, downsample: bool = False, ratio_thresh=None,
                             abs_thresh=None) -> int:
        """Writes the coordinates to a file.

        Parameters
        ==========
        f: str
          The file in which to write the coordinates

        read_write_mode: str
          Use 'w' to write to a new file, or 'a' to append to an existing file

        body_fixed_csys: bool
          Whether to internally transform the airfoil such that :math:`(0,0)` is located at the leading edge and
          :math:`(1,0)` is located at the trailing edge prior to the coordinate output. Default: `False`

        scale_factor: float
          A value by which to internally scale the airfoil uniformly in the :math:`x`- and :math:`y`-directions
          prior to writing the coordinates. Default: `None`

        downsample: bool
          Whether to downsample the airfoil coordinates before writing to the file. Default: `False`

        ratio_thresh: float
          The threshold ratio used by the downsampler (`1.001` by default, ignored if `downsample=False`)

        abs_thresh: float
          The absolute threshold used by the downsampler (`0.1` by default, ignored if `downsample=False`)

        Returns
        =======
        int
          The number of airfoil coordinates
        """
        self.get_coords(body_fixed_csys)
        if downsample:
            ds = fractal_downsampler2(self.coords, ratio_thresh=ratio_thresh, abs_thresh=abs_thresh)
            n_data_pts = len(ds)
            with open(f, read_write_mode) as coord_file:
                for row in ds:
                    coord_file.write(f"{row[0]} {row[1]}\n")
        else:
            n_data_pts = len(self.coords)
            if scale_factor is not None:
                with open(f, read_write_mode) as coord_file:
                    for row in self.coords * scale_factor:
                        coord_file.write(f"{row[0]} {row[1]}\n")
            else:
                with open(f, read_write_mode) as coord_file:
                    for row in self.coords:
                        coord_file.write(f"{row[0]} {row[1]}\n")
        return n_data_pts

    def read_Cl_from_file(self, f: str):
        with open(f, 'r') as Cl_file:
            line = Cl_file.readline()
        str_Cl = ''
        for ch in line:
            if ch.isdigit() or ch in ['.', 'e', 'E', '-']:
                str_Cl += ch
        self.Cl = float(str_Cl)
        return self.Cl

    def read_Cp_from_file(self, f: str):
        df = pd.read_csv(f, names=['x/c', 'Cp'])
        self.Cp = df.to_numpy()
        return self.Cp

    def calculate_Cl_Cp(self, alpha, tool: str = 'panel_fort'):
        """
        Calculates the lift coefficient and surface pressure coefficient distribution for the Airfoil.
        Note that the angle of attack (alpha) should be entered in degrees.
        """
        tool_list = ['panel_fort', 'XFOIL', 'MSES']
        if tool not in tool_list:
            raise ValueError(f"\'tool\' must be one of {tool_list}")
        coord_file_name = 'airfoil_coords_ClCp_calc.dat'
        f = os.path.join(DATA_DIR, coord_file_name)
        n_data_pts = self.write_coords_to_file(f, 'w')
        if tool == 'panel_fort':
            subprocess.run((["panel_fort", DATA_DIR, coord_file_name, str(n_data_pts - 1), str(alpha)]),
                           stdout=subprocess.DEVNULL)
            self.read_Cl_from_file(os.path.join(DATA_DIR, 'LIFT.dat'))
            self.read_Cp_from_file(os.path.join(DATA_DIR, 'CPLV.DAT'))
        elif tool == 'XFOIL':
            subprocess.run((['xfoil', os.path.join(DATA_DIR, coord_file_name)]))

    def plot_control_point_skeleton(self, axs: plt.axes, **plot_kwargs):
        return axs.plot(self.control_point_array[:, 0], self.control_point_array[:, 1], **plot_kwargs)

    def set_curvature_scale_factor(self, scale_factor=None):
        if scale_factor is None and self.curvature_scale_factor is None:
            raise ValueError('Curvature scale factor not initialized for airfoil!')
        if scale_factor is not None:
            self.curvature_scale_factor = scale_factor  # otherwise just use the scale factor that is already set
        self.normalized_curvature_scale_factor = self.curvature_scale_factor / np.max([np.max(abs(curve.k)) for curve in self.curve_list])

    def plot_curvature_comb_normals(self, axs: plt.axes, scale_factor, **plot_kwargs):
        self.set_curvature_scale_factor(scale_factor)
        self.plt_normals = []
        # print(f"Setting plt_normals...")
        for curve in self.curve_list:
            plt_normal = curve.plot_curvature_comb_normals(axs, self.normalized_curvature_scale_factor, **plot_kwargs)
            self.plt_normals.append(plt_normal)
        return self.plt_normals

    def update_curvature_comb_normals(self):
        for curve in self.curve_list:
            # print(f"scale_factor = {curve.scale_factor}")
            curve.update_curvature_comb_normals()

    def plot_curvature_comb_curve(self, axs: plt.axes, scale_factor, **plot_kwargs):
        self.set_curvature_scale_factor(scale_factor)
        self.plt_comb_curves = []
        for curve in self.curve_list:
            plt_comb_curve = curve.plot_curvature_comb_curve(axs, self.normalized_curvature_scale_factor, **plot_kwargs)
            self.plt_comb_curves.append(plt_comb_curve)
        return self.plt_comb_curves

    def update_curvature_comb_curve(self):
        for curve in self.curve_list:
            curve.update_curvature_comb_curve()


if __name__ == '__main__':
    airfoil = Airfoil()
    airfoil.calculate_Cl_Cp(5.0)
