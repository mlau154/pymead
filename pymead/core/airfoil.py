import numpy as np
from pymead.core.param import Param
from pymead.core.anchor_point import AnchorPoint
from pymead.core.free_point import FreePoint
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.bezier import Bezier
from pymead.core.trailing_edge_point import TrailingEdgePoint
from pymead.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import typing
from shapely.geometry import Polygon, LineString
from copy import deepcopy


class Airfoil:

    def __init__(self,
                 number_coordinates: int = 100,
                 base_airfoil_params: BaseAirfoilParams or SymmetricBaseAirfoilParams = None,
                 override_parameters: list = None,
                 tag: str = None
                 ):
        """
        ### Description:

        `pymead.core.airfoil.Airfoil` is the base class for Bézier-parametrized airfoil creation.

        .. image:: complex_airfoil-3.png

        ### Args:

        `number_coordinates`: an `int` representing the number of discrete \\(x\\) - \\(y\\) coordinate pairs in each
        Bézier curve. Gets passed to the `bezier` function.

        `base_airfoil_params`: an instance of either the `pymead.core.base_airfoil_params.BaseAirfoilParams` class or
        the `pymead.symmetric.symmetric_base_airfoil_params.SymmetricBaseAirfoilParams` class which defines the base
        set of parameters to be used

        `anchor_point_tuple`: a `tuple` of `pymead.core.anchor_point.AnchorPoint` objects. To specify a single
        anchor point, use `(pymead.core.anchor_point.AnchorPoint(),)`. Default: `()`

        `free_point_tuple`: a `tuple` of `pymead.core.free_point.FreePoint` objects. To specify a single free
        point, use `(pymead.core.free_point.FreePoint(),)`. Default: `()`

        ### Returns:

        An instance of the `Airfoil` class
        """

        self.nt = number_coordinates
        self.tag = tag
        self.mea = None
        self.base_airfoil_params = base_airfoil_params
        if not self.base_airfoil_params:
            self.base_airfoil_params = BaseAirfoilParams()
        self.override_parameters = override_parameters

        self.override_parameter_start_idx = 0
        self.override_parameter_end_idx = self.base_airfoil_params.n_overrideable_parameters
        if self.override_parameters is not None:
            self.base_airfoil_params.override(
                self.override_parameters[self.override_parameter_start_idx:self.override_parameter_end_idx])
        self.override_parameter_start_idx += self.base_airfoil_params.n_overrideable_parameters

        self.param_dicts = {'Base': {}, 'AnchorPoints': {}, 'FreePoints': {}, 'Custom': {}}

        # self.c = self.base_airfoil_params.c
        # self.alf = self.base_airfoil_params.alf
        # self.R_le = self.base_airfoil_params.R_le
        # self.L_le = self.base_airfoil_params.L_le
        # self.r_le = self.base_airfoil_params.r_le
        # self.phi_le = self.base_airfoil_params.phi_le
        # self.psi1_le = self.base_airfoil_params.psi1_le
        # self.psi2_le = self.base_airfoil_params.psi2_le
        # self.L1_te = self.base_airfoil_params.L1_te
        # self.L2_te = self.base_airfoil_params.L2_te
        # self.theta1_te = self.base_airfoil_params.theta1_te
        # self.theta2_te = self.base_airfoil_params.theta2_te
        # self.t_te = self.base_airfoil_params.t_te
        # self.r_te = self.base_airfoil_params.r_te
        # self.phi_te = self.base_airfoil_params.phi_te
        # self.dx = self.base_airfoil_params.dx
        # self.dy = self.base_airfoil_params.dy

        for bp in ['c', 'alf', 'R_le', 'L_le', 'r_le', 'phi_le', 'psi1_le', 'psi2_le', 'L1_te', 'L2_te', 'theta1_te',
                   'theta2_te', 't_te', 'r_te', 'phi_te', 'dx', 'dy']:
            setattr(self, bp, getattr(self.base_airfoil_params, bp))
            self.param_dicts['Base'][bp] = getattr(self, bp)

        self.control_point_array = None
        self.n_control_points = None
        self.curve_list = None
        self.curve_list_generated = None

        # Ensure that all the trailing edge parameters are no longer active if the trailing edge thickness is set to 0.0
        if self.t_te.value == 0.0:
            self.r_te.active = False
            self.phi_te.active = False

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

        self.needs_update = True

        if self.override_parameters is not None:
            for anchor_point in self.anchor_point_tuple:
                if self.base_airfoil_params.non_dim_by_chord:
                    anchor_point.length_scale_dimension = self.base_airfoil_params.c.value
                self.override_parameter_end_idx = self.override_parameter_start_idx + \
                                                  anchor_point.n_overrideable_parameters
                anchor_point.override(
                    self.override_parameters[self.override_parameter_start_idx:self.override_parameter_end_idx])
                self.override_parameter_start_idx += anchor_point.n_overrideable_parameters

            for free_point in self.free_point_tuple:
                if self.base_airfoil_params.non_dim_by_chord:
                    free_point.length_scale_dimension = self.base_airfoil_params.c.value
                self.override_parameter_end_idx = self.override_parameter_start_idx + \
                                                  free_point.n_overrideable_parameters
                free_point.override(
                    self.override_parameters[self.override_parameter_start_idx:self.override_parameter_end_idx])
                self.override_parameter_start_idx += free_point.n_overrideable_parameters

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

    def extract_parameters(self):
        """
        ### Description:

        This function extracts every parameter from the `pymead.core.base_airfoil_params.BaseAirfoilParams`, all the
        `pymead.core.anchor_point.AnchorPoint`s, and all the `pymead.core.free_point.FreePoint`s with
        `active=True` and `linked=False` as a `list` of parameter values.
        """

        self.params = [var for var in vars(self.base_airfoil_params).values()
                       if isinstance(var, Param) and var.active and not var.linked]

        for anchor_point in self.anchor_point_tuple:

            self.params.extend([var for var in vars(anchor_point).values()
                                if isinstance(var, Param) and var.active and not var.linked])

        for free_point in self.free_point_tuple:

            self.params.extend([var for var in vars(free_point).values()
                                if isinstance(var, Param) and var.active and not var.linked])

    def insert_free_point(self, free_point: FreePoint):
        """
        ### Description:

        Public method used to insert a `pymead.core.free_point.FreePoint` into an already instantiated
        `pymead.core.airfoil.Airfoil`.

        ### Args:

        `free_point`: a `pymead.core.free_point.FreePoint` to add to a Bézier curve
        """
        if free_point.previous_free_point is None:
            free_point.tag = 'FP0'
            free_point.ctrlpt.tag = free_point.tag
            # If there are already free_points assigned to this anchor point, need to change the previous_free_point
            # name of the first free_point in the order from <None> to the name of the current free_point being set
            if len(self.free_points[free_point.anchor_point_tag]) > 0:
                self.free_points[free_point.anchor_point_tag][0].previous_free_point = free_point.tag
            self.free_points[free_point.anchor_point_tag][free_point.tag] = free_point
            self.free_point_order[free_point.anchor_point_tag].insert(0, free_point)
        self.N[free_point.anchor_point_tag] += 1
        self.param_dicts['FreePoints'][free_point.tag] = {'x': free_point.x, 'y': free_point.y}

    def delete_free_point(self, free_point_tag: str, anchor_point_tag: str):
        print(f"free_points before deletion: {self.free_points}")
        print(f"free_point_order before deletion: {self.free_point_order}")
        fp_idx = next((idx for idx, fp in enumerate(self.free_point_order[anchor_point_tag])
                       if fp.tag == free_point_tag))
        print(f"Deleting fp_idx = {fp_idx}...")
        self.free_point_order[anchor_point_tag].pop(fp_idx)
        self.free_points[anchor_point_tag].pop(free_point_tag)
        print(f"free_points after deletion: {self.free_points}")
        print(f"free_point_order after deletion: {self.free_point_order}")

    def insert_anchor_point(self, ap: AnchorPoint):
        order_idx = next((idx for idx, anchor_point in enumerate(self.anchor_point_order)
                          if anchor_point == ap.previous_anchor_point))
        self.anchor_point_order.insert(order_idx, ap.tag)
        self.anchor_points[ap.tag] = ap
        self.N[ap.tag] = 5
        ap_param_list = ['x', 'y', 'L', 'R', 'r', 'phi', 'psi1', 'psi2']
        self.param_dicts['AnchorPoints'][ap.tag] = {p: getattr(ap, p) for p in ap_param_list}

    # def _set_bezier_curve_orders(self):
    #     """
    #     ### Description:
    #
    #     Sets the orders of the Bézier curves properly based on the location of all the
    #     `pymead.core.anchor_point.AnchorPoint`s and `pymead.core.free_point.FreePoint`s.
    #     """
    #     for anchor_point in self.anchor_point_tuple:
    #         self.N[anchor_point.name] = 5
    #         if anchor_point.name not in self.anchor_point_order:
    #             self.anchor_point_order.insert(self.anchor_point_order.index(anchor_point.previous_anchor_point) + 1,
    #                                            anchor_point.name)
    #     if self.anchor_point_order.index('te_2') - self.anchor_point_order.index('le') != 1:
    #         self.N['le'] = 5  # Set the order of the Bézier curve after the leading edge to 5
    #         self.N[self.anchor_point_order[-2]] = 4  # Set the order of the last Bezier curve to 4
    #     for free_point in self.free_point_tuple:
    #         # Increment the order of the modified Bézier curve
    #         self.N[free_point.previous_anchor_point] += 1

    # def _add_anchor_point(self, anchor_point: AnchorPoint):
    #     """
    #     ### Description:
    #
    #     Adds an anchor point between two anchor points, builds the associated control point branch, and inserts the
    #     control point branch into the set of control points. `needs_update` is set to `True`.
    #
    #     ### Args:
    #
    #     `anchor_point`: an `pymead.core.anchor_point.AnchorPoint` from which to build a control point branch
    #     """
    #     self.anchor_points[anchor_point.name] = anchor_point.xy
    #     self._set_slope(anchor_point)
    #     self._set_curvature(anchor_point)
    #     self.needs_update = True

    # def _add_anchor_points(self):
    #     """
    #     ### Description:
    #
    #     This function executes `pymead.core.airfoil.Airfoil.add_anchor_point()` for all the anchor points in the
    #     `anchor_point_tuple`. Enforces leading edge and trailing edge Bézier curve orders.
    #     `needs_update` is set to `True`.
    #     """
    #     for anchor_point in self.anchor_point_tuple:
    #         self._add_anchor_point(anchor_point)
    #     self.needs_update = True

    # def _add_free_points(self):
    #     """
    #     ### Description:
    #
    #     This function executes `pymead.core.airfoil.Airfoil.add_free_point()` for all the anchor points in the
    #     `free_point_tuple`. `needs_update` is set to `True`.
    #     """
    #     for free_point in self.free_point_tuple:
    #         self._add_free_point(free_point)
    #     self.needs_update = True

    # def update_anchor_point_array(self):
    #     r"""
    #     ### Description:
    #
    #     This function updates the `anchor_point_array` attribute of `pymead.core.airfoil.Airfoil`, which is a
    #     `np.ndarray` of `shape=(N, 2)`, where `N` is the number of anchor points in the airfoil, and the columns
    #     represent the \(x\) and \(y\) coordinates.
    #     """
    #     for key in self.anchor_point_order:
    #         xy = self.anchor_points[key]
    #         if key == 'te_1':
    #             self.anchor_point_array = xy
    #         else:
    #             self.anchor_point_array = np.row_stack((self.anchor_point_array, xy))
    #     self.transformed_anchor_points = deepcopy(self.anchor_points)

    def update(self):
        r"""
        ### Description:
        """
        # Translate back to origin if not already at origin
        if self.control_points is not None and self.control_points != []:
            current_le_pos = np.array([[cp.xp, cp.yp] for cp in self.control_points if cp.tag == 'le'])
            self.translate(-current_le_pos[0, 0], -current_le_pos[0, 1])

        # Rotate to zero degree angle of attack
        if self.control_points is not None and self.control_points != []:
            current_te_1_pos = np.array([[cp.xp, cp.yp] for cp in self.control_points if cp.tag == 'te_1']).flatten()
            current_alf = -np.arctan2(current_te_1_pos[1], current_te_1_pos[0])
            # print(current_alf * 180/np.pi)
            self.rotate(current_alf)

        # Scale so that the chord length is equal to 1.0
        if self.control_points is not None and self.control_points != []:
            current_te_1_pos = np.array([[cp.xp, cp.yp] for cp in self.control_points if cp.tag == 'te_1']).flatten()
            # print(f"control_points = {self.control_points}")
            current_chord = current_te_1_pos[0]
            self.scale(1 / current_chord)

        # Generate anchor point branches
        for ap in self.anchor_points:
            if isinstance(ap, AnchorPoint):
                ap.set_minus_plus_bezier_curve_orders(self.N['te_1'], self.N['le'])
                ap.generate_anchor_point_branch(self.anchor_point_order)
            elif isinstance(ap, TrailingEdgePoint):
                ap.generate_anchor_point_branch()  # trailing edge anchor points do not need to know the ap order

        # Get the control points from all the anchor points
        self.control_points = []
        for ap_tag in self.anchor_point_order:
            self.control_points.extend(next((ap.ctrlpt_branch_list for ap in self.anchor_points if ap.tag == ap_tag)))

        for key, fp_dict in self.free_points.items():
            if key == 'te_1':
                insertion_index = 2
            else:
                insertion_index = next((idx for idx, cp in enumerate(self.control_points)
                                        if cp.cp_type == 'g2_plus' and cp.anchor_point_tag == key))
            self.control_points[insertion_index:insertion_index] = [fp.ctrlpt for fp in fp_dict.values()]
            # if key == 'te_1':
            #     print(f"control_points = {self.control_points[insertion_index].xp}, {self.control_points[insertion_index].yp}")

        # Scale airfoil by chord length
        self.scale(self.c.value)

        # Rotate by airfoil angle of attack (alf)
        self.rotate(-self.alf.value)

        # Translate by airfoil dx, dy
        self.translate(self.dx.value, self.dy.value)

        # Get the control point array
        self.control_point_array = np.array([[cp.xp, cp.yp] for cp in self.control_points])

        # Make Bézier curves from the control point array
        self.curve_list_generated = True
        if self.curve_list is None:
            self.curve_list = []
            self.curve_list_generated = False

        cp_end_idx, cp_start_idx = 0, 1
        for idx, ap_tag in enumerate(self.anchor_point_order[:-1]):
            cp_end_idx += self.N[ap_tag] + 1
            P = self.control_point_array[cp_start_idx - 1:cp_end_idx]
            if self.curve_list_generated:
                self.curve_list[idx].update(P, 150)
            else:
                self.curve_list.append(Bezier(P, 150))
            cp_start_idx = deepcopy(cp_end_idx)
        self.curve_list_generated = True

        self.n_control_points = len(self.control_points)

    def override(self, parameters):
        """
        ### Description:

        This function re-initializes the `Airfoil` object using the list of `override_parameters`

        ### Args:

        `parameters`: a list of `override_parameters` generated by `extract_parameters` and possibly modified.
        """
        self.__init__(self.nt, self.base_airfoil_params, override_parameters=parameters)

    def translate(self, dx: float, dy: float):
        """
        ### Description:

        Translates all the control points and anchor points by \\(\\Delta x\\) and \\(\\Delta y\\).

        ### Args:

        `dx`: \\(x\\)-direction translation magnitude

        `dy`: \\(y\\)-direction translation magnitude

        ### Returns:

        The translated control point and anchor point arrays
        """
        for cp in self.control_points:
            cp.xp += dx
            cp.yp += dy

    def rotate(self, angle: float):
        """
        ### Description:

        Rotates all the control points and anchor points by a specified angle. Used to implement the angle of attack.

        ### Args:

        `angle`: Angle (in radians) by which to rotate the airfoil.

        ### Returns:

        The rotated control point and anchor point arrays
        """
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        for cp in self.control_points:
            rotated_point = (rot_mat @ np.array([[cp.xp], [cp.yp]])).flatten()
            cp.xp = rotated_point[0]
            cp.yp = rotated_point[1]

    def scale(self, scale_value):
        """
        ### Description:

        Scales the airfoil about the origin.
        """
        for cp in self.control_points:
            cp.xp *= scale_value
            cp.yp *= scale_value

    def compute_area(self):
        """
        ### Description:

        Computes the area of the airfoil as the area of a many-sided polygon enclosed by the airfoil coordinates using
        the [shapely](https://shapely.readthedocs.io/en/stable/manual.html) library.

        ### Returns:
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
        """
        ### Description:

        Determines whether the airfoil intersects itself using the `is_simple()` function of the
        [`shapely`](https://shapely.readthedocs.io/en/stable/manual.html) library.

        ### Returns:

        A `bool` value describing whether the airfoil intersects itself
        """
        if self.needs_update:
            self.update()
        points_shapely = list(map(tuple, self.coords))
        line_string = LineString(points_shapely)
        is_simple = line_string.is_simple
        return not is_simple

    def compute_thickness(self, n_lines: int = 201):
        r"""
        ### Description:

        Calculates the thickness distribution and maximum thickness of the airfoil.

        ### Args:

        `n_lines`: Optional `int` describing the number of lines evenly spaced along the chordline produced to
        determine the thickness distribution. Default: `201`.

        ### Returns:

        The list of \(x\)-values used for the thickness distribution calculation, the thickness distribution, and the
        maximum value of the thickness distribution.
        """
        points_shapely = list(map(tuple, self.non_transformed_coords))
        airfoil_line_string = LineString(points_shapely)
        x_thickness = np.linspace(0.0, self.c.value, n_lines)
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
        return self.x_thickness, self.thickness, self.max_thickness

    def plot_airfoil(self, axs: plt.axes, **plot_kwargs):
        plt_curves = []
        for curve in self.curve_list:
            plt_curve = curve.plot_curve(axs, **plot_kwargs)
            plt_curves.append(plt_curve)
        return plt_curves

    def init_airfoil_curve_pg(self, v, pen):
        for curve in self.curve_list:
            curve.init_curve_pg(v, pen)

    def update_airfoil_curve(self):
        for curve in self.curve_list:
            curve.update_curve()

    def update_airfoil_curve_pg(self):
        for curve in self.curve_list:
            curve.update_curve_pg()

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
