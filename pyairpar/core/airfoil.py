import numpy as np
from pyairpar.core.param import Param
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.free_point import FreePoint
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
from pyairpar.core.bezier import Bezier
from pyairpar.core.trailing_edge_point import TrailingEdgePoint
from pyairpar.symmetric.symmetric_base_airfoil_params import SymmetricBaseAirfoilParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import typing
from shapely.geometry import Polygon, LineString
from copy import deepcopy


class Airfoil:

    def __init__(self,
                 number_coordinates: int = 100,
                 base_airfoil_params: BaseAirfoilParams or SymmetricBaseAirfoilParams = BaseAirfoilParams(),
                 override_parameters: list = None
                 ):
        """
        ### Description:

        `pyairpar.core.airfoil.Airfoil` is the base class for Bézier-parametrized airfoil creation.

        .. image:: complex_airfoil-3.png

        ### Args:

        `number_coordinates`: an `int` representing the number of discrete \\(x\\) - \\(y\\) coordinate pairs in each
        Bézier curve. Gets passed to the `bezier` function.

        `base_airfoil_params`: an instance of either the `pyairpar.core.base_airfoil_params.BaseAirfoilParams` class or
        the `pyairpar.symmetric.symmetric_base_airfoil_params.SymmetricBaseAirfoilParams` class which defines the base
        set of parameters to be used

        `anchor_point_tuple`: a `tuple` of `pyairpar.core.anchor_point.AnchorPoint` objects. To specify a single
        anchor point, use `(pyairpar.core.anchor_point.AnchorPoint(),)`. Default: `()`

        `free_point_tuple`: a `tuple` of `pyairpar.core.free_point.FreePoint` objects. To specify a single free
        point, use `(pyairpar.core.free_point.FreePoint(),)`. Default: `()`

        ### Returns:

        An instance of the `Airfoil` class
        """

        self.nt = number_coordinates
        self.params = []
        self.base_airfoil_params = base_airfoil_params
        self.override_parameters = override_parameters

        self.override_parameter_start_idx = 0
        self.override_parameter_end_idx = self.base_airfoil_params.n_overrideable_parameters
        if self.override_parameters is not None:
            self.base_airfoil_params.override(
                self.override_parameters[self.override_parameter_start_idx:self.override_parameter_end_idx])
        self.override_parameter_start_idx += self.base_airfoil_params.n_overrideable_parameters

        self.c = base_airfoil_params.c
        self.alf = base_airfoil_params.alf
        self.R_le = base_airfoil_params.R_le
        self.L_le = base_airfoil_params.L_le
        self.r_le = base_airfoil_params.r_le
        self.phi_le = base_airfoil_params.phi_le
        self.psi1_le = base_airfoil_params.psi1_le
        self.psi2_le = base_airfoil_params.psi2_le
        self.L1_te = base_airfoil_params.L1_te
        self.L2_te = base_airfoil_params.L2_te
        self.theta1_te = base_airfoil_params.theta1_te
        self.theta2_te = base_airfoil_params.theta2_te
        self.t_te = base_airfoil_params.t_te
        self.r_te = base_airfoil_params.r_te
        self.phi_te = base_airfoil_params.phi_te
        self.dx = base_airfoil_params.dx
        self.dy = base_airfoil_params.dy
        self.control_point_array = None
        self.n_control_points = None
        self.curve_list = None
        self.curve_list_generated = None

        # Ensure that all the trailing edge parameters are no longer active if the trailing edge thickness is set to 0.0
        if self.t_te.value == 0.0:
            self.r_te.active = False
            self.phi_te.active = False

        self.free_points = {}
        self.param_dicts = {}
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

        if self.override_parameters is not None:
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
        self.free_point_order = {'te_1': [], 'le': []}
        self.anchor_point_array = np.array([])

        self.N = {
            'te_1': 4,
            'le': 4
        }

        self.anchor_points = self.base_airfoil_params.generate_main_anchor_points()
        self.control_points = []

        self.update()

        # fig, axs = plt.subplots(1, 1)
        # self.curve_list[0].plot_curve(axs, color='mediumaquamarine')
        # self.curve_list[1].plot_curve(axs, color='indianred')
        # axs.set_aspect('equal')
        # plt.show()

        # self.update()
        pass

    def extract_parameters(self):
        """
        ### Description:

        This function extracts every parameter from the `pyairpar.core.base_airfoil_params.BaseAirfoilParams`, all the
        `pyairpar.core.anchor_point.AnchorPoint`s, and all the `pyairpar.core.free_point.FreePoint`s with
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

        Public method used to insert a `pyairpar.core.free_point.FreePoint` into an already instantiated
        `pyairpar.core.airfoil.Airfoil`.

        ### Args:

        `free_point`: a `pyairpar.core.free_point.FreePoint` to add to a Bézier curve
        """
        _temp_free_point_list = list(deepcopy(self.free_point_tuple))
        _temp_free_point_list.append(free_point)
        self.free_point_tuple = tuple(deepcopy(_temp_free_point_list))
        self.update()

    def delete_free_point(self, index: int = None, xy_location: tuple = None):
        pass

    def _set_bezier_curve_orders(self):
        """
        ### Description:

        Sets the orders of the Bézier curves properly based on the location of all the
        `pyairpar.core.anchor_point.AnchorPoint`s and `pyairpar.core.free_point.FreePoint`s.
        """
        for anchor_point in self.anchor_point_tuple:
            self.N[anchor_point.name] = 5
            if anchor_point.name not in self.anchor_point_order:
                self.anchor_point_order.insert(self.anchor_point_order.index(anchor_point.previous_anchor_point) + 1,
                                               anchor_point.name)
        if self.anchor_point_order.index('te_2') - self.anchor_point_order.index('le') != 1:
            self.N['le'] = 5  # Set the order of the Bézier curve after the leading edge to 5
            self.N[self.anchor_point_order[-2]] = 4  # Set the order of the last Bezier curve to 4
        for free_point in self.free_point_tuple:
            # Increment the order of the modified Bézier curve
            self.N[free_point.previous_anchor_point] += 1

    def _add_anchor_point(self, anchor_point: AnchorPoint):
        """
        ### Description:

        Adds an anchor point between two anchor points, builds the associated control point branch, and inserts the
        control point branch into the set of control points. `needs_update` is set to `True`.

        ### Args:

        `anchor_point`: an `pyairpar.core.anchor_point.AnchorPoint` from which to build a control point branch
        """
        self.anchor_points[anchor_point.name] = anchor_point.xy
        self._set_slope(anchor_point)
        self._set_curvature(anchor_point)
        self.needs_update = True

    def _add_anchor_points(self):
        """
        ### Description:

        This function executes `pyairpar.core.airfoil.Airfoil.add_anchor_point()` for all the anchor points in the
        `anchor_point_tuple`. Enforces leading edge and trailing edge Bézier curve orders.
        `needs_update` is set to `True`.
        """
        for anchor_point in self.anchor_point_tuple:
            self._add_anchor_point(anchor_point)
        self.needs_update = True

    def _add_free_points(self):
        """
        ### Description:

        This function executes `pyairpar.core.airfoil.Airfoil.add_free_point()` for all the anchor points in the
        `free_point_tuple`. `needs_update` is set to `True`.
        """
        for free_point in self.free_point_tuple:
            self._add_free_point(free_point)
        self.needs_update = True

    def update_anchor_point_array(self):
        r"""
        ### Description:

        This function updates the `anchor_point_array` attribute of `pyairpar.core.airfoil.Airfoil`, which is a
        `np.ndarray` of `shape=(N, 2)`, where `N` is the number of anchor points in the airfoil, and the columns
        represent the \(x\) and \(y\) coordinates.
        """
        for key in self.anchor_point_order:
            xy = self.anchor_points[key]
            if key == 'te_1':
                self.anchor_point_array = xy
            else:
                self.anchor_point_array = np.row_stack((self.anchor_point_array, xy))
        self.transformed_anchor_points = deepcopy(self.anchor_points)

    def update(self):
        r"""
        ### Description:


        """
        # Generate anchor point branches
        for ap in self.anchor_points:
            if isinstance(ap, AnchorPoint):
                ap.set_minus_plus_bezier_curve_orders(self.N['te_1'], self.N['le'])
                ap.generate_anchor_point_branch(self.anchor_point_order)
            elif isinstance(ap, TrailingEdgePoint):
                ap.generate_anchor_point_branch()  # trailing edge anchor points do not need to know the ap order

        # Get the control points from all the anchor points
        self.control_points = []
        for ap_name in self.anchor_point_order:
            print(f"ap_name_control_points = {self.control_points}")
            self.control_points.extend(next((ap.ctrlpt_branch_list for ap in self.anchor_points if ap.name == ap_name)))

        # Get the control point array
        self.control_point_array = np.array([[cp.xp, cp.yp] for cp in self.control_points])

        # Make Bezier curves from the control point array
        self.curve_list_generated = True
        if self.curve_list is None:
            self.curve_list = []
            self.curve_list_generated = False

        cp_end_idx, cp_start_idx = 0, 1
        for idx, ap_name in enumerate(self.anchor_point_order[:-1]):
            cp_end_idx += self.N[ap_name] + 1
            P = self.control_point_array[cp_start_idx - 1:cp_end_idx]
            if self.curve_list_generated:
                self.curve_list[idx].update(P, 150)
            else:
                self.curve_list.append(Bezier(P, 150))
            cp_start_idx = deepcopy(cp_end_idx)

        self.n_control_points = len(self.control_points)

    def override(self, parameters):
        """
        ### Description:

        This function re-initializes the `Airfoil` object using the list of `override_parameters`

        ### Args:

        `parameters`: a list of `override_parameters` generated by `extract_parameters` and possibly modified.
        """
        self.__init__(self.nt, self.base_airfoil_params, self.anchor_point_tuple, self.free_point_tuple,
                      override_parameters=parameters)

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
        self.control_points[:, 0] += dx
        self.control_points[:, 1] += dy
        self.anchor_point_array[:, 0] += dx
        self.anchor_point_array[:, 1] += dy
        for key, anchor_point in self.transformed_anchor_points.items():
            self.transformed_anchor_points[key] = anchor_point + np.array([dx, dy])
        self.needs_update = True
        return self.control_points, self.anchor_point_array

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
        self.control_points = (rot_mat @ self.control_points.T).T
        self.anchor_point_array = (rot_mat @ self.anchor_point_array.T).T
        for key, anchor_point in self.transformed_anchor_points.items():
            self.transformed_anchor_points[key] = (rot_mat @ anchor_point.T).T
        self.needs_update = True
        return self.control_points, self.anchor_point_array

    def generate_coords(self):
        """
        ### Description:

        Generates the Bézier curves through the control points. Also re-casts the Bézier curve points in terms of
        airfoil coordinates by removing the points shared by joined Bézier curves. Curvature information is extracted
        from the `C` dictionary. Helper method for `generate_airfoil_coordinates()` and
        `generate_non_transformed_airfoil_coordinates()`.

        ### Returns:

        The airfoil coordinates and the curvature of the airfoil.
        """
        if self.C:
            self.C = []
        P_start_idx = 0
        for idx in range(len(self.anchor_point_order) - 1):
            P_length = self.N[self.anchor_point_order[idx]] + 1
            P_end_idx = P_start_idx + P_length
            P = self.control_points[P_start_idx:P_end_idx, :]
            C = Bezier(P, self.nt)
            self.C.append(C)
            P_start_idx = P_end_idx - 1
        coords = np.array([])
        curvature = np.array([])
        for idx in range(len(self.C)):
            if idx == 0:
                coords = np.column_stack((self.C[idx].x, self.C[idx].y))
                curvature = np.column_stack((self.C[idx].x, self.C[idx].k))
            else:
                coords = np.row_stack((coords, np.column_stack((self.C[idx].x[1:], self.C[idx].y[1:]))))
                curvature = np.row_stack((curvature, np.column_stack((self.C[idx].x[1:], self.C[idx].k[1:]))))
        return coords, curvature

    def generate_airfoil_coordinates(self):
        """
        ### Description:

        Runs the `generate_coords()` method after the rotation and translation steps and saves the information to
        the `coords` and `curvature` attributes of `Airfoil`. Used in `Airfoil.update()`.

        ### Returns:

        The airfoil coordinates, the `C` dictionary of Bézier curve information, and the curvature.
        """
        coords, curvature = self.generate_coords()
        self.coords = coords
        self.curvature = curvature
        return self.coords, self.C, self.curvature

    def generate_non_transformed_airfoil_coordinates(self):
        """
        ### Description:

        Runs the `generate_coords()` method before the rotation and translation steps and saves the information to the
        `non_transformed_coords` attribute of `Airfoil`. Used in `Airfoil.update()`.

        ### Returns:

        The coordinates of the airfoil before rotation and translation.
        """
        coords, _ = self.generate_coords()
        self.non_transformed_coords = coords
        return self.non_transformed_coords

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

    def plot(self, plot_what: typing.Tuple[str, ...], fig: Figure = None, axs: Axes = None, show_plot: bool = True,
             save_plot: bool = False, save_path: str = None, plot_kwargs: typing.List[dict] or dict = None,
             show_title: bool = True, show_legend: bool = True, figwidth: float = 10.0, figheight: float = 2.5,
             tight_layout: bool = True, axis_equal: bool = True):
        r"""
        ### Description:

        A variety of airfoil plotting options using [`matplotlib`](https://matplotlib.org/). Many wrapper options
        are available here, but custom plots can also be created directly from the `coords`, `curvature`, `C`,
        `control_points`, and `anchor_point_array` attributes of the `Airfoil` class.

        ### Args:

        `plot_what`: One or more of `"airfoil"`, `"anchor-point-skeleton"`, `"control-point-skeleton"`,
        `"chordline"`, `"R-circles"`, or `"curvature"`.

        `fig`: The
        [`matplotlib.figure.Figure`](https://matplotlib.org/stable/api/figure_api.html?highlight=figure%20figure#matplotlib.figure.Figure)
        object as a canvas input. New `Figure` is created if `None`. Default: `None`.

        `axs`: The
        [`matplotlib.axes.Axes`](https://matplotlib.org/stable/api/axes_api.html?highlight=axes%20axes#the-axes-class)
        object as a axes input. New `Axes` is created if `None`. Default: `None`.

        `show_plot`: Whether to show the plot (`bool`). Default: `True`.

        `save_plot`: Whether to save the plot (`bool`). Default: `False`.

        `save_path`: A `str` describing the root directory, filename, and image extension of the path to save. Only
        used if `save_plot` is `True`. Default: `None`.

        `plot_kwargs`: A list of dictionaries with [`matplotlib`](https://matplotlib.org/) keyword arguments to be
        unpacked and applied to each Bézier curve element-by-element. Default: `None`.

        `show_title`: Whether to show a title describing the airfoil chord length, angle of attack, and area (`bool`).
        Default: `True`.

        `show_legend`: Whether to show the legend (`bool`). Default: `True`.

        `figwidth`: The width of the figure, in inches (`float`). Default: `10.0`.

        `figheight`: The height of the figure, in inches (`float`). Default: `2.5`.

        `tight_layout`: Whether to tighten the margins around the plot (`bool`). Default: `True`.

        `axis_equal`: Whether to set the aspect ratio of the plot to equal \(x\) and \(y\) (`bool`). Default: `True`.

        ### Returns:

        A tuple of the
        [`matplotlib.figure.Figure`](https://matplotlib.org/stable/api/figure_api.html?highlight=figure%20figure#matplotlib.figure.Figure)
        object and the
        [`matplotlib.axes.Axes`](https://matplotlib.org/stable/api/axes_api.html?highlight=axes%20axes#the-axes-class)
        object used for plotting
        """
        if self.needs_update:
            self.update()
        if fig is None and axs is None:
            fig, axs = plt.subplots(1, 1)

        for what_to_plot in plot_what:

            if what_to_plot == 'airfoil':
                for idx, C in enumerate(self.C):
                    if plot_kwargs is None:
                        if idx == 0:
                            axs.plot(C.x, C.y, color='cornflowerblue', label='airfoil')
                        else:
                            axs.plot(C.x, C.y, color='cornflowerblue')
                    else:
                        axs.plot(C.x, C.y, **plot_kwargs[idx])

            if what_to_plot == 'anchor-point-skeleton':
                if plot_kwargs is None:
                    axs.plot(self.anchor_point_array[:, 0], self.anchor_point_array[:, 1], ':x', color='black',
                             label='anchor point skeleton')
                else:
                    axs.plot(self.anchor_point_array[:, 0], self.anchor_point_array[:, 1], **plot_kwargs)

            if what_to_plot == 'control-point-skeleton':
                if plot_kwargs is None:
                    axs.plot(self.control_points[:, 0], self.control_points[:, 1], '--*', color='grey',
                             label='control point skeleton')
                else:
                    axs.plot(self.control_points[:, 0], self.control_points[:, 1], **plot_kwargs)
            if what_to_plot == 'chordline':
                if plot_kwargs is None:
                    axs.plot(np.array([0 + self.dx.value, self.c.value * np.cos(self.alf.value) + self.dx.value]),
                             np.array([0 + self.dy.value, -self.c.value * np.sin(self.alf.value) + self.dy.value]),
                             '-.', color='indianred', label='chordline')
                else:
                    axs.plot(np.array([0, self.c.value * np.cos(self.alf.value)]),
                             np.array([0, -self.c.value * np.sin(self.alf.value)]), **plot_kwargs)
            if what_to_plot == 'R-circles':
                line = [[0 + self.dx.value, self.R_le.value *
                         np.cos(self.phi_le.value - self.alf.value) + self.dx.value],
                        [0 + self.dy.value, self.R_le.value *
                         np.sin(self.phi_le.value - self.alf.value) + self.dy.value]]
                circle = plt.Circle((line[0][1], line[1][1]), self.R_le.value, fill=False, color='gold',
                                    label='R circle')
                axs.plot(line[0], line[1], color='gold')
                axs.add_patch(circle)
                for anchor_point in self.anchor_point_tuple:
                    xy = self.anchor_point_array[self.anchor_point_order.index(anchor_point.name), :]
                    if self.anchor_point_order.index(anchor_point.name) > self.anchor_point_order.index('le'):
                        perp_angle = np.pi / 2
                        phi = -anchor_point.phi.value
                    else:
                        perp_angle = -np.pi / 2
                        phi = anchor_point.phi.value
                    line = np.array([xy, xy + anchor_point.R.value *
                                     np.array([np.cos(phi - self.alf.value + perp_angle),
                                               np.sin(phi - self.alf.value + perp_angle)])])
                    circle = plt.Circle((line[1, 0], line[1, 1]), anchor_point.R.value, fill=False, color='gold')
                    axs.plot(line[:, 0], line[:, 1], color='gold')
                    axs.add_patch(circle)
            if what_to_plot == 'curvature':
                if plot_kwargs is None:
                    axs.plot(self.curvature[:, 0], self.curvature[:, 1], color='cornflowerblue', label='curvature')
                else:
                    axs.plot(self.curvature[:, 0], self.curvature[:, 1], **plot_kwargs)

        if axis_equal:
            axs.set_aspect('equal', 'box')
        if show_title:
            area = self.compute_area()
            fig.suptitle(
                fr'Airfoil: $c={self.c.value:.3f}$, $\alpha={np.rad2deg(self.alf.value):.3f}^\circ$, $A={area:.5f}$')
        else:
            fig.suptitle("")
        if tight_layout:
            fig.tight_layout()
        fig.set_figwidth(figwidth)
        fig.set_figheight(figheight)
        if show_legend:
            axs.legend()
        if save_plot:
            fig.savefig(save_path)
        if show_plot:
            plt.show()
        return fig, axs

    def plot_airfoil(self, axs: plt.axes, **plot_kwargs):
        plt_curves = []
        for curve in self.curve_list:
            plt_curve = curve.plot_curve(axs, **plot_kwargs)
            plt_curves.append(plt_curve)
        return plt_curves

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
        print(f"Setting plt_normals...")
        for curve in self.curve_list:
            plt_normal = curve.plot_curvature_comb_normals(axs, self.normalized_curvature_scale_factor, **plot_kwargs)
            self.plt_normals.append(plt_normal)
        return self.plt_normals

    def update_curvature_comb_normals(self):
        for curve in self.curve_list:
            print(f"scale_factor = {curve.scale_factor}")
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
