import numpy as np
from pyairpar.core.param import Param
from pyairpar.core.anchor_point import AnchorPoint
from pyairpar.core.free_point import FreePoint
from pyairpar.core.base_airfoil_params import BaseAirfoilParams
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
                 anchor_point_tuple: typing.Tuple[AnchorPoint, ...] = (),
                 free_point_tuple: typing.Tuple[FreePoint, ...] = (),
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

        # Ensure that all the trailing edge parameters are no longer active if the trailing edge thickness is set to 0.0
        if self.t_te.value == 0.0:
            self.r_te.active = False
            self.phi_te.active = False

        self.C = []
        self.free_points = {}
        self.param_dicts = {}
        self.coords = None
        self.non_transformed_coords = None
        self.curvature = None
        self.area = None
        self.x_thickness = None
        self.thickness = None
        self.max_thickness = None

        self.needs_update = True

        self.anchor_point_tuple = anchor_point_tuple

        if self.override_parameters is not None:
            for anchor_point in self.anchor_point_tuple:
                if self.base_airfoil_params.non_dim_by_chord:
                    anchor_point.length_scale_dimension = self.base_airfoil_params.c.value
                self.override_parameter_end_idx = self.override_parameter_start_idx + \
                                                  anchor_point.n_overrideable_parameters
                anchor_point.override(
                    self.override_parameters[self.override_parameter_start_idx:self.override_parameter_end_idx])
                self.override_parameter_start_idx += anchor_point.n_overrideable_parameters

        self.free_point_tuple = free_point_tuple

        if self.override_parameters is not None:
            for free_point in self.free_point_tuple:
                if self.base_airfoil_params.non_dim_by_chord:
                    free_point.length_scale_dimension = self.base_airfoil_params.c.value
                self.override_parameter_end_idx = self.override_parameter_start_idx + \
                                                  free_point.n_overrideable_parameters
                free_point.override(
                    self.override_parameters[self.override_parameter_start_idx:self.override_parameter_end_idx])
                self.override_parameter_start_idx += free_point.n_overrideable_parameters

        self.anchor_points = {'te_1': self.c.value * np.array([1, 0]) + self.r_te.value * self.t_te.value *
                                      np.array([np.cos(np.pi / 2 + self.phi_te.value),
                                                np.sin(np.pi / 2 + self.phi_te.value)]),
                              'le': np.array([0.0, 0.0]),
                              'te_2': self.c.value * np.array([1, 0]) -
                                      (1 - self.r_te.value) * self.t_te.value *
                                      np.array([np.cos(np.pi / 2 + self.phi_te.value),
                                                np.sin(np.pi / 2 + self.phi_te.value)])}
        self.transformed_anchor_points = None
        self.anchor_point_order = ['te_1', 'le', 'te_2']
        self.anchor_point_array = np.array([])

        self.N = {
            'te_1': 4,
            'le': 4
        }

        self.control_points = np.array([])
        self.n_control_points = len(self.control_points)

        self.g1_minus_points, self.g1_plus_points = self.init_g1_points()
        self.g2_minus_points, self.g2_plus_points = self.init_g2_points()

        self.update()

    def init_g1_points(self):
        """
        ### Description:

        Initializes the "g1_minus" and "g1_plus" points for the leading edge (the neighboring control points to
        the leading edge anchor point). These points are used to enforce \\(G^1\\) continuity. "Minus" points refer to
        control points which occur before the anchor point in the path of the Bézier curve, and "plus" points refer to
        control points which occur before the anchor point.

        ### Returns:

        The neighboring control points to the leading edge anchor point as dictionaries.
        """

        g1_minus_points = {
            'te_2': self.anchor_points['te_2'] +
                  self.L2_te.value * np.array([-np.cos(self.theta2_te.value),
                                               -np.sin(self.theta2_te.value)]),
            'le': self.anchor_points['le'] +
                  self.r_le.value * self.L_le.value *
                  np.array([np.cos(np.pi / 2 + self.phi_le.value),
                            np.sin(np.pi / 2 + self.phi_le.value)])
                    if self.R_le.value != 0 else self.anchor_points['le']
        }

        g1_plus_points = {
            'te_1': self.anchor_points['te_1'] +
                    self.L1_te.value * np.array([-np.cos(self.theta1_te.value),
                                                 np.sin(self.theta1_te.value)]),
            'le': self.anchor_points['le'] -
                  (1 - self.r_le.value) * self.L_le.value *
                  np.array([np.cos(np.pi / 2 + self.phi_le.value),
                            np.sin(np.pi / 2 + self.phi_le.value)])
                    if self.R_le.value != 0 else self.anchor_points['le']
        }

        return g1_minus_points, g1_plus_points

    def init_g2_points(self):
        """
        ### Description:

        Initializes the "g2_minus" and "g2_plus" points for the leading edge (the control points two points from the
        leading edge control point). These points are used to enforce \\(G^2\\) continuity. "Minus" points refer to
        control points which occur before the anchor point in the path of the Bézier curve, and "plus" points refer to
        control points which occur before the anchor point.

        ### Returns:

        The neighboring control points to the leading edge anchor point as dictionaries.
        """
        g2_minus_point_le, g2_plus_point_le = self.set_curvature_le()

        g2_minus_points = {
            'le': g2_minus_point_le
        }

        g2_plus_points = {
            'le': g2_plus_point_le
        }

        return g2_minus_points, g2_plus_points

    def set_slope(self, anchor_point: AnchorPoint):
        r"""
        ### Description:

        This is the function which enforces \(G^1\) continuity for all `pyairpar.core.anchor_point.AnchorPoint`s
        which are added. To keep the length ratios and angles defined in a "nice" way, the neighboring control points
        to the anchor point are defined as follows: For anchor points on the upper surface (where \((x_0,
        y_0)\) precedes the leading edge point):

        $$\begin{align*}
        \begin{bmatrix} x_{-1} \\ y_{-1} \end{bmatrix} &= \begin{bmatrix} x_0 \\ y_0 \end{bmatrix} +
        (1-r)L \begin{bmatrix} \cos{\phi} \\ \sin{\phi} \end{bmatrix} \\
        \begin{bmatrix} x_{+1} \\ y_{+1} \end{bmatrix} &= \begin{bmatrix} x_0 \\ y_0 \end{bmatrix} -
        rL \begin{bmatrix} \cos{\phi} \\ \sin{\phi} \end{bmatrix}
        \end{align*}$$

        For anchor points on the lower surface (where \((x_0,y_0)\) occurs further down the Bézier curve path than
        the leading edge point):

        $$\begin{align*}
        \begin{bmatrix} x_{-1} \\ y_{-1} \end{bmatrix} &= \begin{bmatrix} x_0 \\ y_0 \end{bmatrix} -
        rL \begin{bmatrix} \cos{(-\phi)} \\ \sin{(-\phi)} \end{bmatrix} \\
        \begin{bmatrix} x_{+1} \\ y_{+1} \end{bmatrix} &= \begin{bmatrix} x_0 \\ y_0 \end{bmatrix} +
        (1-r)L \begin{bmatrix} \cos{(-\phi)} \\ \sin{(-\phi)} \end{bmatrix}
        \end{align*}$$

        Here, \((x_{-1},y_{-1})\) represents the coordinates of the "minus" point (the control point before the
        leading edge point), and \((x_{+1},y_{+1})\) represents the coordinates of the "plus" point (the control
        point after the leading edge point). The coordinates of the anchor point itself are \((x_0,y_0)\). With
        these definitions, a ratio \(r>0.5\) biases the neighboring control points toward the leading edge, and
        a ratio \(r<0.5\) biases the neighboring control points toward the leading edge. A positive value of \(\phi\)
        angles the neighboring control points toward the trailing edge, and a negative value of \(\phi\) angles the
        neighboring control points toward the trailing edge. See the diagram for
        `pyairpar.core.anchor_point.AnchorPoint` for a visual description of these definitions.

        ### Returns:

        The generated neighboring control point \(x\) - \(y\) locations as `np.ndarray`s of `shape=2`.
        """
        r = anchor_point.r.value
        L = anchor_point.L.value
        phi = anchor_point.phi.value
        R = anchor_point.R.value

        if R == 0:
            self.g1_minus_points[anchor_point.name] = self.anchor_points[anchor_point.name]
            self.g1_plus_points[anchor_point.name] = self.anchor_points[anchor_point.name]
        else:
            if self.anchor_point_order.index(anchor_point.name) < self.anchor_point_order.index('le'):
                self.g1_minus_points[anchor_point.name] = \
                    self.anchor_points[anchor_point.name] + (1 - r) * L * np.array([np.cos(phi), np.sin(phi)])
                self.g1_plus_points[anchor_point.name] = \
                    self.anchor_points[anchor_point.name] - r * L * np.array([np.cos(phi), np.sin(phi)])
            else:
                self.g1_minus_points[anchor_point.name] = \
                    self.anchor_points[anchor_point.name] - r * L * np.array([np.cos(-phi), np.sin(-phi)])
                self.g1_plus_points[anchor_point.name] = \
                    self.anchor_points[anchor_point.name] + (1 - r) * L * np.array([np.cos(-phi), np.sin(-phi)])
        return self.g1_minus_points[anchor_point.name], self.g1_plus_points[anchor_point.name]

    def set_curvature_le(self):
        r"""
        ### Description:

        See the description of `pyairpar.core.airfoil.Airfoil.set_curvature`. This is just a special case of that
        function tailored for the leading edge of the airfoil. Here, \(\psi_1\) defines the angle of the upper
        curvature control arm at the leading edge, and \(\psi_2\) defines the angle of the lower curvature control
        arm at the leading edge. \(\psi_1\), \(\psi_2\), and \(\phi\) are all referenced differently from the other
        `pyairpar.core.anchor_point.AnchorPoint`s. The difference is given mathematically by \(\psi_{1,LE}=\psi_1 -
        \frac{\pi}{2}\), \(\psi_{2,LE}=\psi_2 - \frac{\pi}{2}\), and \(\phi_{LE} = \phi + \frac{\pi}{2}\). See also
        `pyairpar.core.base_airfoil_params.BaseAirfoilParams` for more details on the definitions of the attribute
        parameters input to this function.

        ### Returns:

        The generated curvature control point \(x\) - \(y\) locations as `np.ndarray`s of `shape=2`.
        """
        R_le = self.R_le.value
        psi1_le = self.psi1_le.value
        psi2_le = self.psi2_le.value
        if R_le in [-np.inf, np.inf] or psi1_le in [-np.pi / 2, np.pi / 2] or psi2_le in [-np.pi / 2, np.pi / 2]:
            g2_minus_point = self.g1_minus_points['le']
            g2_plus_point = self.g1_plus_points['le']
        elif R_le == 0:
            g2_minus_point = self.anchor_points['le']
            g2_plus_point = self.anchor_points['le']
        else:
            n1, n2 = self.N[self.anchor_point_order[self.anchor_point_order.index('le') - 1]], self.N['le']
            theta1, theta2 = self.psi1_le.value, -self.psi2_le.value
            x0, y0 = self.anchor_points['le'][0], self.anchor_points['le'][1]

            x_m1, y_m1 = self.g1_minus_points['le'][0], self.g1_minus_points['le'][1]
            g2_minus_point = np.zeros(2)
            g2_minus_point[0] = x_m1 - 1 / self.R_le.value * ((x0 - x_m1) ** 2 + (y0 - y_m1) ** 2) ** (3/2) / (
                                1 - 1 / n1) / ((x_m1 - x0) * np.tan(theta1) + y0 - y_m1)
            g2_minus_point[1] = np.tan(theta1) * (g2_minus_point[0] - x_m1) + y_m1

            x_p1, y_p1 = self.g1_plus_points['le'][0], self.g1_plus_points['le'][1]
            g2_plus_point = np.zeros(2)
            g2_plus_point[0] = x_p1 - 1 / self.R_le.value * ((x_p1 - x0) ** 2 + (y_p1 - y0) ** 2) ** (3/2) / (
                               1 - 1 / n2) / ((x0 - x_p1) * np.tan(theta2) + y_p1 - y0)
            g2_plus_point[1] = np.tan(theta2) * (g2_plus_point[0] - x_p1) + y_p1

        return g2_minus_point, g2_plus_point

    def set_curvature(self, anchor_point: AnchorPoint):
        r"""
        ### Description:

        This is the function which enforces \(G^2\) continuity for all `pyairpar.core.anchor_point.AnchorPoint`s
        which are added. To keep the length ratios and angles defined in a "nice" way, the neighboring control points
        to the anchor point's slope-control points are defined as follows:

        $$
        \begin{align*}
            \begin{bmatrix} x_{-2} \\ y_{-2} \end{bmatrix} &=
            \begin{cases}
                \begin{bmatrix} x_{-1} + \frac{c_1}{c_2[c_3(\tan{(\theta_1)} + y_0 - y_{-1})]}
                                \\ \tan{(\theta_1)} (x_{-2} - x_{-1}) + y_{-1} \end{bmatrix}
                                ,& \theta_1 \neq \frac{\pi}{2} + k\pi \text{ for integer } k \wedge R \in (-\infty,0) \cup (0, \infty) \\
                \begin{bmatrix} x_{-1} \\ y_{-1} + \frac{c_1}{c_2 c_3}  \end{bmatrix},& \theta_1 = \frac{\pi}{2} + k\pi \text{ for integer } k \wedge
                                                                                                    R \in (-\infty,0)
                                                                                                    \cup (0,\infty) \\
                \begin{bmatrix} x_{-1} \\ y_{-1} \end{bmatrix},& \psi_1 = 0 \vee \psi_2 = 0 \vee \psi_1 = \frac{\pi}{2} \vee \psi_2 = \frac{\pi}{2} \vee R =
                                                                                                            \pm \infty \\
                \begin{bmatrix} x_0 \\ y_0 \end{bmatrix},& R = 0
            \end{cases} \\
            \begin{bmatrix} x_{+2} \\ y_{+2} \end{bmatrix} &=
            \begin{cases}
                \begin{bmatrix} x_{+1} + \frac{c_4}{c_5[c_6(\tan{(\theta_2)} + y_{+1} - y_0)]}
                                \\ \tan{(\theta_2)} (x_{+2} - x_{+1}) + y_{+1} \end{bmatrix}
                                ,& \theta_2 \neq \frac{\pi}{2} + k\pi \text{ for integer } k \wedge R \in (-\infty,0) \cup (0, \infty) \\
                \begin{bmatrix} x_{+1} \\ y_{+1} + \frac{c_4}{c_5 c_6}  \end{bmatrix},& \theta_2 = \frac{\pi}{2} + k\pi \text{ for integer } k \wedge
                                                                                                    R \in (-\infty,0)
                                                                                                    \cup (0,\infty) \\
                \begin{bmatrix} x_{+1} \\ y_{+1} \end{bmatrix},& \psi_1 = 0 \vee \psi_2 = 0 \vee \psi_1 = \frac{\pi}{2} \vee \psi_2 = \frac{\pi}{2} \vee R =
                                                                                                            \pm \infty \\
                \begin{bmatrix} x_0 \\ y_0 \end{bmatrix},& R = 0
            \end{cases}
        \end{align*}
        $$
        where
        $$
        \begin{align*}
        c_1 &= \frac{-1}{R}[(x_0-x_{-1})^2 + (y_0-y_{-1})^2]^{3/2} \\
        c_2 &= 1 - \frac{1}{n_1} \\
        c_3 &= x_{-1} - x_0 \\
        c_4 &= \frac{-1}{R}[(x_{+1}-x_0)^2 + (y_{+1}-y_0)^2]^{3/2} \\
        c_5 &= 1 - \frac{1}{n_2} \\
        c_6 &= x_0 - x_{+1}
        \end{align*}
        $$

        Here, \(n_1\) is the order of the Bézier curve preceding the anchor point, and \(n_2\) is the order of the
        Bézier curve following the anchor point. \((x_0,y_0)\) is the anchor point location, \((x_{-1},y_{-1})\)
        and \((x_{+1},y_{+1})\) are the neighboring control points, and \((x_{-2},y_{-2})\) and \((x_{+2},y_{+2})\)
        are the curvature control points. \(\theta_1\) and \(\theta_2\) are governed by the following relationships:

        $$ \begin{align*} \theta_1 &= \begin{cases} \psi_1 + \phi,& R > 0,\,\text{upper surface} \\ \pi + \psi_2 -
        \phi,& R > 0,\,\text{lower surface} \\ \pi - \psi_1 + \phi,& R < 0,\,\text{upper surface} \\ -\psi_2 - \phi,
        & R < 0,\,\text{lower surface} \end{cases} \\ \theta_2 &= \begin{cases} -\psi_2 + \phi,& R > 0,\,\text{upper
        surface} \\ \psi_1 - \phi,& R > 0,\,\text{lower surface} \\ \pi + \psi_2 + \phi,& R < 0,\,\text{upper
        surface} \\ \psi_1 - \phi,& R < 0,\,\text{lower surface} \end{cases} \end{align*} $$ where \(\psi_1\) and \(
        \psi_2\) are the aft and fore curvature control arm angles, respectively (or the upper and lower curvature
        control arm angles in the case of the leading edge). By these definitions of the curvature control arm
        angles, decreasing \(\psi_1\) or \(\psi_2\) from \(90^{\circ}\) ( \(0^{\circ}\) in the leading edge case) has
        the effect of "tucking" the arms in, and increasing \(\psi_1\) or \(\psi_2\) from \(90^{\circ}\) ( \(0^{
        \circ}\) in the leading edge case) has the effect of "spreading" the arms out. See the documentation for
        `pyairpar.core.anchor_point.AnchorPoint` for further description and a visual.

        ### Returns:

        The generated curvature control point \(x\) - \(y\) locations as `np.ndarray`s of `shape=2`.
        """
        R = anchor_point.R.value
        psi1 = anchor_point.psi1.value
        psi2 = anchor_point.psi2.value
        if R in [-np.inf, np.inf] or psi1 in [0, np.pi] or psi2 in [0, np.pi]:
            self.g2_minus_points[anchor_point.name] = self.g1_minus_points[anchor_point.name]
            self.g2_plus_points[anchor_point.name] = self.g1_plus_points[anchor_point.name]
            g2_minus_point = self.g2_minus_points[anchor_point.name]
            g2_plus_point = self.g2_plus_points[anchor_point.name]
        elif R == 0:
            self.g2_minus_points[anchor_point.name] = self.anchor_points[anchor_point.name]
            self.g2_plus_points[anchor_point.name] = self.anchor_points[anchor_point.name]
            g2_minus_point = self.g2_minus_points[anchor_point.name]
            g2_plus_point = self.g2_plus_points[anchor_point.name]
        else:
            n1, n2 = self.N[self.anchor_point_order[self.anchor_point_order.index(anchor_point.name) - 1]], \
                     self.N[anchor_point.name]
            if R > 0:  # If the radius of curvature is positive,
                if self.anchor_point_order.index(anchor_point.name) < self.anchor_point_order.index('le'):
                    theta1, theta2 = anchor_point.psi1.value + anchor_point.phi.value, \
                                     -anchor_point.psi2.value + anchor_point.phi.value
                else:
                    theta2, theta1 = - anchor_point.psi1.value - anchor_point.phi.value, \
                                     np.pi + anchor_point.psi2.value - anchor_point.phi.value
            else:  # If the radius of curvature is negative,
                if self.anchor_point_order.index(anchor_point.name) < self.anchor_point_order.index('le'):
                    theta1, theta2 = np.pi - anchor_point.psi1.value + anchor_point.phi.value, \
                                     np.pi + anchor_point.psi2.value + anchor_point.phi.value
                else:
                    theta2, theta1 = anchor_point.psi1.value - anchor_point.phi.value, \
                                     -anchor_point.psi2.value - anchor_point.phi.value
            x0, y0 = anchor_point.xy[0], anchor_point.xy[1]

            x_m1, y_m1 = self.g1_minus_points[anchor_point.name][0], self.g1_minus_points[anchor_point.name][1]
            g2_minus_point = np.zeros(2)
            c1 = - 1 / R * ((x0 - x_m1) ** 2 + (y0 - y_m1) ** 2) ** (3 / 2)
            c2 = 1 - 1 / n1
            c3 = x_m1 - x0
            if (theta1 - np.pi / 2) % np.pi != 0:
                g2_minus_point[0] = x_m1 + c1 / c2 / (c3 * np.tan(theta1) + y0 - y_m1)
                g2_minus_point[1] = np.tan(theta1) * (g2_minus_point[0] - x_m1) + y_m1
            else:
                g2_minus_point[0] = x_m1
                g2_minus_point[1] = y_m1 + c1 / c2 / c3

            x_p1, y_p1 = self.g1_plus_points[anchor_point.name][0], self.g1_plus_points[anchor_point.name][1]
            g2_plus_point = np.zeros(2)
            c4 = - 1 / R * ((x_p1 - x0) ** 2 + (y_p1 - y0) ** 2) ** (3 / 2)
            c5 = 1 - 1 / n2
            c6 = x0 - x_p1
            if (theta2 - np.pi / 2) % np.pi != 0:
                g2_plus_point[0] = x_p1 + c4 / c5 / (c6 * np.tan(theta2) + y_p1 - y0)
                g2_plus_point[1] = np.tan(theta2) * (g2_plus_point[0] - x_p1) + y_p1
            else:
                g2_plus_point[0] = x_p1
                g2_plus_point[1] = y_p1 + c4 / c5 / c6

            self.g2_minus_points[anchor_point.name] = g2_minus_point
            self.g2_plus_points[anchor_point.name] = g2_plus_point

        return g2_minus_point, g2_plus_point

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

    def order_control_points(self):
        """
        ### Description:

        This function creates an array of control points based on the anchor, neighboring, and curvature control
        point dictionaries based on the `string`-based `anchor_point_order`.

        ### Returns:

        The control point array and the length of the control point array (number of control points)
        """
        self.control_points = np.array([])
        for idx, anchor_point in enumerate(self.anchor_point_order):

            if idx == 0:
                self.control_points = np.append(self.control_points, self.anchor_points[anchor_point])
                self.control_points = np.row_stack((self.control_points, self.g1_plus_points[anchor_point]))
            else:
                if anchor_point in self.g2_minus_points:
                    self.control_points = np.row_stack((self.control_points, self.g2_minus_points[anchor_point]))
                self.control_points = np.row_stack((self.control_points, self.g1_minus_points[anchor_point]))
                self.control_points = np.row_stack((self.control_points, self.anchor_points[anchor_point]))
                if anchor_point in self.g1_plus_points:
                    self.control_points = np.row_stack((self.control_points, self.g1_plus_points[anchor_point]))
                if anchor_point in self.g2_plus_points:
                    self.control_points = np.row_stack((self.control_points, self.g2_plus_points[anchor_point]))

            if anchor_point in self.free_points:
                if len(self.free_points[anchor_point]) > 0:
                    for fp_idx in range(len(self.free_points[anchor_point])):
                        self.control_points = \
                            np.row_stack((self.control_points, self.free_points[anchor_point][fp_idx, :]))

        self.n_control_points = len(self.control_points)
        return self.control_points, self.n_control_points

    def add_free_point(self, free_point: FreePoint):
        """
        ### Description:

        Adds a free point (and 2 degrees of freedom) to a given Bézier curve (defined by the `previous_anchor_point`)

        ### Args:

        `free_point`: a `pyairpar.core.free_point.FreePoint` to add to a Bézier curve
        """
        if free_point.previous_anchor_point not in self.free_points.keys():
            self.free_points[free_point.previous_anchor_point] = np.array([])
        if len(self.free_points[free_point.previous_anchor_point]) == 0:
            self.free_points[free_point.previous_anchor_point] = free_point.xy.reshape((1, 2))
        else:
            self.free_points[free_point.previous_anchor_point] = np.vstack(
                (self.free_points[free_point.previous_anchor_point], free_point.xy))

        self.needs_update = True

    def set_bezier_curve_orders(self):
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

    def add_anchor_point(self, anchor_point: AnchorPoint):
        """
        ### Description:

        Adds an anchor point between two anchor points, builds the associated control point branch, and inserts the
        control point branch into the set of control points. `needs_update` is set to `True`.

        ### Args:

        `anchor_point`: an `pyairpar.core.anchor_point.AnchorPoint` from which to build a control point branch
        """
        self.anchor_points[anchor_point.name] = anchor_point.xy
        self.set_slope(anchor_point)
        self.set_curvature(anchor_point)
        self.needs_update = True

    def add_anchor_points(self):
        """
        ### Description:

        This function executes `pyairpar.core.airfoil.Airfoil.add_anchor_point()` for all the anchor points in the
        `anchor_point_tuple`. Enforces leading edge and trailing edge Bézier curve orders.
        `needs_update` is set to `True`.
        """
        for anchor_point in self.anchor_point_tuple:
            self.add_anchor_point(anchor_point)
        self.needs_update = True

    def add_free_points(self):
        """
        ### Description:

        This function executes `pyairpar.core.airfoil.Airfoil.add_free_point()` for all the anchor points in the
        `free_point_tuple`. `needs_update` is set to `True`.
        """
        for free_point in self.free_point_tuple:
            self.add_free_point(free_point)
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

        The `update` function adds first all of the anchor points in the `anchor_point_tuple` and then all of the free
        points in the `free_point_tuple`. The parameter information is extracted. The control points are ordered
        based on the `anchor_point_order`, `name` attributes, and `previous_anchor_point` attributes. The
        `anchor_point_array` is updated based on the anchor points added. Airfoil coordinates are generated and saved.
        Rotation to the specified angle of attack
        and translation by the specified \(\Delta x\), \(\Delta y\) are applied, in that order. The Bézier
        curves are generated through the control points and the airfoil coordinates are then calculated again after
        the transformations.
        """
        self.set_bezier_curve_orders()
        self.add_anchor_points()
        self.add_free_points()
        self.extract_parameters()
        self.order_control_points()
        self.update_anchor_point_array()
        self.generate_non_transformed_airfoil_coordinates()
        self.rotate(-self.alf.value)
        self.translate(self.dx.value, self.dy.value)
        self.generate_airfoil_coordinates()
        self.needs_update = False

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
            C = bezier(P, self.nt)
            self.C.append(C)
            P_start_idx = P_end_idx - 1
        coords = np.array([])
        curvature = np.array([])
        for idx in range(len(self.C)):
            if idx == 0:
                coords = np.column_stack((self.C[idx]['x'], self.C[idx]['y']))
                curvature = np.column_stack((self.C[idx]['x'], self.C[idx]['k']))
            else:
                coords = np.row_stack((coords, np.column_stack((self.C[idx]['x'][1:], self.C[idx]['y'][1:]))))
                curvature = np.row_stack((curvature, np.column_stack((self.C[idx]['x'][1:], self.C[idx]['k'][1:]))))
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
                            axs.plot(C['x'], C['y'], color='cornflowerblue', label='airfoil')
                        else:
                            axs.plot(C['x'], C['y'], color='cornflowerblue')
                    else:
                        axs.plot(C['x'], C['y'], **plot_kwargs[idx])

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
            fig.suptitle(
                fr'Airfoil: $c={self.c.value:.3f}$, $\alpha={np.rad2deg(self.alf.value):.3f}^\circ$')
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


def bezier(P: np.ndarray, nt: int) -> dict:
    """
    ### Description:

    Computes the Bézier curve through the control points `P` according to
    $$\\vec{C}(t)=\\sum_{i=0}^n \\vec{P}_i B_{i,n}(t)$$ where \\(B_{i,n}(t)\\) is the Bernstein polynomial, given by
    $$B_{i,n}={n \\choose i} t^i (1-t)^{n-i}$$

    Also included are first derivative, second derivative, and curvature information. These are given by
    $$\\vec{C}'(t)=n \\sum_{i=0}^{n-1} (\\vec{P}_{i+1} - \\vec{P}_i B_{i,n-1}(t)$$
    $$\\vec{C}''(t)=n(n-1) \\sum_{i=0}^{n-2} (\\vec{P}_{i+2}-2\\vec{P}_{i+1}+\\vec{P}_i) B_{i,n-2}(t)$$
    $$\\kappa(t)=\\frac{C'_x(t) C''_y(t) - C'_y(t) C''_x(t)}{[(C'_x)^2(t) + (C'_y)^2(t)]^{3/2}}$$

    Here, the \\('\\) and \\(''\\) are the first and second derivatives with respect to \\(x\\) and \\(y\\), not the
    parameter \\(t\\). The result of \\(\\vec{C}''(t)\\), for example, is a vector with two components, \\(C''_x(t)\\)
    and \\(C''_y(t)\\).

    An example cubic Bézier curve (order \\(n=3\\)) is shown below. Note that the curve passes through the first and
    last control points and has a local slope at \\(P_0\\) equal to the slope of the line passing through \\(P_0\\)
    and \\(P_1\\). Similarly, the local slope at \\(P_3\\) is equal to the slope of the line passing through
    \\(P_2\\) and \\(P_3\\). These properties of Bézier curves allow us to easily enforce \\(G^0\\) and \\(G^1\\)
    continuity at Bézier curve "joints" (common endpoints of connected Bézier curves).

    .. image:: bezier_curve.png

    ### Args:

    `P`: The control point `np.ndarray` of `shape=(n+1, 2)`, where `n` is the order of the Bézier curve and `n+1` is
    the number of control points in the Bézier curve. The two columns represent the \\(x\\)- and \\(y\\) -components of
    the control points.

    `nt`: number of points in the `t` vector (defines the resolution of the curve)

    ### Returns:

    A dictionary of `numpy` arrays of `shape=nt` containing information related to the created Bézier curve:

    $$C_x(t), C_y(t), C'_x(t), C'_y(t), C''_x(t), C''_y(t), \\kappa(t)$$
    where the \\(x\\) and \\(y\\) subscripts represent the \\(x\\) and \\(y\\) components of the vector-valued functions
    \\(\\vec{C}(t)\\), \\(\\vec{C}'(t)\\), and \\(\\vec{C}''(t)\\).
    """

    def nCr(n_, r):
        """
        ### Description:

        Simple function that computes the mathematical combination $$n \\choose r$$

        ### Args:

        `n_`: `n` written with a trailing underscore to avoid conflation with the order `n` of the Bézier curve

        `r'

        ### Returns

        $$n \\choose r$$
        """
        f = np.math.factorial
        return f(n_) / f(r) / f(n_ - r)

    t = np.linspace(0, 1, nt)
    C = {
        't': t,
        'x': 0,
        'y': 0,
        'px': 0,
        'py': 0,
        'ppx': 0,
        'ppy': 0
    }
    n = len(P)
    for i in range(n):
        # Calculate the x- and y-coordinates of the of the Bezier curve given the input vector t
        C['x'] = C['x'] + P[i, 0] * nCr(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
        C['y'] = C['y'] + P[i, 1] * nCr(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
    for i in range(n - 1):
        # Calculate the first derivatives of the Bezier curve with respect to t, that is C_x'(t) and C_y'(t). Here,
        # C_x'(t) is the x-component of the vector derivative dC(t)/dt, and C_y'(t) is the y-component
        C['px'] = C['px'] + (n - 1) * (P[i + 1, 0] - P[i, 0]) * nCr(n - 2, i) * t ** i * (1 - t) ** (
                n - 2 - i)
        C['py'] = C['py'] + (n - 1) * (P[i + 1, 1] - P[i, 1]) * nCr(n - 2, i) * t ** i * (1 - t) ** (
                n - 2 - i)
    for i in range(n - 2):
        # Calculate the second derivatives of the Bezier curve with respect to t, that is C_x''(t) and C_y''(t). Here,
        # C_x''(t) is the x-component of the vector derivative d^2C(t)/dt^2, and C_y''(t) is the y-component
        C['ppx'] = C['ppx'] + (n - 1) * (n - 2) * (P[i + 2, 0] - 2 * P[i + 1, 0] + P[i, 0]) * nCr(n - 3, i) * t ** (
            i) * (1 - t) ** (n - 3 - i)
        C['ppy'] = C['ppy'] + (n - 1) * (n - 2) * (P[i + 2, 1] - 2 * P[i + 1, 1] + P[i, 1]) * nCr(n - 3, i) * t ** (
            i) * (1 - t) ** (n - 3 - i)

        # Calculate the curvature of the Bezier curve (k = kappa = 1 / R, where R is the radius of curvature)
        C['k'] = (C['px'] * C['ppy'] - C['py'] * C['ppx']) / (C['px'] ** 2 + C['py'] ** 2) ** (3 / 2)

    return C
