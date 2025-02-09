import typing

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from rust_nurbs import *

from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point
from pymead.post.fonts_and_colors import font
from pymead.post.plot_formatters import format_axis_scientific
from pymead.utils.nchoosek import nchoosek


class Bezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence or typing.List[Point],
                 default_nt: int or None = None, name: str or None = None,
                 t_start: float = None, t_end: float = None, **kwargs):
        r"""
        Computes the Bézier curve through the control points :math:`\mathbf{P}_i` according to

        .. math::

            \mathbf{C}(t)=\sum_{i=0}^n \mathbf{P}_i B_{i,n}(t)

        where :math:`B_{i,n}(t)` is the Bernstein polynomial, given by

        .. math::

            B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

        .. _cubic-bezier:
        .. figure:: ../images/cubic_bezier_light.*
            :class: only-light
            :width: 600
            :align: center

            Cubic Bézier curve

        .. figure:: ../images/cubic_bezier_dark.*
            :class: only-dark
            :width: 600
            :align: center

            Cubic Bézier curve

        An example cubic Bézier curve (degree :math:`n=3`) is shown above. Note that the curve passes
        through the first and last control points and has a local slope at :math:`P_0` equal to the slope of the
        line passing through :math:`P_0` and :math:`P_1`. Similarly, the local slope at :math:`P_3` is equal to
        the slope of the line passing through :math:`P_2` and :math:`P_3`. These properties of Bézier curves allow us to
        easily enforce :math:`G^0` and :math:`G^1` continuity at Bézier curve "joints" (common endpoints of
        connected Bézier curves).

        Parameters
        ----------
        point_sequence: PointSequence
            Sequence of points defining the control points for the Bézier curve

        name: str or ``None``
            Optional name for the curve. Default: ``None``

        t_start: float or ``None``
            Optional starting parameter vector value for the Bézier curve. Not specifying this value automatically
            gives a value of ``0.0``. Default: ``None``

        t_end: float or ``None``
            Optional ending parameter vector value for the Bézier curve. Not specifying this value automatically
            gives a value of ``1.0``. Default: ``None``
        """
        super().__init__(sub_container="bezier", **kwargs)
        self._point_sequence = None
        self.degree = None
        self.default_nt = default_nt
        point_sequence = PointSequence(point_sequence) if isinstance(point_sequence, list) else point_sequence
        self.set_point_sequence(point_sequence)
        self.t_start = t_start
        self.t_end = t_end
        name = "Bezier-1" if name is None else name
        self.set_name(name)
        self.curve_connections = []
        self._add_references()

    def _add_references(self):
        for idx, point in enumerate(self.point_sequence().points()):
            # If any curves are found at the start point, add their pointers as curve connections
            if idx == 0:
                for curve in point.curves:
                    if not curve.reference:  # Do not include reference curves
                        self.curve_connections.append(curve)

            # If any other curves are found at the end point, add their pointers as curve connections
            elif idx == len(self.point_sequence()) - 1:
                for curve in point.curves:
                    if not curve.reference:  # Do not include reference curves
                        self.curve_connections.append(curve)

            # Add the object reference to each point in the curve
            if self not in point.curves:
                point.curves.append(self)

    def point_sequence(self):
        return self._point_sequence

    def points(self):
        return self.point_sequence().points()

    def get_control_point_array(self):
        return self.point_sequence().as_array()

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence
        self.degree = len(point_sequence) - 1

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def insert_point(self, idx: int, point: Point):
        self.point_sequence().insert_point(idx, point)
        self.degree += 1
        if self not in point.curves:
            point.curves.append(self)
        if self.canvas_item is not None:
            self.canvas_item.point_items.insert(idx, point.canvas_item)
            self.canvas_item.updateCurveItem(self.evaluate())

    def insert_point_after_point(self, point_to_add: Point, preceding_point: Point):
        idx = self.point_sequence().point_idx_from_ref(preceding_point) + 1
        self.insert_point(idx, point_to_add)

    def point_removal_deletes_curve(self):
        return len(self.point_sequence()) <= 3

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)
        self.degree -= 1

        if len(self.point_sequence()) > 2:
            delete_curve = False
        else:
            delete_curve = True

        return delete_curve

    def remove(self):
        if self.canvas_item is not None:
            self.canvas_item.sigRemove.emit(self.canvas_item)

    @staticmethod
    def bernstein_poly(n: int, i: int, t: int or float or np.ndarray):
        r"""
        Calculates the Bernstein polynomial for a given Bézier curve order, index, and parameter vector. The
        Bernstein polynomial is described by

        .. math::

            B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

        Arguments
        =========
        n: int
            Bézier curve degree (one less than the number of control points in the Bézier curve)
        i: int
            Bézier curve index
        t: int, float, or np.ndarray
            Parameter vector for the Bézier curve

        Returns
        =======
        np.ndarray
            Array of values of the Bernstein polynomial evaluated for each point in the parameter vector
        """
        return nchoosek(n, i) * t ** i * (1.0 - t) ** (n - i)

    @staticmethod
    def finite_diff_P(P: np.ndarray, k: int, i: int):
        """Calculates the finite difference of the control points as shown in
        https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html

        Arguments
        =========
        P: np.ndarray
            Array of control points for the Bézier curve
        k: int
            Finite difference level (e.g., k = 1 is the first derivative finite difference)
        i: int
            An index referencing a location in the control point array
        """

        def finite_diff_recursive(_k, _i):
            if _k > 1:
                return finite_diff_recursive(_k - 1, _i + 1) - finite_diff_recursive(_k - 1, _i)
            else:
                return P[_i + 1, :] - P[_i, :]

        return finite_diff_recursive(k, i)

    def hodograph(self) -> "Bezier":
        """
        Generates another ``Bezier`` object representing the derivative ("hodograph") of the original curve

        Returns
        -------
        Bezier
            Hodograph of the curve
        """
        P = self.get_control_point_array()
        if len(P) <= 1:
            point_sequence = PointSequence.generate_from_array(np.array([[0.0, 0.0]]))
            return Bezier(point_sequence, default_nt=self.default_nt)
        point_sequence = PointSequence.generate_from_array(np.array([
            [self.degree * (p1[0] - p0[0]), self.degree * (p1[1] - p0[1])] for p0, p1 in zip(P[:-1, :], P[1:, :])
        ]))
        return Bezier(point_sequence, default_nt=self.default_nt)

    def derivative(self, t: np.ndarray, order: int):
        r"""
        Calculates an arbitrary-order derivative of the Bézier curve.

        Parameters
        ==========
        t: np.ndarray
            The parameter vector
        order: int
            The derivative order. For example, ``order=2`` returns the second derivative.

        Returns
        =======
        np.ndarray
            An array of ``shape=(N,2)`` where ``N`` is the number of evaluated points specified by the :math:`t` vector.
            The columns represent :math:`C^{(m)}_x(t)` and :math:`C^{(m)}_y(t)`, where :math:`m` is the
            derivative order.
        """
        assert order >= 0
        curve = self
        for n in range(order):
            curve = curve.hodograph()
        return curve.evaluate_xy(t)

    def evaluate_xy(self, t: np.ndarray or None = None, **kwargs) -> np.ndarray:
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # Evaluate the curve
        return np.array(bezier_curve_eval_tvec(self.point_sequence().as_array(), t))

    def evaluate(self, t: np.array or None = None, **kwargs) -> PCurveData:
        r"""
        Evaluates the curve using an optionally specified parameter vector. Also included are first derivative,
        second derivative, and curvature information. These are given by

        .. math::

            \mathbf{C}'(t)=n \sum_{i=0}^{n-1} (\mathbf{P}_{i+1} - \mathbf{P}_i) B_{i,n-1}(t)

        .. math::

            \mathbf{C}''(t)=n(n-1) \sum_{i=0}^{n-2} (\mathbf{P}_{i+2}-2\mathbf{P}_{i+1}+\mathbf{P}_i) B_{i,n-2}(t)

        .. math::

            \kappa(t)=\frac{C'_x(t) C''_y(t) - C'_y(t) C''_x(t)}{[(C'_x)^2(t) + (C'_y)^2(t)]^{3/2}}

        Here, the :math:`'` and :math:`''` superscripts are the first and second derivatives with respect to
        :math:`x` and :math:`y`, not the parameter :math:`t`. The result of :math:`\vec{C}''(t)`, for example,
        is a vector with two components, :math:`C''_x(t)` and :math:`C''_y(t)`.

        Parameters
        ----------
        t: np.ndarray or ``None``
            Optional direct specification of the parameter vector for the curve. Not specifying this value
            gives a linearly spaced parameter vector from ``t_start`` or ``t_end`` with the default size.
            Default: ``None``
        kwargs
            Additional keyword arguments to pass to ``ParametricCurve.generate_t_vec``

        Returns
        -------
        PCurveData
            Data class specifying the following information about the Bézier curve:

            .. math::

                    C_x(t), C_y(t), C'_x(t), C'_y(t), C''_x(t), C''_y(t), \kappa(t)

            where the :math:`x` and :math:`y` subscripts represent the :math:`x` and :math:`y` components of the
            vector-valued functions :math:`\mathbf{C}(t)`, :math:`\mathbf{C}'(t)`, and :math:`\mathbf{C}''(t)`.
        """
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # Number of control points, curve degree, control point array
        P = self.point_sequence().as_array()

        # Evaluate the curve
        xy = np.array(bezier_curve_eval_tvec(P, t))

        # Calculate the first derivative
        xpyp = np.array(bezier_curve_dcdt_tvec(P, t))
        xp, yp = xpyp[:, 0], xpyp[:, 1]

        # Calculate the second derivative
        xppypp = np.array(bezier_curve_d2cdt2_tvec(P, t))
        xpp, ypp = xppypp[:, 0], xppypp[:, 1]

        # Combine the derivative x and y data
        xpyp = np.column_stack((xp, yp))
        xppypp = np.column_stack((xpp, ypp))

        # Calculate the curvature
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature of the Bézier curve (k = kappa = 1 / R, where R is the radius of curvature)
            k = np.true_divide((xp * ypp - yp * xpp), (xp ** 2 + yp ** 2) ** (3 / 2))

        # Calculate the radius of curvature: R = 1 / kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.true_divide(1, k)

        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def compute_t_corresponding_to_x(self, x_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_xy(t[0])[0]
            return np.array([point[0] - x_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def compute_t_corresponding_to_y(self, y_seek: float, t0: float = 0.5):
        def bez_root_find_func(t):
            point = self.evaluate_xy(t[0])[0]
            return np.array([point[1] - y_seek])

        return fsolve(bez_root_find_func, x0=np.array([t0]))[0]

    def split(self, t_split: float):

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.point_sequence())
        degree = n_ctrl_points - 1
        P = self.point_sequence().as_array()

        def de_casteljau(i: int, j: int) -> np.ndarray:
            """
            Based on https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm. Recursive algorithm where the
            base case is just the value of the ith original control point.

            Parameters
            ----------
            i: int
                Lower index
            j: int
                Upper index

            Returns
            -------
            np.ndarray
                A one-dimensional array containing the :math:`x` and :math:`y` values of a control point evaluated
                at :math:`(i,j)` for a Bézier curve split at the parameter value ``t_split``
            """
            if j == 0:
                return P[i, :]
            return de_casteljau(i, j - 1) * (1 - t_split) + de_casteljau(i + 1, j - 1) * t_split

        bez_split_1_P = np.array([de_casteljau(i=0, j=i) for i in range(n_ctrl_points)])
        bez_split_2_P = np.array([de_casteljau(i=i, j=degree - i) for i in range(n_ctrl_points)])

        if self.geo_col is None:
            bez_1_points = [self.point_sequence().points()[0]] + [Point(*xy.tolist()) for xy in bez_split_1_P[1:, :]]
            bez_2_points = [bez_1_points[-1]] + [Point(*xy.tolist()) for xy in bez_split_2_P[1:-1, :]] + [
                self.point_sequence().points()[-1]]
        else:
            bez_1_points = [self.point_sequence().points()[0]] + [
                self.geo_col.add_point(*xy.tolist()) for xy in bez_split_1_P[1:, :]]
            bez_2_points = [bez_1_points[-1]] + [
                self.geo_col.add_point(*xy.tolist()) for xy in bez_split_2_P[1:-1, :]] + [
                self.point_sequence().points()[-1]]

        bez_1_point_seq = PointSequence(bez_1_points)
        bez_2_point_seq = PointSequence(bez_2_points)

        if self.geo_col is None:
            return (
                Bezier(point_sequence=bez_1_point_seq),
                Bezier(point_sequence=bez_2_point_seq)
            )
        else:
            for point in self.point_sequence().points()[1:-1]:
                self.geo_col.remove_pymead_obj(point)
            return (
                self.geo_col.add_bezier(point_sequence=bez_1_point_seq, name="BezSplit"),
                self.geo_col.add_bezier(point_sequence=bez_2_point_seq, name="BezSplit")
            )

    def plot(self, ax: plt.Axes or None = None, nt: int = 100,
             show: bool = True, save_file: str or None = None, **plt_kwargs):
        """
        Plots the airfoil to a ``matplotlib`` figure.

        Parameters
        ----------
        ax: plt.Axes or None
            Matplotlib Axes object on which the curve will be plotted. If specified, this method will only.
            If ``None``, a new figure will be created. Default: ``None``
        nt: int
            Number of parametric values to evaluate along the curve. Default: 100
        show: bool
            Whether to immediately show the curve plot. Ignored if ``ax`` is not ``None``. Default: ``True``
        save_file: str or None
            Name of the file to save. If ``None``, the curve image will not be saved to file.
            Ignored if ``ax`` is not ``None``. Default: ``None``
        plt_kwargs
            Additional keyword arguments to pass to ``matplotlib.pyplot.plot``
        """
        ax_specified = ax is not None
        if ax_specified:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=(10, 2))

        # Plot the curves
        curve_data = self.evaluate_xy(np.linspace(0.0, 1.0, nt))
        ax.plot(curve_data[:, 0], curve_data[:, 1], **plt_kwargs)

        if ax_specified:
            return

        # Plot settings
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontdict=font)
        ax.set_ylabel("y", fontdict=font)
        format_axis_scientific(ax=ax)

        # Save and/or show
        if save_file is not None:
            fig.savefig(save_file, bbox_inches="tight")
        if show:
            plt.show()

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()], "default_nt": self.default_nt}
