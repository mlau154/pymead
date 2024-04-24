import numpy as np

from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point
from pymead.utils.nchoosek import nchoosek


class Bezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence, name: str or None = None, t_start: float = None,
                 t_end: float = None, t_start_point: Point = None, t_end_point: Point = None, **kwargs):
        r"""
        Computes the Bézier curve through the control points ``P`` according to

        .. math::

            \vec{C}(t)=\sum_{i=0}^n \vec{P}_i B_{i,n}(t)

        where :math:`B_{i,n}(t)` is the Bernstein polynomial, given by

        .. math::

            B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

        Also included are first derivative, second derivative, and curvature information. These are given by

        .. math::

            \vec{C}'(t)=n \sum_{i=0}^{n-1} (\vec{P}_{i+1} - \vec{P}_i B_{i,n-1}(t)

        .. math::

            \vec{C}''(t)=n(n-1) \sum_{i=0}^{n-2} (\vec{P}_{i+2}-2\vec{P}_{i+1}+\vec{P}_i) B_{i,n-2}(t)

        .. math::

            \kappa(t)=\frac{C'_x(t) C''_y(t) - C'_y(t) C''_x(t)}{[(C'_x)^2(t) + (C'_y)^2(t)]^{3/2}}

        Here, the :math:`'` and :math:`''` superscripts are the first and second derivatives with respect to
        :math:`x` and :math:`y`, not the parameter :math:`t`. The result of :math:`\vec{C}''(t)`, for example,
        is a vector with two components, :math:`C''_x(t)` and :math:`C''_y(t)`.

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

        An example cubic Bézier curve (degree :math:`n=3`) is shown in :numref:`cubic-bezier`. Note that the curve passes
        through the first and last control points and has a local slope at :math:`P_0` equal to the slope of the
        line passing through :math:`P_0` and :math:`P_1`. Similarly, the local slope at :math:`P_3` is equal to
        the slope of the line passing through :math:`P_2` and :math:`P_3`. These properties of Bézier curves allow us to
        easily enforce :math:`G^0` and :math:`G^1` continuity at Bézier curve "joints" (common endpoints of
        connected Bézier curves).

        Parameters
        ==========
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
        self.set_point_sequence(point_sequence)
        self.t_start = t_start
        self.t_end = t_end
        self.t_start_point = t_start_point
        self.t_end_point = t_end_point
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

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence
        self.degree = len(point_sequence) - 1

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def insert_point(self, idx: int, point: Point):
        self.point_sequence().insert_point(idx, point)
        self.degree += 1

    def insert_point_after_point(self, point_to_add: Point, preceding_point: Point):
        idx = self.point_sequence().point_idx_from_ref(preceding_point) + 1
        self.insert_point(idx, point_to_add)
        if self not in point_to_add.curves:
            point_to_add.curves.append(self)
        if self.canvas_item is not None:
            self.canvas_item.point_items.insert(idx, point_to_add.canvas_item)
            self.canvas_item.updateCurveItem(self.evaluate())

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
        """
        Calculates the Bernstein polynomial for a given Bézier curve order, index, and parameter vector

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

    def derivative(self, P: np.ndarray, t: np.ndarray, degree: int, order: int):
        r"""
        Calculates an arbitrary-order derivative of the Bézier curve

        Parameters
        ==========
        P: np.ndarray
            The control point array

        t: np.ndarray
            The parameter vector

        degree: int
            The degree of the Bézier curve

        order: int
            The derivative order. For example, ``order=2`` returns the second derivative.

        Returns
        =======
        np.ndarray
            An array of ``shape=(N,2)`` where ``N`` is the number of evaluated points specified by the :math:`t` vector.
            The columns represent :math:`C^{(m)}_x(t)` and :math:`C^{(m)}_y(t)`, where :math:`m` is the
            derivative order.
        """
        return np.sum(np.array([np.prod(np.array([degree - idx for idx in range(order)])) *
                                np.array([self.finite_diff_P(P, order, i)]).T *
                                np.array([self.bernstein_poly(degree - order, i, t)])
                                for i in range(degree + 1 - order)]), axis=0).T

    def evaluate(self, t: np.array or None = None, **kwargs):
        r"""
        Evaluates the curve using an optionally specified parameter vector.

        Parameters
        ==========
        t: np.ndarray or ``None``
            Optional direct specification of the parameter vector for the curve. Not specifying this value
            gives a linearly spaced parameter vector from ``t_start`` or ``t_end`` with the default size.
            Default: ``None``

        Returns
        =======
        PCurveData
            Data class specifying the following information about the Bézier curve:

            .. math::

                    C_x(t), C_y(t), C'_x(t), C'_y(t), C''_x(t), C''_y(t), \kappa(t)

            where the :math:`x` and :math:`y` subscripts represent the :math:`x` and :math:`y` components of the
            vector-valued functions :math:`\vec{C}(t)`, :math:`\vec{C}'(t)`, and :math:`\vec{C}''(t)`.
        """
        # Pass the starting and ending parameter vector values to the parameter vector generator if they were
        # specified directly
        if self.t_start is not None:
            kwargs["start"] = self.t_start
        if self.t_end is not None:
            kwargs["end"] = self.t_end

        # Generate the parameter vector
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.point_sequence())
        degree = n_ctrl_points - 1
        P = self.point_sequence().as_array()

        # Evaluate the curve
        x, y = np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * self.bernstein_poly(degree, i, t)
            y += P[i, 1] * self.bernstein_poly(degree, i, t)
        xy = np.column_stack((x, y))

        # Calculate the first derivative
        first_deriv = self.derivative(P=P, t=t, degree=degree, order=1)
        xp = first_deriv[:, 0]
        yp = first_deriv[:, 1]

        # Calculate the second derivative
        second_deriv = self.derivative(P=P, t=t, degree=degree, order=2)
        xpp = second_deriv[:, 0]
        ypp = second_deriv[:, 1]

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

    def update_t_start(self):
        if self.t_start_point is not None:
            self.geo_col.remove_pymead_obj(self.t_start_point)
            self.t_start_point = None

        if self.t_start is None or self.t_start == 0.0:
            return

        P_curve_data = self.evaluate(np.array([self.t_start]))
        self.t_start_point = self.geo_col.add_point(
            x=P_curve_data.xy[0, 0], y=P_curve_data.xy[0, 1], name="BezierStartPoint"
        )
        self.t_start_point.curves.append(self)

    def update_t_end(self):

        if self.t_end is None:
            return

        if self.t_end == 1.0 and self.t_end_point is not None:
            self.geo_col.remove_pymead_obj(self.t_end_point)
            self.t_end_point = None
            return

        P_curve_data = self.evaluate(np.array([self.t_end]))

        if self.t_end_point is None:
            self.t_end_point = self.geo_col.add_point(
                x=P_curve_data.xy[0, 0], y=P_curve_data.xy[0, 1], name="BezierEndPoint"
            )
            self.t_end_point.curves.append(self)
        else:
            self.t_end_point.request_move(xp=P_curve_data.xy[0, 0], yp=P_curve_data.xy[0, 1], force=True,
                                          update_curves=False)

    def update(self):
        super().update()
        self.update_t_start()
        self.update_t_end()

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()],
                "t_start": self.t_start, "t_end": self.t_end,
                "t_start_point": self.t_start_point.name() if self.t_start_point is not None else None,
                "t_end_point": self.t_end_point.name() if self.t_end_point is not None else None
                }
