import numpy as np

from pymead.core.parametric_curve2 import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point
from pymead.utils.nchoosek import nchoosek


class Bezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence, name: str or None = None, **kwargs):
        super().__init__(sub_container="bezier", **kwargs)
        self._point_sequence = None
        self.set_point_sequence(point_sequence)
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

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def insert_point(self, idx: int, point: Point):
        self.point_sequence().insert_point(idx, point)

    def point_removal_deletes_curve(self):
        return len(self.point_sequence()) <= 3

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)

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
        """
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
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t
        n_ctrl_points = len(self.point_sequence())
        degree = n_ctrl_points - 1
        P = self.point_sequence().as_array()
        x, y = np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * self.bernstein_poly(degree, i, t)
            y += P[i, 1] * self.bernstein_poly(degree, i, t)
        xy = np.column_stack((x, y))

        first_deriv = self.derivative(P=P, t=t, degree=degree, order=1)
        xp = first_deriv[:, 0]
        yp = first_deriv[:, 1]
        second_deriv = self.derivative(P=P, t=t, degree=degree, order=2)
        xpp = second_deriv[:, 0]
        ypp = second_deriv[:, 1]
        xpyp = np.column_stack((xp, yp))
        xppypp = np.column_stack((xpp, ypp))

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature of the Bézier curve (k = kappa = 1 / R, where R is the radius of curvature)
            k = np.true_divide((xp * ypp - yp * xpp), (xp ** 2 + yp ** 2) ** (3 / 2))

        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.true_divide(1, k)

        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)
