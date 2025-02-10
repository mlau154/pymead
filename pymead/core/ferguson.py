import typing

import numpy as np

from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point


class Ferguson(ParametricCurve):

    def __init__(self, point_sequence: PointSequence or typing.List[Point],
                 default_nt: int or None = None, name: str or None = None,
                 t_start: float = None, t_end: float = None, **kwargs):
        r"""
        Computes the Ferguson curve (see "Multivariable Curve Interpolation" by James Ferguson)
        through the 4 control points ``P``.

        Parameters
        ----------
        point_sequence: PointSequence
            Sequence of points defining the control points for the Ferguson curve. Points 0 and 3 define the starting
            and ending points for the curve, respectively. Point 1 defines the head of the tangent vector at Point 0
            (Point 0 is the tail). Point 2 defines the tail of the tangent vector at Point 3 (point 2 is the head).

        name: str or ``None``
            Optional name for the curve. Default: ``None``

        t_start: float or ``None``
            Optional starting parameter vector value for the Bézier curve. Not specifying this value automatically
            gives a value of ``0.0``. Default: ``None``

        t_end: float or ``None``
            Optional ending parameter vector value for the Bézier curve. Not specifying this value automatically
            gives a value of ``1.0``. Default: ``None``
        """
        super().__init__(sub_container="ferguson", **kwargs)
        self._point_sequence = None
        self.default_nt = default_nt
        point_sequence = PointSequence(point_sequence) if isinstance(point_sequence, list) else point_sequence
        assert len(point_sequence) == 4
        self.set_point_sequence(point_sequence)
        self.t_start = t_start
        self.t_end = t_end
        name = "Ferguson-1" if name is None else name
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

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def point_removal_deletes_curve(self):
        return len(self.point_sequence()) <= 4

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        point_removal_deletes_curve = self.point_removal_deletes_curve()

        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)

        return point_removal_deletes_curve

    def remove(self):
        if self.canvas_item is not None:
            self.canvas_item.sigRemove.emit(self.canvas_item)

    def _get_points_and_tangents(self) -> (Point, Point, Point, Point):
        A = self.point_sequence()[0]
        B = self.point_sequence()[3]
        TA = self.point_sequence()[1] - self.point_sequence()[0]
        TB = self.point_sequence()[3] - self.point_sequence()[2]
        return A, B, TA, TB

    def derivative(self, t: np.ndarray, order: int) -> np.ndarray:
        r"""
        Calculates an arbitrary-order derivative of the Ferguson curve

        Parameters
        ----------
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
        A, B, TA, TB = self._get_points_and_tangents()
        if order == 1:
            return (3.0 * np.outer(t**2, (2.0 * (A - B) + TA + TB).as_array()) +
                    2.0 * np.outer(t, (3.0 * (B - A) - 2.0 * TA - TB).as_array()) + np.outer(np.ones(t.shape), TA.as_array()))
        if order == 2:
            return (6.0 * np.outer(t, (2.0 * (A - B) + TA + TB).as_array()) +
                    2.0 * np.outer(np.ones(t.shape), (3.0 * (B - A) - 2.0 * TA - TB).as_array()))
        if order == 3:
            return 6.0 * np.outer(np.ones(t.shape), (2.0 * (A - B) + TA + TB).as_array())
        if order > 3:
            return np.zeros(shape=(len(t), 2))

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
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # Evaluate the curve
        A, B, TA, TB = self._get_points_and_tangents()
        K0 = A.as_array()
        K1 = TA.as_array()
        K2 = (3.0 * (B - A) - 2.0 * TA - TB).as_array()
        K3 = (2.0 * (A - B) + TA + TB).as_array()
        xy = np.outer(t**3, K3) + np.outer(t**2, K2) + np.outer(t, K1) + np.outer(np.ones(t.shape), K0)

        # Calculate the first derivative
        first_deriv = self.derivative(t=t, order=1)
        xp = first_deriv[:, 0]
        yp = first_deriv[:, 1]

        # Calculate the second derivative
        second_deriv = self.derivative(t=t, order=2)
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

    def evaluate_xy(self, t: np.array or None = None, **kwargs):
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # Evaluate the curve
        A, B, TA, TB = self._get_points_and_tangents()
        K0 = A.as_array()
        K1 = TA.as_array()
        K2 = (3.0 * (B - A) - 2.0 * TA - TB).as_array()
        K3 = (2.0 * (A - B) + TA + TB).as_array()
        xy = np.outer(t**3, K3) + np.outer(t**2, K2) + np.outer(t, K1) + np.outer(np.ones(t.shape), K0)

        return xy

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()], "default_nt": self.default_nt}
