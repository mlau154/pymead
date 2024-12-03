import typing

import numpy as np

from pymead.core.param import ParamSequence, Param
from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point
from pymead.utils.nchoosek import nchoosek


class RBezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence or typing.List[Point],
                 weight_sequence: ParamSequence or typing.List[Param],
                 default_nt: int or None = None, name: str or None = None,
                 **kwargs):
        r"""
        Creates a rational Bézier curve parametrized by the control points :math:`\mathbf{P}_i` and weights :math:`w_i`
        according to

        .. math::

            \mathbf{C}(t)=\frac{\sum_{i=0}^n B_{i,n}(t) w_i \mathbf{P}_i}{\sum_{i=0}^n B_{i,n}(t) w_i}

        where :math:`B_{i,n}(t)` is the Bernstein polynomial, given by

        .. math::

            B_{i,n}(t)={n \choose i} t^i (1-t)^{n-i}

        The weights have the effect of "pulling" the curve toward their corresponding control points when their values
        are increased.

        Parameters
        ----------
        point_sequence: PointSequence
            Sequence of points defining the control points for the Bézier curve
        name: str or ``None``
            Optional name for the curve. Default: ``None``
        t_start: float or ``None``
            Optional starting parameter vector value for the rational Bézier curve. Not specifying this value
            automatically gives a value of ``0.0``. Default: ``None``
        t_end: float or ``None``
            Optional ending parameter vector value for the rational Bézier curve. Not specifying this value
            automatically gives a value of ``1.0``. Default: ``None``
        """
        super().__init__(sub_container="bezier", **kwargs)
        self._point_sequence = None
        self._weight_sequence = None
        self.degree = None
        self.default_nt = default_nt
        point_sequence = PointSequence(point_sequence) if isinstance(point_sequence, list) else point_sequence
        weight_sequence = ParamSequence(weight_sequence) if isinstance(weight_sequence, list) else weight_sequence
        self.set_point_sequence(point_sequence)
        self.set_weight_sequence(weight_sequence)
        name = "RBezier-1" if name is None else name
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

    def weight_sequence(self):
        return self._weight_sequence

    def points(self):
        return self.point_sequence().points()

    def weights(self):
        return self.weight_sequence().params()

    def get_control_point_array(self):
        return self.point_sequence().as_array()

    def get_weight_vector(self):
        return self.weight_sequence().as_array()

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence
        self.degree = len(point_sequence) - 1

    def set_weight_sequence(self, weight_sequence: ParamSequence):
        self._weight_sequence = weight_sequence

    def reverse_point_sequence(self):
        self.point_sequence().reverse()
        self.weight_sequence().reverse()

    def reverse_weight_sequence(self):
        self.reverse_point_sequence()

    def insert_point(self, idx: int, point: Point, weight: Param):
        self.point_sequence().insert_point(idx, point)
        self.weight_sequence().insert_param(idx, weight)
        self.degree += 1
        if self not in point.curves:
            point.curves.append(self)
        if self.canvas_item is not None:
            self.canvas_item.point_items.insert(idx, point.canvas_item)
            self.canvas_item.updateCurveItem(self.evaluate())

    def insert_point_after_point(self, point_to_add: Point, preceding_point: Point, weight: Param):
        idx = self.point_sequence().point_idx_from_ref(preceding_point) + 1
        self.insert_point(idx, point_to_add, weight)

    def point_removal_deletes_curve(self):
        return len(self.point_sequence()) <= 3

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)
        self.weight_sequence().remove_param(idx)
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

    def _evaluate_denominator(self, t: np.ndarray):
        w = self.get_weight_vector()
        return np.sum(
            np.array([w[i] * self.bernstein_poly(self.degree, i, t) for i in range(len(self.points()))]), axis=0
        )

    def hodograph(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluates the hodograph of the rational Bézier curve using a specified parameter vector. Note that unlike
        in the case of the non-rational Bézier, the hodograph is not itself a rational Bézier curve. Therefore,
        only an array of :math:`x`- and :math:`y`-values is returned

        t: np.ndarray
            Parameter vector along which the hodograph will be evaluated

        Returns
        -------
        np.ndarray
            Evaluated hodograph of the curve, dimensions :math:`N_t \times 2`, where :math:`N_t` is the length
            of the input parameter vector, and columns represent the first derivative with respect to :math:`x` and
            :math:`y`, respectively.
        """
        P = self.get_control_point_array()
        w = self.get_weight_vector()
        if len(P) <= 1:
            return np.zeros((len(t), 2))
        D2 = self._evaluate_denominator(t) ** 2
        hodo = np.zeros((len(t), 2))
        for k in range(0, 2 * self.degree - 1):
            print(f"{k = }")
            Rk = np.array([0.0, 0.0])
            # print(f"{int(np.floor(k * 0.5)) + 1 = }")
            for i in range(max(0, k - self.degree + 1), int(np.floor(k * 0.5)) + 1):
                print(f"{i = }")
                print(f"{P[i, :] = }")
                print(f"{k - i + 1 = }")
                print(f"{P[k - i + 1, :] = }")
                Rk += (k - 2 * i + 1) * nchoosek(self.degree, i) * nchoosek(
                    self.degree, k - i + 1) * w[i] * w[k - i + 1] * (P[k - i + 1, :] - P[i, :])
            Rk /= nchoosek(2 * self.degree - 2, k)
            print(f"{Rk.shape = }")
            hodo += np.outer(self.bernstein_poly(2 * self.degree - 2, k, t), Rk)
            print(f"{hodo.shape = }")
        print("Made it here")
        return hodo / np.column_stack((D2, D2))

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
        if order == 0:
            return self.evaluate_xy(t)
        if order == 1:
            return self.hodograph(t)
        return np.ones((len(t), 2))

    def evaluate_xy(self, t: np.ndarray or None = None, **kwargs) -> np.ndarray:
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # Number of control points, curve degree, control point array
        n_ctrl_points = len(self.point_sequence())
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()
        w = self.get_weight_vector()

        # Evaluate the curve
        x, y = np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += w[i] * P[i, 0] * self.bernstein_poly(degree, i, t)
            y += w[i] * P[i, 1] * self.bernstein_poly(degree, i, t)
        D = self._evaluate_denominator(t)
        return np.column_stack((x, y)) / np.column_stack((D, D))

    def evaluate(self, t: np.array or None = None, **kwargs):
        r"""
        Evaluates the curve using an optionally specified parameter vector.

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
        n_ctrl_points = len(self.point_sequence())
        degree = n_ctrl_points - 1
        P = self.get_control_point_array()
        w = self.get_weight_vector()

        # Evaluate the curve
        x, y = np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += w[i] * P[i, 0] * self.bernstein_poly(degree, i, t)
            y += w[i] * P[i, 1] * self.bernstein_poly(degree, i, t)
        D = self._evaluate_denominator(t)
        xy = np.column_stack((x, y)) / np.column_stack((D, D))

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

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()], "default_nt": self.default_nt}


def main():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    points = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    weights = np.array([1.0, 1 / np.sqrt(2.0), 1.0])
    rbez = RBezier(PointSequence.generate_from_array(points), ParamSequence.generate_from_array(weights),
                   default_nt=151)
    hodo = rbez.hodograph(np.linspace(0.0, 1.0, rbez.default_nt))
    data = rbez.evaluate()
    data.plot(ax)
    x_circ = np.cos(2 * np.pi * np.linspace(0.0, 0.25, 25))
    y_circ = np.sin(2 * np.pi * np.linspace(0.0, 0.25, 25))
    ax.plot(x_circ, y_circ, ls="none", color="black", marker="o")
    ax.plot(hodo[:, 0], hodo[:, 1])
    # ax.plot((hodo[:, 1] / hodo[:, 0])[5:-5])
    tails, heads = data.get_curvature_comb(0.05)
    for tail, head in zip(tails, heads):
        ax.plot([tail[0], head[0]], [tail[1], head[1]], color="steelblue")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
