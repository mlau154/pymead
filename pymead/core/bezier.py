import numpy as np

from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point
from pymead.utils.nchoosek import nchoosek


class Bezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence, name: str or None = None, t_start: float = None,
                 t_end: float = None, **kwargs):
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

    def get_dict_rep(self):
        return {"points": [pt.name() for pt in self.point_sequence().points()]}


def main():
    points = np.array([
        [0.0, 0.0],
        [0.1, -0.15],
        [0.2, 0.13],
        [0.4, 0.2],
        [0.7, -0.1],
        [0.9, -0.05],
        [1.0, 0.0]
    ])
    original_point_seq = PointSequence.generate_from_array(points)
    bez = Bezier(original_point_seq)
    bez1, bez2 = bez.split(0.7)
    bez_xy = bez.evaluate().xy
    bez1_xy = bez1.evaluate().xy
    bez2_xy = bez2.evaluate().xy

    import matplotlib.pyplot as plt
    plt.plot(bez_xy[:, 0], bez_xy[:, 1], color="steelblue", ls="solid", lw=4)
    plt.plot(bez1_xy[:, 0], bez1_xy[:, 1], color="white", ls="dotted", lw=2)
    plt.plot(bez2_xy[:, 0], bez2_xy[:, 1], color="indianred", ls="dashdot", lw=2)
    plt.plot(original_point_seq.as_array()[:, 0], original_point_seq.as_array()[:, 1], color="grey", ls="dotted", marker="x")
    plt.plot(bez1.point_sequence().as_array()[:, 0], bez1.point_sequence().as_array()[:, 1], color="black", ls="dotted", marker="s")
    plt.plot(bez2.point_sequence().as_array()[:, 0], bez2.point_sequence().as_array()[:, 1], color="gold", ls="dotted", marker="d")
    plt.show()


if __name__ == "__main__":
    main()
