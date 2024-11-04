import typing

import numpy as np
from pymead.core.param import Param, ParamSequence

from pymead.core.point import PointSequence, Point

from pymead.core.parametric_curve import ParametricCurve, PCurveData


class BSpline(ParametricCurve):

    def __init__(self,
                 point_sequence: PointSequence or typing.List[Point],
                 knot_sequence: ParamSequence or typing.List[Param],
                 default_nt: int or None = None,
                 **kwargs):
        """
        Non-uniform rational B-spline (NURBS) curve evaluation class
        """
        super().__init__(sub_container="bspline", **kwargs)
        assert len(knot_sequence) >= len(point_sequence) + 1
        self.degree = len(knot_sequence) - len(point_sequence) - 1
        self.default_nt = default_nt
        self._point_sequence = None
        self._knot_sequence = None
        point_sequence = PointSequence(point_sequence) if isinstance(point_sequence, list) else point_sequence
        knot_sequence = ParamSequence(knot_sequence) if isinstance(knot_sequence, list) else knot_sequence
        self.set_point_sequence(point_sequence)
        self.set_knot_sequence(knot_sequence)
        self.weights = np.ones(len(self.point_sequence()))
        self.possible_spans, self.possible_span_indices = self._get_possible_spans()

    def point_sequence(self):
        return self._point_sequence

    def knot_sequence(self):
        return self._knot_sequence

    def points(self):
        return self.point_sequence().points()

    def knots(self):
        return self.knot_sequence().params()

    def get_control_point_array(self):
        return self.point_sequence().as_array()

    def get_knot_vector(self):
        return self.knot_sequence().as_array()

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence

    def set_knot_sequence(self, knot_sequence: ParamSequence):
        self._knot_sequence = knot_sequence

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def reverse_knot_sequence(self):
        self.knot_sequence().reverse()

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

        if len(self.point_sequence()) > 2:
            delete_curve = False
        else:
            delete_curve = True

        return delete_curve

    def remove(self):
        if self.canvas_item is not None:
            self.canvas_item.sigRemove.emit(self.canvas_item)

    def is_clamped(self) -> bool:
        """
        Determines whether the knot sequence of the B-spline is clamped. A knot sequence
        is clamped when the first :math:`p+1` knots are equal to 0 and the last :math:`p+1` knots
        are equal to 1.

        Returns
        -------
        bool
            Whether the B-spline has a clamped knot sequence
        """
        return all([ti.value() == 0.0 for ti in self.knots()[:self.degree + 1]]) and all(
            [ti.value() == 1.0 for ti in self.knots()[len(self.knot_sequence()) - self.degree - 1:]]
        )

    def hodograph(self) -> "BSpline":
        P = self.get_control_point_array()
        if self.degree <= 1:
            point_sequence = PointSequence.generate_from_array(np.array([[0.0, 0.0]]))
            knot_sequence = ParamSequence.generate_from_array(np.array([0.0]))
            return BSpline(point_sequence, knot_sequence, default_nt=self.default_nt)
        point_sequence = PointSequence.generate_from_array(np.array([
            [
                self.degree / (self.knots()[i + self.degree + 1].value() - self.knots()[i + 1].value()) * (
                        P[i + 1, 0] - P[i, 0]),
                self.degree / (self.knots()[i + self.degree + 1].value() - self.knots()[i + 1].value()) * (
                        P[i + 1, 0] - P[i, 0])
            ] for i in range(len(self.points()) - 1)
        ]))
        knot_sequence = ParamSequence.generate_from_slice(self.knot_sequence(), slice(1, -1))
        return BSpline(point_sequence, knot_sequence, default_nt=self.default_nt)

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

    def evaluate_xy(self, t: np.array or None = None, **kwargs) -> np.ndarray:
        """
        Evaluate the NURBS curve at parameter t
        """
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        xy = np.zeros(shape=(len(t), 2))
        for idx, t_val in enumerate(t):
            B = self._basis_functions(t_val, self.degree)
            point = np.dot(B, self.get_control_point_array())
            xy[idx, 0] = point[0]
            xy[idx, 1] = point[1]
        return xy

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

        xy = np.zeros(shape=(len(t), 2))
        for idx, t_val in enumerate(t):
            B = self._basis_functions(t_val, self.degree)
            point = np.dot(B, self.get_control_point_array())
            xy[idx, 0] = point[0]
            xy[idx, 1] = point[1]

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

    def _get_possible_spans(self) -> (np.ndarray, np.ndarray):
        possible_span_indices = np.array([], dtype=int)
        possible_spans = []
        knot_vector = self.get_knot_vector()
        for knot_idx, (knot_1, knot_2) in enumerate(zip(knot_vector[:-1], knot_vector[1:])):
            if knot_1 == knot_2:
                continue
            possible_span_indices = np.append(possible_span_indices, knot_idx)
            possible_spans.append([knot_1, knot_2])
        return np.array(possible_spans), possible_span_indices

    def _cox_de_boor(self, t: float, i: int, p: int) -> float:
        if p == 0:
            return 1.0 if i in self.possible_span_indices and self._find_span(t) == i else 0.0
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                knot_vector = self.get_knot_vector()
                f = (t - knot_vector[i]) / (knot_vector[i + p] - knot_vector[i])
                g = (knot_vector[i + p + 1] - t) / (knot_vector[i + p + 1] - knot_vector[i + 1])
                if np.isinf(f) or np.isnan(f):
                    f = 0.0
                if np.isinf(g) or np.isnan(g):
                    g = 0.0
                if f == 0.0 and g == 0.0:
                    return 0.0
                elif f != 0.0 and g == 0.0:
                    return f * self._cox_de_boor(t, i, p - 1)
                elif f == 0.0 and g != 0.0:
                    return g * self._cox_de_boor(t, i + 1, p - 1)
                else:
                    return f * self._cox_de_boor(t, i, p - 1) + g * self._cox_de_boor(t, i + 1, p - 1)

    def _basis_functions(self, t: float, p: int):
        """
        Compute the non-zero basis functions at parameter t
        """
        print(f"{len(self.points()) = }, {p = }")
        return np.array([self._cox_de_boor(t, i, p) for i in range(len(self.points()))])

    def _find_span(self, t: float):
        """
        Find the knot span index
        """
        for knot_span, knot_span_idx in zip(self.possible_spans, self.possible_span_indices):
            if knot_span[0] <= t < knot_span[1]:
                return knot_span_idx
        if t == self.possible_spans[-1][1]:
            return self.possible_span_indices[-1]

    def get_dict_rep(self) -> dict:
        return {
            "points": [pt.name() for pt in self.point_sequence().points()],
            "knots": [knot.name() for knot in self.knot_sequence().params()],
            "default_nt": self.default_nt
        }


def main():
    point_sequence = PointSequence.generate_from_array(np.array([
        [0.0, 0.0],
        [0.0, 0.2],
        [0.2, 0.3],
        [0.5, 0.1],
        [0.8, 0.2],
        [1.0, 0.0]
    ]))
    knot_sequence = ParamSequence.generate_from_array(
        np.array([0.0, 0.0, 0.0, 0.0, 1/3, 2/3, 1.0, 1.0, 1.0, 1.0])
    )
    bspline = BSpline(point_sequence, knot_sequence)
    data = bspline.evaluate()
    knot_sequence2 = ParamSequence.generate_from_array(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    bspline2 = BSpline(point_sequence, knot_sequence2)
    data2 = bspline2.evaluate()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    data.plot(ax)
    ax.plot(bspline.get_control_point_array()[:, 0], bspline.get_control_point_array()[:, 1],
            ls=":", color="grey", marker="o")
    data2.plot(ax)
    plt.show()


if __name__ == "__main__":
    main()
