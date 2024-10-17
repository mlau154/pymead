import typing

import matplotlib.pyplot as plt
import numpy as np

from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.param import Param
from pymead.core.point import PointSequence, Point


class Wagner(ParametricCurve):

    def __init__(self, A: typing.List[Param], point_sequence: PointSequence or typing.List[Point],
                 default_nt: int or None = None, name: str or None = None,
                 t_start: float = None, t_end: float = None, **kwargs):
        r"""
        Computes a Wagner curve by the coefficients.

        Parameters
        ----------
        point_sequence: PointSequence
            Sequence of points defining the transformation for the curve. The first point defines the origin of
            the local coordinate system, and the position of the second point relative to the first point defines
            the scale and rotation angle.

        name: str or ``None``
            Optional name for the curve. Default: ``None``

        t_start: float or ``None``
            Optional starting parameter vector value for the Wagner curve. Not specifying this value automatically
            gives a value of ``0.0``. Default: ``None``

        t_end: float or ``None``
            Optional ending parameter vector value for the Wagner curve. Not specifying this value automatically
            gives a value of ``1.0``. Default: ``None``
        """
        super().__init__(sub_container="ferguson", **kwargs)
        self._point_sequence = None
        self.default_nt = default_nt
        self.A = A
        point_sequence = PointSequence(point_sequence) if isinstance(point_sequence, list) else point_sequence
        assert len(point_sequence) == 2
        assert len(A) > 0
        self.set_point_sequence(point_sequence)
        self.t_start = t_start
        self.t_end = t_end
        name = "Wagner-1" if name is None else name
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

    def get_A_as_array(self):
        return np.array([a.value() for a in self.A])

    def get_control_point_array(self):
        return self.point_sequence().as_array()

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence

    def reverse_point_sequence(self):
        self.point_sequence().reverse()

    def point_removal_deletes_curve(self):
        return len(self.point_sequence()) <= 2

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        point_removal_deletes_curve = self.point_removal_deletes_curve()

        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)

        return point_removal_deletes_curve

    def remove(self):
        if self.canvas_item is not None:
            self.canvas_item.sigRemove.emit(self.canvas_item)

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

        A = self.get_A_as_array()
        theta = np.pi * t

        def dx_dt(p: int):
            K1 = int((p - 1) % 2 == 0) * (-1) ** int((p - 1) % 4 != 0)
            K2 = int(p % 2 == 0) * (-1) ** int(p % 4 == 0)
            print(f"{K1 = }, {K2 = }")
            return np.pi ** p / 2 * (K1 * np.sin(np.pi * t) + K2 * np.cos(np.pi * t))

        def dy_dt(p: int):
            out = -A[0] * dx_dt(p)
            for n in range(len(A)):
                K1 = (n + 1) ** (p - 1) * int(p % 2 == 0) * (-1) ** int(p % 4 != 0)
                K2 = (n + 1) ** (p - 1) * int((p - 1) % 2 == 0) * (-1) ** int((p - 1) % 4 != 0)
                K3 = n ** (p - 1) * int(p % 2 == 0) * (-1) ** int(p % 4 != 0)
                K4 = n ** (p - 1) * int((p - 1) % 2 == 0) * (-1) ** int((p - 1) % 4 != 0)
                print(f"{K1 = }, {K2 = }, {K3 = }, {K4 = }")
                out += np.pi ** (p - 1) * A[n] * (K1 * np.sin((n + 1) * theta) + K2 * np.cos((n + 1) * theta) +
                                                  K3 * np.sin(n * theta) + K4 * np.cos(n * theta))
            return out

        return np.column_stack((dx_dt(order), dy_dt(order)))

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
        A = self.get_A_as_array()
        theta = np.pi * t
        x = 0.5 * (1 - np.cos(theta))
        y = -A[0] * np.sin(0.5 * theta) ** 2 + A[0] / np.pi * (theta + np.sin(theta))
        for A_idx, A_val in enumerate(A[1:]):
            n = A_idx + 1
            y += A_val / np.pi * (np.sin((n + 1) * theta) / (n + 1) + np.sin(n * theta) / n)
        xy = np.column_stack((x, y))

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

        # Use curvature with x-parametrization instead of t at t=1 because of the zero value of xp and yp
        # for t_idx, t_val in enumerate(t):
        #     if not np.isclose(t_val, 1.0):
        #         continue
        #     k[t_idx] = xppypp[t_idx, 1] / xppypp[t_idx, 0] / (1 + (-A[0])**2) ** 1.5

        # Calculate the radius of curvature: R = 1 / kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.true_divide(1, k)

        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R)

    def get_dict_rep(self):
        return {
            "A": [param.name() for param in self.A],
            "points": [pt.name() for pt in self.point_sequence().points()],
            "default_nt": self.default_nt
        }


def main():
    # # xy = wagner_contour(np.array([0.10148, 0.019233, 0.0044033, 0.008108]))
    # A = np.array([0.071049, 0.011098, 0.0051307])
    # xy = wagner_contour(A)
    # xy2 = wagner_contour(-A)
    # n_theta = 150
    # dxdydt = wagner_contour_deriv(A, n_theta=n_theta)
    # plt.plot(xy[:, 0], xy[:, 1], color="indianred")
    #
    # for i in range(n_theta):
    #     tail = xy[i, :]
    #     head = tail - 0.03 * dxdydt[i, 1:] / np.linalg.norm(dxdydt[i, 1:])
    #     print(f"{tail = }, {head = }")
    #     plt.plot([tail[0], head[0]], [tail[1], head[1]], color="steelblue")
    #
    # plt.show()

    wagner = Wagner([Param(0.10148, name="A0"), Param(0.019233, name="A1"), Param(0.0044033, name="A2"),
                     Param(0.008108, name="A3")],
                    PointSequence.generate_from_array(np.array([[0.0, 0.0], [1.0, 0.0]])))
    data = wagner.evaluate(nt=200)

    print(f"Leading edge radius: {data.R[0]}")
    print(f"Trailing edge radius: {data.R[-1]}")
    print(f"Leading edge data: {data.xpyp[0] = }, {data.xppypp[0] = }")
    print(f"Middle data: {data.xpyp[75], data.xppypp[75] = }")
    print(f"Trailing edge data: {data.xpyp[-1] = }, {data.xppypp[-1] = }")
    plt.plot(data.xy[:, 0], data.xy[:, 1], color="steelblue")
    comb_tails, comb_heads = data.get_curvature_comb(0.0005)
    for comb_tail, comb_head in zip(comb_tails, comb_heads):
        plt.plot([comb_tail[0], comb_head[0]], [comb_tail[1], comb_head[1]], color="indianred")

    wagner = Wagner([Param(-0.10148, name="A0"), Param(-0.019233, name="A1"), Param(-0.0044033, name="A2"),
                     Param(-0.008108, name="A3")],
                    PointSequence.generate_from_array(np.array([[0.0, 0.0], [1.0, 0.0]])))
    data = wagner.evaluate()
    plt.plot(data.xy[:, 0], data.xy[:, 1], color="mediumaquamarine")
    comb_tails, comb_heads = data.get_curvature_comb(0.0005)
    for comb_tail, comb_head in zip(comb_tails, comb_heads):
        plt.plot([comb_tail[0], comb_head[0]], [comb_tail[1], comb_head[1]], color="gold", ls=":")

    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
