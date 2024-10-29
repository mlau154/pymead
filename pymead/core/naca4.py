import numpy as np

from pymead.core.parametric_curve import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, Point
from pymead.core.param import Param


class NACA4(ParametricCurve):

    def __init__(self, max_camber: Param, max_camber_loc: Param, max_thickness: Param, leading_edge: Point,
                 trailing_edge: Point, upper: bool,
                 default_nt: int or None = None, name: str or None = None,
                 cosine_spacing: bool = True,
                 sharp_trailing_edge: bool = False,
                 t_start: float = None, t_end: float = None, **kwargs):
        r"""
        Computes the profile for either the lower or upper surface of a NACA 4-series airfoil.

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
        super().__init__(sub_container="naca4", **kwargs)
        self.a = [0.29690, -0.12600, -0.35160, 0.28430, -0.1036 if sharp_trailing_edge else -0.10150]
        self.sharp_trailing_edge = sharp_trailing_edge
        self.t0 = 0.2
        self._point_sequence = None
        self.max_camber = max_camber
        self.max_camber_loc = max_camber_loc
        self.max_thickness = max_thickness
        self.leading_edge = leading_edge
        self.trailing_edge = trailing_edge
        self.upper = upper
        self.cosine_spacing = cosine_spacing
        self.default_nt = default_nt
        point_sequence = PointSequence(points=[self.leading_edge, self.trailing_edge])
        self.set_point_sequence(point_sequence)
        self.t_start = t_start
        self.t_end = t_end
        name = "NACA4-1" if name is None else name
        self.set_name(name)
        self.curve_connections = []
        self._add_references()

    def get_4_digit_designation(self) -> int:
        digit_1 = int(round(100 * self.max_camber.value(), 0))
        digit_2 = int(round(10 * self.max_camber_loc.value(), 0))
        digits_34 = int(round(100 * self.max_thickness.value(), 0))
        return digit_1 * 1000 + digit_2 * 100 + digits_34

    def get_4_digit_name(self) -> str:
        return str(self.get_4_digit_designation())

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

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence

    def point_removal_deletes_curve(self):
        return len(self.point_sequence()) <= 2

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

    def compute_camber(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the camber distribution of the NACA 4-series airfoil.

        Parameters
        ----------
        x: np.ndarray
            The pre-computed distribution of :math:`x`-values

        Returns
        -------
        np.ndarray
            Array of the same size as ``x`` containing the camber distribution.
        """
        m, p = self.max_camber.value(), self.max_camber_loc.value()
        yc = np.zeros(x.shape)
        for idx in range(len(x)):
            if x[idx] < p:
                yc[idx] = m / p**2 * (2 * p * x[idx] - x[idx]**2)
            else:
                yc[idx] = m / (1 - p)**2 * ((1 - 2*p) + 2*p*x[idx] - x[idx]**2)
        return yc

    @staticmethod
    def _compute_camber_gradient(x: np.ndarray, m: float, p: float, order: int) -> np.ndarray:
        """
        Computes the gradient of the camber distribution with respect to :math:`x`.

        Parameters
        ----------
        x: np.ndarray
            The pre-computed distribution of :math:`x`-values
        m: float
            Maximum camber
        p: float
            Maximum camber location
        order: int
            The order of the derivative. If the order is not 1 or 2, an error is raised.

        Returns
        -------
        np.ndarray
            An array of the same size as ``x`` containing the values of the camber gradient at each value of :math:`x`.
        """
        if order == 1:
            dyc_dt = np.zeros(x.shape)
            for idx in range(len(x)):
                if x[idx] < p:
                    dyc_dt[idx] = 2 * m / p**2 * (p - x[idx])
                else:
                    dyc_dt[idx] = 2 * m / (1 - p)**2 * (p - x[idx])
            return dyc_dt
        elif order == 2:
            d2yc_dt2 = np.zeros(x.shape)
            for idx in range(len(x)):
                if x[idx] < p:
                    d2yc_dt2[idx] = -2 * m / p ** 2
                else:
                    d2yc_dt2[idx] = -2 * m / (1 - p) ** 2
            return d2yc_dt2
        raise ValueError(f"Invalid camber gradient order {order}. Must be either 1 or 2.")

    @staticmethod
    def _compute_camber_angle_gradient(x: np.ndarray, m: float, p: float, order: int) -> np.ndarray:
        """
        Computes the gradient of the camber angle distribution with respect to :math:`x`.

        Parameters
        ----------
        x: np.ndarray
            The pre-computed distribution of :math:`x`-values
        m: float
            Maximum camber
        p: float
            Maximum camber location
        order: int
            The order of the derivative. If the order is not 1 or 2, an error is raised.

        Returns
        -------
        np.ndarray
            An array of the same size as ``x`` containing the values of the camber angle gradient at each value of
            :math:`x`.
        """
        if order == 1:
            dtheta_dt = np.zeros(x.shape)
            for idx in range(len(x)):
                if x[idx] < p:
                    # dtheta_dt[idx] = 2 * p * (p - x[idx]) / (m * (2 * p * x[idx] - x[idx]**2)**2 + 1)
                    dtheta_dt[idx] = -2 * m * p**2 / (4 * m**2 * (p - x[idx])**2 + p**4)
                else:
                    # dtheta_dt[idx] = 2 * (1 - p) * (p - x[idx]) / (m * (1 - 2 * p + 2 * p * x[idx] - x[idx]**2)**2 + 1)
                    dtheta_dt[idx] = -2 * m / (1 - p)**2 / (4 * m**2 * (p - x[idx])**2 / (1 - p)**4 + 1)
            return dtheta_dt
        elif order == 2:
            d2theta_dt2 = np.zeros(x.shape)
            for idx in range(len(x)):
                if x[idx] < p:
                    # k1 = m * x[idx] * (8 * p**3 - 16 * p**2 * x[idx] + 12 * p * x[idx]**2 - 3 * x[idx]**3) + 1
                    # k2 = m * x[idx]**2 * (x[idx] - 2 * p)**2 + 1
                    # d2theta_dt2[idx] = 2 * p * k1 / k2**2
                    d2theta_dt2[idx] = -16 * m**3 * p**2 * (p - x[idx]) / (4 * m**2 * (p - x[idx])**2 + p**4)**2
                else:
                    # k1 = m * (2 * p * x[idx] - 2 * p - x[idx]**2 + 1)**2 + 1
                    # k2 = 4 * m * (1 - p) * (2 * p - 2 * x[idx])
                    # k3 = (p - x[idx]) * (2 * p * x[idx] - 2 * p - x[idx]**2 + 1)
                    # k4 = (m * (2 * p * x[idx] - 2 * p - x[idx]**2 + 1)**2 + 1)**2
                    # d2theta_dt2[idx] = -2 * (1 - p) / k1 - (k2 * k3) / k4
                    d2theta_dt2[idx] = -16 * m**3 * (p - x[idx]) / (1 - p)**6 / (4 * m**2 * (p - x[idx])**2 / (1 - p)**4 + 1)**2
            return d2theta_dt2
        raise ValueError(f"Invalid camber angle gradient order {order}. Must be either 1 or 2.")

    def derivative(self, theta: np.ndarray, x: np.ndarray, yt: np.ndarray, order: int) -> np.ndarray:
        r"""
        Calculates an arbitrary-order derivative of the Bézier curve

        Parameters
        ----------
        theta: np.ndarray
            The pre-computed camber angle distribution
        x: np.ndarray
            The pre-computed distribution of :math:`x`-values
        yt: np.ndarray
            The pre-computed thickness distribution
        order: int
            The order of the derivative. The only valid orders are ``1`` and ``2``.

        Returns
        -------
        np.ndarray
            An array of ``shape=(N,2)`` where ``N`` is the number of evaluated points specified by the :math:`x` vector.
            The columns represent :math:`C^{(m)}_x(t)` and :math:`C^{(m)}_y(t)`, where :math:`m` is the
            derivative order.
        """
        # NACA 4-series parameters
        m = self.max_camber.value()
        p = self.max_camber_loc.value()
        t_max = self.max_thickness.value()
        t0 = self.t0
        a = self.a

        error = NotImplementedError(
            "Derivatives of order greater than 2 are not implemented for NACA 4-series airfoil curves"
        )

        # Compute the thickness, camber, and camber angle first-order derivatives
        with np.errstate(divide="ignore", invalid="ignore"):
            dyt_dt = t_max / t0 * (0.5 * a[0] * np.true_divide(
                1, x ** 0.5) + a[1] + 2 * a[2] * x + 3 * a[3] * x ** 2 + 4 * a[4] * x ** 3)
        dyc_dt = self._compute_camber_gradient(x, m, p, order=1)
        dtheta_dt = self._compute_camber_angle_gradient(x, m, p, order=1)

        # Compute the first- and second-order parametric derivatives for either the upper or lower surface
        if self.upper:
            if order == 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    dxu_dt = 1 - dyt_dt * np.sin(theta) - yt * dtheta_dt * np.cos(theta)
                dyu_dt = dyc_dt + dyt_dt * np.cos(theta) - yt * dtheta_dt * np.sin(theta)
                return np.column_stack((dxu_dt, dyu_dt))
            elif order == 2:
                with np.errstate(divide="ignore", invalid="ignore"):
                    d2yt_dt2 = t_max / t0 * (-0.25 * a[0] * np.true_divide(
                        1, x**1.5) + 2 * a[2] + 6 * a[3] * x + 12 * a[4] * x**2)
                d2yc_dt2 = self._compute_camber_gradient(x, m, p, order=order)
                d2theta_dt2 = self._compute_camber_angle_gradient(x, m, p, order=order)
                with np.errstate(divide="ignore", invalid="ignore"):
                    d2xu_dt2 = (-d2yt_dt2 * np.sin(theta) - 2 * dyt_dt * dtheta_dt * np.cos(theta) -
                                yt * d2theta_dt2 * np.cos(theta) + yt * dtheta_dt**2 * np.sin(theta))
                    d2yu_dt2 = (d2yc_dt2 + d2yt_dt2 * np.cos(theta) - 2 * dyt_dt * dtheta_dt * np.sin(theta) -
                                yt * d2theta_dt2 * np.sin(theta) - yt * dtheta_dt**2 * np.cos(theta))
                return np.column_stack((d2xu_dt2, d2yu_dt2))
            else:
                raise error
        else:
            if order == 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    dxl_dt = 1 + dyt_dt * np.sin(theta) + yt * dtheta_dt * np.cos(theta)
                dyl_dt = dyc_dt - dyt_dt * np.cos(theta) + yt * dtheta_dt * np.sin(theta)
                return np.column_stack((dxl_dt, dyl_dt))
            elif order == 2:
                with np.errstate(divide="ignore", invalid="ignore"):
                    d2yt_dt2 = t_max / t0 * (-0.25 * a[0] * np.true_divide(
                        1, x ** 1.5) + 2 * a[2] + 6 * a[3] * x + 12 * a[4] * x ** 2)
                d2yc_dt2 = self._compute_camber_gradient(x, m, p, order=order)
                d2theta_dt2 = self._compute_camber_angle_gradient(x, m, p, order=order)
                with np.errstate(divide="ignore", invalid="ignore"):
                    d2xl_dt2 = (d2yt_dt2 * np.sin(theta) + 2 * dyt_dt * dtheta_dt * np.cos(theta) +
                                yt * d2theta_dt2 * np.cos(theta) - yt * dtheta_dt ** 2 * np.sin(theta))
                    d2yl_dt2 = (d2yc_dt2 - d2yt_dt2 * np.cos(theta) + 2 * dyt_dt * dtheta_dt * np.sin(theta) +
                                yt * d2theta_dt2 * np.sin(theta) + yt * dtheta_dt ** 2 * np.cos(theta))
                return np.column_stack((d2xl_dt2, d2yl_dt2))
            else:
                raise error

    def evaluate(self, t: np.array or None = None, **kwargs):
        r"""
        Evaluates the curve using an optionally specified parameter vector.

        Parameters
        ----------
        t: np.ndarray or ``None``
            Optional direct specification of the parameter vector for the curve. Not specifying this value
            gives a vector from ``t_start`` or ``t_end`` with the default size and with spacing determined by the value
            of ``cosine_spacing`` specified in the class constructor.
            Default: ``None``

        Returns
        -------
        PCurveData
            Data class specifying the following information about the NACA 4-series airfoil curve:

            .. math::

                    C_x(t), C_y(t), C'_x(t), C'_y(t), C''_x(t), C''_y(t), \kappa(t)

            where the :math:`x` and :math:`y` subscripts represent the :math:`x` and :math:`y` components of the
            vector-valued functions :math:`\vec{C}(t)`, :math:`\vec{C}'(t)`, and :math:`\vec{C}''(t)`.
        """
        # Generate the parameter vector
        if self.default_nt is not None:
            kwargs["nt"] = self.default_nt
        if self.cosine_spacing:
            kwargs["spacing"] = "cosine"
        t = ParametricCurve.generate_t_vec(**kwargs) if t is None else t

        # NACA 4-series parameters
        m = self.max_camber.value()
        p = self.max_camber_loc.value()
        t_max = self.max_thickness.value()
        t0 = self.t0
        a = self.a

        # Compute x-locations, camber, and thickness
        x = t
        yc = self.compute_camber(x)
        yt = t_max / t0 * (a[0] * np.sqrt(x) + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4)

        # Compute theta
        dyc_dx = self._compute_camber_gradient(x, m, p, order=1)
        theta = np.arctan2(dyc_dx, 1)
        # if self.upper:
        #     import matplotlib.pyplot as plt
        #     plt.plot(x, theta)
        #     plt.show()

        # Evaluate the curve
        if self.upper:
            xu = x - yt * np.sin(theta)
            yu = yc + yt * np.cos(theta)
            xy = np.column_stack((xu, yu))
        else:
            xl = x + yt * np.sin(theta)
            yl = yc - yt * np.cos(theta)
            xy = np.column_stack((xl, yl))

        # Calculate the first derivative
        first_deriv = self.derivative(theta, x, yt, order=1)
        xp = first_deriv[:, 0]
        yp = first_deriv[:, 1]

        # Calculate the second derivative
        second_deriv = self.derivative(theta, x, yt, order=2)
        xpp = second_deriv[:, 0]
        ypp = second_deriv[:, 1]

        # Combine the derivative x and y data
        xpyp = np.column_stack((xp, yp))
        xppypp = np.column_stack((xpp, ypp))

        # Calculate the curvature
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature (k = kappa = 1 / R, where R is the radius of curvature)
            k = np.abs(np.true_divide((xp * ypp - yp * xpp), (xp ** 2 + yp ** 2) ** (3 / 2)))
            if x[0] == 0.0:
                k[0] = 1 / (1.1090 * t_max**2)

        # Calculate the radius of curvature: R = 1 / kappa
        with np.errstate(divide='ignore', invalid='ignore'):
            R = np.true_divide(1, k)

        # dx/dy at t=0 (x=0):
        dxdy_start = np.array([np.cos(theta[0]), np.sin(theta[0])])

        return PCurveData(t=t, xy=xy, xpyp=xpyp, xppypp=xppypp, k=k, R=R, dxdy_start=dxdy_start)

    def get_dict_rep(self):
        return {
            "max_camber": self.max_camber.name(),
            "max_camber_loc": self.max_camber_loc.name(),
            "max_thickness": self.max_thickness.name(),
            "leading_edge": self.leading_edge.name(),
            "trailing_edge": self.trailing_edge.name(),
            "upper": self.upper,
            "cosine_spacing": self.cosine_spacing,
            "default_nt": self.default_nt
        }


def main():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    for upper, k_factor in zip([True, False], [-0.005, 0.005]):
        naca4 = NACA4(
            max_camber=Param(0.05, "NACA4-1.m"),
            max_camber_loc=Param(0.5, "NACA4-1.p"),
            max_thickness=Param(0.12, "NACA4-1.t"),
            leading_edge=Point.generate_from_array(np.array([0.0, 0.0])),
            trailing_edge=Point.generate_from_array(np.array([1.0, 0.0])),
            upper=upper,
            sharp_trailing_edge=False,
            default_nt=200,
            cosine_spacing=True
        )
        data = naca4.evaluate()
        ax.plot(data.xy[:, 0], data.xy[:, 1], color="steelblue")
        comb_tails, comb_heads = data.get_curvature_comb(k_factor, flip_leading_edge_normal=not upper)
        for comb_tail, comb_head in zip(comb_tails, comb_heads):
            ax.plot([comb_tail[0], comb_head[0]], [comb_tail[1], comb_head[1]], color="indianred")
        if upper:
            x = np.linspace(0.0, 1.0, 200)
            ax.plot(x, naca4.compute_camber(x), color="mediumaquamarine")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
