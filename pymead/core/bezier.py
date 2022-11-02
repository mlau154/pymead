from pymead.utils.nchoosek import nchoosek
from pymead.core.parametric_curve import ParametricCurve
import numpy as np


class Bezier(ParametricCurve):
    """Bezier Class"""

    def __init__(self, P, nt: int = 100, t: np.ndarray = None):
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

        self.P = P
        self.n = len(self.P) - 1

        if t is not None:
            if np.min(t) != 0 or np.max(t) != 1:
                raise ValueError('\'t\' array must have a minimum at 0 and a maximum at 1')
            else:
                self.t = t
        else:
            self.t = np.linspace(0, 1, nt)

        n_ctrl_points = len(P)

        self.x, self.y = np.zeros(self.t.shape), np.zeros(self.t.shape)

        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            self.x += P[i, 0] * self.bernstein_poly(self.n, i, self.t)
            self.y += P[i, 1] * self.bernstein_poly(self.n, i, self.t)

        first_deriv = self.derivative(1)
        self.px = first_deriv[:, 0]
        self.py = first_deriv[:, 1]
        second_deriv = self.derivative(2)
        self.ppx = second_deriv[:, 0]
        self.ppy = second_deriv[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature of the Bézier curve (k = kappa = 1 / R, where R is the radius of curvature)
            self.k = np.true_divide((self.px * self.ppy - self.py * self.ppx),
                                    (self.px ** 2 + self.py ** 2) ** (3 / 2))

        with np.errstate(divide='ignore', invalid='ignore'):
            self.R = np.true_divide(1, self.k)

        super().__init__(self.t, self.x, self.y, self.px, self.py, self.ppx, self.ppy, self.k, self.R)

    @staticmethod
    def bernstein_poly(n: int, i: int, t):
        """Calculates the Bernstein polynomial for a given Bézier curve order, index, and parameter vector

        Arguments
        =========
        n: int
            Bézier curve order (one less than the number of control points in the Bézier curve)
        i: int
            Bézier curve index
        t: int, float, or np.ndarray
            Parameter vector for the Bézier curve

        Returns
        =======
        np.ndarray
            Array of values of the Bernstein polynomial evaluated for each point in the parameter vector
        """
        return nchoosek(n, i) * t ** i * (1 - t) ** (n - i)

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
        x_or_y: str
            Choose 'x' to return the 'x'-value of the point, or choose 'y' to return the 'y'-value of the point
        """
        def finite_diff_recursive(_k, _i):
            if _k > 1:
                return finite_diff_recursive(_k - 1, _i + 1) - finite_diff_recursive(_k - 1, _i)
            else:
                return P[_i + 1, :] - P[_i, :]

        return finite_diff_recursive(k, i)

    def derivative(self, order: int):
        n_ctrlpts = len(self.P)
        return np.sum(np.array([np.prod(np.array([self.n - idx for idx in range(order)])) *
                                np.array([self.finite_diff_P(self.P, order, i)]).T *
                                np.array([self.bernstein_poly(self.n - order, i, self.t)])
                                for i in range(n_ctrlpts - order)]), axis=0).T

    @staticmethod
    def approximate_arc_length(P, nt):
        # nchoosek_array = nchoosek_matrix(np.ones(shape=nt) * (len(P) - 1), np.arange(len(P)))
        # xy_array = np.sum(P * nchoosek_array)

        x, y = np.zeros(nt), np.zeros(nt)
        n = len(P)
        t = np.linspace(0, 1, nt)
        for i in range(n):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * nchoosek(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
            y += P[i, 1] * nchoosek(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
        return np.sum(np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]))

    def update(self, P, nt, t: np.ndarray = None):
        self.P = P
        self.n = len(self.P) - 1

        if t is not None:
            if np.min(t) != 0 or np.max(t) != 1:
                raise ValueError('\'t\' array must have a minimum at 0 and a maximum at 1')
            else:
                self.t = t
        else:
            self.t = np.linspace(0, 1, nt)

        n_ctrl_points = len(P)

        self.x, self.y = np.zeros(self.t.shape), np.zeros(self.t.shape)

        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            self.x += P[i, 0] * self.bernstein_poly(self.n, i, self.t)
            self.y += P[i, 1] * self.bernstein_poly(self.n, i, self.t)

        first_deriv = self.derivative(1)
        self.px = first_deriv[:, 0]
        self.py = first_deriv[:, 1]
        second_deriv = self.derivative(2)
        self.ppx = second_deriv[:, 0]
        self.ppy = second_deriv[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate the curvature of the Bézier curve (k = kappa = 1 / R, where R is the radius of curvature)
            self.k = np.true_divide((self.px * self.ppy - self.py * self.ppx),
                                    (self.px ** 2 + self.py ** 2) ** (3 / 2))

        with np.errstate(divide='ignore', invalid='ignore'):
            self.R = np.true_divide(1, self.k)

    def get_curvature_comb(self, max_k_normalized_scale_factor, interval: int = 1):
        comb_heads_x = self.x - self.py / np.sqrt(self.px**2 + self.py**2) * self.k * max_k_normalized_scale_factor
        comb_heads_y = self.y + self.px / np.sqrt(self.px**2 + self.py**2) * self.k * max_k_normalized_scale_factor
        # Stack the x and y columns (except for the last x and y values) horizontally and keep only the rows by the
        # specified interval:
        self.comb_tails = np.column_stack((self.x, self.y))[:-1:interval, :]
        self.comb_heads = np.column_stack((comb_heads_x, comb_heads_y))[:-1:interval, :]
        # Add the last x and y values onto the end (to make sure they do not get skipped with input interval)
        self.comb_tails = np.row_stack((self.comb_tails, np.array([self.x[-1], self.y[-1]])))
        self.comb_heads = np.row_stack((self.comb_heads, np.array([comb_heads_x[-1], comb_heads_y[-1]])))

    def plot_control_point_skeleton(self, *plt_args, **plt_kwargs):
        pass
