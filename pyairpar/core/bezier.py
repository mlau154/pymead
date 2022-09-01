from pyairpar.utils.nchoosek import nchoosek
from pyairpar.core.parametric_curve import ParametricCurve
import numpy as np


class Bezier(ParametricCurve):

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

        .. image:: bezier_curve.png

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

        t = self.t

        n = len(P)

        self.x, self.y, self.px, self.py, self.ppx, self.ppy = np.zeros(self.t.shape), np.zeros(self.t.shape), \
                                                               np.zeros(self.t.shape), np.zeros(self.t.shape), \
                                                               np.zeros(self.t.shape), np.zeros(self.t.shape)

        for i in range(n):
            # Calculate the x- and y-coordinates of the of the Bezier curve given the input vector t
            self.x += P[i, 0] * nchoosek(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
            self.y += P[i, 1] * nchoosek(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
        for i in range(n - 1):
            # Calculate the first derivatives of the Bezier curve with respect to t, that is C_x'(t) and C_y'(t). Here,
            # C_x'(t) is the x-component of the vector derivative dC(t)/dt, and C_y'(t) is the y-component
            self.px += (n - 1) * (P[i + 1, 0] - P[i, 0]) * nchoosek(n - 2, i) * t ** i * (1 - t) ** (n - 2 - i)
            self.py += (n - 1) * (P[i + 1, 1] - P[i, 1]) * nchoosek(n - 2, i) * t ** i * (1 - t) ** (n - 2 - i)
        for i in range(n - 2):
            # Calculate the second derivatives of the Bezier curve with respect to t, that is C_x''(t) and C_y''(t).
            # Here, C_x''(t) is the x-component of the vector derivative d^2C(t)/dt^2, and C_y''(t) is the y-component.
            self.ppx += (n - 1) * (n - 2) * (P[i + 2, 0] - 2 * P[i + 1, 0] + P[i, 0]) * nchoosek(n - 3, i) * t ** (
                           i) * (1 - t) ** (n - 3 - i)
            self.ppy += (n - 1) * (n - 2) * (P[i + 2, 1] - 2 * P[i + 1, 1] + P[i, 1]) * nchoosek(n - 3, i) * t ** (
                           i) * (1 - t) ** (n - 3 - i)

            # Calculate the curvature of the Bezier curve (k = kappa = 1 / R, where R is the radius of curvature)
            self.k = (self.px * self.ppy - self.py * self.ppx) / (self.px ** 2 + self.py ** 2) ** (3 / 2)

        with np.errstate(divide='ignore'):
            self.R = np.true_divide(1, self.k)

        super().__init__(self.t, self.x, self.y, self.px, self.py, self.ppx, self.ppy, self.k, self.R)
        # print('Finished initialization of Bezier curve.')

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

        t = self.t

        n = len(P)

        self.x, self.y, self.px, self.py, self.ppx, self.ppy = np.zeros(self.t.shape), np.zeros(self.t.shape), \
                                                               np.zeros(self.t.shape), np.zeros(self.t.shape), \
                                                               np.zeros(self.t.shape), np.zeros(self.t.shape)

        for i in range(n):
            # Calculate the x- and y-coordinates of the of the Bezier curve given the input vector t
            self.x += P[i, 0] * nchoosek(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
            self.y += P[i, 1] * nchoosek(n - 1, i) * t ** i * (1 - t) ** (n - 1 - i)
        for i in range(n - 1):
            # Calculate the first derivatives of the Bezier curve with respect to t, that is C_x'(t) and C_y'(t). Here,
            # C_x'(t) is the x-component of the vector derivative dC(t)/dt, and C_y'(t) is the y-component
            self.px += (n - 1) * (P[i + 1, 0] - P[i, 0]) * nchoosek(n - 2, i) * t ** i * (1 - t) ** (n - 2 - i)
            self.py += (n - 1) * (P[i + 1, 1] - P[i, 1]) * nchoosek(n - 2, i) * t ** i * (1 - t) ** (n - 2 - i)
        for i in range(n - 2):
            # Calculate the second derivatives of the Bezier curve with respect to t, that is C_x''(t) and C_y''(t).
            # Here, C_x''(t) is the x-component of the vector derivative d^2C(t)/dt^2, and C_y''(t) is the y-component.
            self.ppx += (n - 1) * (n - 2) * (P[i + 2, 0] - 2 * P[i + 1, 0] + P[i, 0]) * nchoosek(n - 3, i) * t ** (
                i) * (1 - t) ** (n - 3 - i)
            self.ppy += (n - 1) * (n - 2) * (P[i + 2, 1] - 2 * P[i + 1, 1] + P[i, 1]) * nchoosek(n - 3, i) * t ** (
                i) * (1 - t) ** (n - 3 - i)

            # Calculate the curvature of the Bezier curve (k = kappa = 1 / R, where R is the radius of curvature)
            self.k = (self.px * self.ppy - self.py * self.ppx) / (self.px ** 2 + self.py ** 2) ** (3 / 2)

        with np.errstate(divide='ignore'):
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
