import numpy as np
import matplotlib.pyplot as plt

from pymead.core.param2 import ParamCollection
from pymead.core.parametric_curve2 import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, SurfPointSequence
from pymead.utils.nchoosek import nchoosek


class Bezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence):
        self._point_sequence = None
        self.set_point_sequence(point_sequence)

    def point_sequence(self):
        return self._point_sequence

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence

    @staticmethod
    def bernstein_poly(n: int, i: int, t):
        """Calculates the Bernstein polynomial for a given Bézier curve order, index, and parameter vector

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
        t = t.as_array().flatten() if isinstance(t, ParamCollection) else t
        return nchoosek(n, i) * t ** i * (1.0 - t) ** (n - i)

    def evaluate(self, t: ParamCollection or None = None, **kwargs):
        t = ParametricCurve.generate_t_collection(**kwargs) if t is None else t
        n_ctrl_points = len(self.point_sequence())
        degree = n_ctrl_points - 1
        P = self.point_sequence().as_array()
        x, y = np.zeros(t.shape), np.zeros(t.shape)
        for i in range(n_ctrl_points):
            # Calculate the x- and y-coordinates of the Bézier curve given the input vector t
            x += P[i, 0] * self.bernstein_poly(degree, i, t.as_array().flatten())
            y += P[i, 1] * self.bernstein_poly(degree, i, t.as_array().flatten())
        xy = SurfPointSequence.generate_from_array(np.column_stack((x, y)))
        return PCurveData(t=t, xy=xy)


if __name__ == "__main__":
    _P = np.array([[0.0, 0.0], [0.3, 0.2], [0.5, -0.1], [0.7, -0.05], [0.9, 0.15], [1.0, 0.0]])
    _point_seq = PointSequence.generate_from_array(_P)
    bez = Bezier(point_sequence=_point_seq)
    data = bez.evaluate(nt=150)
    fig, ax = plt.subplots()
    data.plot(ax, color="steelblue")
    plt.show()
