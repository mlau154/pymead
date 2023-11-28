import numpy as np
import matplotlib.pyplot as plt

from pymead.core.param2 import ParamCollection
from pymead.core.parametric_curve2 import ParametricCurve, PCurveData
from pymead.core.point import PointSequence, SurfPointSequence, Point
from pymead.utils.nchoosek import nchoosek


class Bezier(ParametricCurve):

    def __init__(self, point_sequence: PointSequence, *args, **kwargs):
        self._point_sequence = None
        self.gui_obj = None
        self.set_point_sequence(point_sequence)
        for point in self.point_sequence().points():
            if self not in point.curves:
                point.curves.append(self)
        super().__init__(*args, **kwargs)

    def point_sequence(self):
        return self._point_sequence

    def set_point_sequence(self, point_sequence: PointSequence):
        self._point_sequence = point_sequence

    def insert_point(self, idx: int, point: Point):
        self.point_sequence().insert_point(idx, point)

    def remove_point(self, idx: int or None = None, point: Point or None = None):
        if isinstance(point, Point):
            idx = self.point_sequence().point_idx_from_ref(point)
        self.point_sequence().remove_point(idx)

        if len(self.point_sequence()) > 1:
            delete_curve = False
        else:
            delete_curve = True

        return delete_curve

    def remove(self):
        if self.gui_obj is not None:
            self.gui_obj.sigRemove.emit(self.gui_obj)

    def update(self):
        p_curve_data = self.evaluate()
        if self.gui_obj is not None:
            self.gui_obj.updateGUIObj(curve_data=p_curve_data)

    @staticmethod
    def bernstein_poly(n: int, i: int, t: int or float or np.ndarray):
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
    fig, ax = plt.subplots(1, 3)
    _P = np.array([[0.0, 0.0], [0.3, 0.2], [0.5, -0.1], [0.7, -0.05], [0.9, 0.15], [1.0, 0.0]])
    _point_seq = PointSequence.generate_from_array(_P)
    bez = Bezier(point_sequence=_point_seq)
    data = bez.evaluate(nt=150)
    data.plot(ax[0], color="steelblue")
    ax[0].plot(_P[:, 0], _P[:, 1], ls="--", marker="o", color="gray", mfc="indianred", mec="gray")
    bez.insert_point(2, point=Point.generate_from_array(np.array([0.5, 0.5])))
    new_data = bez.evaluate(nt=150)
    new_data.plot(ax[1], color="steelblue")
    new_P = bez.point_sequence().as_array()
    ax[1].plot(new_P[:, 0], new_P[:, 1], ls="--", marker="o", color="gray", mfc="indianred", mec="gray")
    bez.remove_point(2)
    remove_data = bez.evaluate(nt=150)
    remove_data.plot(ax[2], color="steelblue")
    newest_P = bez.point_sequence().as_array()
    ax[2].plot(newest_P[:, 0], newest_P[:, 1], ls="--", marker="o", color="gray", mfc="indianred", mec="gray")
    plt.show()
