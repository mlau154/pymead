import numpy as np

from pymead.core.param2 import ParamCollection, Param
from pymead.core.point import PointSequence, SurfPointSequence, Point
from pymead.core.parametric_curve2 import ParametricCurve, PCurveData


class LineSegment(ParametricCurve):

    def __init__(self, point_sequence: PointSequence, name: str or None = None, **kwargs):
        self._point_sequence = None
        self.geo_col = None
        self.set_point_sequence(point_sequence)
        name = "Line" if name is None else name
        self._add_references()
        super().__init__(name=name, **kwargs)

    def _add_references(self):
        for idx, point in enumerate(self.point_sequence().points()):
            # Add the object reference to each point in the curve
            if self not in point.curves:
                point.curves.append(self)

    def set_name(self, name: str):
        # Rename the reference in the geometry collection
        if self.geo_col is not None and self.name() in self.geo_col.container()["lines"]:
            self.geo_col.container()["lines"][name] = self.geo_col.container()["lines"][self.name()]
            self.geo_col.container()["lines"].pop(self.name())

        self._name = name

    def point_sequence(self):
        return self._point_sequence

    def set_point_sequence(self, point_sequence: PointSequence):
        if len(point_sequence) != 2:
            raise ValueError("Point sequence must contain exactly two points")
        self._point_sequence = point_sequence

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

    def evaluate(self, t: ParamCollection or None = None, **kwargs):
        if "nt" not in kwargs.keys() and t is None:
            kwargs["nt"] = 2  # Set the default parameter vector for the line to be [0.0, 1.0]
        t = ParametricCurve.generate_t_collection(**kwargs) if t is None else t
        p1 = self.point_sequence().points()[0]
        p2 = self.point_sequence().points()[1]
        x1 = p1.x().value()
        y1 = p1.y().value()
        x2 = p2.x().value()
        y2 = p2.y().value()
        theta = np.arctan2(y2 - y1, x2 - x1)
        r = np.hypot(x2 - x1, y2 - y1)
        x = x1 + t.as_array().flatten() * r * np.cos(theta)
        y = y1 + t.as_array().flatten() * r * np.sin(theta)
        xy_arr = np.column_stack((x, y))
        surf_points = SurfPointSequence.generate_from_array(xy_arr)
        return PCurveData(t=t, xy=surf_points)
