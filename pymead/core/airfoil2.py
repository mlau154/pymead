from pymead.core.point import Point


class Airfoil:
    def __init__(self, leading_edge: Point, trailing_edge: Point,
                 upper_surf_end: Point, lower_surf_end: Point, name: str or None = None):

        # Point inputs
        self.leading_edge = leading_edge
        self.trailing_edge = trailing_edge
        self.upper_surf_end = upper_surf_end
        self.lower_surf_end = lower_surf_end

        # References
        self.geo_col = None
        self.tree_item = None

        # Name the airfoil
        name = "Airfoil-1" if name is None else name
        self._name = None
        self.set_name(name)

        # Properties to set during closure check
        self.upper_te_curve = None
        self.lower_te_curve = None
        self.curves = []
        self.curves_to_reverse = []

        # Check if the curves in the curve list form a single closed loop
        self.check_closed()

    def name(self):
        return self._name

    def set_name(self, name: str):
        # Rename the reference in the geometry collection
        if self.geo_col is not None and self.name() in self.geo_col.container()["airfoils"].keys():
            self.geo_col.container()["airfoils"][name] = self.geo_col.container()["airfoils"][self.name()]
            self.geo_col.container()["airfoils"].pop(self.name())

        self._name = name

    def check_closed(self):
        # Get the trailing edge upper curve
        if self.trailing_edge is self.upper_surf_end:
            self.upper_te_curve = None
        else:
            for curve in self.trailing_edge.curves:
                if self.upper_surf_end in curve.point_sequence().points():
                    self.upper_te_curve = curve
                    break
            if self.upper_te_curve is None:
                raise ClosureError("Could not identify the upper trailing edge line/curve")

        # Get the trailing edge lower curve
        if self.trailing_edge is self.lower_surf_end:
            self.lower_te_curve = None
        else:
            for curve in self.trailing_edge.curves:
                if self.lower_surf_end in curve.point_sequence().points():
                    self.lower_te_curve = curve
                    break
            if self.lower_te_curve is None:
                raise ClosureError("Could not identify the lower trailing edge line/curve")
        #
        # # Check that there is only one curve attached to upper_surf_end if upper_te_curve is not found
        # if self.upper_te_curve is None and len(self.upper_surf_end.curves) > 1:
        #     raise BranchError("Detected multiple curves branching from a sharp upper trailing edge point")
        #
        # # Check that there is only one curve attached to lower_surf_end if lower_te_curve is not found
        # if self.lower_te_curve is None and len(self.lower_surf_end.curves) > 1:
        #     raise BranchError("Detected multiple curves branching from a sharp lower trailing edge point")

        # Loop through the rest of the curves
        current_curve = None
        if self.upper_te_curve is None:
            current_curve = self.upper_surf_end.curves[0]
        else:
            for curve in self.upper_surf_end.curves:
                if curve is not self.upper_te_curve:
                    current_curve = curve
                    break

        if current_curve is None:
            raise ClosureError("Curve loop is not closed")

        for point in current_curve.point_sequence().points():
            if len(point.curves) > 2:
                raise BranchError("Found more than two curves associated with a curve endpoint in the loop")

        previous_point = self.upper_surf_end
        closed = False
        while not closed:
            self.curves.append(current_curve)

            idx_of_prev_point = current_curve.point_sequence().points().index(previous_point)
            if idx_of_prev_point == 0:
                idx_of_next_point = -1
            else:
                self.curves_to_reverse.append(current_curve)
                idx_of_next_point = 0

            next_point = current_curve.point_sequence().points()[idx_of_next_point]

            if next_point is self.lower_surf_end:
                closed = True
                break

            for curve in next_point.curves:
                if curve is not current_curve:
                    current_curve = curve
                    break

            previous_point = next_point
            pass

        if not closed:
            raise ClosureError("Curve loop not closed")


class BranchError(Exception):
    pass


class ClosureError(Exception):
    pass
