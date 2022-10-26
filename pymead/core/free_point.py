import numpy as np
from pymead.core.param import Param
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import translate, rotate, scale, transform


class FreePoint(ControlPoint):

    def __init__(self,
                 x: Param,
                 y: Param,
                 previous_anchor_point: str,
                 airfoil_tag: str,
                 previous_free_point: str or None = None,
                 tag: str or None = None,
                 length_scale_dimension: float or None = None
                 ):
        """
        ### Description:

        The `FreePoint` in `pymead` is the way to add a control point to a Bézier curve within an `Airfoil`
        without requiring the Bézier curve to pass through that particular point. In other words, a FreePoint allows
        an \\(x\\) - \\(y\\) coordinate pair to be added to the `P` matrix (see `pymead.core.airfoil.bezier` for
        usage). An example showing some possible locations of `FreePoint`s is shown below.

        ### Args:

        `x`: `Param` describing the x-location of the `FreePoint`

        `y`: `Param` describing the y-location of the `FreePoint`

        `previous_anchor_point`: a `str` representing the previous `AnchorPoint` (counter-clockwise ordering)

        `length_scale_dimension`: a `float` giving the length scale by which to non-dimensionalize the `x` and `y`
        values (optional)

        ### Returns:

        An instance of the `FreePoint` class
        """

        super().__init__(x.value, y.value, tag, previous_anchor_point, cp_type='free_point')

        self.ctrlpt = ControlPoint(x.value, y.value, tag, previous_anchor_point, cp_type='free_point')

        self.x = x
        self.y = y
        self.xp = Param(self.x.value)
        self.yp = Param(self.y.value)
        self.x.free_point = self
        self.y.free_point = self
        self.xp.free_point = self
        self.yp.free_point = self
        self.airfoil_transformation = None
        self.airfoil_tag = airfoil_tag
        self.tag = tag
        self.previous_free_point = previous_free_point
        self.length_scale_dimension = length_scale_dimension
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()

    def set_tag(self, tag: str):
        self.tag = tag
        self.ctrlpt.tag = tag

    def set_xy(self, x=None, y=None, xp=None, yp=None):
        # other_airfoils_affected = []
        # print("Made it to the start")
        # print(f"Made it 2")
        for xy in [['x', 'y'], ['xp', 'yp'], ['x', 'yp'], ['xp', 'y']]:
            if getattr(getattr(self, xy[0]), 'linked') and getattr(getattr(self, xy[1]), 'linked') or (
                    not getattr(getattr(self, xy[0]), 'active')) and (not getattr(getattr(self, xy[1]), 'active')):
                return []  # Early return if both x and y degrees of freedom are locked
        if (x is not None or y is not None) and xp is None and yp is None:
            # if self.xp.linked and self.yp.linked or (not self.xp.active) and (not self.yp.active):
            #     return []
            if x is not None and (not self.x.linked) and self.x.active:
                self.x.value = x
            if y is not None and (not self.y.linked) and self.y.active:
                self.y.value = y
            if self.xp.linked or not self.xp.active:
                self.yp.value = self.get_yp_from_x_y(x, y)
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            elif self.yp.linked or not self.yp.active:
                self.xp.value = self.get_xp_from_x_y(x, y)
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            else:
                self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
                                                         self.airfoil_transformation['dx'].value,
                                                         self.airfoil_transformation['dy'].value,
                                                         -self.airfoil_transformation['alf'].value,
                                                         self.airfoil_transformation['c'].value,
                                                         ['scale', 'rotate', 'translate'])
            # If other airfoils are affected by this change in FreePoint location, we need to mark the airfoil for
            # change:
            # other_airfoils_affected.extend(self.update_xy())
        elif (xp is not None or yp is not None) and x is None and y is None:
            # if self.x.linked and self.y.linked or (not self.x.active) and (not self.y.active):
            #     return []
            # if xp is not None and (not self.xp.linked) and self.xp.active:
            #     self.xp.value = xp
            #     print(f"Setting xp value! xp = {self.xp.value}")
            # if yp is not None and (not self.yp.linked) and self.yp.active:
            #     self.yp.value = yp
            if self.x.linked or not self.x.active:
                self.y.value = self.get_y_from_xp_yp(xp, yp)
                self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
                                                         self.airfoil_transformation['dx'].value,
                                                         self.airfoil_transformation['dy'].value,
                                                         -self.airfoil_transformation['alf'].value,
                                                         self.airfoil_transformation['c'].value,
                                                         ['scale', 'rotate', 'translate'])
            elif self.y.linked or not self.y.active:
                self.x.value = self.get_x_from_xp_yp(xp, yp)
                self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
                                                         self.airfoil_transformation['dx'].value,
                                                         self.airfoil_transformation['dy'].value,
                                                         -self.airfoil_transformation['alf'].value,
                                                         self.airfoil_transformation['c'].value,
                                                         ['scale', 'rotate', 'translate'])
            elif self.xp.linked or not self.xp.active:
                self.yp.value = yp
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            elif self.yp.linked or not self.yp.active:
                self.xp.value = xp
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
            else:
                self.xp.value = xp
                self.yp.value = yp
                self.x.value, self.y.value = transform(self.xp.value, self.yp.value,
                                                       -self.airfoil_transformation['dx'].value,
                                                       -self.airfoil_transformation['dy'].value,
                                                       self.airfoil_transformation['alf'].value,
                                                       1 / self.airfoil_transformation['c'].value,
                                                       ['translate', 'rotate', 'scale'])
        # elif only_update_xy:
        #     _, self.yp.value = transform(self.x.value, self.y.value,
        #                                              self.airfoil_transformation['dx'].value,
        #                                              self.airfoil_transformation['dy'].value,
        #                                              -self.airfoil_transformation['alf'].value,
        #                                              self.airfoil_transformation['c'].value,
        #                                              ['scale', 'rotate', 'translate'])
            # self.ctrlpt.xp = self.xp.value
            # If other airfoils are affected by this change in FreePoint location, we need to mark the airfoil for
            # change:
            # other_airfoils_affected.extend(self.update_xy())
        # elif only_update_xp_yp:
        #     print(f"only updating xp yp")
        #     self.xp.value, self.yp.value = transform(self.x.value, self.y.value,
        #                                              self.airfoil_transformation['dx'].value,
        #                                              self.airfoil_transformation['dy'].value,
        #                                              -self.airfoil_transformation['alf'].value,
        #                                              self.airfoil_transformation['c'].value,
        #                                              ['scale', 'rotate', 'translate'])
        #     # other_airfoils_affected.extend(self.update_xy())
        else:
            raise ValueError("Either (\'x\' or \'y\') or (\'xp\' or \'yp\') must be specified or \'only_update_xp_yp\' "
                             "must be set to True")
        other_airfoils_affected = self.update_xy()
        # print(f"other_airfoils_affected now is {other_airfoils_affected}")
        # if only_update_xy:
        #     self.set_ctrlpt_value2()
        # else:
        self.set_ctrlpt_value()
        return other_airfoils_affected

    def update_xy(self):
        other_airfoils_affected = []
        for xy in ['x', 'y', 'xp', 'yp']:
            getattr(self, xy).update()
            # print(f"updating affects")
            # print(f"affects = {getattr(self, xy).affects}")
            for affects in getattr(self, xy).affects:
                # print(f"affects = {affects}")
                affects.update()
                if affects.free_point is not None:
                    fp = affects.free_point
                    if fp.airfoil_tag != self.airfoil_tag:
                        other_airfoils_affected.append(fp.airfoil_tag)
                    if affects.x or affects.y:
                        fp.xp.value, fp.yp.value = transform(fp.x.value, fp.y.value,
                                                             fp.airfoil_transformation['dx'].value,
                                                             fp.airfoil_transformation['dy'].value,
                                                             -fp.airfoil_transformation['alf'].value,
                                                             fp.airfoil_transformation['c'].value,
                                                             ['scale', 'rotate', 'translate'])
                    if affects.xp or affects.yp:
                        fp.x.value, fp.y.value = transform(fp.xp.value, fp.yp.value,
                                                           -fp.airfoil_transformation['dx'].value,
                                                           -fp.airfoil_transformation['dy'].value,
                                                           fp.airfoil_transformation['alf'].value,
                                                           1 / fp.airfoil_transformation['c'].value,
                                                           ['translate', 'rotate', 'scale'])
                    if affects.x or affects.y or affects.xp or affects.yp:
                        fp.set_ctrlpt_value()
                if affects.anchor_point is not None:
                    ap = affects.anchor_point
                    if ap.airfoil_tag != self.airfoil_tag:
                        other_airfoils_affected.append(ap.airfoil_tag)
                    if affects.x or affects.y:
                        ap.xp.value, ap.yp.value = transform(ap.x.value, ap.y.value,
                                                             ap.airfoil_transformation['dx'].value,
                                                             ap.airfoil_transformation['dy'].value,
                                                             -ap.airfoil_transformation['alf'].value,
                                                             ap.airfoil_transformation['c'].value,
                                                             ['scale', 'rotate', 'translate'])
                    if affects.xp or affects.yp:
                        ap.x.value, ap.y.value = transform(ap.xp.value, ap.yp.value,
                                                           -ap.airfoil_transformation['dx'].value,
                                                           -ap.airfoil_transformation['dy'].value,
                                                           ap.airfoil_transformation['alf'].value,
                                                           1 / ap.airfoil_transformation['c'].value,
                                                           ['translate', 'rotate', 'scale'])
                    if affects.x or affects.y or affects.xp or affects.yp:
                        ap.set_ctrlpt_value()
        return other_airfoils_affected

    def get_x_from_xp_yp(self, xp, yp):
        x, _ = transform(xp, yp, -self.airfoil_transformation['dx'].value,
                         -self.airfoil_transformation['dy'].value,
                         self.airfoil_transformation['alf'].value,
                         1 / self.airfoil_transformation['c'].value,
                         ['translate', 'rotate', 'scale'])
        return x

    def get_y_from_xp_yp(self, xp, yp):
        _, y = transform(xp, yp, -self.airfoil_transformation['dx'].value, -self.airfoil_transformation['dy'].value,
                         self.airfoil_transformation['alf'].value, 1 / self.airfoil_transformation['c'].value,
                         ['translate', 'rotate', 'scale'])
        return y

    def get_xp_from_x_y(self, x, y):
        xp, _ = transform(x, y, self.airfoil_transformation['dx'].value,
                          self.airfoil_transformation['dy'].value,
                          -self.airfoil_transformation['alf'].value,
                          self.airfoil_transformation['c'].value,
                          ['scale', 'rotate', 'translate'])
        return xp

    def get_yp_from_x_y(self, x, y):
        _, yp = transform(x, y, self.airfoil_transformation['dx'].value, self.airfoil_transformation['dy'].value,
                          -self.airfoil_transformation['alf'].value, self.airfoil_transformation['c'].value,
                          ['scale', 'rotate', 'translate'])
        return yp

    def more_than_one_xy_linked_or_inactive(self):
        linked_or_inactive_counter = 0
        for xy in ['x', 'y', 'xp', 'yp']:
            if getattr(self, xy).linked or not getattr(self, xy).active:
                linked_or_inactive_counter += 1
        if linked_or_inactive_counter > 1:
            return True
        else:
            return False

    def x_or_y_linked_or_inactive(self):
        for xy in ['x', 'y']:
            if getattr(self, xy).linked or not getattr(self, xy).active:
                return True
        return False

    def xp_or_yp_linked_or_inactive(self):
        for xy in ['xp', 'yp']:
            if getattr(self, xy).linked or not getattr(self, xy).active:
                return True
        return False

    def set_x_value(self, value):
        # print(f"set_x_value called!")
        if value is not None:
            self.x.value = value
        self.xp.value, self.yp.value = scale(self.x.value, self.y.value,
                                             self.airfoil_transformation['c'].value)
        self.xp.value, self.yp.value = rotate(self.xp.value, self.yp.value, -self.airfoil_transformation['alf'].value)
        self.xp.value, self.yp.value = translate(self.xp.value, self.yp.value, self.airfoil_transformation['dx'].value,
                                                 self.airfoil_transformation['dy'].value)
        if value is not None:
            self.set_ctrlpt_value()

    def set_y_value(self, value):
        if value is not None:
            self.y.value = value
        self.xp.value, self.yp.value = scale(self.x.value, self.y.value,
                                             self.airfoil_transformation['c'].value)
        self.xp.value, self.yp.value = rotate(self.xp.value, self.yp.value, -self.airfoil_transformation['alf'].value)
        self.xp.value, self.yp.value = translate(self.xp.value, self.yp.value, self.airfoil_transformation['dx'].value,
                                                 self.airfoil_transformation['dy'].value)
        if value is not None:
            self.set_ctrlpt_value()

    def set_xp_value(self, value):
        if value is not None:
            self.xp.value = value

        # if not self.x.active or self.x.linked:
        #     skip_x = True
        # else:
        #     skip_x = False
        #
        # if not self.y.active or self.y.linked:
        #     skip_y = True
        # else:
        #     skip_y = False
        skip_x = False
        skip_y = False

        # print(f"skip_x = {skip_x}, skip_y = {skip_y}")

        if not skip_x or not skip_y:
            # print(f"not skip_x or not skip_y!")
            self.x.value, self.y.value = transform(self.xp.value, self.yp.value, -self.airfoil_transformation['dx'].value,
                                                   -self.airfoil_transformation['dy'].value,
                                                   self.airfoil_transformation['alf'].value,
                                                   1 / self.airfoil_transformation['c'].value,
                                                   ['translate', 'rotate', 'scale'], skip_x=skip_x, skip_y=skip_y)
            # print(f"self.x.value now is {self.x.value}")
            # print(f"self.y.value now is {self.y.value}")
            # print(f"self.xp.value now is {self.xp.value}")
            # print(f"self.yp.value now is {self.yp.value}")
        self.set_ctrlpt_value()

    def set_yp_value(self, value):
        # print(f"set_yp called")
        if value is not None:
            self.yp.value = value

        # if not self.x.active or self.x.linked:
        #     skip_x = True
        # else:
        #     skip_x = False
        #
        # if not self.y.active or self.y.linked:
        #     skip_y = True
        # else:
        #     skip_y = False

        skip_x = False
        skip_y = False

        if not skip_x or not skip_y:
            self.x.value, self.y.value = transform(self.xp.value, self.yp.value, -self.airfoil_transformation['dx'].value,
                                                   -self.airfoil_transformation['dy'].value,
                                                   self.airfoil_transformation['alf'].value,
                                                   1 / self.airfoil_transformation['c'].value,
                                                   ['translate', 'rotate', 'scale'], skip_x=skip_x, skip_y=skip_y)

        self.set_ctrlpt_value()

    def set_ctrlpt_value(self):
        self.ctrlpt.xp = self.xp.value
        self.ctrlpt.yp = self.yp.value
        self.ctrlpt.x_val = self.x.value
        self.ctrlpt.y_val = self.y.value

    def set_ctrlpt_value2(self):
        self.ctrlpt.xp = self.x.value
        self.ctrlpt.yp = self.y.value
        self.ctrlpt.x_val = self.x.value
        self.ctrlpt.y_val = self.y.value

    def scale_vars(self):
        """
        ### Description:

        Scales all the `pymead.core.param.Param`s in the `FreePoint` with `units == 'length'` by the
        `length_scale_dimension`. Scaling only occurs for each parameter if the `pymead.core.param.Param` has not yet
        been scaled.
        """
        if self.length_scale_dimension is not None:  # only scale if the anchor point has a length scale dimension
            for param in [var for var in vars(self).values()  # For each parameter in the anchor point,
                          if isinstance(var, Param) and var.units == 'length']:
                if param.scale_value is None:  # only scale if the parameter has not yet been scaled
                    param.value = param.value * self.length_scale_dimension

    def __repr__(self):
        return f"free_point_{self.tag}"

    def count_overrideable_variables(self):
        """
        ### Description:

        Counts all the overrideable `pymead.core.param.Param`s in the `FreePoint` (criteria:
        `pymead.core.param.Param().active == True`, `pymead.core.param.Param().linked == False`)

        ### Returns:

        Number of overrideable variables (`int`)
        """
        n_overrideable_variables = len([var for var in vars(self).values()
                                        if isinstance(var, Param) and var.active and not var.linked])
        return n_overrideable_variables

    def override(self, parameters: list):
        """
        ### Description:

        Overrides all the `pymead.core.param.Param`s in `FreePoint` which are active and not linked using a list of
        parameters. This list of parameters is likely a subset of parameters passed to either
        `pymead.core.airfoil.Airfoil` or `pymead.core.parametrization.AirfoilParametrization`. This function is
        useful whenever iteration over only the relevant parameters is required.

        ### Args:

        `parameters`: a `list` of parameters
        """
        override_param_obj_list = [var for var in vars(self).values()
                                   if isinstance(var, Param) and var.active and not var.linked]
        if len(parameters) != len(override_param_obj_list):
            raise Exception('Number of base airfoil parameters does not match length of input override parameter list')
        param_idx = 0
        for param in override_param_obj_list:
            setattr(param, 'value', parameters[param_idx])
            param_idx += 1

        self.scale_vars()
        self.xy = np.array([self.x.value, self.y.value])

    def set_all_as_linked(self):
        """
        ### Description:

        Sets `linked=True` on all `pymead.core.param.Param`s in the `FreePoint`
        """
        for param in [var for var in vars(self).values() if isinstance(var, Param)]:
            param.linked = True
