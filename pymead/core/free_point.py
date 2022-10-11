import numpy as np
from pymead.core.param import Param
from pymead.core.control_point import ControlPoint
from pymead.utils.transformations import translate, rotate, scale, transform


class FreePoint(ControlPoint):

    def __init__(self,
                 x: Param,
                 y: Param,
                 previous_anchor_point: str,
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

        .. image:: complex_airfoil_free_points.png

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
        self.airfoil_transformation = None
        self.tag = tag
        self.previous_free_point = previous_free_point
        self.length_scale_dimension = length_scale_dimension
        self.n_overrideable_parameters = self.count_overrideable_variables()
        self.scale_vars()

    def set_tag(self, tag: str):
        self.tag = tag
        self.ctrlpt.tag = tag

    def set_x_value(self, value):
        # print(f"set_x_value called!")
        if value is not None:
            self.x.value = value
        # print(f"Before x call, self.xp.value was {self.xp.value}")
        self.xp.value, self.yp.value = scale(self.x.value, self.y.value,
                                             self.airfoil_transformation['c'].value)
        self.xp.value, self.yp.value = rotate(self.xp.value, self.yp.value, -self.airfoil_transformation['alf'].value)
        self.xp.value, self.yp.value = translate(self.xp.value, self.yp.value, self.airfoil_transformation['dx'].value,
                                                 self.airfoil_transformation['dy'].value)
        # print(f"After x call, self.xp.value is {self.xp.value}")
        if value is not None:
            self.set_ctrlpt_value()

    def set_y_value(self, value):
        # print(f"set_y_value called!")
        if value is not None:
            self.y.value = value
        # print(f"Before y call, self.xp.value was {self.xp.value}")
        self.xp.value, self.yp.value = scale(self.x.value, self.y.value,
                                             self.airfoil_transformation['c'].value)
        self.xp.value, self.yp.value = rotate(self.xp.value, self.yp.value, -self.airfoil_transformation['alf'].value)
        self.xp.value, self.yp.value = translate(self.xp.value, self.yp.value, self.airfoil_transformation['dx'].value,
                                                 self.airfoil_transformation['dy'].value)
        # print(f"After y call, self.xp.value is {self.xp.value}")
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
