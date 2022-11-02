import numpy as np
import math
from pymead.utils.transformations import transform


class Param:

    def __init__(self, value: float, units: str or None = None,
                 bounds: list or np.ndarray = np.array([-np.inf, np.inf]), scale_value: float or None = None,
                 active: bool = True, linked: bool = False, func_str: str = None, x: bool = False, y: bool = False,
                 xp: bool = False, yp: bool = False):
        """
        ### Description:

        This is the class used to define parameters used for the airfoil and airfoil parametrization definitions
        in `pymead`.

        ### Args:

        `value`: a `float` representing the value of the parameter

        `units`: a `str` where, if not set to `None`, scales the parameters by the value contained in
        `length_scale_dimension`. Must be one of `"length"`, `"inverse-length"`, or `None`. Default: `None`.

        `bounds`: a `list` or 1D `np.ndarray` with two elements of the form `[<lower bound>, <upper bound>]`. Used in
        `pymead.utils.airfoil_matching` and for normalization during parameter extraction. Default:
        `np.array([-np.inf, np.inf])` (no normalization).

        `scale_value`: length scale used to non-dimensionalize the parameter if `units` is not `None`.
        Default value: `None`.

        `active`: a `bool` stating whether the parameter is active. If `False`, direct and indirect write access to the
         parameter are restricted. Default: `True`.

        `linked`: a `bool` stating whether the parameter is linked to another parameter (i.e., whether the parameter
        has an active function). If `True`, direct write access to the parameter is restricted, but indirect write
        access is still allowed (via modification of parameters to which the current parameter is linked).
        Default: `False`.

        `name`: an optional `str` that gives the name of the parameter. Can be useful in identifying extracted
        parameters.

        ### Returns:

        An instance of the `pymead.core.param.Param` class.
        """

        self.units = units
        self.scale_value = scale_value
        self._value = None
        self.bounds = bounds

        if self.units == 'length' and self.scale_value is not None:
            self._value = value * self.scale_value
        elif self.units == 'inverse-length' and self.scale_value is not None:
            self._value = value / self.scale_value
        else:
            self._value = value

        self.active = active
        self.linked = linked
        # self.tag = tag
        self.func_str = func_str
        self.func = None
        if self.func_str is not None:
            self.linked = True
        self.function_dict = {}
        self.depends_on = {}
        self.affects = []
        self.tag_matrix = None
        self.tag_list = None
        self.airfoil_tag = None
        self.mea = None
        self.deactivated_for_airfoil_matching = False
        self.at_boundary = False
        self.x = x
        self.y = y
        self.xp = xp
        self.yp = yp
        self.free_point = None
        self.anchor_point = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        old_value = self._value
        if v < self.bounds[0]:
            self._value = self.bounds[0]
            self.at_boundary = True
        elif v > self.bounds[1]:
            self._value = self.bounds[1]
            self.at_boundary = True
        else:
            self._value = v
            self.at_boundary = False
        # self.update_function(False)
        # self.update_value()
        if len(self.affects) > 0:
            idx = 0
            any_affected_params_at_boundary = False
            old_affected_param_values = []
            for idx, affected_param in enumerate(self.affects):
                old_affected_param_values.append(affected_param.value)
                affected_param.update()
                if affected_param.at_boundary:
                    any_affected_params_at_boundary = True
                    break
            if any_affected_params_at_boundary:
                self._value = old_value
                for idx2, affected_param in enumerate(self.affects[:idx + 1]):
                    affected_param._value = old_affected_param_values[idx2]

    def set_func_str(self, func_str: str):
        self.func_str = func_str
        self.linked = True

    def remove_func(self):
        self.func_str = None
        self.function_dict = {}
        self.linked = False
        for parameter in self.depends_on.values():
            if self in parameter.affects:
                parameter.affects.remove(self)
        self.depends_on = {}

    def update_function(self, show_q_error_messages: bool):
        if self.func_str is None:
            pass
        else:
            # Convert the function string into a Python function and determine parameters present in string:
            math_function_list = self.parse_update_function_str()

            # Add any math functions detected from the func_str:
            for s in math_function_list:
                if s not in self.function_dict.keys():
                    if s in vars(math).keys():
                        self.function_dict[s] = vars(math)[s]

            # Add the variables the function depends on to the function_dict and detect whether the function should be
            # executed:
            execute = self.add_dependencies(show_q_error_messages)

            # Update the function (not the result) in the function_dict
            if execute:
                exec(self.func, self.function_dict)

    def update_value(self):
        if self.func_str is None:
            pass
        else:
            self.value = self.function_dict['f']()  # no parameters passed as inputs (inputs all stored and updated
            # inside self.function_dict )
            # print(f"self.value now is {self.value}")

    def update(self, show_q_error_messages: bool = True):
        self.update_function(show_q_error_messages)
        self.update_value()
        self.update_fp_ap()

    def update_fp_ap(self):
        if self.free_point is not None:
            fp = self.free_point
            if self.x or self.y:
                fp.xp.value, fp.yp.value = transform(fp.x.value, fp.y.value,
                                                     fp.airfoil_transformation['dx'].value,
                                                     fp.airfoil_transformation['dy'].value,
                                                     -fp.airfoil_transformation['alf'].value,
                                                     fp.airfoil_transformation['c'].value,
                                                     ['scale', 'rotate', 'translate'])
            if self.xp or self.yp:
                fp.x.value, fp.y.value = transform(fp.xp.value, fp.yp.value,
                                                   -fp.airfoil_transformation['dx'].value,
                                                   -fp.airfoil_transformation['dy'].value,
                                                   fp.airfoil_transformation['alf'].value,
                                                   1 / fp.airfoil_transformation['c'].value,
                                                   ['translate', 'rotate', 'scale'])
            if self.x or self.y or self.xp or self.yp:
                fp.set_ctrlpt_value()
        if self.anchor_point is not None:
            ap = self.anchor_point
            if self.x or self.y:
                ap.xp.value, ap.yp.value = transform(ap.x.value, ap.y.value,
                                                     ap.airfoil_transformation['dx'].value,
                                                     ap.airfoil_transformation['dy'].value,
                                                     -ap.airfoil_transformation['alf'].value,
                                                     ap.airfoil_transformation['c'].value,
                                                     ['scale', 'rotate', 'translate'])
            if self.xp or self.yp:
                ap.x.value, ap.y.value = transform(ap.xp.value, ap.yp.value,
                                                   -ap.airfoil_transformation['dx'].value,
                                                   -ap.airfoil_transformation['dy'].value,
                                                   ap.airfoil_transformation['alf'].value,
                                                   1 / ap.airfoil_transformation['c'].value,
                                                   ['translate', 'rotate', 'scale'])
            if self.x or self.y or self.xp or self.yp:
                ap.set_ctrlpt_value()

    def parse_update_function_str(self):
        self.tag_matrix = []
        self.func = 'def f(): return '
        math_functions_to_include = []
        appending = False
        append_new_to_math_function_list = True
        for ch in self.func_str:
            if appending:
                if ch.isalnum() or ch == '_':
                    self.tag_matrix[-1][-1] += ch
                elif ch == '.':
                    self.tag_matrix[-1].append('')
                else:
                    appending = False
            if ch == '$':
                self.tag_matrix.append([''])
                appending = True
            elif ch == '.' and appending:
                self.func += '_'
            else:
                self.func += ch
            if not appending and ch.isalnum():
                if append_new_to_math_function_list:
                    math_functions_to_include.append('')
                math_functions_to_include[-1] += ch
                append_new_to_math_function_list = False
            if not appending and not ch.isalnum():
                append_new_to_math_function_list = True

        def concatenate_strings(lst: list):
            tag = ''
            for idx, s in enumerate(lst):
                tag += s
                if idx < len(lst) - 1:
                    tag += '_'
            return tag

        self.tag_list = [concatenate_strings(tl) for tl in self.tag_matrix]
        for t in self.tag_list:
            self.depends_on[t] = None

        return math_functions_to_include

    def add_dependencies(self, show_q_error_messages: bool):

        def get_nested_dict_val(d: dict, lst: list):
            if isinstance(d, dict):
                if lst[0] in d.keys():
                    return get_nested_dict_val(d[lst[0]], lst[1:])
                else:
                    return None
            else:
                return d

        for idx, t in enumerate(self.tag_list):
            self.depends_on[t] = get_nested_dict_val(self.mea.param_dict, self.tag_matrix[idx])
            if self.depends_on[t] is not None:
                if self not in self.depends_on[t].affects:
                    self.depends_on[t].affects.append(self)
            if self.depends_on[t] is None:
                self.depends_on = {}
                message = f"Could not compile input function string: {self.func_str}"
                self.remove_func()
                if show_q_error_messages:
                    from PyQt5.QtWidgets import QErrorMessage
                    err = QErrorMessage()
                    print("Showing error message")
                    err.showMessage(message)
                else:
                    print(message)
                return False

        for key, value in self.depends_on.items():
            # print(f"updating depends on! val = {value.value}")
            # print(f"param_dict = {self.param_dict}")
            # print(f"key, val = {key}, {value}")
            # print(f"function_dict = {self.function_dict}")
            # if value is not None:
            self.function_dict[key] = value.value
            # else:
            #     self.depends_on.pop(key)

        return True


if __name__ == '__main__':
    from pymead.core.mea import MEA
    from pymead.core.airfoil import Airfoil
    from pymead.core.free_point import FreePoint
    mea = MEA(airfoils=Airfoil())
    mea.airfoils['A0'].insert_free_point(FreePoint(x=Param(0.5), y=Param(0.1), previous_anchor_point='te_1'))
    mea.param_dict['A0']['FP0']['x'].func_str = '6.0 * $A0.FP0.y + 0.01'
    mea.param_dict['A0']['FP0']['y'].value = 0.15
    mea.param_dict['A0']['FP0']['x'].mea = mea
    mea.param_dict['A0']['FP0']['x'].update()
    mea.add_custom_parameters({'r_htf': dict(value=0.38)})
    mea.param_dict['A0']['FP0']['x'].func_str = '$CUSTOM.r_htf + cos(0.005)'
    mea.param_dict['A0']['FP0']['x'].update()
    pass
