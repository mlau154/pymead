import numpy as np
import math
from pymead.utils.misc import count_dollar_signs


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

        if self.units == 'length' and self.scale_value is not None:
            self.value = value * self.scale_value
        elif self.units == 'inverse-length' and self.scale_value is not None:
            self.value = value / self.scale_value
        else:
            self.value = value

        self.bounds = bounds
        self.active = active
        self.linked = linked
        # self.tag = tag
        self.func_str = func_str
        self.func = None
        if self.func_str is not None:
            self.linked = True
        self.function_dict = {}
        self.depends_on = {}
        self.tag_matrix = None
        self.tag_list = None
        self.mea = None
        self.x = x
        self.y = y
        self.xp = xp
        self.yp = yp

    def set_func_str(self, func_str: str):
        self.func_str = func_str
        self.linked = True

    def remove_func(self):
        self.func_str = None
        self.linked = False
        self.depends_on = {}

    def update_function(self, show_q_error_messages: bool):
        if self.func_str is None:
            pass
        else:
            if 'math' not in self.function_dict.values():
                self.function_dict = {**self.function_dict, **vars(math)}
            self.parse_update_function_str()  # Convert the function string into a Python function and extract vars
            execute = self.add_dependencies(show_q_error_messages)  # Add the variables the function depends on to the function_dict
            if execute:
                exec(self.func, self.function_dict)  # Update the function (not the result) in the function_dict

    def update_value(self):
        if self.func_str is None:
            pass
        else:
            self.value = self.function_dict['f']()  # no parameters passed as inputs (inputs all stored and updated
            # inside self.function_dict )

    def update(self, show_q_error_messages: bool = True):
        self.update_function(show_q_error_messages)
        self.update_value()

    def parse_update_function_str(self):
        self.tag_matrix = []
        self.func = 'def f(): return '
        appending = False
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

    def add_dependencies(self, show_q_error_messages: bool):


        # def iter_through_dict_recursively(d: dict):
        #     for k, v in d.items():
        #         if isinstance(v, dict):
        #
        #             iter_through_dict_recursively(v)
        #         else:
        #             print(k, ":", v)

        def get_nested_dict_val(d: dict, lst: list):
            if isinstance(d, dict):
                # print(lst[0] in d.keys())
                if lst[0] in d.keys():
                    # print(f"Good to go")
                    return get_nested_dict_val(d[lst[0]], lst[1:])
                else:
                    # print(f'Returning None')
                    return None
            else:
                return d

        for idx, t in enumerate(self.tag_list):
            # print(f"tag_matrix_idx = {self.tag_matrix[idx]}")
            self.depends_on[t] = get_nested_dict_val(self.mea.param_dict, self.tag_matrix[idx])
            # print(f"self.depends_on = {self.depends_on}")
            if self.depends_on[t] is None:
                self.depends_on = {}
                message = f"Could not compile input function string: {self.func_str}"
                self.remove_func()
                # print(f"depends_on = {self.depends_on}")
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
