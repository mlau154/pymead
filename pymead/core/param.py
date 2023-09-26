import benedict
import numpy as np
import math
import typing
from pymead.core.transformation import AirfoilTransformation
from pymead.utils.geometry import map_angle_m180_p180


class Param:

    def __init__(self, value: float or typing.Tuple[float], bounds: tuple = (-np.inf, np.inf),
                 active: bool or typing.Tuple[bool] = True, linked: bool or typing.Tuple[bool] = False,
                 func_str: str = None, name: str = None, periodic: bool = False):
        """
        This is the class used to define parameters used for the airfoil and airfoil parametrization definitions
        in pymead.

        ### Args:

        `value`: a `float` representing the value of the parameter

        `bounds`: a `list` or 1D `np.ndarray` with two elements of the form `[<lower bound>, <upper bound>]`. Used in
        `pymead.utils.airfoil_matching` and for normalization during parameter extraction. Default:
        `np.array([-np.inf, np.inf])` (no normalization).

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
        self._value = list(value) if isinstance(value, tuple) else value
        self.bounds = list(bounds) if isinstance(bounds, tuple) else bounds
        self._name = None
        self.name = name

        self.active = list(active) if isinstance(active, tuple) or isinstance(active, list) else active
        self.linked = list(linked) if isinstance(linked, tuple) or isinstance(active, list) else linked
        self.periodic = periodic
        self.func_str = func_str
        self.func = None
        if self.func_str is not None:
            if isinstance(self.linked, list):
                self.linked = [True, True]
            else:
                self.linked = True
        self.function_dict = {'depends': {}, 'name': self.name.split('.')[-1] if self.name is not None else None}
        self.depends_on = {}
        self.affects = []
        self.tag_matrix = None
        self.user_func_strs = None
        self.tag_list = None
        self.airfoil_tag = None
        self.sets_airfoil_csys = False
        self.mea = None
        self.deactivated_for_airfoil_matching = False
        self.at_boundary = False
        self.free_point = None
        self.anchor_point = None

    # def __setattr__(self, key, value):
    #     return super().__setattr__(key, value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, n: str):
        self._name = n
        if n is not None and n.split('.')[-1] in ['c', 'alf', 'dx', 'dy']:
            self.sets_airfoil_csys = True

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        if self.active:
            old_transformation, new_transformation, airfoil = None, None, None
            if self.sets_airfoil_csys and self.mea is not None and self.airfoil_tag is not None:
                airfoil = self.mea.airfoils[self.airfoil_tag]
                old_transformation = AirfoilTransformation(dx=airfoil.dx.value, dy=airfoil.dy.value,
                                                           alf=airfoil.alf.value, c=airfoil.c.value)
            old_value = self._value
            if self.periodic:
                # Treatment of the periodic case (only applies to angles)
                two_pi = 2 * np.pi
                v = map_angle_m180_p180(v)

                if not (np.isinf(self.bounds[0]) or np.isinf(self.bounds[1])):

                    if self.bounds[0] % two_pi < self.bounds[1] % two_pi:

                        # Calculate the angle half-way between the bounds in the out-of-bounds region
                        dist_b1_2pi = two_pi - self.bounds[1]
                        dist_0_b0 = self.bounds[0]
                        dist_total = dist_b1_2pi + dist_0_b0
                        mid_ob_angle = self.bounds[1] + dist_total / 2

                        if self.bounds[0] <= v <= self.bounds[1]:
                            self._value = v
                            self.at_boundary = False
                        elif v % two_pi < self.bounds[0] % two_pi or v % two_pi > mid_ob_angle % two_pi:
                            self._value = self.bounds[0]
                            self.at_boundary = True
                        elif self.bounds[1] % two_pi < v % two_pi <= mid_ob_angle % two_pi:
                            self._value = self.bounds[1]
                            self.at_boundary = True
                        else:
                            raise ValueError("Somehow found an angle value whose 2*pi modulo was "
                                             "outside the range [0, 2*pi]")
                    else:  # This is usually the case when the lower bound is a negative angle and the upper bound is a
                        # positive angle

                        # Calculate the angle half-way between the bounds in the out-of-bounds region
                        mid_ob_angle = np.mean([self.bounds[0] % two_pi, self.bounds[1] % two_pi])

                        if v % two_pi >= self.bounds[0] % two_pi or v % two_pi <= self.bounds[1] % two_pi:
                            self._value = v
                            self.at_boundary = False
                        elif mid_ob_angle % two_pi < v % two_pi < self.bounds[0] % two_pi:
                            self._value = self.bounds[0]
                            self.at_boundary = True
                        elif self.bounds[1] % two_pi < v % two_pi <= mid_ob_angle % two_pi:
                            self._value = self.bounds[1]
                            self.at_boundary = True
                else:
                    if np.isinf(self.bounds[0]) and np.isfinite(self.bounds[1]):
                        if v > self.bounds[1]:
                            self._value = self.bounds[1]
                            self.at_boundary = True
                        else:
                            self._value = v
                            self.at_boundary = False
                    elif np.isfinite(self.bounds[0]) and np.isinf(self.bounds[1]):
                        if v < self.bounds[0]:
                            self._value = self.bounds[0]
                            self.at_boundary = True
                        else:
                            self._value = v
                            self.at_boundary = False
                    elif np.isinf(self.bounds[0]) and np.isinf(self.bounds[1]):
                        self._value = v
                        self.at_boundary = False
            else:
                # Non-periodic treatment
                if v < self.bounds[0]:
                    self._value = self.bounds[0]
                    self.at_boundary = True
                elif v > self.bounds[1]:
                    self._value = self.bounds[1]
                    self.at_boundary = True
                else:
                    self._value = v
                    self.at_boundary = False

            self.update_affected_params(old_value)
            if self.sets_airfoil_csys and self.mea is not None and self.airfoil_tag is not None:
                new_transformation = AirfoilTransformation(dx=airfoil.dx.value, dy=airfoil.dy.value,
                                                           alf=airfoil.alf.value, c=airfoil.c.value)
                self.update_ap_fp(self, old_transformation=old_transformation, new_transformation=new_transformation)

    def update_affected_params(self, old_value):
        if len(self.affects) > 0:
            idx = 0
            any_affected_params_at_boundary = False
            old_affected_param_values = []
            for idx, affected_param in enumerate(self.affects):
                old_affected_param_values.append(affected_param.value)
                affected_param.update()
                self.update_ap_fp(affected_param)
                if affected_param.at_boundary:
                    any_affected_params_at_boundary = True
                    break
            if any_affected_params_at_boundary:
                self._value = old_value
                for idx2, affected_param in enumerate(self.affects[:idx + 1]):
                    affected_param._value = old_affected_param_values[idx2]
                    self.update_ap_fp(affected_param)

    @staticmethod
    def update_ap_fp(param, old_transformation: AirfoilTransformation = None,
                     new_transformation: AirfoilTransformation = None):
        if old_transformation is not None and new_transformation is not None:
            for ap_tag in param.mea.airfoils[param.airfoil_tag].free_points.keys():
                for fp in param.mea.airfoils[param.airfoil_tag].free_points[ap_tag].values():
                    old_coords = np.array([fp.xy.value])
                    new_coords = new_transformation.transform_abs(old_transformation.transform_rel(old_coords))
                    # print(f"{old_coords = }, {new_coords = }")
                    if fp.xy.linked[0] or not fp.xy.active[0]:
                        new_coords[0][0] = old_coords[0][0]
                    if fp.xy.linked[1] or not fp.xy.active[1]:
                        new_coords[0][1] = old_coords[0][1]
                    fp.xy.value = new_coords[0].tolist()
                    fp.set_ctrlpt_value()
            for ap in param.mea.airfoils[param.airfoil_tag].anchor_points:
                if ap.tag not in ['te_1', 'le', 'te_2']:
                    old_coords = np.array([ap.xy.value])
                    new_coords = new_transformation.transform_abs(old_transformation.transform_rel(old_coords))
                    # print(f"{old_coords = }, {new_coords = }")
                    if ap.xy.linked[0] or not ap.xy.active[0]:
                        new_coords[0][0] = old_coords[0][0]
                    if ap.xy.linked[1] or not ap.xy.active[1]:
                        new_coords[0][1] = old_coords[0][1]
                    ap.xy.value = new_coords[0].tolist()
                    ap.set_ctrlpt_value()
        else:
            if param.free_point is not None:
                param.free_point.set_ctrlpt_value()
            elif param.anchor_point is not None:
                param.anchor_point.set_ctrlpt_value()

    def set_func_str(self, func_str: str):
        if len(func_str) == 0:
            self.remove_func()
        else:
            if len(self.function_dict) == 0:
                self.function_dict = {'depends': {}, 'name': self.name.split('.')[-1] if self.name is not None else None}
            self.func_str = func_str
            self.linked = True

    def remove_func(self):
        self.func_str = None
        self.function_dict = {'depends': {}, 'name': self.name.split('.')[-1] if self.name is not None else None}
        self.linked = False
        for parameter in self.depends_on.values():
            if self in parameter.affects:
                parameter.affects.remove(self)
        self.depends_on = {}

    def update_function(self, show_q_error_messages: bool, func_str_changed: bool = False):
        if self.func_str is None:
            pass
        else:
            if func_str_changed:
                # Convert the function string into a Python function and determine parameters present in string:
                math_function_list, user_function_list = self.parse_update_function_str()

                # Add any math functions detected from the func_str:
                for s in math_function_list:
                    if s not in self.function_dict.keys():
                        if s in vars(math).keys():
                            self.function_dict[s] = vars(math)[s]

                for s in user_function_list:
                    if s not in self.function_dict.keys():
                        s_list = s.split('.')
                        mod_name = s_list[0]
                        func_name = s_list[1]
                        if self.mea.param_tree is not None:
                            self.function_dict[func_name] = getattr(self.mea.param_tree.user_mods[mod_name], func_name)
                        else:
                            self.function_dict[func_name] = getattr(self.mea.user_mods[mod_name], func_name)

                # Add the variables the function depends on to the function_dict:
                self.add_dependencies(show_q_error_messages)

            self.update_dependencies()

            if self.func is not None:
                self.function_dict['__builtins__'] = {}
                exec(self.func, self.function_dict)

    def update_value(self):
        if self.func_str is None:
            pass
        else:
            if 'f' not in self.function_dict.keys():  # TODO: test this fix for affected Params not updating
                self.update_function(show_q_error_messages=True, func_str_changed=True)
            self.value = self.function_dict['f']()  # no parameters passed as inputs (inputs all stored and updated

    def update(self, show_q_error_messages: bool = True, func_str_changed: bool = False):
        self.update_function(show_q_error_messages, func_str_changed)
        self.update_value()

    def parse_update_function_str(self):
        self.tag_matrix = []
        self.user_func_strs = []
        self.func = 'def f(): return '
        math_functions_to_include = []
        appending, appending_user_func = False, False
        append_new_to_math_function_list = True
        for ch_idx, ch in enumerate(self.func_str):
            if appending:
                if ch.isalnum() or ch == '_':
                    self.tag_matrix[-1][-1] += ch
                elif ch == '.':
                    self.tag_matrix[-1].append('')
                else:
                    self.func += '"]'
                    appending = False
            if appending_user_func:
                if ch == '(' and appending_user_func:
                    appending_user_func = False
            if ch == '$':
                self.tag_matrix.append([''])
                appending = True
                self.func += 'depends["'
            elif ch == '.':
                self.func += '.'
                if appending_user_func:
                    self.user_func_strs[-1] += '.'
            elif ch == '^':
                self.user_func_strs.append('')
                appending_user_func = True
            else:
                self.func += ch
            if appending and ch_idx == len(self.func_str) - 1:
                self.func += '"]'
            if not appending and ch.isalnum():
                if append_new_to_math_function_list:
                    math_functions_to_include.append('')
                math_functions_to_include[-1] += ch
                append_new_to_math_function_list = False
                if appending_user_func:
                    self.user_func_strs[-1] += ch
            if not appending and not ch.isalnum():
                append_new_to_math_function_list = True

        for user_func_str in self.user_func_strs:
            self.func = self.func.replace(user_func_str, user_func_str.split('.')[-1])

        def concatenate_strings(lst: list):
            tag = ''
            for idx, s in enumerate(lst):
                tag += s
                if idx < len(lst) - 1:
                    tag += '.'
            return tag

        self.tag_list = [concatenate_strings(tl) for tl in self.tag_matrix]
        for t in self.tag_list:
            self.depends_on[t] = None

        return math_functions_to_include, self.user_func_strs

    def add_dependencies(self, show_q_error_messages: bool):

        def get_nested_dict_val(d: dict, tag):
            dben = benedict.benedict(d)
            return dben[tag]

        for idx, t in enumerate(self.tag_list):
            self.depends_on[t] = get_nested_dict_val(self.mea.param_dict, t)
            if self.depends_on[t] is not None:
                if self not in self.depends_on[t].affects:
                    self.depends_on[t].affects.append(self)
            if self.depends_on[t] is None:
                self.depends_on = {}
                message = f"Could not compile input function string: {self.func_str}"
                self.remove_func()
                if not show_q_error_messages:
                    print(message)
                return False

        return True

    def update_dependencies(self):
        for key, value in self.depends_on.items():
            self.function_dict['depends'][key] = value.value

    @classmethod
    def from_param_dict(cls, param_dict: dict):
        """Generates a Param from a JSON-saved param_dict (aids in backward/forward compatibility)"""
        temp_dict = {'value': param_dict['_value']}
        for attr_name, attr_value in param_dict.items():
            if attr_name in ['bounds', 'active', 'linked', 'func_str', 'name']:
                temp_dict[attr_name] = attr_value
        return cls(**temp_dict)


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
