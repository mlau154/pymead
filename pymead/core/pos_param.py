from pymead.core.param import Param
import numpy as np
import typing
import re


class PosParam(Param):
    def __init__(self, value: tuple, bounds: typing.Tuple[tuple] = ((-np.inf, np.inf), (-np.inf, np.inf)),
                 active: typing.Tuple[bool] = (True, True), linked: typing.Tuple[bool] = (False, False),
                 func_str: str = None, name: str = None):
        super().__init__(value=value, active=active, bounds=bounds, linked=linked, func_str=func_str, name=name)
        if self.func_str is not None:
            self.parse_func_str_for_linked()  # override the singularly-valued linked bool with a list of bools

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        update_x = self.active[0]
        update_y = self.active[1]
        old_value = self._value
        if update_x:
            if v[0] < self.bounds[0][0]:
                self._value[0] = self.bounds[0][0]
                self.at_boundary = True
            elif v[0] > self.bounds[0][1]:
                self._value[0] = self.bounds[0][1]
                self.at_boundary = True
            else:
                self._value[0] = v[0]
                self.at_boundary = False
        if update_y:
            if v[1] < self.bounds[1][0]:
                self._value[1] = self.bounds[1][0]
                self.at_boundary = True
            elif v[1] > self.bounds[1][1]:
                self._value[1] = self.bounds[1][1]
                self.at_boundary = True
            else:
                self._value[1] = v[1]
                self.at_boundary = False

        if update_x or update_y:
            self.update_affected_params(old_value)

        pass

    def set_func_str(self, func_str: str):
        if len(func_str) == 0:
            self.remove_func()
        else:
            if len(self.function_dict) == 0:
                self.function_dict = {'depends': {}, 'name': self.name.split('.')[-1] if self.name is not None else None}
            self.func_str = func_str
            self.parse_func_str_for_linked()

    def remove_func(self):
        self.func_str = None
        self.function_dict = {'depends': {}, 'name': self.name.split('.')[-1] if self.name is not None else None}
        self.linked = [False, False]
        for parameter in self.depends_on.values():
            if self in parameter.affects:
                parameter.affects.remove(self)
        self.depends_on = {}

    def parse_func_str_for_linked(self):
        str_before_comma = ''
        str_after_comma = ''
        appending_before_comma, appending_after_comma = False, False
        for ch in self.func_str:
            if appending_before_comma:
                str_before_comma += ch
            elif appending_after_comma:
                str_after_comma += ch
            if ch == '{':
                appending_before_comma = True
            elif ch == ',':
                appending_before_comma = False
                appending_after_comma = True
            elif ch == '}':
                break

        if len(str_before_comma) > 0 and 'None' not in str_before_comma:
            self.linked[0] = True
        else:
            self.linked[0] = False
        if len(str_after_comma) > 0 and 'None' not in str_after_comma:
            self.linked[1] = True
        else:
            self.linked[1] = False

    def update_value(self):
        if self.func_str is None:
            pass
        else:
            temp_value = self.function_dict['f']()
            if temp_value[0] is None and temp_value[1] is None:
                raise ValueError('Must set at least one of the two cells in the PosParam equation.')
            if temp_value[0] is None:
                self.value = [self._value[0], temp_value[1]]
            elif temp_value[1] is None:
                self.value = [temp_value[0], self._value[1]]
            else:
                self.value = temp_value
            # self.value = self.function_dict['f']()  # no parameters passed as inputs (inputs all stored and updated
            # inside self.function_dict )

    # def update(self, show_q_error_messages: bool = True, func_str_changed: bool = False):
    #     self.update_function(show_q_error_messages, func_str_changed)
    #     if func_str_changed:
    #         match1 = re.search(r'\{\s+,', self.func)
    #         match2 = re.search(r',\s+\}', self.func)
    #         if match1:
    #             self.func = self.func.replace(match1.group(0), '{None, ')
    #         if match2:
    #             self.func = self.func.replace(match2.group(0), ', None}')
    #     self.update_value()

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
            elif ch == '{':
                self.func += '['
            elif ch == '}':
                self.func += ']'
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


def main():
    from pymead.core.mea import MEA
    from pymead.core.airfoil import Airfoil
    mea = MEA(airfoils=[Airfoil()])
    pos_param_1 = {'value': (0.5, 0.3)}
    pos_param_2 = {'value': (0.2, 0.1)}
    mea.add_custom_parameters({'xy1': pos_param_1, 'xy2': pos_param_2})
    mea.param_dict['Custom']['xy2'].set_func_str('{$Custom.xy1[0] + 0.1, $Custom.xy1[1] + 0.5}')
    mea.param_dict['Custom']['xy2'].update(func_str_changed=True)
    mea.param_dict['Custom']['xy2'].set_func_str('{$Custom.xy1[0] + 0.3, None}')
    mea.param_dict['Custom']['xy2'].update(func_str_changed=True)
    # pos_param_2.set_func_str('')
    pass


if __name__ == '__main__':
    main()
