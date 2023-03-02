import math


class ObjectiveConstraint:
    def __init__(self, func_str: str):
        self.func_str = func_str
        self.func = None
        self.function_dict = {}
        self.depends_on = {}
        self.value = None
        self.tag_matrix = None
        self.tag_list = None

    def add_or_set_dependencies(self, dependencies: dict):
        for k, v in dependencies.items():
            # print(f"{k = }")
            self.depends_on[k] = v

    def set_func_str(self, func_str: str):
        self.func_str = func_str

    def remove_func(self):
        self.func_str = None
        self.func = None
        self.function_dict = {}

    def update_function(self):
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
            execute = self.add_dependencies()

            # Update the function (not the result) in the function_dict
            if execute:
                try:
                    exec(self.func, self.function_dict)
                except (SyntaxError, NameError, TypeError):
                    raise FunctionCompileError('Error in function compilation')

    def update_value(self):
        if self.func_str is None:
            pass
        else:
            try:
                self.value = self.function_dict['f']()  # no parameters passed as inputs (inputs all stored and updated
                # inside self.function_dict )
            except (SyntaxError, NameError, TypeError):
                raise FunctionCompileError('Error in function update')

    def update(self, dependencies: dict):
        self.add_or_set_dependencies(dependencies)
        self.update_function()
        self.update_value()

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
            self.function_dict[t] = None

        return math_functions_to_include

    def add_dependencies(self):

        for idx, t in enumerate(self.tag_list):
            if t in self.depends_on.keys():
                self.function_dict[t] = self.depends_on[t]
            else:
                self.function_dict = {}
                self.remove_func()
                raise FunctionCompileError(f"Could not compile input function string: {self.func_str}")

        return True


class Objective(ObjectiveConstraint):
    def __init__(self, func_str: str):
        super().__init__(func_str)


class Constraint(ObjectiveConstraint):
    def __init__(self, func_str: str):
        super().__init__(func_str)


class FunctionCompileError(Exception):
    pass


if __name__ == '__main__':
    obj = Objective('$Cd')
    obj.update({'Cd': 0.02, 'Cl': 0.35, 'Cl_target': 0.08, 'Cl_tol': 0.005})
    constraint = Constraint('abs($Cl - $Cl_target) - $Cl_tol')
    constraint.update({'Cd': 0.02, 'Cl': 0.35, 'Cl_target': 0.3455, 'Cl_tol': 0.005})
    pass
