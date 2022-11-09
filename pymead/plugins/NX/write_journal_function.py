import math
import os
import typing
import numpy as np

import benedict
from pymead.core.mea import MEA
from pymead.core.param import Param
from pymead.core.airfoil import Airfoil


NX_DIR = os.path.dirname(os.path.abspath(__file__))


def write_import_statements_to_file(file_path: str):
    imports = ['math', 'NXOpen', 'NXOpen.Features', 'NXOpen.GeometricUtilities', 'NXOpen.Preferences',
               'NXOpen.Annotations', 'NXOpen.Drawings']
    import_as = []
    with open(file_path, 'w') as f:
        for import_ in imports:
            f.write(f'import {import_}\n')
        for import_as_ in import_as:
            f.write(f'import {import_as_[0]} as {import_as_[1]}\n')
        f.write('\n\n')


def dump_journal_functions_to_file(file_path: str):
    with open(os.path.join(NX_DIR, 'journal_functions.py'), 'r') as f:
        journal_function_lines = f.readlines()

    with open(file_path, 'a') as f:
        f.writelines(journal_function_lines)


def write_parameters_to_equations(param_dict: dict):
    key_matrix = []
    equations = []
    d_ben = benedict.benedict(param_dict)
    keypaths = d_ben.keypaths()
    for k in keypaths:
        split = k.split('.')
        if len(split) > 1 and split[-1] not in ['AnchorPoints', 'Base', 'FreePoints', 'Custom']:
            if not (split[1] == 'FreePoints' and len(split) < 5):
                if not (split[1] == 'AnchorPoints' and len(split) < 4):
                    key_matrix.append(split)

    def get_dict_value_recursively(list_of_keys: typing.List[str], d: dict):
        k_ = list_of_keys[0]
        if isinstance(d[k_], dict):
            return get_dict_value_recursively(list_of_keys[1:], d[k_])
        else:
            if isinstance(d[k_], Param):
                if any([angle in k_ for angle in ['alf', 'theta', 'phi']]):
                    return d[k_].value * 180 / math.pi
                else:
                    return d[k_].value
            else:
                raise ValueError('Found value in dictionary that was not of type Param')

    for key_list in key_matrix:
        if any([angle in key_list[-1] for angle in ['alf', 'theta', 'phi']]):
            unit = "deg"
        else:
            unit = None
        equations.append((f"{'_'.join(key_list)} = {get_dict_value_recursively(key_list, param_dict)}", unit))
    return equations


def dump_equations_to_file(equations: typing.List[tuple], file_path: str, mea: MEA):
    scale_factor = 100
    airfoil = mea.airfoils['A0']
    control_point_array = airfoil.control_point_array * scale_factor
    control_point_array = np.insert(control_point_array, 1, 0.0, axis=1)
    control_point_list_string = str(control_point_array.tolist())
    curve_orders = [val for val in airfoil.N.values()]
    with open(file_path, 'a') as f:
        f.write('\n\ndef main():\n')
        # f.write('    user_expressions = [\n')
        # for equation in equations:
        #     if equation[1] is None:
        #         quote1 = ""
        #         quote2 = ""
        #     else:
        #         quote1 = "\""
        #         quote2 = "\""
        #     f.write(f"        (\"{equation[0]}\", {quote1}{equation[1]}{quote2}),\n")
        # f.write('    ]\n\n')
        # f.write('    for expression in user_expressions:\n')
        # f.write('        write_user_expression(expression[0], expression[1])\n\n')
        f.write(f'    add_sketch({control_point_list_string}, {str(curve_orders)})\n\n')
        # f.write('    create_connected_lines_from_points([[0.0, 0.0, 0.0], [0.2, 0.0, 0.2], '
        #         '[0.5, 0.0, 0.0]])\n\n\n')
        f.write('if __name__ == \'__main__\':\n')
        f.write('    main()\n')


def main(mea: MEA, output_file_path: str):
    write_import_statements_to_file(output_file_path)
    dump_journal_functions_to_file(output_file_path)
    equations = write_parameters_to_equations(mea.param_dict)
    dump_equations_to_file(equations, output_file_path, mea)


if __name__ == '__main__':
    mea_ = MEA()
    output_file_path = 'C:\\Users\\mlauer2\\Documents\\test_journal.py'
    main(mea_, output_file_path)
