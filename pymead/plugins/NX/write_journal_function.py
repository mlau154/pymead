import math
import os
import typing
import numpy as np

import benedict
from pymead.core.mea import MEA
from pymead.core.param import Param


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


def write_airfoil_info_to_file(mea: MEA, file_path: str):
    airfoil_info = {}
    for airfoil_name, airfoil in mea.airfoils.items():
        control_point_array = airfoil.control_point_array
        control_point_array = np.insert(control_point_array, 1, 0.0, axis=1)
        control_point_list = control_point_array.tolist()
        curve_orders = [val for val in airfoil.N.values()]
        line_tags = [['pt2pt_linear', 'chord_line'], ['pt2pt_linear', 'te_thickness_upper']]
        for idx, cp in enumerate(airfoil.control_points[:-1]):
            print(f"cp tag = {cp.tag}")
            if any(substr in cp.tag for substr in ['te_1', 'g1_plus', 'g1_minus', 'g2_plus', 'g2_minus', 'le']) and \
                    'te_1_g1_plus' not in cp.tag and 'le_g2_plus' not in cp.tag:
                line_tags.append(['pt2pt_linear'])
            else:
                line_tags.append([])
        for idx, cp in enumerate(airfoil.control_points[:-1]):
            if cp.tag == 'te_1':
                line_tags[idx + 2].append('theta_te_upper')
            elif cp.tag == 'anchor_point_te_2_g1_minus':
                line_tags[idx + 2].append('theta_te_lower')
            elif cp.tag == 'anchor_point_le_g2_minus':
                line_tags[idx + 2].append('psi1_le')
            elif cp.tag == 'anchor_point_le_g1_plus':
                line_tags[idx + 2].append('psi2_le')
            elif cp.tag == 'anchor_point_le_g1_minus':
                line_tags[idx + 2].append('le_angle_180')
        line_tags.append(['pt2pt_linear', 'te_thickness_lower'])
        print(f"line tags = {line_tags}")
        assert len(line_tags) == len(control_point_array) + 2
        airfoil_info[airfoil_name] = {
            'control_point_list_string': control_point_list,
            'curve_orders': curve_orders,
            'line_tags': line_tags,
            'dx': airfoil.dx.value,
            'dy': airfoil.dy.value,
            'c': airfoil.c.value,
            'alf': airfoil.alf.value,
        }
    with open(file_path, 'a') as f:
        f.write('def fetch_airfoil_info_for_NX():\n')
        f.write('    airfoil_info = {\n')
        for airfoil_name, airfoil_info_single in airfoil_info.items():
            f.write(f'        \'{airfoil_name}\': ')
            f.write('{\n')
            for k, v in airfoil_info_single.items():
                f.write(f'            \'{k}\': {str(v)},\n')
            f.write('        }\n')
        f.write('    }\n')
        f.write('    return airfoil_info\n\n\n')


def dump_equations_to_file(equations: typing.List[tuple], file_path: str, mea: MEA):
    with open(file_path, 'a') as f:
        f.write('\n\ndef main():\n')
        f.write('    user_expressions = [\n')
        for equation in equations:
            if equation[1] is None:
                quote1 = ""
                quote2 = ""
            else:
                quote1 = "\""
                quote2 = "\""
            f.write(f"        (\"{equation[0]}\", {quote1}{equation[1]}{quote2}),\n")
        f.write('    ]\n\n')
        f.write('    for expression in user_expressions:\n')
        f.write('        write_user_expression(expression[0], expression[1])\n\n')
        f.write(f'    airfoil_data = fetch_airfoil_info_for_NX()\n')
        f.write(f'    add_sketch(airfoil_data)\n\n')
        f.write('if __name__ == \'__main__\':\n')
        f.write('    main()\n')


def main(mea: MEA, output_file_path: str):
    write_import_statements_to_file(output_file_path)
    write_airfoil_info_to_file(mea, output_file_path)
    dump_journal_functions_to_file(output_file_path)
    equations = write_parameters_to_equations(mea.param_dict)
    dump_equations_to_file(equations, output_file_path, mea)


if __name__ == '__main__':
    from pymead.core.airfoil import Airfoil
    from pymead.core.base_airfoil_params import BaseAirfoilParams
    airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(c=Param(100.0), t_te=Param(0.02), alf=Param(0.1),
                                                            dx=Param(0.5), dy=Param(0.5)))
    mea_ = MEA(airfoils=[airfoil])
    output_file_path = 'C:\\Users\\mlauer2\\Documents\\test_journal.py'
    main(mea_, output_file_path)
