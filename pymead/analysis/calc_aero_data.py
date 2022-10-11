import subprocess
import os
from pymead.analysis.read_aero_data import read_Cl_from_file_panel_fort, read_Cp_from_file_panel_fort, \
    read_aero_data_from_xfoil, read_Cp_from_file_xfoil
from pymead.core.airfoil import Airfoil
from pymead.core.base_airfoil_params import BaseAirfoilParams
from pymead.core.param import Param
from pymead import DATA_DIR
import re
import time


def calculate_aero_data(airfoil_coord_dir: str, airfoil_name: str, alpha, airfoil: Airfoil,
                        tool: str = 'panel_fort', xfoil_settings: dict = None, export_Cp: bool = True):
    tool_list = ['panel_fort', 'xfoil', 'mses']
    if tool not in tool_list:
        raise ValueError(f"\'tool\' must be one of {tool_list}")
    airfoil.get_coords(body_fixed_csys=True)
    if airfoil.check_self_intersection():
        raise GeometryError("Airfoil is self-intersecting")
    aero_data = {}
    base_dir = os.path.join(airfoil_coord_dir, airfoil_name)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if tool == 'panel_fort':
        f = os.path.join(base_dir, airfoil_name + '.dat')
        n_data_pts = airfoil.write_coords_to_file(f, 'w', body_fixed_csys=True)
        subprocess.run((["panel_fort", airfoil_coord_dir, airfoil_name + '.dat', str(n_data_pts - 1), str(alpha)]),
                       stdout=subprocess.DEVNULL)  # stdout=subprocess.DEVNULL suppresses output to console
        aero_data['Cl'] = read_Cl_from_file_panel_fort(os.path.join(airfoil_coord_dir, 'LIFT.dat'))
        if export_Cp:
            aero_data['Cp'] = read_Cp_from_file_panel_fort(os.path.join(airfoil_coord_dir, 'CPLV.DAT'))
        return aero_data
    elif tool == 'xfoil':
        if xfoil_settings is None:
            raise ValueError(f"\'xfoil_settings\' must be set if \'xfoil\' tool is selected")
        f = os.path.join(base_dir, airfoil_name + ".dat")
        # print(repr(f).replace('\\\\', '\\\\\\\\'))
        airfoil.write_coords_to_file(f, 'w', body_fixed_csys=True)
        xfoil_input_file = os.path.join(base_dir, 'xfoil_input.txt')
        xfoil_input_list = ['', 'oper', f'iter {xfoil_settings["iter"]}', 'visc', str(xfoil_settings['Re'])]
        if not isinstance(alpha, list):
            alpha = [alpha]
        for idx, alf in enumerate(alpha):
            xfoil_input_list.append('alfa ' + str(alf))
        if export_Cp:
            xfoil_input_list.append('cpwr ' + f"{airfoil_name}_Cp.dat")
        xfoil_input_list.append('')
        xfoil_input_list.append('quit')
        write_input_file(xfoil_input_file, xfoil_input_list)
        xfoil_log = os.path.join(base_dir, 'xfoil.log')
        # with open(xfoil_log, 'wb') as h:
        with open(xfoil_input_file, 'r') as g:
            process = subprocess.Popen(['xfoil', f"{airfoil_name}.dat"], stdin=g, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, cwd=base_dir, shell=False)
            aero_data['converged'] = False
            aero_data['timed_out'] = False
            aero_data['errored_out'] = False
            try:
                # print(f"communicating")
                outs, errs = process.communicate(timeout=xfoil_settings['timeout'])
                # print(f"done communicating")
                with open(xfoil_log, 'wb') as h:
                    h.write('Output:\n'.encode('utf-8'))
                    h.write(outs)
                    h.write('\nErrors:\n'.encode('utf-8'))
                    h.write(errs)
                aero_data['timed_out'] = False
                aero_data['converged'] = True
            except subprocess.TimeoutExpired:
                process.kill()
                outs, errs = process.communicate()
                with open(xfoil_log, 'wb') as h:
                    h.write('After timeout, \nOutput: \n'.encode('utf-8'))
                    h.write(outs)
                    h.write('\nErrors:\n'.encode('utf-8'))
                    h.write(errs)
                aero_data['timed_out'] = True
                aero_data['converged'] = False
            finally:
                if not aero_data['timed_out']:
                    # time.sleep(3)
                    line1, line2 = read_aero_data_from_xfoil(xfoil_log, aero_data)
                    if line1 is not None:
                        convert_xfoil_string_to_aero_data(line1, line2, aero_data)
                        if export_Cp:
                            aero_data['Cp'] = read_Cp_from_file_xfoil(os.path.join(base_dir, f"{airfoil_name}_Cp.dat"))

        return aero_data, xfoil_log


def write_input_file(input_file: str, input_list: list):
    with open(input_file, 'w') as f:
        for input_ in input_list:
            f.write(input_)
            f.write('\n')


def convert_xfoil_string_to_aero_data(line1: str, line2: str, aero_data: dict):
    new_str = line1.replace(' ', '') + line2.replace(' ', '')
    new_str = new_str.replace('=>', '')
    appending = False
    data_list = []
    for ch in new_str:
        if ch.isdigit() or ch == '.' or ch == '-':
            if appending:
                data_list[-1] += ch
        else:
            appending = False
        last_ch = ch
        if last_ch == '=' and not ch == '>':
            appending = True
            data_list.append('')
    aero_data['Cm'] = float(data_list[0])
    aero_data['Cd'] = float(data_list[1])
    aero_data['Cdf'] = float(data_list[2])
    aero_data['Cdp'] = float(data_list[3])
    aero_data['Cl'] = float(data_list[5])
    aero_data['L/D'] = aero_data['Cl'] / aero_data['Cd']
    return aero_data


class GeometryError(Exception):
    pass


if __name__ == '__main__':
    d = DATA_DIR
    airfoil = Airfoil(base_airfoil_params=BaseAirfoilParams(R_le=Param(0.04), L_le=Param(0.12)))
    a_name = "test_airfoil"
    xfoil_settings = {'Re': 1e5, 'timeout': 15, 'iter': 150}
    # line1, line2 = read_aero_data_from_xfoil(os.path.join(d, a_name, 'xfoil.log'), {})
    aero_data, xfoil_log = calculate_aero_data(d, a_name, [2.0], airfoil, 'xfoil', xfoil_settings)
    print(aero_data)
    # print(xfoil_log)
    pass